"""
tools/da_pipeline.py
====================
Model-agnostic domain-adaptation evaluation protocol.

A detector plugs in by subclassing ``DetectorAdapter`` (see ``rtdetr_metrics.py``
for RT-DETRv4). Everything else -- threshold selection on the source validation
split, per-target metrics, confidence statistics, embeddings, t-SNE, MMD, CSV
export and W&B logging -- lives here once and is shared by every model.

Protocol (one source checkpoint)
--------------------------------
1. Source ``val`` split  -> confidence curves -> ``best_f2_conf`` threshold.
2. Every target ``test`` split (source included) -> detection metrics at that
   threshold, confidence statistics and image embeddings.
3. Source ``train`` split (optional) -> embeddings.
4. Combined t-SNE + pairwise MMD (source test vs each target test).

Sections
--------
1.  Adapter interface (SplitOutput, DetectorAdapter)
2.  Dataset config helpers
3.  Single-split evaluation (evaluate_split)
4.  Multi-dataset orchestration (evaluate_source_against_targets)
5.  Top-level orchestration (run_eval, run_all_sources)
6.  CLI helpers
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tools.metrics_common import (
    calculate_conf_curves,
    calculate_metrics,
    compute_confidence_stats,
    compute_domain_gap_mmd,
    get_dataset_plot_label,
    log_dataset_config,
    log_metrics_table,
    log_results_table,
    log_target_curves,
    log_target_scalars,
    plot_labeled_tsne,
    save_results_csv,
)


# ---------------------------------------------------------------------------
# 1. Adapter Interface
# ---------------------------------------------------------------------------

@dataclass
class SplitOutput:
    """Everything a detector produces for one dataset split.

    predictions : COCO results format
        [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]
        with ``image_id``/``category_id`` matching the split's ``ann_file``.
    coco_gt     : pycocotools COCO object, or None to let the pipeline load it
                  from the split's ``ann_file``.
    image_ids / file_names / embeddings : row-aligned; ``embeddings`` is
                  (num_images, feature_dim) or None when not requested.
    """
    predictions: list[dict] = field(default_factory=list)
    coco_gt: Any = None
    image_ids: list[int] = field(default_factory=list)
    file_names: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None


class DetectorAdapter(ABC):
    """Model-specific inference behind the shared evaluation protocol.

    Subclasses implement ``_load`` (weights for one source checkpoint) and
    ``run_split`` (one inference pass producing predictions and/or embeddings).
    Model-level settings (device, batch size, config files, ...) belong in the
    subclass constructor, not in ``dataset_configs``.
    """

    name: str = "detector"   # human-readable, used in plot titles / W&B config
    slug: str = "detector"   # short identifier, used in W&B tags

    def __init__(self) -> None:
        self.checkpoint_path: str | None = None

    def load(self, checkpoint_path: str | Path) -> None:
        """Load the weights of one source checkpoint (called once per source)."""
        self._load(str(checkpoint_path))
        self.checkpoint_path = str(checkpoint_path)

    @abstractmethod
    def _load(self, checkpoint_path: str) -> None:
        ...

    @abstractmethod
    def run_split(
        self,
        split: dict,
        *,
        min_score: float = 0.001,
        with_predictions: bool = True,
        with_embeddings: bool = False,
        verbose: bool = True,
    ) -> SplitOutput:
        """Run inference on ``split`` ({"img_folder", "ann_file"}) in a single pass."""
        ...

    def describe(self) -> dict[str, Any]:
        """Model metadata stored in every result row and in the W&B run config."""
        return {"model_name": self.name, "checkpoint_path": self.checkpoint_path}


# ---------------------------------------------------------------------------
# 2. Dataset Config Helpers
# ---------------------------------------------------------------------------

def get_split(dataset_cfg: dict, split_name: str, dataset_name: str = "") -> dict:
    """Return ``dataset_cfg['splits'][split_name]`` or raise a descriptive error."""
    splits = dataset_cfg.get("splits")
    split = splits.get(split_name) if isinstance(splits, dict) else None
    if not isinstance(split, dict) or "img_folder" not in split or "ann_file" not in split:
        raise ValueError(
            f"Dataset '{dataset_name}' must define splits['{split_name}'] with 'img_folder' and 'ann_file'"
        )
    return split


def _split_available(dataset_cfg: dict, split_name: str) -> dict | None:
    """Return the split if it is configured and exists on disk, else None."""
    try:
        split = get_split(dataset_cfg, split_name)
    except ValueError:
        return None
    if os.path.exists(split["img_folder"]) and os.path.exists(split["ann_file"]):
        return split
    return None


def _ensure_coco_gt(output: SplitOutput, split: dict) -> Any:
    if output.coco_gt is None:
        from pycocotools.coco import COCO
        output.coco_gt = COCO(str(split["ann_file"]))
    return output.coco_gt


# ---------------------------------------------------------------------------
# 3. Single-Split Evaluation (evaluate_split)
# ---------------------------------------------------------------------------

def evaluate_split(
    adapter: DetectorAdapter,
    dataset_cfg: dict,
    conf_threshold: float,
    run_name: str = "",
    output_dir: str | Path | None = None,
    extract_embeddings_flag: bool = True,
    verbose: bool = True,
) -> tuple[dict, np.ndarray | None, list[str]]:
    """Evaluate the loaded model on a dataset's ``test`` split.

    Runs one inference pass (predictions + optional embeddings), then detection
    metrics at ``conf_threshold`` and confidence statistics.

    Returns
    -------
    tuple[dict, np.ndarray | None, list[str]]
        (result_dict, embeddings, file_names)
    """
    test_split = get_split(dataset_cfg, "test")
    iou_match = float(dataset_cfg.get("iou", 0.50))
    min_score = float(dataset_cfg.get("min_score", 0.001))
    out_dir_path = Path(output_dir) if output_dir is not None else None

    output = adapter.run_split(
        test_split,
        min_score=min_score,
        with_predictions=True,
        with_embeddings=extract_embeddings_flag,
        verbose=verbose,
    )
    predictions = output.predictions
    coco_gt = _ensure_coco_gt(output, test_split)

    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)
        with open(out_dir_path / "predictions.json", "w") as fh:
            json.dump(predictions, fh)
        if output.embeddings is not None:
            np.save(out_dir_path / "embeddings.npy", output.embeddings)

    det_metrics = calculate_metrics(
        pred_list=predictions,
        coco_gt=coco_gt,
        iou_match=iou_match,
        conf_threshold=conf_threshold,
        output_dir=out_dir_path,
    )
    conf_metrics = compute_confidence_stats(predictions)
    th_conf_metrics = compute_confidence_stats(predictions, threshold=conf_threshold, prefix="th")

    result = {
        "run_name": run_name,
        **adapter.describe(),
        "img_folder": test_split["img_folder"],
        "ann_file": test_split["ann_file"],
        **det_metrics,
        **conf_metrics,
        **th_conf_metrics,
    }
    return result, output.embeddings, output.file_names


# ---------------------------------------------------------------------------
# 4. Multi-Dataset Orchestration (evaluate_source_against_targets)
# ---------------------------------------------------------------------------

def select_threshold_on_source_val(
    adapter: DetectorAdapter,
    dataset_configs: dict,
    source_name: str,
    result_folder: Path,
    run: Any = None,
    verbose: bool = True,
) -> float:
    """Compute confidence curves on the source ``val`` split and return ``best_f2_conf``."""
    source_cfg = dataset_configs[source_name]
    val_split = get_split(source_cfg, "val", source_name)

    if verbose:
        print(f"[*] Calculating confidence curves on source validation split: {source_name}")
    output = adapter.run_split(
        val_split,
        min_score=float(source_cfg.get("min_score", 0.001)),
        with_predictions=True,
        with_embeddings=False,
        verbose=verbose,
    )
    curves = calculate_conf_curves(
        pred_list=output.predictions,
        coco_gt=_ensure_coco_gt(output, val_split),
        iou_match=source_cfg.get("iou", 0.50),
        output_dir=result_folder / source_name / "val",
    )
    if run is not None:
        log_target_curves(run, f"{source_name}/val", curves["curve_data"])
    threshold = curves["best_f2_conf"]
    if verbose:
        print(f"[*] Source validation best_f2_conf: {threshold:.4f}")
    return threshold


def evaluate_source_against_targets(
    adapter: DetectorAdapter,
    checkpoint_path: str,
    dataset_configs: dict,
    source_name: str,
    target_names: list[str] | None = None,
    result_path: str | Path = "results",
    extract_embeddings_flag: bool = True,
    run: Any = None,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Evaluate one source checkpoint across multiple target datasets.

    Extracts multi-group embeddings (including optional source_train) and generates
    a combined t-SNE visualization and pairwise domain gap (MMD) metrics.
    """
    result_folder = Path(result_path)
    result_folder.mkdir(parents=True, exist_ok=True)
    target_names = target_names or list(dataset_configs.keys())

    adapter.load(checkpoint_path)
    model_info = adapter.describe()

    if run is not None:
        log_dataset_config(
            run=run,
            dataset_configs=dataset_configs,
            source_name=source_name,
            config_path=model_info.get("config_path"),
            checkpoint_path=model_info.get("checkpoint_path"),
            model_name=model_info.get("model_name"),
        )

    source_conf_threshold = select_threshold_on_source_val(
        adapter, dataset_configs, source_name, result_folder, run=run, verbose=verbose,
    )

    results: list[dict] = []
    embeddings_by_dataset: dict[str, np.ndarray] = {}

    # Evaluate each target test split
    for target_name in target_names:
        cfg_t = dataset_configs[target_name]

        if verbose:
            print(f"\n[*] Evaluating: source={source_name} -> target={target_name} ({cfg_t.get('label', '')})")

        result, emb, _ = evaluate_split(
            adapter,
            dataset_cfg=cfg_t,
            conf_threshold=source_conf_threshold,
            run_name=f"source{source_name}_to_{target_name}",
            output_dir=result_folder / target_name,
            extract_embeddings_flag=extract_embeddings_flag,
            verbose=verbose,
        )

        result["source_dataset_name"] = source_name
        result["source_dataset_label"] = dataset_configs[source_name].get("label", source_name)
        result["target_dataset_name"] = target_name
        result["target_dataset_label"] = cfg_t.get("label", target_name)
        results.append(result)

        if emb is not None:
            embeddings_by_dataset[target_name] = emb
            np.save(result_folder / f"embeddings_{source_name}_to_{target_name}.npy", emb)

        if run is not None:
            log_target_scalars(run, target_name, result)

    # Optional: source training embeddings for complete domain visualization
    src_train_split = _split_available(dataset_configs[source_name], "train")
    if extract_embeddings_flag and src_train_split is not None:
        if verbose:
            print(f"[*] Extracting embeddings for source train split: {source_name}")
        output = adapter.run_split(
            src_train_split, with_predictions=False, with_embeddings=True, verbose=verbose,
        )
        if output.embeddings is not None:
            np.save(result_folder / f"embeddings_{source_name}_train.npy", output.embeddings)
            embeddings_by_dataset["source_train"] = output.embeddings

    # Save summary CSV
    csv_path = result_folder / f"source{source_name}_results.csv"
    save_results_csv(results, str(csv_path))

    if run is not None:
        log_metrics_table(run, results, table_name="Table 1: Metrics")
        log_results_table(run, results, table_name="results_table")
        try:
            import wandb
            csv_art = wandb.Artifact(f"source{source_name}_results", type="evaluation")
            csv_art.add_file(str(csv_path))
            run.log_artifact(csv_art)
        except ImportError:
            pass

    # Single combined t-SNE (source train + every dataset's test split) and Domain Gap (MMD)
    if extract_embeddings_flag and source_name in embeddings_by_dataset:
        _log_embedding_space(
            adapter, dataset_configs, source_name, target_names, embeddings_by_dataset, run, verbose,
        )

    return results, embeddings_by_dataset


def _log_embedding_space(
    adapter: DetectorAdapter,
    dataset_configs: dict,
    source_name: str,
    target_names: list[str],
    embeddings_by_dataset: dict[str, np.ndarray],
    run: Any,
    verbose: bool,
) -> None:
    source_label = dataset_configs[source_name].get("label", source_name)
    tsne_entries: list[dict[str, Any]] = []

    if "source_train" in embeddings_by_dataset:
        tsne_entries.append({
            "dataset_label": get_dataset_plot_label(source_label, "train"),
            "embeddings": embeddings_by_dataset["source_train"],
            "marker": "o",
        })

    tsne_entries.append({
        "dataset_label": get_dataset_plot_label(source_label, "test"),
        "embeddings": embeddings_by_dataset[source_name],
        "marker": "D",
    })

    for target_name in target_names:
        if target_name != source_name and target_name in embeddings_by_dataset:
            target_label = dataset_configs[target_name].get("label", target_name)
            tsne_entries.append({
                "dataset_label": get_dataset_plot_label(target_label, "test"),
                "embeddings": embeddings_by_dataset[target_name],
                "marker": "D",
            })

    if run is not None:
        tsne_image = plot_labeled_tsne(
            tsne_entries,
            title=f"{adapter.name} Embeddings: {source_label} vs Targets (t-SNE)",
        )
        run.log({"tsne": tsne_image})

    # Pairwise MMD calculation (source test vs target test)
    for target_name in target_names:
        if target_name == source_name or target_name not in embeddings_by_dataset:
            continue
        gap = compute_domain_gap_mmd(
            embeddings_by_dataset[source_name],
            embeddings_by_dataset[target_name],
        )
        if verbose:
            print(f"[*] Domain Gap MMD ({source_name} -> {target_name}): {gap:.6f}")
        if run is not None:
            run.summary[f"{target_name}/domain_gap_mmd"] = gap


# ---------------------------------------------------------------------------
# 5. Top-Level Orchestration (run_eval, run_all_sources)
# ---------------------------------------------------------------------------

def init_wandb_run(
    run_name: str,
    wandb_project: str | None,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
) -> Any:
    """Start a W&B eval run, or return None if W&B is disabled/unavailable."""
    if wandb_project is None:
        return None

    # Load .env credentials if present
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.expanduser("~/.env"))
    except ImportError:
        pass

    if not os.environ.get("WANDB_API_KEY"):
        print("[!] WANDB_API_KEY not found in environment or ~/.env — W&B logging disabled.")
        return None

    try:
        import wandb
        wandb.login()
        return wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type="eval",
            tags=wandb_tags,
            name=f"eval_{run_name}",
            group=run_name,
        )
    except Exception as exc:
        print(f"[!] W&B init failed (continuing without it): {exc}")
        return None


def run_eval(
    adapter: DetectorAdapter,
    run_name: str,
    checkpoint_path: str,
    dataset_configs: dict,
    source_name: str,
    result_folder: str | Path = "results",
    extract_embeddings_flag: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
) -> list[dict]:
    """Evaluate one source checkpoint against all datasets, with optional W&B logging."""
    run = init_wandb_run(
        run_name,
        wandb_project,
        wandb_entity,
        wandb_tags or [f"{adapter.slug}_metrics"],
    )

    try:
        results, _ = evaluate_source_against_targets(
            adapter,
            checkpoint_path=checkpoint_path,
            dataset_configs=dataset_configs,
            source_name=source_name,
            result_path=result_folder,
            extract_embeddings_flag=extract_embeddings_flag,
            run=run,
        )
    finally:
        if run is not None:
            run.finish()

    return results


def run_all_sources(
    adapter: DetectorAdapter,
    checkpoints: dict[str, str],
    dataset_configs: dict,
    result_root: str | Path = "results",
    extract_embeddings_flag: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Full cross-domain matrix: for each ``source -> checkpoint``, evaluate on every dataset.

    Outputs go to ``{result_root}/source_{label}``; one W&B run per source.
    """
    all_results: dict[str, list[dict]] = {}
    for source_name, checkpoint_path in checkpoints.items():
        label = dataset_configs[source_name].get("label", source_name)
        all_results[source_name] = run_eval(
            adapter,
            run_name=f"{label}_metrics",
            checkpoint_path=checkpoint_path,
            dataset_configs=dataset_configs,
            source_name=source_name,
            result_folder=Path(result_root) / f"source_{label}",
            extract_embeddings_flag=extract_embeddings_flag,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
        )
    return all_results


# ---------------------------------------------------------------------------
# 6. CLI Helpers
# ---------------------------------------------------------------------------

def build_common_cli_parser(description: str) -> argparse.ArgumentParser:
    """Arguments shared by every model's CLI (single source dataset)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--checkpoint", "-r", required=True, help="Path to model checkpoint")
    p.add_argument("--img-folder", required=True, help="Path to test images directory")
    p.add_argument("--ann-file", required=True, help="Path to test COCO annotations (.json)")
    p.add_argument("--val-img-folder", required=True, help="Path to validation images (threshold selection)")
    p.add_argument("--val-ann-file", required=True, help="Path to validation COCO annotations (.json)")
    p.add_argument("--output-dir", default="eval_output", help="Directory to save outputs")
    p.add_argument("--device", default=None, help="Torch device (default: cuda:0 if available)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--iou-match", type=float, default=0.50, help="IoU threshold for curves and fixed operating point")
    p.add_argument("--min-score", type=float, default=0.001, help="Filter predictions below score")
    p.add_argument("--extract-embeddings", action="store_true", help="Extract and save image embeddings")
    p.add_argument("--run-name", default="cli_eval", help="Evaluation run name")
    p.add_argument("--dataset-label", default="dataset", help="Human-readable label for dataset")
    p.add_argument("--source-name", default="A", help="Dataset identifier key")
    p.add_argument("--wandb-project", default=None, help="Optional W&B project name")
    p.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team name")
    return p


def dataset_configs_from_args(args: argparse.Namespace) -> dict:
    return {
        args.source_name: {
            "label": args.dataset_label,
            "splits": {
                "val": {"ann_file": args.val_ann_file, "img_folder": args.val_img_folder},
                "test": {"ann_file": args.ann_file, "img_folder": args.img_folder},
            },
            "iou": args.iou_match,
            "min_score": args.min_score,
        }
    }


def run_cli(adapter: DetectorAdapter, args: argparse.Namespace) -> list[dict]:
    """Run a single-dataset evaluation from parsed common CLI args and print a summary."""
    results = run_eval(
        adapter,
        run_name=args.run_name,
        checkpoint_path=args.checkpoint,
        dataset_configs=dataset_configs_from_args(args),
        source_name=args.source_name,
        result_folder=args.output_dir,
        extract_embeddings_flag=args.extract_embeddings,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    print_results_summary(results)
    print(f"[✓] All evaluation outputs saved to: {os.path.abspath(args.output_dir)}")
    return results


_SUMMARY_SKIP_KEYS = {
    "run_name", "model_name", "config_path", "checkpoint_path", "ann_file", "img_folder",
    "all_iou_thresholds", "precision_per_class", "recall_per_class",
    "source_dataset_name", "source_dataset_label", "target_dataset_name", "target_dataset_label",
}


def print_results_summary(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY")
    print("=" * 60)
    for res in results:
        target = res.get("target_dataset_name", "?")
        label = res.get("target_dataset_label", "")
        print(f"\n  Target: {target} ({label})")
        for k, v in res.items():
            if k in _SUMMARY_SKIP_KEYS:
                continue
            if isinstance(v, float):
                print(f"    {k:<28s}: {v:.4f}")
            elif isinstance(v, (int, str)):
                print(f"    {k:<28s}: {v}")
    print("=" * 60 + "\n")
