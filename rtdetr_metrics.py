"""
rtdetr_metrics.py
=================
Domain-adaptation-relevant metrics for RT-DETRv4 models.

Mirrors the structure and public API of ``da_metrics.py`` (the Ultralytics/YOLO
counterpart) while integrating with RT-DETRv4 YAML configs, PyTorch checkpoints,
and COCO annotations.

Sections
--------
1.  Model loading & Category mapping
2.  Embedding hook utilities
3.  Standalone prediction generation (rtdetr_predict)
4.  Standalone embedding extraction (calculate_embeddings)
5.  Confidence summary
6.  Detection metrics wrapper (evaluate_detection_metrics)
7.  Single-dataset pipeline (run_full_evaluation)
8.  Multi-dataset orchestration (evaluate_source_against_targets)
9.  Top-level orchestration (run_eval)
10. CLI entry-point
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Shared metrics utilities
from tools.metrics_common import (
    calculate_metrics,
    compute_domain_gap_mmd,
    compute_improvement,
    get_dataset_plot_label,
    log_dataset_config,
    log_metrics_table,
    log_operating_points_table,
    log_results_table,
    log_target_curves,
    log_target_plots,
    log_target_scalars,
    plot_labeled_tsne,
    save_results_csv,
    calculate_conf_curves,
)


# ---------------------------------------------------------------------------
# 1. Model Loading & Category Mapping
# ---------------------------------------------------------------------------

MSCOCO_CATEGORY2LABEL: dict[int, int] = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
    14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20,
    23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
    44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47,
    54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56,
    63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
    86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}
MSCOCO_LABEL2CATEGORY: dict[int, int] = {v: k for k, v in MSCOCO_CATEGORY2LABEL.items()}


def get_label_to_category_map(coco_gt: Any, remap_mscoco: bool = False) -> dict[int, int]:
    """
    Map model predicted class index (0..num_classes-1) to dataset category_id.
    - If remap_mscoco is True and category IDs match MS COCO (80 classes, max 90): uses MSCOCO mapping.
    - For custom datasets (e.g. single-class 0:swimmer): 0th model output maps to the first sorted category id.
    """
    cat_ids = sorted(coco_gt.getCatIds())
    if remap_mscoco and len(cat_ids) == 80 and max(cat_ids) == 90:
        return MSCOCO_LABEL2CATEGORY
    return {idx: cat_id for idx, cat_id in enumerate(cat_ids)}


def get_yaml_config(repo_root: str = "."):
    """Locate and return the YAMLConfig class from RT-DETRv4 repository."""
    sys.path.insert(0, os.path.abspath(repo_root))
    for mod in ("engine.core", "src.core"):
        try:
            m = __import__(mod, fromlist=["YAMLConfig"])
            return m.YAMLConfig
        except ImportError:
            continue
    raise ImportError("Could not import YAMLConfig from engine.core or src.core.")


def load_rtdetr_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    repo_root: str = ".",
    img_folder: str | None = None,
    ann_file: str | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Load RT-DETRv4 model, postprocessor, dataloader, and ground truth.

    Returns
    -------
    tuple
        (model, postprocessor, val_dataloader, coco_gt, cfg)
    """
    device = torch.device(device)
    YAMLConfig = get_yaml_config(repo_root)
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    # Override dataset paths and batch size in config
    if "val_dataloader" in cfg.yaml_cfg:
        ds_cfg = cfg.yaml_cfg["val_dataloader"]
        if img_folder is not None:
            ds_cfg["dataset"]["img_folder"] = img_folder
        if ann_file is not None:
            ds_cfg["dataset"]["ann_file"] = ann_file
        ds_cfg["total_batch_size"] = batch_size
        ds_cfg["num_workers"] = num_workers

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state_dict)

    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    val_dataloader = cfg.val_dataloader
    coco_gt = val_dataloader.dataset.coco

    return model, postprocessor, val_dataloader, coco_gt, cfg


# ---------------------------------------------------------------------------
# 2. Embedding Hook Utilities
# ---------------------------------------------------------------------------

class EmbeddingHook:
    """Forward hook capturing feature representations."""

    def __init__(self):
        self.features: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        self.features = out


def pool_features(feat_output) -> torch.Tensor:
    """Pool arbitrary layer feature outputs to flat [B, C] vectors."""
    feats = feat_output if isinstance(feat_output, (list, tuple)) else [feat_output]
    pooled = []
    for f in feats:
        if isinstance(f, torch.Tensor):
            if f.dim() == 4:  # [B, C, H, W] -> GAP to [B, C]
                pooled.append(F.adaptive_avg_pool2d(f, (1, 1)).flatten(1))
            elif f.dim() == 3:  # [B, N, C] -> Mean to [B, C]
                pooled.append(f.mean(dim=1))
    if not pooled:
        raise ValueError("Could not pool features from hooked layer.")
    return torch.cat(pooled, dim=1)


# ---------------------------------------------------------------------------
# 3. Standalone Prediction Generation (rtdetr_predict)
# ---------------------------------------------------------------------------

def rtdetr_predict(
    config_path: str,
    checkpoint_path: str,
    img_folder: str,
    ann_file: str,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str | torch.device = "cuda:0",
    min_score: float = 0.001,
    remap_mscoco: bool = False,
    output_predictions_path: str | Path | None = None,
    repo_root: str = ".",
    verbose: bool = True,
) -> tuple[list[dict], Any, list[int]]:
    """Generate model predictions on a dataset split in COCO format.

    Parameters
    ----------
    config_path : str
        Path to RT-DETRv4 YAML config.
    checkpoint_path : str
        Path to model weights checkpoint.
    img_folder : str
        Path to images directory.
    ann_file : str
        Path to COCO annotations JSON.
    batch_size : int
    num_workers : int
    device : str | torch.device
    min_score : float
        Filter detections below this score threshold.
    remap_mscoco : bool
        Whether to apply standard MS COCO category remapping. False for 1-class swimmer.
    output_predictions_path : str | Path | None
        Optional file path to save predictions to JSON.
    repo_root : str
    verbose : bool

    Returns
    -------
    tuple[list[dict], Any, list[int]]
        (predictions, coco_gt, image_ids)
    """
    model, postprocessor, val_dataloader, coco_gt, _ = load_rtdetr_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        repo_root=repo_root,
        img_folder=img_folder,
        ann_file=ann_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    label_to_cat = get_label_to_category_map(coco_gt, remap_mscoco=remap_mscoco)

    try:
        from tqdm import tqdm
        progress = tqdm(val_dataloader, desc="Predicting", leave=False)
    except ImportError:
        progress = val_dataloader

    predictions: list[dict] = []
    image_ids: list[int] = []
    dev = torch.device(device)

    with torch.no_grad():
        for samples, targets in progress:
            samples = samples.to(dev)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(dev)
            batch_img_ids = [int(t["image_id"]) for t in targets]
            image_ids.extend(batch_img_ids)

            outputs = model(samples)
            results = postprocessor(outputs, orig_sizes)

            for img_id, res in zip(batch_img_ids, results):
                labels = res["labels"].detach().cpu().numpy()
                boxes = res["boxes"].detach().cpu().numpy()
                scores = res["scores"].detach().cpu().numpy()

                for lbl, box, score in zip(labels, boxes, scores):
                    if score < min_score:
                        continue
                    cat_id = label_to_cat.get(int(lbl), int(lbl))
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    })

    if verbose:
        print(f"[*] Total predictions collected: {len(predictions)}")

    if output_predictions_path is not None:
        p_path = Path(output_predictions_path)
        p_path.parent.mkdir(parents=True, exist_ok=True)
        with open(p_path, "w") as fh:
            json.dump(predictions, fh)
        if verbose:
            print(f"[*] Predictions saved to: {p_path}")

    return predictions, coco_gt, image_ids


# ---------------------------------------------------------------------------
# 4. Standalone Embedding Extraction (calculate_embeddings)
# ---------------------------------------------------------------------------

def calculate_embeddings(
    config_path: str,
    checkpoint_path: str,
    img_folder: str,
    ann_file: str,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str | torch.device = "cuda:0",
    embedding_module: str = "encoder",
    output_path: str | Path | None = None,
    repo_root: str = ".",
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Extract per-image feature embeddings using a forward hook.

    Parameters
    ----------
    config_path : str
    checkpoint_path : str
    img_folder : str
    ann_file : str
    batch_size : int
    num_workers : int
    device : str | torch.device
    embedding_module : str
        Target module name to hook (e.g., 'encoder', 'hybrid_encoder', 'backbone').
    output_path : str | Path | None
        Optional file path to save embeddings (.npy or .npz).
    repo_root : str
    verbose : bool

    Returns
    -------
    tuple[np.ndarray, list[str], list[int]]
        (embeddings, file_names, image_ids)
    """
    model, _, val_dataloader, coco_gt, _ = load_rtdetr_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        repo_root=repo_root,
        img_folder=img_folder,
        ann_file=ann_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Locate target module
    submodules = dict(model.named_modules())
    target_mod = submodules.get(embedding_module)
    if target_mod is None:
        for cand in ("encoder", "hybrid_encoder", "backbone"):
            if cand in submodules:
                target_mod = submodules[cand]
                embedding_module = cand
                break
    if target_mod is None:
        raise ValueError(
            f"Module '{embedding_module}' not found in model. Available: {list(submodules.keys())[:20]}"
        )

    hook = EmbeddingHook()
    handle = target_mod.register_forward_hook(hook)

    emb_list: list[np.ndarray] = []
    file_names: list[str] = []
    image_ids: list[int] = []
    dev = torch.device(device)

    try:
        from tqdm import tqdm
        progress = tqdm(val_dataloader, desc="Extracting embeddings", leave=False)
    except ImportError:
        progress = val_dataloader

    with torch.no_grad():
        for samples, targets in progress:
            samples = samples.to(dev)
            batch_img_ids = [int(t["image_id"]) for t in targets]

            _ = model(samples)

            if hook.features is not None:
                batch_emb = pool_features(hook.features).detach().cpu().numpy()
                for k, img_id in enumerate(batch_img_ids):
                    image_ids.append(img_id)
                    file_names.append(coco_gt.imgs[img_id]["file_name"])
                    emb_list.append(batch_emb[k])

    handle.remove()

    if not emb_list:
        raise RuntimeError("No embeddings were captured during extraction.")

    embeddings = np.stack(emb_list)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".npz":
            np.savez_compressed(
                out_p,
                embeddings=embeddings,
                file_names=np.array(file_names),
                image_ids=np.array(image_ids),
            )
        else:
            np.save(out_p, embeddings)
        if verbose:
            print(f"[*] Embeddings saved to: {out_p} (shape: {embeddings.shape})")

    return embeddings, file_names, image_ids


# ---------------------------------------------------------------------------
# 5. Confidence Summary
# ---------------------------------------------------------------------------

def compute_confidence_stats(
    predictions: list[dict],
    threshold: float = 0.0,
    prefix: str = "",
) -> dict[str, Any]:
    """Calculate confidence statistics for predictions above a threshold."""

    confs = np.array(
        [p["score"] for p in predictions if p["score"] >= threshold],
        dtype=np.float32,
    )

    prefix_space = f"{prefix}_" if prefix else ""
    if len(confs) == 0:
        return {
            f"{prefix_space}confidence_mean": None,
            f"{prefix_space}confidence_std": None,
            f"{prefix_space}confidence_median": None,
            f"{prefix_space}n_detections": 0,
        }

    return {
        f"{prefix_space}confidence_mean": float(confs.mean()),
        f"{prefix_space}confidence_std": float(confs.std()),
        f"{prefix_space}confidence_median": float(np.median(confs)),
        f"{prefix_space}n_detections": int(len(confs)),
    }


# ---------------------------------------------------------------------------
# 6. Detection Metrics Wrapper (evaluate_detection_metrics)
# ---------------------------------------------------------------------------

def evaluate_detection_metrics(
    config_path: str,
    checkpoint_path: str,
    img_folder: str,
    ann_file: str,
    device: str | torch.device = "cuda:0",
    min_score: float = 0.001,
    iou_thrs: np.ndarray | None = None,
    iou_match: float = 0.50,
    conf_threshold: float = 0.50,
    remap_mscoco: bool = False,
    output_dir: str | Path | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    repo_root: str = ".",
    verbose: bool = True,
) -> tuple[dict[str, Any], list[dict], Path | None]:
    """Run inference and calculate detection metrics at a fixed threshold.

    Returns
    -------
    tuple[dict[str, Any], list[dict], Path | None]
        (metrics_dict, predictions, plots_dir)
    """
    pred_save_path = Path(output_dir) / "predictions.json" if output_dir is not None else None

    predictions, coco_gt, _ = rtdetr_predict(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        img_folder=img_folder,
        ann_file=ann_file,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        min_score=min_score,
        remap_mscoco=remap_mscoco,
        output_predictions_path=pred_save_path,
        repo_root=repo_root,
        verbose=verbose,
    )

    metrics = calculate_metrics(
        pred_list=predictions,
        coco_gt=coco_gt,
        iou_thrs=iou_thrs,
        iou_match=iou_match,
        conf_threshold=conf_threshold,
        output_dir=output_dir,
    )

    plots_dir = Path(output_dir) if output_dir is not None else None
    return metrics, predictions, plots_dir


# ---------------------------------------------------------------------------
# 7. Single-Dataset Pipeline (run_full_evaluation)
# ---------------------------------------------------------------------------

def run_full_evaluation(
    config_path: str,
    checkpoint_path: str,
    dataset_cfg: dict,
    run_name: str = "",
    device: str | torch.device = "cuda:0",
    repo_root: str = ".",
    output_dir: str | Path | None = None,
    extract_embeddings_flag: bool = True,
    embedding_module: str = "encoder",
    conf_threshold: float = 0.50,
    fixed_conf_threshold: float | None = None,
    verbose: bool = True,
) -> tuple[dict, np.ndarray | None, list[str], Path | None]:
    """Runs prediction, metric evaluation, confidence estimation, and embedding extraction.

    Returns
    -------
    tuple[dict, np.ndarray | None, list[str], Path | None]
        (result_dict, embeddings, file_names, plots_dir)
    """
    splits = dataset_cfg.get("splits")
    if not isinstance(splits, dict) or "test" not in splits:
        raise ValueError("dataset_cfg must define a 'splits' dictionary with a 'test' split")

    test_split = splits["test"]
    if not isinstance(test_split, dict) or "img_folder" not in test_split or "ann_file" not in test_split:
        raise ValueError("dataset_cfg['splits']['test'] must define 'img_folder' and 'ann_file'")

    img_folder = test_split["img_folder"]
    ann_file = test_split["ann_file"]
    iou_match = float(dataset_cfg.get("iou", 0.50))
    conf = float(dataset_cfg.get("conf", conf_threshold)) if fixed_conf_threshold is None else fixed_conf_threshold
    min_score = float(dataset_cfg.get("min_score", 0.001))
    remap_mscoco = bool(dataset_cfg.get("remap_mscoco", False))
    batch_size = int(dataset_cfg.get("batch_size", 8))
    num_workers = int(dataset_cfg.get("num_workers", 4))

    out_dir_path = Path(output_dir) if output_dir is not None else None

    # 1. Detection metrics & predictions
    det_metrics, predictions, plots_dir = evaluate_detection_metrics(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        img_folder=img_folder,
        ann_file=ann_file,
        device=device,
        min_score=min_score,
        iou_match=iou_match,
        conf_threshold=conf,
        remap_mscoco=remap_mscoco,
        output_dir=out_dir_path,
        batch_size=batch_size,
        num_workers=num_workers,
        repo_root=repo_root,
        verbose=verbose,
    )

    # 2. Confidence stats
    conf_metrics = compute_confidence_stats(predictions)
    th_conf_metrics = compute_confidence_stats(predictions, threshold=conf, prefix="th")

    # 3. Embeddings
    embeddings = None
    file_names: list[str] = []
    if extract_embeddings_flag:
        emb_save_path = out_dir_path / "embeddings.npy" if out_dir_path is not None else None
        embeddings, file_names, _ = calculate_embeddings(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            img_folder=img_folder,
            ann_file=ann_file,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            embedding_module=embedding_module,
            output_path=emb_save_path,
            repo_root=repo_root,
            verbose=verbose,
        )

    result = {
        "run_name": run_name,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "img_folder": img_folder,
        "ann_file": ann_file,
        **det_metrics,
        **conf_metrics,
        **th_conf_metrics,
    }

    return result, embeddings, file_names, plots_dir


# ---------------------------------------------------------------------------
# 8. Multi-Dataset Orchestration (evaluate_source_against_targets)
# ---------------------------------------------------------------------------

def evaluate_source_against_targets(
    config_path: str,
    checkpoint_path: str,
    dataset_configs: dict,
    source_name: str,
    target_names: list[str] | None = None,
    device: str | torch.device = "cuda:0",
    repo_root: str = ".",
    result_path: str = "rtdetr_results",
    embedding_module: str = "encoder",
    extract_embeddings_flag: bool = True,
    run: Any = None,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Evaluate source checkpoint across multiple target datasets.

    Extracts multi-group embeddings (including optional source_train) and generates
    multi-group t-SNE visualizations and pairwise domain gap (MMD) metrics.
    """
    result_folder = Path(result_path)
    result_folder.mkdir(parents=True, exist_ok=True)
    target_names = target_names or list(dataset_configs.keys())

    if run is not None:
        log_dataset_config(
            run=run,
            dataset_configs=dataset_configs,
            source_name=source_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

    results: list[dict] = []
    embeddings_by_dataset: dict[str, np.ndarray] = {}

    source_cfg = dataset_configs[source_name]
    source_splits = source_cfg.get("splits", {})
    source_val_split = source_splits.get("val") if isinstance(source_splits, dict) else None
    if not isinstance(source_val_split, dict) or "img_folder" not in source_val_split or "ann_file" not in source_val_split:
        raise ValueError("The source dataset must define splits['val'] with 'img_folder' and 'ann_file'")

    if verbose:
        print(f"[*] Calculating confidence curves on source validation split: {source_name}")
    source_val_predictions, source_val_gt, _ = rtdetr_predict(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        img_folder=source_val_split["img_folder"],
        ann_file=source_val_split["ann_file"],
        batch_size=source_cfg.get("batch_size", 8),
        num_workers=source_cfg.get("num_workers", 4),
        device=device,
        min_score=source_cfg.get("min_score", 0.001),
        remap_mscoco=source_cfg.get("remap_mscoco", False),
        repo_root=repo_root,
        verbose=verbose,
    )
    source_val_curve_dir = result_folder / source_name / "val"
    source_val_curves = calculate_conf_curves(
        pred_list=source_val_predictions,
        coco_gt=source_val_gt,
        iou_match=source_cfg.get("iou", 0.50),
        output_dir=source_val_curve_dir,
    )
    source_conf_threshold = source_val_curves["best_f2_conf"]
    if run is not None:
        log_target_curves(run, f"{source_name}/val", source_val_curves["curve_data"])
    if verbose:
        print(f"[*] Source validation best_f2_conf: {source_conf_threshold:.4f}")

    # Evaluate each target split
    for target_name in target_names:
        cfg_t = dataset_configs[target_name]
        target_out_dir = result_folder / target_name

        if verbose:
            print(f"\n[*] Evaluating: source={source_name} -> target={target_name} ({cfg_t.get('label', '')})")

        result, emb, paths, plots_dir = run_full_evaluation(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            dataset_cfg=cfg_t,
            run_name=f"source{source_name}_to_{target_name}",
            device=device,
            repo_root=repo_root,
            output_dir=target_out_dir,
            extract_embeddings_flag=extract_embeddings_flag,
            embedding_module=embedding_module,
            fixed_conf_threshold=source_conf_threshold,
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
            if "curve_data" in result:
                log_target_curves(run, target_name, result["curve_data"])

    # Optional: Extract source training embeddings for complete domain visualization
    src_cfg = dataset_configs.get(source_name, {})
    src_splits = src_cfg.get("splits", {})
    src_train_split = src_splits.get("train") if isinstance(src_splits, dict) else None
    if (
        extract_embeddings_flag
        and isinstance(src_train_split, dict)
        and "img_folder" in src_train_split
        and "ann_file" in src_train_split
        and os.path.exists(src_train_split["img_folder"])
        and os.path.exists(src_train_split["ann_file"])
    ):
        if verbose:
            print(f"[*] Extracting embeddings for source train split: {source_name}")
        src_train_emb, _, _ = calculate_embeddings(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            img_folder=src_train_split["img_folder"],
            ann_file=src_train_split["ann_file"],
            batch_size=src_cfg.get("batch_size", 8),
            num_workers=src_cfg.get("num_workers", 4),
            device=device,
            embedding_module=embedding_module,
            output_path=result_folder / f"embeddings_{source_name}_train.npy",
            repo_root=repo_root,
            verbose=verbose,
        )
        embeddings_by_dataset["source_train"] = src_train_emb

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
                title=f"RT-DETRv4 Embeddings: {source_label} vs Targets (t-SNE)",
            )
            run.log({f"tsne": tsne_image})

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

    return results, embeddings_by_dataset


# ---------------------------------------------------------------------------
# 9. Top-Level Orchestration (run_eval)
# ---------------------------------------------------------------------------

def run_eval(
    run_name: str,
    config_path: str,
    checkpoint_path: str,
    dataset_configs: dict,
    source_name: str = "A",
    result_folder: str = "rtdetr_results",
    device: str = "cuda:0",
    repo_root: str = ".",
    embedding_module: str = "encoder",
    extract_embeddings_flag: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
) -> list[dict]:
    """Top-level evaluation entry-point with optional W&B logging."""
    # Load .env credentials if present
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.expanduser("~/.env"))
    except ImportError:
        pass

    run = None
    if wandb_project is not None and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb.login()
            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                job_type="eval",
                tags=wandb_tags or ["rtdetr_metrics"],
                name=f"eval_{run_name}",
                group=run_name,
            )
        except Exception as exc:
            print(f"[!] W&B init failed (continuing without it): {exc}")
            run = None
    elif wandb_project is not None:
        print("[!] WANDB_API_KEY not found in environment or ~/.env — W&B logging disabled.")

    results, _ = evaluate_source_against_targets(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_configs=dataset_configs,
        source_name=source_name,
        device=device,
        repo_root=repo_root,
        result_path=result_folder,
        embedding_module=embedding_module,
        extract_embeddings_flag=extract_embeddings_flag,
        run=run,
    )

    if run is not None:
        run.finish()

    return results


# ---------------------------------------------------------------------------
# 10. CLI Entry-Point
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RT-DETRv4 and extract DA metrics & embeddings.")
    p.add_argument("--repo-root", default=".", help="Path to RT-DETRv4 repository root")
    p.add_argument("--config", "-c", required=True, help="YAML config file")
    p.add_argument("--checkpoint", "-r", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--img-folder", required=True, help="Path to images directory")
    p.add_argument("--ann-file", required=True, help="Path to COCO annotations (.json)")
    p.add_argument("--output-dir", default="rtdetr_eval_output", help="Directory to save outputs")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--iou-match", type=float, default=0.50, help="IoU threshold for curves and fixed operating point")
    p.add_argument("--conf-threshold", type=float, default=0.20, help="Fixed operating-point confidence threshold")
    p.add_argument("--min-score", type=float, default=0.001, help="Filter predictions below score")
    p.add_argument("--remap-mscoco", action="store_true", help="Set flag for standard MS COCO 80-class mapping")
    p.add_argument("--extract-embeddings", action="store_true", help="Extract and save image embeddings")
    p.add_argument("--embedding-module", default="encoder", help="Module to hook for embeddings")
    p.add_argument("--run-name", default="cli_eval", help="Evaluation run name")
    p.add_argument("--dataset-label", default="dataset", help="Human-readable label for dataset")
    p.add_argument("--source-name", default="A", help="Dataset identifier key")
    p.add_argument("--wandb-project", default=None, help="Optional W&B project name")
    p.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team name")
    return p


def _main_cli():
    args = _build_cli_parser().parse_args()

    dataset_configs = {
        args.source_name: {
            "label": args.dataset_label,
            "splits": {
                "test": {
                    "ann_file": args.ann_file,
                    "img_folder": args.img_folder,
                },
            },
            "iou": args.iou_match,
            "conf": args.conf_threshold,
            "min_score": args.min_score,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "remap_mscoco": args.remap_mscoco,
        }
    }

    results = run_eval(
        run_name=args.run_name,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_configs=dataset_configs,
        source_name=args.source_name,
        result_folder=args.output_dir,
        device=args.device,
        repo_root=args.repo_root,
        embedding_module=args.embedding_module,
        extract_embeddings_flag=args.extract_embeddings,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY")
    print("=" * 60)
    for res in results:
        target = res.get("target_dataset_name", "?")
        label = res.get("target_dataset_label", "")
        print(f"\n  Target: {target} ({label})")
        for k, v in res.items():
            if k in (
                "run_name", "config_path", "checkpoint_path", "ann_file", "img_folder",
                "all_iou_thresholds", "precision_per_class", "recall_per_class",
                "source_dataset_name", "source_dataset_label", "target_dataset_name", "target_dataset_label"
            ):
                continue
            if isinstance(v, float):
                print(f"    {k:<28s}: {v:.4f}")
            elif isinstance(v, (int, str)):
                print(f"    {k:<28s}: {v}")
    print("=" * 60 + "\n")
    print(f"[✓] All evaluation outputs saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    _main_cli()
