"""
rtdetr_metrics.py
=================
Domain-adaptation-relevant metrics for RT-DETRv4 models.

Mirrors the structure and public API of ``da_metrics.py`` (the Ultralytics/YOLO
counterpart) so that both pipelines share the same evaluation conventions,
W&B logging patterns, and ``dataset_configs`` schema.

Sections
--------
1.  COCO evaluation helpers     (custom IoU range, AP dict, curves)
2.  Embedding utilities         (forward hook, feature pooling)
3.  Model loading
4.  Detection metrics           (COCO inference loop → mAP dict + plots dir)
5.  Confidence mean             (from collected predictions)
6.  Embedding extraction        (forward-hook based)
7.  Full single-dataset pipeline
8.  Multi-dataset orchestration (source vs N targets)
9.  Top-level ``run_eval`` entry-point (optional W&B)

Shared utilities (domain gap MMD, t-SNE, CSV, W&B helpers) live in
``tools.metrics_common`` and are imported here.

``dataset_configs`` schema
--------------------------
Both ``da_metrics.py`` (YOLO) and this module use the same top-level schema;
RT-DETRv4-specific keys replace / extend the YOLO-specific ones:

.. code-block:: python

    dataset_configs = {
        "A": {
            "label":       "SeaDronesSee",          # human-readable name
            # RT-DETRv4 specific
            "config":      "configs/rtdetr/...",     # YAML config path
            "ann_file":    "datasets/A/test.json",   # COCO annotations
            "img_folder":  "datasets/A/images/test", # image directory
            # shared / tunable
            "iou":         0.20,    # IoU threshold for PR/F1/F2 curves
            "conf":        0.001,   # min-score filter for predictions
            "batch_size":  8,
            "num_workers": 4,
        },
        ...
    }

Usage
-----
Library mode (import from another script)::

    from rtdetr_metrics import run_full_evaluation, evaluate_source_against_targets

CLI mode::

    python rtdetr_metrics.py \\
        --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \\
        --checkpoint weights/model_best.pth \\
        --ann-file    datasets/A/annotations/test.json \\
        --img-folder  datasets/A/images/test \\
        --output-dir  eval_output_A \\
        [--wandb-project my_project --wandb-run-name eval_A]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou

# Shared pure utilities (no ML-framework dependency)
from tools.metrics_common import (
    compute_domain_gap_mmd,
    compute_improvement,
    log_dataset_config,
    log_results_table,
    log_target_plots,
    log_target_scalars,
    plot_tsne,
    save_results_csv,
)


# ---------------------------------------------------------------------------
# MS COCO 80-class category mapping
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


# ---------------------------------------------------------------------------
# Section 1 – COCO evaluation helpers
# ---------------------------------------------------------------------------

def _get_label_to_category_map(coco_gt, remap_mscoco: bool) -> dict[int, int]:
    """Map model predicted class index (0-based) → dataset category_id."""
    cat_ids = sorted(coco_gt.getCatIds())
    if remap_mscoco and len(cat_ids) == 80 and max(cat_ids) == 90:
        return MSCOCO_LABEL2CATEGORY
    return {idx: cat_id for idx, cat_id in enumerate(cat_ids)}


def _run_custom_cocoeval(coco_gt, predictions: list[dict], iou_thrs: np.ndarray):
    """Run COCOeval with an arbitrary array of IoU thresholds."""
    if not predictions:
        return None
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array(iou_thrs, dtype=np.float64)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _compute_extended_ap_dict(coco_eval, iou_thrs: np.ndarray) -> dict:
    """Extract per-IoU AP values and standard summary metrics from COCOeval."""
    if coco_eval is None:
        return {
            "map20_95": 0.0, "map50_95": 0.0,
            "map20": 0.0, "map50": 0.0, "map75": 0.0,
            "all_iou_thresholds": {},
        }

    prec = coco_eval.eval["precision"]   # [T, R, K, A, M]
    aps: dict[str, float] = {}
    for idx, t in enumerate(iou_thrs):
        p = prec[idx, :, :, 0, -1]
        valid = p[p > -1]
        aps[f"AP@{t:.2f}"] = float(valid.mean()) if valid.size > 0 else 0.0

    def _find(target: float) -> float:
        idx = int(np.argmin(np.abs(np.asarray(iou_thrs) - target)))
        return aps[f"AP@{iou_thrs[idx]:.2f}"]

    std_thrs = [aps[f"AP@{t:.2f}"] for t in iou_thrs if t >= 0.499]

    return {
        "map20_95":            float(np.mean(list(aps.values()))),
        "map50_95":            float(np.mean(std_thrs)) if std_thrs else 0.0,
        "map20":               _find(0.20),
        "map50":               _find(0.50),
        "map75":               _find(0.75),
        "all_iou_thresholds":  aps,
    }


def _compute_per_class_metrics(coco_eval) -> dict:
    """Extract mean precision and recall across classes from COCOeval results."""
    if coco_eval is None:
        return {"precision_mean": 0.0, "recall_mean": 0.0,
                "precision_per_class": [], "recall_per_class": []}

    # precision shape: [T, R, K, A, M]  (T=IoU, R=recall, K=class, A=area, M=maxDets)
    # Take the IoU=0.50 slice (index where iou_thr ~ 0.50) for class-level P/R
    iou_thrs = coco_eval.params.iouThrs
    iou50_idx = int(np.argmin(np.abs(iou_thrs - 0.50)))

    prec_slice = coco_eval.eval["precision"][iou50_idx, :, :, 0, -1]  # [R, K]
    # Per-class precision = mean over recall points where valid (> -1)
    per_class_p = []
    for k in range(prec_slice.shape[1]):
        valid = prec_slice[:, k]
        valid = valid[valid > -1]
        per_class_p.append(float(valid.mean()) if valid.size > 0 else 0.0)

    # recall shape: [T, K, A, M]
    rec_slice = coco_eval.eval["recall"][iou50_idx, :, 0, -1]   # [K]
    per_class_r = [float(r) if r > -1 else 0.0 for r in rec_slice]

    return {
        "precision_mean":      float(np.mean(per_class_p)) if per_class_p else 0.0,
        "recall_mean":         float(np.mean(per_class_r)) if per_class_r else 0.0,
        "precision_per_class": per_class_p,
        "recall_per_class":    per_class_r,
    }


# ---------------------------------------------------------------------------
# Precision / Recall / F1 / F2 vs confidence curves
# ---------------------------------------------------------------------------

def _compute_curves(
    coco_gt,
    predictions: list[dict],
    iou_thresh: float = 0.5,
    num_conf_steps: int = 100,
) -> dict:
    """Compute per-confidence-threshold P/R/F1/F2 metrics."""
    conf_thresholds = np.linspace(0.01, 0.99, num_conf_steps)

    gt_by_img: dict = defaultdict(lambda: defaultdict(list))
    total_gt = 0
    for ann in coco_gt.dataset["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_img[ann["image_id"]][ann["category_id"]].append([x, y, x + w, y + h])
        total_gt += 1

    pred_by_img: dict = defaultdict(lambda: defaultdict(list))
    for p in predictions:
        pred_by_img[p["image_id"]][p["category_id"]].append(p)

    precision_list, recall_list, f1_list, f2_list = [], [], [], []

    for conf in conf_thresholds:
        tp = fp = 0

        for img_id, cat_dict in gt_by_img.items():
            for cat_id, gt_boxes_list in cat_dict.items():
                gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32)
                matched = np.zeros(len(gt_boxes), dtype=bool)
                preds = [p for p in pred_by_img[img_id].get(cat_id, [])
                         if p["score"] >= conf]
                preds = sorted(preds, key=lambda x: -x["score"])

                for p in preds:
                    x, y, w, h = p["bbox"]
                    p_box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
                    ious = box_iou(p_box, gt_boxes)[0].numpy()
                    ious[matched] = -1.0
                    best = int(ious.argmax()) if ious.size > 0 else -1
                    if best >= 0 and ious[best] >= iou_thresh:
                        tp += 1
                        matched[best] = True
                    else:
                        fp += 1

        # FP for predicted classes not present in that GT image
        for img_id, cat_dict in pred_by_img.items():
            for cat_id, p_list in cat_dict.items():
                if cat_id not in gt_by_img.get(img_id, {}):
                    fp += sum(1 for p in p_list if p["score"] >= conf)

        prec = tp / max(tp + fp, 1e-9)
        rec  = tp / max(total_gt, 1e-9)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        f2   = 5 * prec * rec / max(4 * prec + rec, 1e-9)

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        f2_list.append(f2)

    return {
        "conf_thresholds": conf_thresholds,
        "precision":       np.array(precision_list),
        "recall":          np.array(recall_list),
        "f1":              np.array(f1_list),
        "f2":              np.array(f2_list),
        "total_gt":        total_gt,
        "iou_thresh":      iou_thresh,
    }


def _save_curves_and_plots(curve_data: dict, output_dir: str | Path) -> dict:
    """Save P/R/F1/F2 CSV + PNG plots and return operating-point summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    confs = curve_data["conf_thresholds"]
    prec  = curve_data["precision"]
    rec   = curve_data["recall"]
    f1    = curve_data["f1"]
    f2    = curve_data["f2"]

    import csv as _csv
    csv_file = output_dir / "metrics_curves.csv"
    with open(csv_file, "w", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow(["confidence_threshold", "precision", "recall", "f1", "f2"])
        for c, p, r, _f1, _f2 in zip(confs, prec, rec, f1, f2):
            writer.writerow([round(c, 4), round(p, 4), round(r, 4),
                             round(_f1, 4), round(_f2, 4)])

    def _plot(x, y, xlabel, ylabel, title, fname):
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, color="#1f77b4", linewidth=2.0)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=180)
        plt.close()

    _plot(confs, prec, "Confidence Threshold", "Precision",
          "Precision vs. Confidence", "precision_curve.png")
    _plot(confs, rec,  "Confidence Threshold", "Recall",
          "Recall vs. Confidence",   "recall_curve.png")
    _plot(confs, f1,   "Confidence Threshold", "F1 Score",
          "F1 Score vs. Confidence", "f1_curve.png")
    _plot(confs, f2,   "Confidence Threshold", "F2 Score",
          "F2 Score vs. Confidence", "f2_curve.png")
    _plot(rec, prec, "Recall", "Precision",
          f"Precision-Recall Curve (IoU={curve_data['iou_thresh']})",
          "pr_curve.png")

    best_f1_idx = int(np.argmax(f1))
    best_f2_idx = int(np.argmax(f2))
    summary = {
        "best_f1_operating_point": {
            "optimal_confidence_threshold": float(confs[best_f1_idx]),
            "f1":        float(f1[best_f1_idx]),
            "precision": float(prec[best_f1_idx]),
            "recall":    float(rec[best_f1_idx]),
        },
        "best_f2_operating_point": {
            "optimal_confidence_threshold": float(confs[best_f2_idx]),
            "f2":        float(f2[best_f2_idx]),
            "precision": float(prec[best_f2_idx]),
            "recall":    float(rec[best_f2_idx]),
        },
    }
    with open(output_dir / "operating_points_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Section 2 – Embedding utilities
# ---------------------------------------------------------------------------

class _EmbeddingHook:
    """Simple forward-hook that captures the last output tensor of a module."""

    def __init__(self):
        self.features: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        self.features = out


def _pool_features(feat_output) -> torch.Tensor:
    """Pool an arbitrary module output to a flat [B, C] feature vector."""
    feats = feat_output if isinstance(feat_output, (list, tuple)) else [feat_output]
    pooled = []
    for f in feats:
        if isinstance(f, torch.Tensor):
            if f.dim() == 4:   # [B, C, H, W] → GAP → [B, C]
                pooled.append(F.adaptive_avg_pool2d(f, (1, 1)).flatten(1))
            elif f.dim() == 3:  # [B, N, C] → mean → [B, C]
                pooled.append(f.mean(dim=1))
    if not pooled:
        raise ValueError("Could not pool features from hooked layer.")
    return torch.cat(pooled, dim=1)


# ---------------------------------------------------------------------------
# Section 3 – Model loading
# ---------------------------------------------------------------------------

def _get_yaml_config(repo_root: str):
    """Locate and return the ``YAMLConfig`` class from the RT-DETRv4 repo."""
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
    """Load an RT-DETRv4 model + postprocessor + dataloader from a YAML config
    and checkpoint file.

    Parameters
    ----------
    config_path : str
        Path to the RT-DETRv4 YAML config file.
    checkpoint_path : str
        Path to the ``.pth`` checkpoint file.
    device : torch.device | str
    repo_root : str
        Root of the RT-DETRv4 repository (needed for ``sys.path`` insertion).
    img_folder : str | None
        Override the image folder in the config (optional).
    ann_file : str | None
        Override the annotation file in the config (optional).
    batch_size : int
        Override batch size in the config.
    num_workers : int
        Override number of dataloader workers in the config.

    Returns
    -------
    tuple
        ``(model, postprocessor, val_dataloader, coco_gt, cfg)``
    """
    device = torch.device(device)
    YAMLConfig = _get_yaml_config(repo_root)
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    # Override dataset paths / loader settings when provided
    if "val_dataloader" in cfg.yaml_cfg:
        ds_cfg = cfg.yaml_cfg["val_dataloader"]
        if img_folder is not None:
            ds_cfg["dataset"]["img_folder"] = img_folder
        if ann_file is not None:
            ds_cfg["dataset"]["ann_file"] = ann_file
        ds_cfg["total_batch_size"] = batch_size
        ds_cfg["num_workers"] = num_workers

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state_dict)

    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    val_dataloader = cfg.val_dataloader
    coco_gt = val_dataloader.dataset.coco

    return model, postprocessor, val_dataloader, coco_gt, cfg


# ---------------------------------------------------------------------------
# Section 4 – Detection metrics
# ---------------------------------------------------------------------------

def evaluate_detection_metrics(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    val_dataloader,
    coco_gt,
    device: torch.device | str = "cpu",
    min_score: float = 0.001,
    iou_thrs: np.ndarray | None = None,
    iou_match: float = 0.5,
    remap_mscoco: bool = False,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[dict], Path | None]:
    """Run inference on the full validation dataloader and compute COCO metrics.

    This is the RT-DETRv4 equivalent of ``da_metrics.evaluate_detection_metrics``
    for Ultralytics/YOLO.  The return dict has the **same keys** so that shared
    orchestration code works unchanged:
    ``map20_95``, ``map50_95``, ``map20``, ``map50``, ``map75``,
    ``precision_mean``, ``recall_mean``,
    ``precision_per_class``, ``recall_per_class``.

    Parameters
    ----------
    model, postprocessor : torch.nn.Module
        Loaded RT-DETRv4 model and postprocessor (in ``.eval()`` mode).
    val_dataloader :
        Validation dataloader produced by ``load_rtdetr_model``.
    coco_gt :
        ``pycocotools.coco.COCO`` ground-truth object.
    device : torch.device | str
    min_score : float
        Discard predictions below this confidence (keeps predictions list lean).
    iou_thrs : np.ndarray | None
        IoU thresholds for the COCOeval sweep (default: 0.20 → 0.95 step 0.05).
    iou_match : float
        IoU threshold used for the P/R/F1/F2 vs confidence curves.
    remap_mscoco : bool
        Set ``True`` for standard MS COCO 80-class evaluation.
    output_dir : str | Path | None
        If provided, saves prediction JSON + curve plots to this directory.
    verbose : bool

    Returns
    -------
    tuple
        ``(metrics_dict, predictions, plots_dir)``
        ``plots_dir`` is ``Path(output_dir)`` when ``output_dir`` is given,
        else ``None``.
    """
    device = torch.device(device)
    if iou_thrs is None:
        iou_thrs = np.round(np.arange(0.20, 0.951, 0.05), 2)

    label_to_cat = _get_label_to_category_map(coco_gt, remap_mscoco)

    # ---- inference loop ----
    try:
        from tqdm import tqdm
        progress = tqdm(val_dataloader, desc="Evaluating", leave=False)
    except ImportError:
        progress = val_dataloader

    predictions: list[dict] = []
    model.eval()
    postprocessor.eval()

    with torch.no_grad():
        for samples, targets in progress:
            samples = samples.to(device)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            image_ids = [int(t["image_id"]) for t in targets]

            outputs = model(samples)
            results = postprocessor(outputs, orig_sizes)

            for img_id, res in zip(image_ids, results):
                labels = res["labels"].detach().cpu().numpy()
                boxes  = res["boxes"].detach().cpu().numpy()
                scores = res["scores"].detach().cpu().numpy()

                for lbl, box, score in zip(labels, boxes, scores):
                    if score < min_score:
                        continue
                    cat_id = label_to_cat.get(int(lbl), int(lbl))
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id":   img_id,
                        "category_id": cat_id,
                        "bbox":        [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score":       float(score),
                    })

    if verbose:
        print(f"[*] Total predictions collected: {len(predictions)}")

    # ---- optional disk output ----
    plots_dir: Path | None = None
    if output_dir is not None:
        plots_dir = Path(output_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        with open(plots_dir / "predictions.json", "w") as fh:
            json.dump(predictions, fh)

    # ---- COCO metrics ----
    if verbose:
        print("[*] Computing COCO metrics...")
    coco_eval = _run_custom_cocoeval(coco_gt, predictions, iou_thrs)
    ap_dict   = _compute_extended_ap_dict(coco_eval, iou_thrs)
    pr_dict   = _compute_per_class_metrics(coco_eval)

    metrics: dict[str, Any] = {**ap_dict, **pr_dict}

    # ---- curves (if output_dir provided) ----
    if output_dir is not None:
        curve_data     = _compute_curves(coco_gt, predictions, iou_thresh=iou_match)
        op_summary     = _save_curves_and_plots(curve_data, output_dir)
        metrics["best_f1_operating_point"] = op_summary["best_f1_operating_point"]
        metrics["best_f2_operating_point"] = op_summary["best_f2_operating_point"]
        with open(plots_dir / "metrics_summary.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

    return metrics, predictions, plots_dir


# ---------------------------------------------------------------------------
# Section 5 – Confidence mean
# ---------------------------------------------------------------------------

def compute_confidence_mean(predictions: list[dict]) -> dict:
    """Compute mean and std confidence from collected COCO-format predictions.

    This is the RT-DETRv4 equivalent of ``da_metrics.compute_confidence_mean``.
    Because RT-DETRv4 has no ``model.predict(image_dir)`` convenience API,
    confidence statistics are derived from the predictions already gathered
    during the COCO inference loop — no second inference pass is needed.

    Parameters
    ----------
    predictions : list[dict]
        List of COCO-format prediction dicts (each must have a ``"score"`` key).

    Returns
    -------
    dict
        Keys: ``confidence_mean``, ``confidence_std``, ``n_detections``.
    """
    if not predictions:
        return {"confidence_mean": None, "confidence_std": None, "n_detections": 0}

    confs = np.array([p["score"] for p in predictions], dtype=np.float32)
    return {
        "confidence_mean": float(confs.mean()),
        "confidence_std":  float(confs.std()),
        "n_detections":    int(len(confs)),
    }


# ---------------------------------------------------------------------------
# Section 6 – Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: torch.nn.Module,
    val_dataloader,
    coco_gt,
    device: torch.device | str = "cpu",
    embedding_module: str = "encoder",
) -> tuple[np.ndarray, list[str]]:
    """Extract one embedding vector per image using a forward hook.

    This is the RT-DETRv4 equivalent of ``da_metrics.extract_embeddings``
    (which uses Ultralytics' ``model.embed()``).

    Parameters
    ----------
    model : torch.nn.Module
        RT-DETRv4 model in ``.eval()`` mode.
    val_dataloader :
        Validation dataloader.
    coco_gt :
        ``pycocotools.coco.COCO`` ground-truth object (for ``file_name`` lookup).
    device : torch.device | str
    embedding_module : str
        Sub-module name to hook (``"encoder"``, ``"backbone"``, etc.).

    Returns
    -------
    tuple
        ``(embeddings, image_paths)`` where ``embeddings`` is an ndarray of
        shape ``(N, feature_dim)`` and ``image_paths`` is a list of filenames.
    """
    device = torch.device(device)

    # Locate hook target
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
            f"Module '{embedding_module}' not found in model. "
            f"Available top-level modules: {list(submodules.keys())[:20]}"
        )

    hook = _EmbeddingHook()
    handle = target_mod.register_forward_hook(hook)

    embedding_records: dict[str, list] = {"image_id": [], "file_name": [], "embedding": []}

    try:
        from tqdm import tqdm
        progress = tqdm(val_dataloader, desc="Extracting embeddings", leave=False)
    except ImportError:
        progress = val_dataloader

    model.eval()
    with torch.no_grad():
        for samples, targets in progress:
            samples   = samples.to(device)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            image_ids  = [int(t["image_id"]) for t in targets]

            _ = model(samples)   # trigger hook; we don't need detections here

            if hook.features is not None:
                batch_emb = _pool_features(hook.features).detach().cpu().numpy()
                for k, img_id in enumerate(image_ids):
                    embedding_records["image_id"].append(img_id)
                    embedding_records["file_name"].append(
                        coco_gt.imgs[img_id]["file_name"]
                    )
                    embedding_records["embedding"].append(batch_emb[k])

    handle.remove()

    if not embedding_records["embedding"]:
        raise RuntimeError("No embeddings were captured. Check embedding_module name.")

    embeddings = np.stack(embedding_records["embedding"])
    return embeddings, embedding_records["file_name"]


# ---------------------------------------------------------------------------
# Section 7 – Full single-dataset pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    config_path: str,
    checkpoint_path: str,
    dataset_cfg: dict,
    run_name: str = "",
    device: str | torch.device = "cuda:0",
    repo_root: str = ".",
    output_dir: str | None = None,
    extract_embeddings_flag: bool = True,
    embedding_module: str = "encoder",
    verbose: bool = True,
) -> tuple[dict, np.ndarray | None, list[str], Path | None]:
    """Runs detection metrics + confidence stats + (optionally) embeddings.

    This is the RT-DETRv4 equivalent of ``da_metrics.run_full_evaluation``.
    It returns the **same tuple shape** so orchestration code is interchangeable.

    Parameters
    ----------
    config_path : str
        Path to the RT-DETRv4 YAML config.
    checkpoint_path : str
        Path to the ``.pth`` checkpoint.
    dataset_cfg : dict
        One entry from ``dataset_configs``. Required keys:
        ``ann_file``, ``img_folder``. Optional: ``iou``, ``conf``,
        ``batch_size``, ``num_workers``.
    run_name : str
        Human-readable identifier for this evaluation run.
    device : str | torch.device
    repo_root : str
    output_dir : str | None
        If given, predictions JSON + curve plots are saved here.
    extract_embeddings_flag : bool
        Whether to run the embedding extraction pass.
    embedding_module : str
        Sub-module to hook for embedding extraction.
    verbose : bool

    Returns
    -------
    tuple
        ``(metrics_dict, embeddings_array, image_paths, plots_dir)``
        ``embeddings_array`` is ``None`` when ``extract_embeddings_flag=False``.
        ``plots_dir`` is ``None`` when ``output_dir`` is not specified.
    """
    model, postprocessor, val_dataloader, coco_gt, cfg = load_rtdetr_model(
        config_path    = config_path,
        checkpoint_path= checkpoint_path,
        device         = device,
        repo_root      = repo_root,
        img_folder     = dataset_cfg.get("img_folder"),
        ann_file       = dataset_cfg.get("ann_file"),
        batch_size     = dataset_cfg.get("batch_size", 8),
        num_workers    = dataset_cfg.get("num_workers", 4),
    )

    remap = bool(cfg.yaml_cfg.get("remap_mscoco_category", False))

    # ---- detection metrics + predictions ----
    det_metrics, predictions, plots_dir = evaluate_detection_metrics(
        model          = model,
        postprocessor  = postprocessor,
        val_dataloader = val_dataloader,
        coco_gt        = coco_gt,
        device         = device,
        min_score      = dataset_cfg.get("conf", 0.001),
        iou_match      = dataset_cfg.get("iou", 0.5),
        remap_mscoco   = remap,
        output_dir     = output_dir,
        verbose        = verbose,
    )

    # ---- confidence statistics (from already-collected predictions) ----
    conf_metrics = compute_confidence_mean(predictions)

    # ---- embeddings (second pass with hook) ----
    embeddings: np.ndarray | None = None
    image_paths: list[str] = []
    if extract_embeddings_flag:
        embeddings, image_paths = extract_embeddings(
            model           = model,
            val_dataloader  = val_dataloader,
            coco_gt         = coco_gt,
            device          = device,
            embedding_module= embedding_module,
        )

    result: dict = {
        "run_name":       run_name,
        "config_path":    config_path,
        "checkpoint_path": checkpoint_path,
        "ann_file":       dataset_cfg.get("ann_file", ""),
        "img_folder":     dataset_cfg.get("img_folder", ""),
        **det_metrics,
        **conf_metrics,
    }

    return result, embeddings, image_paths, plots_dir


# ---------------------------------------------------------------------------
# Section 8 – Multi-dataset orchestration
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
    """Evaluate one checkpoint against every dataset in ``target_names``.

    This is the RT-DETRv4 equivalent of ``da_metrics.evaluate_source_against_targets``.
    Works for any number of datasets — just add entries to ``dataset_configs``.

    Parameters
    ----------
    config_path : str
        RT-DETRv4 YAML config (the architecture definition; dataset paths will
        be overridden per-target via ``dataset_cfg``).
    checkpoint_path : str
        Path to the ``.pth`` checkpoint.
    dataset_configs : dict
        Full ``dataset_configs`` mapping.
    source_name : str
        Key in ``dataset_configs`` used as the in-domain baseline.
    target_names : list[str] | None
        Targets to evaluate. Defaults to ALL keys in ``dataset_configs``.
    device : str | torch.device
    repo_root : str
    result_path : str
        Root folder for saving per-target embeddings, CSV, t-SNE plots, etc.
    embedding_module : str
    extract_embeddings_flag : bool
    run : wandb.Run | None
        Active W&B run. If ``None``, W&B logging is skipped.
    verbose : bool

    Returns
    -------
    tuple
        ``(results_list, embeddings_by_dataset)``
    """
    result_folder = Path(result_path)
    result_folder.mkdir(parents=True, exist_ok=True)
    target_names = target_names or list(dataset_configs.keys())

    if run is not None:
        log_dataset_config(run, dataset_configs, source_name)

    results: list[dict] = []
    embeddings_by_dataset: dict[str, np.ndarray] = {}

    for target_name in target_names:
        cfg_t = dataset_configs[target_name]
        out_dir = result_folder / target_name

        if verbose:
            print(f"\n[*] Evaluating: source={source_name} → target={target_name}")

        result, emb, paths, plots_dir = run_full_evaluation(
            config_path             = config_path,
            checkpoint_path         = checkpoint_path,
            dataset_cfg             = cfg_t,
            run_name                = f"source{source_name}_to_{target_name}",
            device                  = device,
            repo_root               = repo_root,
            output_dir              = str(out_dir),
            extract_embeddings_flag = extract_embeddings_flag,
            embedding_module        = embedding_module,
            verbose                 = verbose,
        )
        result["source_dataset_name"]  = source_name
        result["source_dataset_label"] = dataset_configs[source_name]["label"]
        result["target_dataset_name"]  = target_name
        result["target_dataset_label"] = cfg_t["label"]

        results.append(result)

        if emb is not None:
            embeddings_by_dataset[target_name] = emb
            np.save(result_folder / f"embeddings_{source_name}_to_{target_name}.npy", emb)

        # ---- W&B per-target logging ----
        if run is not None:
            log_target_scalars(run, target_name, result)
            log_target_plots(run, target_name, plots_dir)

    # ---- aggregate W&B table + CSV ----
    if run is not None:
        log_results_table(run, results)

    csv_path = result_folder / f"source{source_name}_results.csv"
    save_results_csv(results, str(csv_path))

    if run is not None:
        try:
            import wandb
            csv_art = wandb.Artifact(f"source{source_name}_results", type="evaluation")
            csv_art.add_file(str(csv_path))
            run.log_artifact(csv_art)
        except ImportError:
            pass

    # ---- domain gap + t-SNE ----
    if extract_embeddings_flag and source_name in embeddings_by_dataset:
        for target_name in target_names:
            if target_name == source_name:
                continue
            if target_name not in embeddings_by_dataset:
                continue
            gap = compute_domain_gap_mmd(
                embeddings_by_dataset[source_name],
                embeddings_by_dataset[target_name],
            )
            if verbose:
                print(f"[*] Domain gap MMD ({source_name} → {target_name}): {gap:.6f}")
            if run is not None:
                run.summary[f"{target_name}/domain_gap_mmd"] = gap

            tsne_path = plot_tsne(
                source_embeddings = embeddings_by_dataset[source_name],
                target_embeddings = embeddings_by_dataset[target_name],
                out_path = str(result_folder / f"tsne_{source_name}_vs_{target_name}.png"),
                title = f"RT-DETRv4 embeddings: {source_name} vs {target_name} (t-SNE)",
            )
            if run is not None:
                try:
                    import wandb
                    run.log({f"{target_name}/tsne": wandb.Image(tsne_path)})
                except ImportError:
                    pass

    return results, embeddings_by_dataset


# ---------------------------------------------------------------------------
# Section 9 – Top-level entry-point (optional W&B)
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
    """Top-level convenience function mirroring ``da_metrics.run_eval``.

    W&B logging is **optional** — activated only when ``wandb_project`` is
    provided AND ``WANDB_API_KEY`` is set in the environment (loaded from
    ``~/.env`` via ``python-dotenv`` if available).

    Parameters
    ----------
    run_name : str
    config_path : str
    checkpoint_path : str
    dataset_configs : dict
    source_name : str
    result_folder : str
    device : str
    repo_root : str
    embedding_module : str
    extract_embeddings_flag : bool
    wandb_project : str | None
        W&B project name. Pass ``None`` to skip W&B entirely.
    wandb_entity : str | None
        W&B entity (team or username). Defaults to the logged-in user.
    wandb_tags : list[str] | None

    Returns
    -------
    list[dict]
        All per-target result dicts.
    """
    # Load .env for WANDB_API_KEY (silently ignored if dotenv not installed)
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
                project  = wandb_project,
                entity   = wandb_entity,
                job_type = "eval",
                tags     = wandb_tags or ["rtdetr_metrics"],
                name     = f"eval_{run_name}",
                group    = run_name,
            )
        except Exception as exc:
            print(f"[!] W&B init failed (continuing without it): {exc}")
            run = None
    elif wandb_project is not None:
        print("[!] WANDB_API_KEY not found — W&B logging disabled. "
              "Add it to ~/.env to enable.")

    results, _ = evaluate_source_against_targets(
        config_path             = config_path,
        checkpoint_path         = checkpoint_path,
        dataset_configs         = dataset_configs,
        source_name             = source_name,
        device                  = device,
        repo_root               = repo_root,
        result_path             = result_folder,
        embedding_module        = embedding_module,
        extract_embeddings_flag = extract_embeddings_flag,
        run                     = run,
    )

    if run is not None:
        run.finish()

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RT-DETRv4 evaluation — mirrors da_metrics.py API."
    )
    p.add_argument("--repo-root", default=".", help="RT-DETRv4 repository root")
    p.add_argument("--config", "-c", required=True, help="YAML config file")
    p.add_argument("--checkpoint", "-r", required=True, help="Checkpoint .pth file")
    p.add_argument("--ann-file",   required=True, help="COCO annotations JSON")
    p.add_argument("--img-folder", required=True, help="Image directory")
    p.add_argument("--output-dir", default="rtdetr_eval_output",
                   help="Directory to save results (default: rtdetr_eval_output)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size",   type=int, default=8)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--iou-match",    type=float, default=0.5,
                   help="IoU threshold for PR/F1/F2 curves")
    p.add_argument("--min-score",    type=float, default=0.001,
                   help="Minimum prediction score to keep")
    p.add_argument("--run-name",     default="cli_eval", help="Evaluation run name")
    p.add_argument("--dataset-label", default="dataset",
                   help="Human-readable label for this dataset")
    p.add_argument("--source-name",  default="A",
                   help="Dataset key used as source label in results")
    p.add_argument("--extract-embeddings", action="store_true",
                   help="Extract and save image embeddings")
    p.add_argument("--embedding-module", default="encoder",
                   help="Sub-module to hook for embeddings")
    # W&B
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name (optional; enables W&B logging)")
    p.add_argument("--wandb-entity",  default=None, help="W&B entity / team")
    return p


def _main_cli():
    args = _build_cli_parser().parse_args()

    # Build a single-entry dataset_configs dict from CLI args
    dataset_configs = {
        args.source_name: {
            "label":      args.dataset_label,
            "config":     args.config,
            "ann_file":   args.ann_file,
            "img_folder": args.img_folder,
            "iou":        args.iou_match,
            "conf":       args.min_score,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
    }

    results = run_eval(
        run_name                = args.run_name,
        config_path             = args.config,
        checkpoint_path         = args.checkpoint,
        dataset_configs         = dataset_configs,
        source_name             = args.source_name,
        result_folder           = args.output_dir,
        device                  = args.device,
        repo_root               = args.repo_root,
        embedding_module        = args.embedding_module,
        extract_embeddings_flag = args.extract_embeddings,
        wandb_project           = args.wandb_project,
        wandb_entity            = args.wandb_entity,
    )

    # Print summary table
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"\n  Target : {result.get('target_dataset_name', '?')} "
              f"({result.get('target_dataset_label', '')})")
        for k, v in result.items():
            if k in ("run_name", "config_path", "checkpoint_path", "ann_file",
                     "img_folder", "all_iou_thresholds",
                     "precision_per_class", "recall_per_class",
                     "best_f1_operating_point", "best_f2_operating_point",
                     "source_dataset_name", "source_dataset_label",
                     "target_dataset_name", "target_dataset_label"):
                continue
            if isinstance(v, float):
                print(f"    {k:<30s}: {v:.4f}")
    print("=" * 60 + "\n")
    print(f"[✓] Results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    _main_cli()

