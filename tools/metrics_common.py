"""
tools/metrics_common.py
=======================
Framework-agnostic shared utilities for detection evaluation pipelines.

These functions are reusable across RT-DETRv4 (rtdetr_metrics.py),
Ultralytics/YOLO (da_metrics.py), and future architectures (e.g. D-FINE).

Sections
--------
1.  Detection metrics & curves (COCOeval, mAP sweep, P/R curves, fixed & optimal operating points)
2.  Improvement deltas         (compare adapted vs. baseline run)
3.  Domain gap & t-SNE         (MMD metric + multi-group t-SNE visualization)
4.  CSV export
5.  W&B logging helpers        (import-guarded; require ``wandb`` installed)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal Helpers for IoU and Curves
# ---------------------------------------------------------------------------

def _box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.
    boxes1: (N, 4) in [x1, y1, x2, y2]
    boxes2: (M, 4) in [x1, y1, x2, y2]
    returns: (N, M) IoU matrix
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = np.clip(rb - lt, a_min=0.0, a_max=None)               # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]                         # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-9)

    return inter / union


def _match_at_confidence(
    gt_by_img: dict[int, dict[int, list[list[float]]]],
    pred_by_img: dict[int, dict[int, list[dict]]],
    conf_thresh: float,
    iou_thresh: float,
    total_gt: int,
) -> tuple[int, int, int, int, float, float, float, float]:
    """
    Match predictions against ground truth at a specific confidence threshold.
    Returns (tp, fp, fn, tn, precision, recall, f1, f2).
    """
    tp = 0
    fp = 0

    # Match predictions against GT for categories present in GT
    for img_id, cat_dict in gt_by_img.items():
        for cat_id, gt_boxes_list in cat_dict.items():
            gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
            matched = np.zeros(len(gt_boxes), dtype=bool)

            preds = [p for p in pred_by_img[img_id].get(cat_id, []) if p["score"] >= conf_thresh]
            preds = sorted(preds, key=lambda x: -x["score"])

            for p in preds:
                x, y, w, h = p["bbox"]
                p_box = np.array([[x, y, x + w, y + h]], dtype=np.float32)
                ious = _box_iou_numpy(p_box, gt_boxes)[0]
                ious[matched] = -1.0
                best_idx = int(ious.argmax()) if ious.size > 0 else -1

                if best_idx >= 0 and ious[best_idx] >= iou_thresh:
                    tp += 1
                    matched[best_idx] = True
                else:
                    fp += 1

    # Add false positives for predicted categories not present in GT for that image
    for img_id, cat_dict in pred_by_img.items():
        for cat_id, p_list in cat_dict.items():
            if cat_id not in gt_by_img.get(img_id, {}):
                fp += sum(1 for p in p_list if p["score"] >= conf_thresh)

    fn = max(0, total_gt - tp)
    tn = 0  # In object detection bounding box evaluation, True Negatives are undefined/0

    p = tp / max(tp + fp, 1e-9)
    r = tp / max(total_gt, 1e-9)
    f1 = 2 * p * r / max(p + r, 1e-9)
    f2 = 5 * p * r / max(4 * p + r, 1e-9)

    return tp, fp, fn, tn, float(p), float(r), float(f1), float(f2)


# ---------------------------------------------------------------------------
# 1. Detection Metrics & Curves
# ---------------------------------------------------------------------------

def calculate_conf_curves(
    pred_list: list[dict],
    coco_gt: Any,
    iou_match: float = 0.50,
    output_dir: str | Path | None = None,
    num_conf_steps: int = 100,
) -> dict[str, Any]:
    """Calculate confidence-threshold curves and select the best F2 threshold."""
    from pycocotools.coco import COCO

    if isinstance(coco_gt, (str, Path)):
        coco_gt = COCO(str(coco_gt))

    gt_by_img = defaultdict(lambda: defaultdict(list))
    total_gt = 0
    for ann in coco_gt.dataset.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_img[ann["image_id"]][ann["category_id"]].append([x, y, x + w, y + h])
        total_gt += 1

    pred_by_img = defaultdict(lambda: defaultdict(list))
    for prediction in pred_list:
        pred_by_img[prediction["image_id"]][prediction["category_id"]].append(prediction)

    conf_thresholds = np.linspace(0.01, 0.99, num_conf_steps)
    precision_list = []
    recall_list = []
    f1_list = []
    f2_list = []
    for conf in conf_thresholds:
        _, _, _, _, precision, recall, f1, f2 = _match_at_confidence(
            gt_by_img=gt_by_img,
            pred_by_img=pred_by_img,
            conf_thresh=conf,
            iou_thresh=iou_match,
            total_gt=total_gt,
        )
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        f2_list.append(f2)

    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)
    f1_array = np.array(f1_list)
    f2_array = np.array(f2_list)
    best_f1_idx = int(np.argmax(f1_array)) if len(f1_array) > 0 else 0
    best_f2_idx = int(np.argmax(f2_array)) if len(f2_array) > 0 else 0

    curves = {
        "confidence": conf_thresholds.tolist(),
        "precision": precision_array.tolist(),
        "recall": recall_array.tolist(),
        "f1": f1_array.tolist(),
        "f2": f2_array.tolist(),
    }
    result = {
        "best_f1_conf": float(conf_thresholds[best_f1_idx]),
        "best_f1": float(f1_array[best_f1_idx]),
        "best_f1_precision": float(precision_array[best_f1_idx]),
        "best_f1_recall": float(recall_array[best_f1_idx]),
        "best_f2_conf": float(conf_thresholds[best_f2_idx]),
        "best_f2": float(f2_array[best_f2_idx]),
        "best_f2_precision": float(precision_array[best_f2_idx]),
        "best_f2_recall": float(recall_array[best_f2_idx]),
        "curve_data": curves,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "metrics_curves.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["confidence_threshold", "precision", "recall", "f1", "f2"])
            for values in zip(conf_thresholds, precision_array, recall_array, f1_array, f2_array):
                writer.writerow([round(value, 4) for value in values])

        def _plot(x, y, xlabel, ylabel, title, filename):
            plt.figure(figsize=(7, 5))
            plt.plot(x, y, color="#1f77b4", linewidth=2.0)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title(title, fontsize=13)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tight_layout()
            plt.savefig(out_path / filename, dpi=180)
            plt.close()

        _plot(conf_thresholds, precision_array, "Confidence Threshold", "Precision", "Precision vs. Confidence", "precision_curve.png")
        _plot(conf_thresholds, recall_array, "Confidence Threshold", "Recall", "Recall vs. Confidence", "recall_curve.png")
        _plot(conf_thresholds, f1_array, "Confidence Threshold", "F1 Score", "F1 Score vs. Confidence", "f1_curve.png")
        _plot(conf_thresholds, f2_array, "Confidence Threshold", "F2 Score", "F2 Score vs. Confidence", "f2_curve.png")
        _plot(recall_array, precision_array, "Recall", "Precision", f"Precision-Recall Curve (IoU={iou_match})", "pr_curve.png")

    return result


def calculate_metrics(
    pred_list: list[dict],
    coco_gt: Any,
    iou_thrs: np.ndarray | None = None,
    iou_match: float = 0.50,
    conf_threshold: float = 0.50,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute detection metrics and fixed-threshold operating points.

    Parameters
    ----------
    pred_list : list[dict]
        List of detections in COCO format:
        [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]
    coco_gt : pycocotools.coco.COCO or str or Path
        Ground-truth COCO object or path to annotations JSON.
    iou_thrs : np.ndarray | None
        Array of IoU thresholds for evaluation (default: 0.20 to 0.95 with 0.05 step).
    iou_match : float
        IoU threshold for curves and fixed operating point matching (default: 0.50).
    conf_threshold : float
        Configurable fixed confidence threshold for operating-point counts (TP, FP, FN, TN).
    output_dir : str | Path | None
        Directory where curve plots, CSVs, and JSON summaries are saved.
    Returns
    -------
    dict[str, Any]
        Complete dictionary of metrics, AP values, per-class stats, and operating points.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if isinstance(coco_gt, (str, Path)):
        coco_gt = COCO(str(coco_gt))

    if iou_thrs is None:
        iou_thrs = np.round(np.arange(0.20, 0.951, 0.05), 2)

    # 1. Organize Ground Truth
    gt_by_img = defaultdict(lambda: defaultdict(list))
    total_gt = 0
    for ann in coco_gt.dataset.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_img[ann["image_id"]][ann["category_id"]].append([x, y, x + w, y + h])
        total_gt += 1

    # 2. Organize Predictions
    pred_by_img = defaultdict(lambda: defaultdict(list))
    for p in pred_list:
        pred_by_img[p["image_id"]][p["category_id"]].append(p)

    # 3. Fixed Operating Point (TP, FP, FN, TN at conf_threshold & iou_match)
    tp, fp, fn, tn, p_at_conf, r_at_conf, f1_at_conf, f2_at_conf = _match_at_confidence(
        gt_by_img=gt_by_img,
        pred_by_img=pred_by_img,
        conf_thresh=conf_threshold,
        iou_thresh=iou_match,
        total_gt=total_gt,
    )

    # 4. Standard COCOeval calculation
    aps: dict[str, float] = {}
    per_class_p: list[float] = []
    per_class_r: list[float] = []

    if pred_list and total_gt > 0:
        coco_dt = coco_gt.loadRes(pred_list)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.iouThrs = np.array(iou_thrs, dtype=np.float64)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        prec = coco_eval.eval["precision"]  # [T, R, K, A, M]
        for idx, t in enumerate(iou_thrs):
            p_slice = prec[idx, :, :, 0, -1]
            valid_p = p_slice[p_slice > -1]
            aps[f"AP@{t:.2f}"] = float(valid_p.mean()) if valid_p.size > 0 else 0.0

        # Class-level P/R at IoU 0.50
        iou50_idx = int(np.argmin(np.abs(iou_thrs - 0.50)))
        prec_slice = coco_eval.eval["precision"][iou50_idx, :, :, 0, -1]  # [R, K]
        for k in range(prec_slice.shape[1]):
            valid_k = prec_slice[:, k][prec_slice[:, k] > -1]
            per_class_p.append(float(valid_k.mean()) if valid_k.size > 0 else 0.0)

        rec_slice = coco_eval.eval["recall"][iou50_idx, :, 0, -1]          # [K]
        per_class_r = [float(r) if r > -1 else 0.0 for r in rec_slice]
    else:
        for t in iou_thrs:
            aps[f"AP@{t:.2f}"] = 0.0

    def _find_ap(target: float) -> float:
        idx = int(np.argmin(np.abs(np.asarray(iou_thrs) - target)))
        return aps.get(f"AP@{iou_thrs[idx]:.2f}", 0.0)

    std_thrs = [aps[f"AP@{t:.2f}"] for t in iou_thrs if t >= 0.499]
    map50_95 = float(np.mean(std_thrs)) if std_thrs else 0.0
    map20_95 = float(np.mean(list(aps.values()))) if aps else 0.0

    # 5. AP thresholds from 0.20 to 0.95 with 0.05 step
    sweep_16_thrs = np.round(np.arange(0.20, 0.951, 0.05), 2)
    map_20_to_95 = [_find_ap(t) for t in sweep_16_thrs]

    metrics: dict[str, Any] = {
        # mAP Metrics
        "map20":               _find_ap(0.20),
        "map50":               _find_ap(0.50),
        "map75":               _find_ap(0.75),
        "map95":               _find_ap(0.95),
        "map20_95":            map20_95,
        "map50_95":            map50_95,
        "map_20_to_95":        map_20_to_95,
        "all_iou_thresholds":  aps,
        # Class aggregates (at IoU 0.50)
        "precision_mean":      float(np.mean(per_class_p)) if per_class_p else 0.0,
        "recall_mean":         float(np.mean(per_class_r)) if per_class_r else 0.0,
        "precision_per_class": per_class_p,
        "recall_per_class":    per_class_r,
        # Configurable fixed operating point
        "conf_threshold":      float(conf_threshold),
        "iou_match":           float(iou_match),
        "tp":                  int(tp),
        "fp":                  int(fp),
        "fn":                  int(fn),
        "tn":                  int(tn),
        "precision_at_conf":   round(p_at_conf, 4),
        "recall_at_conf":      round(r_at_conf, 4),
        "f1_at_conf":          round(f1_at_conf, 4),
        "f2_at_conf":          round(f2_at_conf, 4),
    }

    # 6. Save Plots & Curve Data if output_dir is given
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save operating points summary
        op_summary = {
            "fixed_operating_point": {
                "conf_threshold": float(conf_threshold),
                "iou_match": float(iou_match),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": round(p_at_conf, 4),
                "recall": round(r_at_conf, 4),
                "f1": round(f1_at_conf, 4),
                "f2": round(f2_at_conf, 4),
            },
        }
        with open(out_path / "operating_points_summary.json", "w") as fh:
            json.dump(op_summary, fh, indent=2)

        with open(out_path / "metrics_summary.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# 2. Improvement Deltas
# ---------------------------------------------------------------------------

def compute_improvement(baseline_result: dict, adapted_result: dict) -> dict:
    """Compare an adapted-model run against a source-only baseline run on the
    SAME target dataset/split.

    Returns dict with keys prefixed by 'delta_'.
    """
    def _delta(key: str):
        b, a = baseline_result.get(key), adapted_result.get(key)
        return None if (b is None or a is None) else float(a) - float(b)

    return {
        "run_name": f"{adapted_result.get('run_name', 'adapted')}_vs_{baseline_result.get('run_name', 'baseline')}",
        "delta_map50_95":          _delta("map50_95"),
        "delta_map20_95":          _delta("map20_95"),
        "delta_map20":             _delta("map20"),
        "delta_map50":             _delta("map50"),
        "delta_map75":             _delta("map75"),
        "delta_precision_mean":    _delta("precision_mean"),
        "delta_recall_mean":       _delta("recall_mean"),
        "delta_confidence_mean":   _delta("confidence_mean"),
    }


# ---------------------------------------------------------------------------
# 3. Domain Gap (MMD) & Multi-Group t-SNE Visualization
# ---------------------------------------------------------------------------

def compute_domain_gap_mmd(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """RBF-kernel Maximum Mean Discrepancy (MMD).

    Lower values indicate that source and target feature distributions are
    closer together (less domain gap).
    """
    def _rbf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_sq = np.sum(x ** 2, axis=1, keepdims=True)
        y_sq = np.sum(y ** 2, axis=1, keepdims=True)
        dist = x_sq + y_sq.T - 2.0 * (x @ y.T)
        return np.exp(-gamma * dist)

    k_ss = _rbf(source_embeddings, source_embeddings).mean()
    k_tt = _rbf(target_embeddings, target_embeddings).mean()
    k_st = _rbf(source_embeddings, target_embeddings).mean()
    return float(k_ss + k_tt - 2.0 * k_st)

def get_dataset_plot_label(dataset_label: str, split: str, context: str | None = None) -> str:
    """Build a consistent t-SNE/legend label, e.g. 'SeaDronesSee test (afo_humans_060)'.

    dataset_label : human-readable dataset name (e.g. cfg["label"]), not the
                     internal key ("A"/"B"/"C").
    split         : "train" or "test".
    context       : optional extra disambiguator shown in parentheses
                     (e.g. a dataset-folder name). Omit if not available.
    """
    if context:
        return f"{dataset_label} {split} ({context})"
    return f"{dataset_label} {split}"

def plot_tsne(
    embeddings_or_groups: dict[str, np.ndarray] | np.ndarray,
    target_or_out_path: np.ndarray | str | Path,
    out_path: str | Path | None = None,
    title: str = "Embedding Space (t-SNE)",
    perplexity: int = 30,
    random_state: int = 42,
) -> str:
    """Visualize embeddings projected to 2-D via t-SNE.

    Supports two calling conventions:
    1. Multi-group (preferred):
       plot_tsne(
           {"source_train": arr1, "source_test": arr2, "target_B": arr3},
           out_path="tsne.png",
           title="...",
       )
    2. Backward-compatible two-array:
       plot_tsne(source_emb, target_emb, out_path="tsne.png", title="...")

    Returns
    -------
    str
        Absolute path to the saved figure PNG.
    """
    from sklearn.manifold import TSNE

    # Normalize arguments
    if isinstance(embeddings_or_groups, dict):
        embedding_groups = embeddings_or_groups
        save_path = str(target_or_out_path)
    else:
        embedding_groups = {
            "source": embeddings_or_groups,
            "target": target_or_out_path,
        }
        save_path = str(out_path)

    # Filter out empty groups
    valid_groups = {k: v for k, v in embedding_groups.items() if v is not None and len(v) > 0}
    if not valid_groups:
        raise ValueError("No valid embeddings provided to plot_tsne.")

    # Combine data and labels
    combined_list = []
    labels_list = []
    for name, arr in valid_groups.items():
        combined_list.append(arr)
        labels_list.extend([name] * len(arr))

    combined = np.vstack(combined_list)
    labels = np.array(labels_list)

    # Adaptive perplexity for small sample sets
    eff_perplexity = max(1, min(perplexity, len(combined) - 1))

    proj = TSNE(
        n_components=2,
        init="pca",
        perplexity=eff_perplexity,
        random_state=random_state,
    ).fit_transform(combined)

    plt.figure(figsize=(8, 7))
    cmap = mpl.colormaps["tab10"].resampled(len(valid_groups))

    for idx, name in enumerate(valid_groups.keys()):
        mask = labels == name
        color = cmap(idx)
        plt.scatter(
            proj[mask, 0],
            proj[mask, 1],
            label=name,
            alpha=0.65,
            s=18,
            color=color,
        )

    # CAMBIO: Leyenda cuadrada, dentro del gráfico y con borde negro sutil
    plt.legend(
        loc="upper right", 
        fancybox=False, 
        edgecolor="black",
        framealpha=0.9
    )
    
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()

    return str(Path(save_path).resolve())


def plot_labeled_tsne(
    embedding_groups: list[dict[str, Any]],
    title: str = "Embedding Space (t-SNE)",
    perplexity: int = 30,
    random_state: int = 42,
):
    """Visualize multiple embedding groups in 2D as a single wandb.Image.

    Unlike ``plot_tsne`` (which writes a PNG to disk and gives every group a
    single flat style), this groups points by ``dataset_label`` while letting
    each *entry* carry its own marker -- e.g. a dataset's train split can be
    "o" and its test split "D" while still using its own legend/color. Meant
    to replace one-off/individual t-SNE calls with a single richer plot
    logged straight to W&B.

    Each entry in embedding_groups must include:
      - dataset_label: label shown in the legend
      - embeddings: np.ndarray of shape (n_samples, feature_dim)
      - marker: matplotlib marker symbol for the points in that entry

    Returns
    -------
    wandb.Image
        Ready to pass straight into ``run.log({...})``.
    """
    _require_wandb()
    import wandb
    from sklearn.manifold import TSNE

    entries = [e for e in embedding_groups if e.get("embeddings") is not None and len(e["embeddings"])]
    if not entries:
        raise ValueError("No embeddings provided for t-SNE plotting.")

    combined = np.vstack([entry["embeddings"] for entry in entries])
    eff_perplexity = max(1, min(perplexity, len(combined) - 1))
    proj = TSNE(
        n_components=2,
        init="pca",
        perplexity=eff_perplexity,
        random_state=random_state,
    ).fit_transform(combined)

    colors = plt.cm.tab10.colors
    cursor = 0
    grouped_entries: dict[str, list[tuple[dict[str, Any], int, int]]] = {}

    for entry in entries:
        n_points = len(entry["embeddings"])
        grouped_entries.setdefault(entry["dataset_label"], []).append((entry, cursor, cursor + n_points))
        cursor += n_points

    plt.figure(figsize=(7, 7))
    for index, (dataset_label, slices) in enumerate(grouped_entries.items()):
        x_values, y_values = [], []
        marker = slices[0][0]["marker"]
        for entry, start, end in slices:
            x_values.extend(proj[start:end, 0].tolist())
            y_values.extend(proj[start:end, 1].tolist())
        plt.scatter(
            x_values, y_values,
            label=dataset_label,
            alpha=0.75, s=22,
            c=[colors[index % len(colors)]],
            marker=marker,
            edgecolors="none",
        )

    plt.title(title, fontsize=13)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Dataset",  loc="upper right", borderaxespad=0.0)
    # # plt.tight_layout()

    fig = plt.gcf()
    image = wandb.Image(fig)
    plt.close(fig)
    return image
# ---------------------------------------------------------------------------
# 4. CSV Export
# ---------------------------------------------------------------------------

def save_results_csv(results_list: list[dict], out_path: str) -> None:
    """Write a list of result dicts to a CSV file."""
    if not results_list:
        return
    cleaned_rows = []
    for r in results_list:
        row = {}
        for k, v in r.items():
            if k == "curve_data":
                continue
            elif isinstance(v, (list, tuple, dict)):
                row[k] = json.dumps(v)
            else:
                row[k] = v
        cleaned_rows.append(row)
    keys = sorted(set().union(*(r.keys() for r in cleaned_rows)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cleaned_rows)


# ---------------------------------------------------------------------------
# 5. W&B Logging Helpers (import-guarded)
# ---------------------------------------------------------------------------

def _require_wandb():
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "wandb is required for W&B logging. Install it with: pip install wandb"
        ) from exc


def log_dataset_config(
    run: Any,
    dataset_configs: dict,
    source_name: str,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
) -> None:
    """Summarize run and dataset configurations in run.config."""
    _require_wandb()
    config_update: dict = {
        "source_dataset_name":  source_name,
        "source_dataset_label": dataset_configs[source_name].get("label", source_name),
        "target_dataset_names": [n for n in dataset_configs if n != source_name],
    }
    if config_path is not None:
        config_update["model_config_path"] = str(config_path)
    if checkpoint_path is not None:
        config_update["weights_path"] = str(checkpoint_path)

    for name, cfg in dataset_configs.items():
        config_update[f"dataset_{name}_label"] = cfg.get("label", name)
    run.config.update(config_update)


def log_target_scalars(run: Any, target_name: str, result: dict) -> None:
    """Log final numeric metrics to run.summary."""
    _require_wandb()
    for key, value in result.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            run.summary[f"{target_name}/{key}"] = value


def log_target_plots(run: Any, target_name: str, plots_dir: Path | None) -> None:
    """Log PNG/JPG plots from plots_dir to W&B under {target_name}/plots/*."""
    _require_wandb()
    import wandb

    if plots_dir is None or not Path(plots_dir).exists():
        return
    plots_dir = Path(plots_dir)
    image_paths = sorted(plots_dir.glob("*.png")) + sorted(plots_dir.glob("*.jpg"))
    log_dict = {
        f"{target_name}/plots/{p.stem}": wandb.Image(str(p)) for p in image_paths
    }
    if log_dict:
        run.log(log_dict)


def log_target_curves(run: Any, target_name: str, curve_data: dict[str, list[float]]) -> None:
    """Log interactive W&B native line charts for precision, recall, F1, F2, and PR curves."""
    _require_wandb()
    import wandb

    if not curve_data or "confidence" not in curve_data:
        return

    confs = curve_data["confidence"]
    precs = curve_data["precision"]
    recs = curve_data["recall"]
    f1s = curve_data["f1"]
    f2s = curve_data["f2"]

    # Interactive table for confidence-sweep metrics
    curve_table = wandb.Table(
        data=[[c, p, r, f1, f2] for c, p, r, f1, f2 in zip(confs, precs, recs, f1s, f2s)],
        columns=["confidence", "precision", "recall", "f1", "f2"],
    )

    # Interactive table for PR curve
    pr_table = wandb.Table(
        data=[[r, p] for r, p in zip(recs, precs)],
        columns=["recall", "precision"],
    )

    run.log({
        f"{target_name}/curves/precision_curve": wandb.plot.line(
            curve_table, "confidence", "precision", title=f"Precision vs. Confidence ({target_name})"
        ),
        f"{target_name}/curves/recall_curve": wandb.plot.line(
            curve_table, "confidence", "recall", title=f"Recall vs. Confidence ({target_name})"
        ),
        f"{target_name}/curves/f1_curve": wandb.plot.line(
            curve_table, "confidence", "f1", title=f"F1 vs. Confidence ({target_name})"
        ),
        f"{target_name}/curves/f2_curve": wandb.plot.line(
            curve_table, "confidence", "f2", title=f"F2 vs. Confidence ({target_name})"
        ),
        f"{target_name}/curves/pr_curve": wandb.plot.line(
            pr_table, "recall", "precision", title=f"PR Curve ({target_name})"
        ),
    })


def log_metrics_table(
    run: Any,
    results_list: list[dict],
    table_name: str = "Table 1: Metrics",
) -> None:
    """Log interactive browsable metrics table with exact ordered columns."""
    _require_wandb()
    import wandb

    if not results_list:
        return

    columns = [
        "source_name",
        "target_name",
        "map20_95",
        "map50_95",
        "map_20_to_95",
        "conf_threshold",
        "precision_mean",
        "recall_mean",
        "confidence_mean",
        "confidence_median",
        "confidence_std",
        "n_detections",
        "th_confidence_mean",
        "th_confidence_median",
        "th_confidence_std",
        "th_n_detections"
    ]

    rows = []
    for r in results_list:
        source_id = r.get("source_dataset_name", r.get("source_name", ""))
        target_id = r.get("target_dataset_name", r.get("target_name", ""))
        row = [
            source_id,
            target_id,
            r.get("map20_95", 0.0),
            r.get("map50_95", 0.0),
            r.get("map_20_to_95", []),
            r.get("conf_threshold", None),
            r.get("precision_mean", 0.0),
            r.get("recall_mean", 0.0),
            r.get("confidence_mean", None),
            r.get("confidence_median", None),
            r.get("confidence_std", None),
            r.get("n_detections", 0),
            r.get("th_confidence_mean", None),
            r.get("th_confidence_std", None),
            r.get("th_confidence_median", None),
            r.get("th_n_detections", 0),
        ]
        rows.append(row)

    table = wandb.Table(columns=columns, data=rows)
    run.log({table_name: table})


def log_operating_points_table(
    run: Any,
    results_list: list[dict],
    table_name: str = "Table 2: Operating Points & Domain Gap",
) -> None:
    """Log supplemental table for fixed operating points, curve optimal points, and domain gap."""
    _require_wandb()
    import wandb

    if not results_list:
        return

    columns = [
        "source_name",
        "target_name",
        "conf_threshold",
        "iou_match",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision_at_conf",
        "recall_at_conf",
        "f1_at_conf",
        "f2_at_conf",
        "best_f1_conf",
        "best_f1",
        "best_f2_conf",
        "best_f2",
        "domain_gap_mmd",
    ]

    rows = []
    for r in results_list:
        source_id = r.get("source_dataset_name", r.get("source_name", ""))
        target_id = r.get("target_dataset_name", r.get("target_name", ""))
        row = [
            source_id,
            target_id,
            r.get("conf_threshold", None),
            r.get("iou_match", None),
            r.get("tp", 0),
            r.get("fp", 0),
            r.get("fn", 0),
            r.get("tn", 0),
            r.get("precision_at_conf", None),
            r.get("recall_at_conf", None),
            r.get("f1_at_conf", None),
            r.get("f2_at_conf", None),
            r.get("best_f1_conf", None),
            r.get("best_f1", None),
            r.get("best_f2_conf", None),
            r.get("best_f2", None),
            r.get("domain_gap_mmd", None),
        ]
        rows.append(row)

    table = wandb.Table(columns=columns, data=rows)
    run.log({table_name: table})


def log_results_table(
    run: Any,
    results_list: list[dict],
    table_name: str = "results_table",
) -> None:
    """Log a browsable W&B Table covering all evaluated targets."""
    _require_wandb()
    import wandb

    if not results_list:
        return
    keys = sorted(set().union(*(r.keys() for r in results_list if isinstance(r, dict))))
    # Filter out large curve_data from flat table
    keys = [k for k in keys if k != "curve_data"]
    table = wandb.Table(
        columns=keys,
        data=[[r.get(k) for k in keys] for r in results_list],
    )
    run.log({table_name: table})
