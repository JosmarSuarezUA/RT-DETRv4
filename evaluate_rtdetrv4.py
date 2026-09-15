#!/usr/bin/env python
"""
evaluate_rtdetrv4.py
====================
Fixed and optimized evaluation + embedding extraction tool for RT-DETRv4.
Supports:
  - Custom COCO datasets & MS COCO
  - mAP@[0.20:0.95], mAP@0.20, mAP@0.50, mAP@0.75
  - Precision, Recall, F1, F2 vs. Confidence threshold curves + PR curve
  - Fixed-size image embedding extraction (saved to .npz and .pt)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou


# --------------------------------------------------------------------------
# Import helper
# --------------------------------------------------------------------------
def get_yaml_config(repo_root: str):
    sys.path.insert(0, os.path.abspath(repo_root))
    for mod in ["engine.core", "src.core"]:
        try:
            m = __import__(mod, fromlist=["YAMLConfig"])
            return m.YAMLConfig
        except ImportError:
            continue
    raise ImportError("Could not import YAMLConfig from engine.core or src.core.")


# --------------------------------------------------------------------------
# Embedding hook
# --------------------------------------------------------------------------
class EmbeddingHook:
    def __init__(self):
        self.features = None

    def __call__(self, module, inp, out):
        self.features = out


def pool_features(feat_output) -> torch.Tensor:
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


# --------------------------------------------------------------------------
# Standard MS COCO 80-class label-to-category mapping
# --------------------------------------------------------------------------
MSCOCO_CATEGORY2LABEL = {
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
MSCOCO_LABEL2CATEGORY = {v: k for k, v in MSCOCO_CATEGORY2LABEL.items()}


def get_label_to_category_map(coco_gt, remap_mscoco: bool):
    """
    Returns a dict mapping model predicted class index (0..num_classes-1)
    to the dataset's actual category_id.
    """
    cat_ids = sorted(coco_gt.getCatIds())
    if remap_mscoco and len(cat_ids) == 80 and max(cat_ids) == 90:
        return MSCOCO_LABEL2CATEGORY
    # For custom datasets: 0-th model output maps to the first sorted category id, etc.
    return {idx: cat_id for idx, cat_id in enumerate(cat_ids)}


# --------------------------------------------------------------------------
# Evaluator: Custom IoU Range COCO Evaluation
# --------------------------------------------------------------------------
def run_custom_cocoeval(coco_gt, predictions, iou_thrs):
    if not predictions:
        return None
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array(iou_thrs, dtype=np.float64)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def compute_extended_ap_dict(coco_eval, iou_thrs):
    if coco_eval is None:
        return {"error": "No valid predictions available to evaluate."}

    prec = coco_eval.eval["precision"]  # [T, R, K, A, M]
    aps = {}
    for idx, t in enumerate(iou_thrs):
        p = prec[idx, :, :, 0, -1]
        valid_p = p[p > -1]
        aps[f"AP@{t:.2f}"] = float(valid_p.mean()) if valid_p.size > 0 else 0.0

    def find_nearest_ap(target_iou):
        idx = int(np.argmin(np.abs(np.array(iou_thrs) - target_iou)))
        return aps[f"AP@{iou_thrs[idx]:.2f}"]

    std_coco_thrs = [aps[f"AP@{t:.2f}"] for t in iou_thrs if t >= 0.499]

    return {
        "mAP_0.20_0.95": float(np.mean(list(aps.values()))),
        "mAP_0.50_0.95": float(np.mean(std_coco_thrs)) if std_coco_thrs else 0.0,
        "mAP_0.20": find_nearest_ap(0.20),
        "mAP_0.50": find_nearest_ap(0.50),
        "mAP_0.75": find_nearest_ap(0.75),
        "all_iou_thresholds": aps,
    }


# --------------------------------------------------------------------------
# Curves: Precision, Recall, F1, F2 vs. Confidence & PR Curve
# --------------------------------------------------------------------------
def compute_curves(coco_gt, predictions, iou_thresh=0.5, num_conf_steps=100):
    conf_thresholds = np.linspace(0.01, 0.99, num_conf_steps)

    # Organize GT by image_id and category_id
    gt_by_img = defaultdict(lambda: defaultdict(list))
    total_gt = 0
    for ann in coco_gt.dataset["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_img[ann["image_id"]][ann["category_id"]].append([x, y, x + w, y + h])
        total_gt += 1

    # Organize predictions by image_id and category_id
    pred_by_img = defaultdict(lambda: defaultdict(list))
    for p in predictions:
        pred_by_img[p["image_id"]][p["category_id"]].append(p)

    precision_list = []
    recall_list = []
    f1_list = []
    f2_list = []

    for conf in conf_thresholds:
        tp = 0
        fp = 0

        for img_id, cat_dict in gt_by_img.items():
            for cat_id, gt_boxes_list in cat_dict.items():
                gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32)
                matched = np.zeros(len(gt_boxes), dtype=bool)

                preds = [p for p in pred_by_img[img_id].get(cat_id, []) if p["score"] >= conf]
                preds = sorted(preds, key=lambda x: -x["score"])

                for p in preds:
                    x, y, w, h = p["bbox"]
                    p_box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
                    ious = box_iou(p_box, gt_boxes)[0].numpy()
                    ious[matched] = -1.0
                    best_idx = int(ious.argmax()) if ious.size > 0 else -1

                    if best_idx >= 0 and ious[best_idx] >= iou_thresh:
                        tp += 1
                        matched[best_idx] = True
                    else:
                        fp += 1

        # Also count false positives for predicted classes not present in that image
        for img_id, cat_dict in pred_by_img.items():
            for cat_id, p_list in cat_dict.items():
                if cat_id not in gt_by_img.get(img_id, {}):
                    fp += sum(1 for p in p_list if p["score"] >= conf)

        p = tp / max(tp + fp, 1e-9)
        r = tp / max(total_gt, 1e-9)
        f1 = 2 * p * r / max(p + r, 1e-9)
        f2 = 5 * p * r / max(4 * p + r, 1e-9)

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)
        f2_list.append(f2)

    return {
        "conf_thresholds": conf_thresholds,
        "precision": np.array(precision_list),
        "recall": np.array(recall_list),
        "f1": np.array(f1_list),
        "f2": np.array(f2_list),
        "total_gt": total_gt,
        "iou_thresh": iou_thresh,
    }


def save_curves_and_plots(curve_data, output_dir):
    confs = curve_data["conf_thresholds"]
    prec = curve_data["precision"]
    rec = curve_data["recall"]
    f1 = curve_data["f1"]
    f2 = curve_data["f2"]

    # 1. Save CSV
    csv_file = os.path.join(output_dir, "metrics_curves.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["confidence_threshold", "precision", "recall", "f1", "f2"])
        for c, p, r, _f1, _f2 in zip(confs, prec, rec, f1, f2):
            writer.writerow([round(c, 4), round(p, 4), round(r, 4), round(_f1, 4), round(_f2, 4)])

    # 2. Plotting helper
    def make_plot(x, y, xlabel, ylabel, title, fname):
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, color="#1f77b4", linewidth=2.0)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=180)
        plt.close()

    make_plot(confs, prec, "Confidence Threshold", "Precision", "Precision vs. Confidence", "precision_curve.png")
    make_plot(confs, rec, "Confidence Threshold", "Recall", "Recall vs. Confidence", "recall_curve.png")
    make_plot(confs, f1, "Confidence Threshold", "F1 Score", "F1 Score vs. Confidence", "f1_curve.png")
    make_plot(confs, f2, "Confidence Threshold", "F2 Score", "F2 Score vs. Confidence", "f2_curve.png")
    make_plot(rec, prec, "Recall", "Precision", f"Precision-Recall Curve (IoU={curve_data['iou_thresh']})", "pr_curve.png")

    # 3. Find optimal points
    best_f1_idx = int(np.argmax(f1))
    best_f2_idx = int(np.argmax(f2))

    summary = {
        "best_f1_operating_point": {
            "optimal_confidence_threshold": float(confs[best_f1_idx]),
            "f1": float(f1[best_f1_idx]),
            "precision": float(prec[best_f1_idx]),
            "recall": float(rec[best_f1_idx]),
        },
        "best_f2_operating_point": {
            "optimal_confidence_threshold": float(confs[best_f2_idx]),
            "f2": float(f2[best_f2_idx]),
            "precision": float(prec[best_f2_idx]),
            "recall": float(rec[best_f2_idx]),
        },
    }

    with open(os.path.join(output_dir, "operating_points_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# --------------------------------------------------------------------------
# Main Workflow
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate RT-DETRv4 and extract metrics/embeddings.")
    parser.add_argument("--repo-root", default=".", help="Path to RT-DETRv4 repository root")
    parser.add_argument("--config", "-c", required=True, help="YAML config file")
    parser.add_argument("--checkpoint", "-r", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--img-folder", required=True, help="Path to COCO images directory")
    parser.add_argument("--ann-file", required=True, help="Path to COCO annotations (.json)")
    parser.add_argument("--output-dir", default="eval_output", help="Directory to save results")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--iou-match", type=float, default=0.5, help="IoU threshold for PR/F1/F2 curves")
    parser.add_argument("--min-score", type=float, default=0.001, help="Score threshold for saving predictions")
    parser.add_argument("--remap-mscoco", type=lambda s: s.lower() == "true", default=None,
                        help="Explicitly set True for MS COCO 80-class remapping, False for custom datasets.")
    parser.add_argument("--extract-embeddings", action="store_true", help="Extract and save image embeddings")
    parser.add_argument("--embedding-module", default="encoder", help="Module to hook (e.g., 'encoder' or 'backbone')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load Config and update dataset paths before building dataloader
    YAMLConfig = get_yaml_config(args.repo_root)
    cfg = YAMLConfig(args.config, resume=args.checkpoint)

    # Override paths in config dictionary
    if "val_dataloader" in cfg.yaml_cfg:
        cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = args.img_folder
        cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = args.ann_file
        if args.batch_size:
            cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
        if args.num_workers is not None:
            cfg.yaml_cfg["val_dataloader"]["num_workers"] = args.num_workers

    # 2. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state_dict)

    # Use standard eval mode (DO NOT call .deploy() on postprocessor for native PyTorch evaluation!)
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    # 3. Dataloader & Dataset
    val_dataloader = cfg.val_dataloader
    dataset = val_dataloader.dataset
    coco_gt = dataset.coco

    # Determine whether MS COCO remapping applies
    remap_mscoco = args.remap_mscoco
    if remap_mscoco is None:
        remap_mscoco = bool(cfg.yaml_cfg.get("remap_mscoco_category", False))

    label_to_cat = get_label_to_category_map(coco_gt, remap_mscoco)
    print(f"[*] Category mapping initialized. Total dataset categories: {len(coco_gt.getCatIds())}")

    # 4. Attach Embedding Hook
    hook_handle = None
    embedding_records = {"image_id": [], "file_name": [], "embedding": []}
    if args.extract_embeddings:
        submodules = dict(model.named_modules())
        target_mod = submodules.get(args.embedding_module)
        if target_mod is None:
            # Fallback to backbone or first child
            for cand in ["encoder", "backbone", "hybrid_encoder"]:
                if cand in submodules:
                    target_mod = submodules[cand]
                    args.embedding_module = cand
                    break
        if target_mod is None:
            raise ValueError(f"Module '{args.embedding_module}' not found in model.")
        hook = EmbeddingHook()
        hook_handle = target_mod.register_forward_hook(hook)
        print(f"[*] Attached embedding hook to: '{args.embedding_module}'")

    # 5. Inference loop
    try:
        from tqdm import tqdm
        progress = tqdm(val_dataloader, desc="Evaluating")
    except ImportError:
        progress = val_dataloader

    predictions = []
    print("[*] Running inference...")

    with torch.no_grad():
        for samples, targets in progress:
            samples = samples.to(device)

            # Pull native orig_size [h, w] directly from targets (crucial fix!)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            image_ids = [int(t["image_id"]) for t in targets]

            outputs = model(samples)
            results = postprocessor(outputs, orig_target_sizes)

            # Extract embeddings if requested
            if args.extract_embeddings and hook.features is not None:
                batch_emb = pool_features(hook.features).detach().cpu().numpy()
                for k, img_id in enumerate(image_ids):
                    embedding_records["image_id"].append(img_id)
                    embedding_records["file_name"].append(coco_gt.imgs[img_id]["file_name"])
                    embedding_records["embedding"].append(batch_emb[k])

            # Process detections per image
            for img_id, res in zip(image_ids, results):
                labels = res["labels"].detach().cpu().numpy()
                boxes = res["boxes"].detach().cpu().numpy()
                scores = res["scores"].detach().cpu().numpy()

                for lbl, box, score in zip(labels, boxes, scores):
                    if score < args.min_score:
                        continue
                    cat_id = label_to_cat.get(int(lbl), int(lbl))
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    })

    if hook_handle:
        hook_handle.remove()

    print(f"[*] Total predictions collected: {len(predictions)}")
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f)

    # 6. Compute Extended Range mAP
    print("[*] Computing COCO metrics (mAP@[0.20:0.95], mAP@0.20, mAP@0.50, mAP@0.75)...")
    iou_thrs = np.round(np.arange(0.20, 0.951, 0.05), 2)
    coco_eval = run_custom_cocoeval(coco_gt, predictions, iou_thrs)
    metrics_summary = compute_extended_ap_dict(coco_eval, iou_thrs)

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 50)
    print("                EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in metrics_summary.items():
        if k != "all_iou_thresholds":
            print(f"  {k:28s} : {v:.4f}" if isinstance(v, float) else f"  {k:28s} : {v}")
    print("=" * 50 + "\n")

    # 7. Compute Curves
    print("[*] Computing Precision, Recall, F1, and F2 curves...")
    curve_data = compute_curves(coco_gt, predictions, iou_thresh=args.iou_match)
    operating_summary = save_curves_and_plots(curve_data, args.output_dir)

    print("[*] Best Operating Points:")
    print(json.dumps(operating_summary, indent=2))

    # 8. Save Embeddings
    if args.extract_embeddings and embedding_records["image_id"]:
        print("[*] Saving embeddings...")
        embs = np.stack(embedding_records["embedding"])
        np.savez_compressed(
            os.path.join(args.output_dir, "embeddings.npz"),
            embeddings=embs,
            image_ids=np.array(embedding_records["image_id"]),
            file_names=np.array(embedding_records["file_name"]),
        )
        torch.save(
            {img_id: torch.from_numpy(emb) for img_id, emb in zip(embedding_records["image_id"], embs)},
            os.path.join(args.output_dir, "embeddings.pt")
        )
        print(f"[*] Embeddings successfully saved. Shape: {embs.shape}")

    print(f"\n[✓] All results and plots generated in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()