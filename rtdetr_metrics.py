"""
rtdetr_metrics.py
=================
RT-DETRv4 adapter for the shared domain-adaptation evaluation protocol.

Only the RT-DETRv4-specific parts live here: YAML config + checkpoint loading,
dataloader construction, postprocessor output -> COCO results, and the forward
hook used for image embeddings. The evaluation protocol itself is in
``tools/da_pipeline.py`` and the metrics in ``tools/metrics_common.py``.

Sections
--------
1.  Category mapping & config loading
2.  Embedding hook utilities
3.  RTDETRAdapter
4.  CLI entry-point
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tools.da_pipeline import (
    DetectorAdapter,
    SplitOutput,
    build_common_cli_parser,
    run_cli,
)


# ---------------------------------------------------------------------------
# 1. Category Mapping & Config Loading
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
    """Locate and return the YAMLConfig class (RT-DETRv4 ``engine`` or D-FINE-style ``src``)."""
    sys.path.insert(0, os.path.abspath(repo_root))
    for mod in ("engine.core", "src.core"):
        try:
            m = __import__(mod, fromlist=["YAMLConfig"])
            return m.YAMLConfig
        except ImportError:
            continue
    raise ImportError("Could not import YAMLConfig from engine.core or src.core.")


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


def find_embedding_module(model: torch.nn.Module, embedding_module: str) -> tuple[str, torch.nn.Module]:
    """Resolve the module to hook, falling back to common RT-DETR component names."""
    submodules = dict(model.named_modules())
    if embedding_module in submodules:
        return embedding_module, submodules[embedding_module]
    for cand in ("encoder", "hybrid_encoder", "backbone"):
        if cand in submodules:
            return cand, submodules[cand]
    raise ValueError(
        f"Module '{embedding_module}' not found in model. Available: {list(submodules.keys())[:20]}"
    )


# ---------------------------------------------------------------------------
# 3. RTDETRAdapter
# ---------------------------------------------------------------------------

class RTDETRAdapter(DetectorAdapter):
    """RT-DETRv4 inference through the YAMLConfig engine.

    The model is built and loaded once per checkpoint; a dataloader is built per
    split, and predictions + embeddings come from the same forward pass.
    """

    name = "RT-DETRv4"
    slug = "rtdetr"

    def __init__(
        self,
        config_path: str,
        device: str | torch.device | None = None,
        repo_root: str = ".",
        embedding_module: str = "encoder",
        batch_size: int = 8,
        num_workers: int = 4,
        remap_mscoco: bool = False,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.repo_root = repo_root
        self.embedding_module = embedding_module
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.remap_mscoco = remap_mscoco

        self.cfg = None
        self.model: torch.nn.Module | None = None
        self.postprocessor: torch.nn.Module | None = None
        self._hook = EmbeddingHook()
        self._hook_handle = None

    def _load(self, checkpoint_path: str) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        YAMLConfig = get_yaml_config(self.repo_root)
        self.cfg = YAMLConfig(self.config_path, resume=checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
        self.cfg.model.load_state_dict(state_dict)

        self.model = self.cfg.model.to(self.device).eval()
        self.postprocessor = self.cfg.postprocessor.to(self.device).eval()

        self.embedding_module, target_mod = find_embedding_module(self.model, self.embedding_module)
        self._hook_handle = target_mod.register_forward_hook(self._hook)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "config_path": self.config_path}

    def build_dataloader(self, img_folder: str, ann_file: str):
        """Build a fresh val-style dataloader for the given split."""
        if self.cfg is None:
            raise RuntimeError("Call load(checkpoint_path) before running inference.")
        ds_cfg = self.cfg.yaml_cfg["val_dataloader"]
        ds_cfg["dataset"]["img_folder"] = img_folder
        ds_cfg["dataset"]["ann_file"] = ann_file
        ds_cfg["total_batch_size"] = self.batch_size
        ds_cfg["num_workers"] = self.num_workers
        return self.cfg.build_dataloader("val_dataloader")

    def run_split(
        self,
        split: dict,
        *,
        min_score: float = 0.001,
        with_predictions: bool = True,
        with_embeddings: bool = False,
        verbose: bool = True,
    ) -> SplitOutput:
        dataloader = self.build_dataloader(split["img_folder"], split["ann_file"])
        coco_gt = dataloader.dataset.coco
        label_to_cat = get_label_to_category_map(coco_gt, remap_mscoco=self.remap_mscoco)

        try:
            from tqdm import tqdm
            progress = tqdm(dataloader, desc="Inference", leave=False)
        except ImportError:
            progress = dataloader

        predictions: list[dict] = []
        image_ids: list[int] = []
        file_names: list[str] = []
        emb_list: list[np.ndarray] = []

        with torch.no_grad():
            for samples, targets in progress:
                samples = samples.to(self.device)
                batch_img_ids = [int(t["image_id"]) for t in targets]
                image_ids.extend(batch_img_ids)

                self._hook.features = None
                outputs = self.model(samples)

                if with_predictions:
                    orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(self.device)
                    results = self.postprocessor(outputs, orig_sizes)
                    for img_id, res in zip(batch_img_ids, results):
                        predictions.extend(self._to_coco(img_id, res, label_to_cat, min_score))

                if with_embeddings and self._hook.features is not None:
                    batch_emb = pool_features(self._hook.features).detach().cpu().numpy()
                    for k, img_id in enumerate(batch_img_ids):
                        file_names.append(coco_gt.imgs[img_id]["file_name"])
                        emb_list.append(batch_emb[k])

        embeddings = None
        if with_embeddings:
            if not emb_list:
                raise RuntimeError("No embeddings were captured during extraction.")
            embeddings = np.stack(emb_list)

        if verbose:
            if with_predictions:
                print(f"[*] Total predictions collected: {len(predictions)}")
            if embeddings is not None:
                print(f"[*] Embeddings extracted (shape: {embeddings.shape})")

        return SplitOutput(
            predictions=predictions,
            coco_gt=coco_gt,
            image_ids=image_ids,
            file_names=file_names,
            embeddings=embeddings,
        )

    @staticmethod
    def _to_coco(img_id: int, res: dict, label_to_cat: dict[int, int], min_score: float) -> list[dict]:
        labels = res["labels"].detach().cpu().numpy()
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()

        out = []
        for lbl, box, score in zip(labels, boxes, scores):
            if score < min_score:
                continue
            x1, y1, x2, y2 = box.tolist()
            out.append({
                "image_id": img_id,
                "category_id": label_to_cat.get(int(lbl), int(lbl)),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })
        return out


# ---------------------------------------------------------------------------
# 4. CLI Entry-Point
# ---------------------------------------------------------------------------

def _main_cli():
    p = build_common_cli_parser("Evaluate RT-DETRv4 and extract DA metrics & embeddings.")
    p.add_argument("--config", "-c", required=True, help="YAML config file")
    p.add_argument("--repo-root", default=".", help="Path to RT-DETRv4 repository root")
    p.add_argument("--embedding-module", default="encoder", help="Module to hook for embeddings")
    p.add_argument("--remap-mscoco", action="store_true", help="Set flag for standard MS COCO 80-class mapping")
    args = p.parse_args()

    adapter = RTDETRAdapter(
        config_path=args.config,
        device=args.device,
        repo_root=args.repo_root,
        embedding_module=args.embedding_module,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        remap_mscoco=args.remap_mscoco,
    )
    run_cli(adapter, args)


if __name__ == "__main__":
    _main_cli()
