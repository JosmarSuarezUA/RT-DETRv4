#!/usr/bin/env python
"""
evaluate_rtdetrv4.py
====================
Cross-domain evaluation of RT-DETRv4: every source checkpoint is evaluated on
the test split of every dataset. The protocol, metrics and adapter live in the
shared ``da-eval`` package (https://github.com/JosmarSuarezUA/da-eval).

- Datasets are defined once in ``da-eval/configs/datasets.yaml`` (shared by all models).
- ``CHECKPOINTS`` maps each source dataset to the checkpoint trained on it.
- Model settings (config, batch size, device, ...) go to ``RTDETRv4Adapter``.

Equivalent CLI::

    uv run da-eval --model rtdetrv4 \\
        --config configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml \\
        --datasets ../da-eval/configs/datasets.yaml \\
        --checkpoint A=outputs/sds_jp_transfer_rtv4_hgnetv2_s_coco/best_stg1.pth \\
        --checkpoint B=outputs/synbase_rtv4_hgnetv2_s_coco/best_stg1.pth \\
        --result-root rtdetr_results --wandb-project rtdetrv4
"""

from pathlib import Path

from da_eval import load_dataset_configs, run_all_sources
from da_eval.adapters.yaml_engine import RTDETRv4Adapter

DATASETS = Path(__file__).resolve().parent.parent / "da-eval" / "configs" / "datasets.yaml"

CHECKPOINTS = {
    "A": "outputs/sds_jp_transfer_rtv4_hgnetv2_s_coco/best_stg1.pth",
    "B": "outputs/synbase_rtv4_hgnetv2_s_coco/best_stg1.pth",
    "C": "outputs/afo_rtv4_hgnetv2_s_coco/best_stg1.pth"
}

if __name__ == "__main__":
    adapter = RTDETRv4Adapter(config_path="configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml")
    run_all_sources(
        adapter,
        checkpoints=CHECKPOINTS,
        dataset_configs=load_dataset_configs(DATASETS),
        result_root="rtdetr_results",
        wandb_project="rtdetrv4",   # optional
    )
