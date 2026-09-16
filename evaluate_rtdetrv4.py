#!/usr/bin/env python
"""
evaluate_rtdetrv4.py
====================
Thin CLI wrapper around ``rtdetr_metrics.run_eval``.

All heavy logic lives in ``rtdetr_metrics.py``.  Use this script for quick
command-line evaluations; import ``rtdetr_metrics`` directly for programmatic
/ notebook use.

Supports
--------
- Custom COCO datasets & MS COCO
- mAP@[0.20:0.95], mAP@0.20, mAP@0.50, mAP@0.75
- Precision, Recall, F1, F2 vs. confidence-threshold curves + PR curve
- Image embedding extraction (saved as .npy per dataset)
- Optional W&B logging (set WANDB_API_KEY in ~/.env)

Example
-------
Single-dataset evaluation (disk output only)::

    python evaluate_rtdetrv4.py \\
        --config   configs/rtdetr/rtdetr_r50vd_6x_coco.yml \\
        --checkpoint weights/best.pth \\
        --ann-file   datasets/A/annotations/test.json \\
        --img-folder datasets/A/images/test \\
        --output-dir eval_output_A

With W&B logging::

    python evaluate_rtdetrv4.py ... --wandb-project my_project

Multi-target evaluation (programmatic, not CLI)::

    from rtdetr_metrics import run_eval

    dataset_configs = {
        "A": {
            "label":      "SeaDronesSee",
            "config":     "configs/rtdetr/rtdetr_r50vd.yml",
            "ann_file":   "datasets/A/test.json",
            "img_folder": "datasets/A/images/test",
            "iou": 0.20, "conf": 0.001,
        },
        "B": {
            "label":      "SynBase",
            "config":     "configs/rtdetr/rtdetr_r50vd.yml",
            "ann_file":   "datasets/B/test.json",
            "img_folder": "datasets/B/images/test",
            "iou": 0.20, "conf": 0.001,
        },
    }

    run_eval(
        run_name        = "my_experiment",
        config_path     = "configs/rtdetr/rtdetr_r50vd.yml",
        checkpoint_path = "weights/best.pth",
        dataset_configs = dataset_configs,
        source_name     = "A",
        result_folder   = "rtdetr_results",
        wandb_project   = "my_project",   # optional
    )
"""

# Re-export the CLI entry point from rtdetr_metrics so this file remains
# a thin, importable wrapper with no duplicated logic.
from rtdetr_metrics import _main_cli, run_eval

if __name__ == "__main__":
    # _main_cli()  
    dataset_configs = {
        "A": {
            "label":      "SeaDronesSee",
            "config":     "configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml",
            "ann_file":   "datasets/processed/sds_jp_coco/instances_test.json",
            "img_folder": "datasets/raw/SeaDronesSee_Juanpe/images/test",
            "iou": 0.20, "conf": 0.001,
        },
        "B": {
            "label":      "SynBase",
            "config":     "configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml",
            "ann_file":   "datasets/processed/synbase_yolov5/instances_test.json",
            "img_folder": "datasets/processed/synbase_yolov5/images/test",
            "iou": 0.20, "conf": 0.001,
        },
    }
    
    run_eval(
        run_name        = "sds_jp_metrics",
        config_path     = "configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml",
        checkpoint_path = "outputs/sds_jp_transfer_rtv4_hgnetv2_s_coco/best_stg1.pth",
        dataset_configs = dataset_configs,
        source_name     = "A",
        result_folder   = "rtdetr_results",
        wandb_project   = "rtdetr",   # optional
    )