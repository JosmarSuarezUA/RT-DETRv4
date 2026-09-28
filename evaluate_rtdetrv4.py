#!/usr/bin/env python
"""
evaluate_rtdetrv4.py
====================
Cross-domain evaluation of RT-DETRv4: every source checkpoint is evaluated on
the test split of every dataset (see ``tools/da_pipeline.py`` for the protocol).

- ``dataset_configs`` describes data only (splits + evaluation thresholds) and
  is shared across models.
- ``CHECKPOINTS`` maps each source dataset to the checkpoint trained on it.
- Model settings (config, batch size, device, ...) go to ``RTDETRAdapter``.

For a single-dataset CLI evaluation use ``python rtdetr_metrics.py --help``.
"""

from rtdetr_metrics import RTDETRAdapter
from tools.da_pipeline import run_all_sources

dataset_configs = {
    "A": {
        "label": "SeaDronesSee",
        "splits": {
            "train": {
                "ann_file": "/data2/detection_datasets/processed/sds_jp_coco/instances_train.json",
                "img_folder": "/data2/detection_datasets/raw/SeaDronesSee_Juanpe/images/training",
            },
            "val": {
                "ann_file": "/data2/detection_datasets/processed/sds_jp_coco/instances_val.json",
                "img_folder": "/data2/detection_datasets/raw/SeaDronesSee_Juanpe/images/validation",
            },
            "test": {
                "ann_file": "/data2/detection_datasets/processed/sds_jp_coco/instances_test.json",
                "img_folder": "/data2/detection_datasets/raw/SeaDronesSee_Juanpe/images/test",
            },
        },
        "iou": 0.20,          # IoU for matching (curves + fixed operating point)
        "min_score": 0.001,   # drop predictions below this score
    },
    "B": {
        "label": "SynBase",
        "splits": {
            "train": {
                "ann_file": "/data2/detection_datasets/processed/synbase_coco/instances_train.json",
                "img_folder": "/data2/detection_datasets/processed/synbase_yolov5/images/train",
            },
            "val": {
                "ann_file": "/data2/detection_datasets/processed/synbase_coco/instances_val.json",
                "img_folder": "/data2/detection_datasets/processed/synbase_yolov5/images/val",
            },
            "test": {
                "ann_file": "/data2/detection_datasets/processed/synbase_coco/instances_test.json",
                "img_folder": "/data2/detection_datasets/processed/synbase_yolov5/images/test",
            },
        },
        "iou": 0.20,
        "min_score": 0.001,
    },
}

CHECKPOINTS = {
    "A": "outputs/sds_jp_transfer_rtv4_hgnetv2_s_coco/best_stg1.pth",
    "B": "outputs/synbase_rtv4_hgnetv2_s_coco/best_stg1.pth",
}

if __name__ == "__main__":
    adapter = RTDETRAdapter(config_path="configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml")
    run_all_sources(
        adapter,
        checkpoints=CHECKPOINTS,
        dataset_configs=dataset_configs,
        result_root="rtdetr_results",
        wandb_project="rtdetrv4",   # optional
    )
