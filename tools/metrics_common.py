"""
tools/metrics_common.py
=======================
Framework-agnostic shared utilities for detection evaluation pipelines.

These functions have **zero ML-framework dependencies** (only numpy,
matplotlib, csv, pathlib) so they can be safely imported from both the
Ultralytics/YOLO pipeline (da_metrics.py) and the RT-DETRv4 pipeline
(rtdetr_metrics.py) without pulling in unneeded heavy dependencies.

Sections
--------
1.  Improvement deltas  (compare adapted vs. baseline run)
2.  Domain gap          (MMD metric + t-SNE visualization)
3.  CSV export
4.  W&B logging helpers (import-guarded; require ``wandb`` installed)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. Improvement deltas
# ---------------------------------------------------------------------------

def compute_improvement(baseline_result: dict, adapted_result: dict) -> dict:
    """Compare an adapted-model run against a source-only baseline run on the
    SAME target dataset/split.

    This is what "improved precision/recall" means in most DA papers: not the
    raw value, but the *delta* over naïve transfer.

    Parameters
    ----------
    baseline_result : dict
        Metrics dict produced by ``run_full_evaluation`` for the baseline model.
    adapted_result : dict
        Metrics dict produced by ``run_full_evaluation`` for the adapted model.

    Returns
    -------
    dict
        Keys prefixed with ``delta_`` holding ``adapted - baseline`` values.
        Any key missing in either input is reported as ``None``.
    """
    def _delta(key: str):
        b, a = baseline_result.get(key), adapted_result.get(key)
        return None if (b is None or a is None) else float(a) - float(b)

    return {
        "run_name": f"{adapted_result['run_name']}_vs_{baseline_result['run_name']}",
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
# 2. Domain gap: MMD metric + t-SNE visualization
# ---------------------------------------------------------------------------

def compute_domain_gap_mmd(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """RBF-kernel Maximum Mean Discrepancy (MMD).

    Lower values indicate that source and target feature distributions are
    closer together (less domain gap).

    Parameters
    ----------
    source_embeddings : np.ndarray, shape (N_s, D)
    target_embeddings : np.ndarray, shape (N_t, D)
    gamma : float
        RBF kernel bandwidth parameter.

    Returns
    -------
    float
        MMD² estimate.
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


def plot_tsne(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    out_path: str,
    title: str = "Source vs Target embeddings (t-SNE)",
) -> str:
    """Visualize source vs target embeddings projected to 2-D via t-SNE.

    Requires ``scikit-learn`` and ``matplotlib``.

    Parameters
    ----------
    source_embeddings : np.ndarray, shape (N_s, D)
    target_embeddings : np.ndarray, shape (N_t, D)
    out_path : str
        File path to save the PNG (created/overwritten).
    title : str
        Plot title.

    Returns
    -------
    str
        Absolute path to the saved figure (same as ``out_path``).
    """
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = np.vstack([source_embeddings, target_embeddings])
    labels = np.array(
        ["source"] * len(source_embeddings) + ["target"] * len(target_embeddings)
    )
    proj = TSNE(n_components=2, init="pca", random_state=42).fit_transform(combined)

    plt.figure(figsize=(6, 6))
    for label, color in [("source", "tab:blue"), ("target", "tab:orange")]:
        mask = labels == label
        plt.scatter(proj[mask, 0], proj[mask, 1], label=label, alpha=0.6, s=15, c=color)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(Path(out_path).resolve())


# ---------------------------------------------------------------------------
# 3. CSV export
# ---------------------------------------------------------------------------

def save_results_csv(results_list: list[dict], out_path: str) -> None:
    """Write a list of result dicts to a CSV file.

    All unique keys across all dicts are used as columns (sorted). Missing
    values for any particular row are left blank.

    Parameters
    ----------
    results_list : list[dict]
        Each dict is one evaluation result row.
    out_path : str
        Destination CSV path.
    """
    if not results_list:
        return
    keys = sorted(set().union(*(r.keys() for r in results_list)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_list)


# ---------------------------------------------------------------------------
# 4. W&B logging helpers  (import-guarded)
# ---------------------------------------------------------------------------

def _require_wandb():
    """Lazy import guard — raises a helpful error if wandb is not installed."""
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "wandb is required for W&B logging. Install it with: pip install wandb"
        ) from exc


def log_dataset_config(run: Any, dataset_configs: dict, source_name: str) -> None:
    """Summarize every dataset's label in ``run.config``.

    This gives a single place (the W&B Overview tab) that spells out what
    dataset labels A, B, C … N mean, regardless of how many datasets you are
    running with.

    Parameters
    ----------
    run : wandb.Run
    dataset_configs : dict
        Full ``dataset_configs`` mapping (as defined in the calling script).
    source_name : str
        Key in ``dataset_configs`` corresponding to the source/training dataset.
    """
    _require_wandb()
    config_update: dict = {
        "source_dataset_name":  source_name,
        "source_dataset_label": dataset_configs[source_name]["label"],
        "target_dataset_names": [n for n in dataset_configs if n != source_name],
    }
    for name, cfg in dataset_configs.items():
        config_update[f"dataset_{name}_label"] = cfg["label"]
    run.config.update(config_update)


def log_target_scalars(run: Any, target_name: str, result: dict) -> None:
    """Log final numeric metrics to ``run.summary`` (not ``run.log``).

    These are single final values (not a time series), so ``summary`` is the
    semantically correct place. It also makes them sortable/filterable columns
    when comparing this run against others in the project.

    Parameters
    ----------
    run : wandb.Run
    target_name : str
        Short identifier for the target dataset (e.g. ``"B"``).
    result : dict
        Flat dict of metric name → scalar value.
    """
    _require_wandb()
    for key, value in result.items():
        if isinstance(value, (int, float)):
            run.summary[f"{target_name}/{key}"] = value


def log_target_plots(run: Any, target_name: str, plots_dir: Path | None) -> None:
    """Log PNG/JPG plots from ``plots_dir`` to W&B under ``{target_name}/plots/*``.

    Parameters
    ----------
    run : wandb.Run
    target_name : str
    plots_dir : Path | None
        Directory containing ``*.png`` / ``*.jpg`` files to upload.
    """
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


def log_results_table(
    run: Any,
    results_list: list[dict],
    table_name: str = "results_table",
) -> None:
    """Log a single browsable W&B Table covering all targets in this run.

    Handles array-valued columns (e.g. per-class precision/recall) that
    ``run.summary`` scalars cannot hold.

    Parameters
    ----------
    run : wandb.Run
    results_list : list[dict]
    table_name : str
        Key under which the table appears inside the run.
    """
    _require_wandb()
    import wandb

    if not results_list:
        return
    keys = sorted(set().union(*(r.keys() for r in results_list)))
    table = wandb.Table(
        columns=keys,
        data=[[r.get(k) for k in keys] for r in results_list],
    )
    run.log({table_name: table})

