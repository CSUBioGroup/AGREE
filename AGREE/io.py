"""Input/output helpers for AGREE command-line workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_partition_table(
    path: str | Path,
    method_columns: list[str] | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None, pd.DataFrame]:
    """Read base partitions from a CSV/TSV table."""

    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    table = pd.read_csv(path, sep=sep)
    if method_columns is None:
        exclude = {x_col, y_col, "ground_truth", "GroundTruth", "label"}
        method_columns = [col for col in table.columns if col not in exclude]
    labels = [table[col].to_numpy() for col in method_columns]
    spatial = None
    if x_col is not None and y_col is not None:
        spatial = table[[x_col, y_col]].to_numpy(dtype=float)
    return labels, spatial, table


def read_h5ad_partitions(
    path: str | Path,
    method_columns: list[str],
    spatial_key: str = "spatial",
) -> tuple[list[np.ndarray], np.ndarray | None, object]:
    """Read base partitions and spatial coordinates from an AnnData file."""

    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Reading .h5ad files requires scanpy.") from exc

    adata = sc.read_h5ad(path)
    missing = [col for col in method_columns if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing obs columns in {path}: {missing}")
    labels = [adata.obs[col].to_numpy() for col in method_columns]
    spatial = adata.obsm[spatial_key] if spatial_key in adata.obsm else None
    return labels, spatial, adata
