"""AGREE: spatial transcriptomics ensemble consensus clustering."""

from .model import (
    AGREEConfig,
    compute_partition_weights,
    refine_labels,
    run_agree,
    run_deepwalk_ec,
)

__all__ = [
    "AGREEConfig",
    "compute_partition_weights",
    "refine_labels",
    "run_agree",
    "run_deepwalk_ec",
]

__version__ = "0.1.0"
