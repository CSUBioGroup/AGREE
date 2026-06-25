"""Minimal synthetic example for AGREE."""

from __future__ import annotations

import numpy as np

from AGREE import AGREEConfig, run_agree


def main() -> None:
    rng = np.random.default_rng(7)
    n_per_cluster = 30
    centers = np.array([[0, 0], [4, 0], [2, 3.5]], dtype=float)
    spatial = np.vstack([
        center + rng.normal(scale=0.35, size=(n_per_cluster, 2))
        for center in centers
    ])
    truth = np.repeat(np.arange(len(centers)), n_per_cluster)

    base_partitions = []
    for _ in range(5):
        labels = truth.copy()
        flip = rng.choice(len(labels), size=8, replace=False)
        labels[flip] = rng.integers(0, len(centers), size=len(flip))
        base_partitions.append(labels)

    config = AGREEConfig(seed=7, num_walks=20, walk_length=12, embedding_dim=16)
    pred, details = run_agree(
        base_partitions,
        n_clusters=3,
        spatial_coords=spatial,
        config=config,
        return_details=True,
        verbose=True,
    )
    print("Predicted labels:", pred[:12].tolist(), "...")
    print("Details:", {k: v for k, v in details.items() if k != "config"})


if __name__ == "__main__":
    main()
