import numpy as np

from AGREE import AGREEConfig, compute_partition_weights, run_agree


def test_compute_partition_weights_shape():
    labels = [
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 0, 0, 2, 2]),
    ]
    weights = compute_partition_weights(labels)
    assert weights.shape == (3,)
    assert np.isfinite(weights).all()


def test_run_agree_synthetic_smoke():
    spatial = np.array(
        [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]],
        dtype=float,
    )
    labels = [
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([0, 0, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 0, 1]),
    ]
    config = AGREEConfig(
        seed=1,
        embedding_dim=8,
        walk_length=6,
        num_walks=5,
        word2vec_epochs=5,
        spatial_k=2,
    )
    pred = run_agree(labels, n_clusters=2, spatial_coords=spatial, config=config)
    assert len(pred) == 6
    assert len(set(pred.tolist())) == 2
