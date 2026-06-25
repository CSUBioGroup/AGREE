"""Core implementation of AGREE.

AGREE builds a cluster-association graph from multiple base partitions,
embeds cluster nodes with weighted DeepWalk, projects the embeddings back to
spots/cells, and obtains a final consensus partition with optional spatial
label refinement.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable, Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import normalize

ClusteringMethod = Literal["agglomerative", "kmeans"]


@dataclass
class AGREEConfig:
    """Configuration for AGREE consensus clustering."""

    seed: int = 42
    embedding_dim: int = 32
    walk_length: int = 20
    num_walks: int = 100
    window_size: int = 2
    spatial_k: int = 6
    spatial_beta: float = 0.3
    graph_edge_threshold: float = 0.001
    clustering_method: ClusteringMethod = "agglomerative"
    refine: bool = True
    refine_alpha: float = 2.0
    refine_consensus_threshold: float = 0.6
    word2vec_epochs: int = 30


def fix_seed(seed: int = 42) -> None:
    """Set Python and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _encode_labels(labels: Iterable[object]) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(labels).astype(str), sort=True)
    return codes.astype(int)


def _validate_labels(labels_list: list[np.ndarray]) -> None:
    if len(labels_list) == 0:
        raise ValueError("labels_list must contain at least one base partition.")
    lengths = {len(labels) for labels in labels_list}
    if len(lengths) != 1:
        raise ValueError("All base partitions must have the same number of samples.")


def _prepare_labels(labels_list: Iterable[Iterable[object]]) -> list[np.ndarray]:
    encoded = [_encode_labels(labels) for labels in labels_list]
    _validate_labels(encoded)
    return encoded


def compute_partition_weights(labels_list: Iterable[Iterable[object]]) -> np.ndarray:
    """Estimate base-partition reliability by average mutual consistency.

    Each partition receives the mean NMI against all other partitions, then the
    vector is normalized to have mean 1.0. This preserves the scale of the
    sample-cluster incidence matrix while up-weighting mutually supported base
    partitions.
    """

    labels = _prepare_labels(labels_list)
    n_partitions = len(labels)
    if n_partitions <= 1:
        return np.ones(n_partitions, dtype=float)

    nmi_matrix = np.zeros((n_partitions, n_partitions), dtype=float)
    for i in range(n_partitions):
        for j in range(i, n_partitions):
            score = normalized_mutual_info_score(labels[i], labels[j])
            nmi_matrix[i, j] = score
            nmi_matrix[j, i] = score

    weights = nmi_matrix.mean(axis=1)
    return weights / (weights.mean() + 1e-10)


def build_cluster_graph(
    labels_list: Iterable[Iterable[object]],
    partition_weights: Iterable[float] | None = None,
    spatial_coords: np.ndarray | None = None,
    spatial_k: int = 6,
    beta: float = 0.3,
    edge_threshold: float = 0.001,
) -> tuple[nx.Graph, sparse.csc_matrix, int]:
    """Build the weighted global cluster graph used by AGREE.

    Nodes are clusters from all base partitions. Edges combine weighted Jaccard
    overlap between cluster memberships and optional spatial-neighborhood
    support.
    """

    labels = _prepare_labels(labels_list)
    n_samples = len(labels[0])
    if partition_weights is None:
        weights = compute_partition_weights(labels)
    else:
        weights = np.asarray(list(partition_weights), dtype=float)
        if len(weights) != len(labels):
            raise ValueError("partition_weights must match labels_list length.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    offset = 0

    for partition_idx, partition_labels in enumerate(labels):
        _, inverse = np.unique(partition_labels, return_inverse=True)
        n_clusters = int(inverse.max()) + 1
        rows.extend(range(n_samples))
        cols.extend((inverse + offset).tolist())
        data.extend([float(weights[partition_idx])] * n_samples)
        offset += n_clusters

    n_cluster_nodes = offset
    membership = sparse.csc_matrix((data, (rows, cols)), shape=(n_samples, n_cluster_nodes))

    intersection = (membership.T @ membership).toarray()
    cluster_sizes = np.diag(intersection)
    union = cluster_sizes[:, None] + cluster_sizes[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        w_jaccard = intersection / union
    w_jaccard = np.nan_to_num(w_jaccard)

    w_final = w_jaccard.copy()
    if spatial_coords is not None and beta > 0:
        spatial = np.asarray(spatial_coords, dtype=float)
        if spatial.shape[0] != n_samples:
            raise ValueError("spatial_coords must have one row per sample.")
        adjacency = kneighbors_graph(
            spatial,
            n_neighbors=spatial_k,
            mode="connectivity",
            include_self=False,
        )
        spatial_conn = (membership.T @ adjacency @ membership).toarray()
        diag = np.diag(spatial_conn)
        denominator = np.sqrt(np.outer(diag, diag))
        denominator[denominator == 0] = 1.0
        w_spatial = spatial_conn / denominator
        np.fill_diagonal(w_spatial, 1.0)
        w_final = (1.0 - beta) * w_jaccard + beta * w_spatial

    graph = nx.Graph()
    graph.add_nodes_from(range(n_cluster_nodes))
    row_idx, col_idx = np.triu_indices(n_cluster_nodes, k=1)
    edge_weights = w_final[row_idx, col_idx]
    valid = edge_weights > edge_threshold
    if np.any(valid):
        graph.add_weighted_edges_from(
            zip(row_idx[valid].tolist(), col_idx[valid].tolist(), edge_weights[valid].tolist())
        )

    return graph, membership, n_cluster_nodes


def _generate_weighted_walks(
    graph: nx.Graph,
    n_cluster_nodes: int,
    num_walks: int,
    walk_length: int,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    walks: list[list[str]] = []
    nodes = list(range(n_cluster_nodes))
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start_node in nodes:
            walk = [start_node]
            while len(walk) < walk_length:
                current = walk[-1]
                neighbors = list(graph.neighbors(current))
                if not neighbors:
                    break
                weights = [float(graph[current][node]["weight"]) for node in neighbors]
                if sum(weights) == 0:
                    break
                walk.append(rng.choices(neighbors, weights=weights, k=1)[0])
            walks.append([str(node) for node in walk])
    return walks


def _run_deepwalk_skipgram(
    walks: list[list[str]],
    n_cluster_nodes: int,
    embedding_dim: int,
    window_size: int,
    seed: int,
    epochs: int,
) -> np.ndarray:
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "AGREE requires gensim for DeepWalk embeddings. Install it with "
            "`pip install gensim` or `pip install -r requirements.txt`."
        ) from exc

    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_dim,
        window=window_size,
        min_count=0,
        sg=1,
        workers=1,
        epochs=epochs,
        seed=seed,
    )
    embedding = np.zeros((n_cluster_nodes, embedding_dim), dtype=float)
    for node in range(n_cluster_nodes):
        key = str(node)
        if key in model.wv:
            embedding[node] = model.wv[key]
    return embedding


def _compute_sample_vectors(
    membership: sparse.csc_matrix,
    cluster_embeddings: np.ndarray,
    method: ClusteringMethod,
) -> np.ndarray:
    sample_vectors = membership @ cluster_embeddings
    if method.lower() == "kmeans":
        sample_vectors = normalize(sample_vectors, norm="l2", axis=1)
    return np.asarray(sample_vectors)


def _run_clustering_in_vector_space(
    sample_vectors: np.ndarray,
    n_clusters: int,
    seed: int,
    method: ClusteringMethod,
) -> np.ndarray:
    if method.lower() == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=20,
            random_state=seed,
        )
        return clusterer.fit_predict(sample_vectors)

    if method.lower() == "agglomerative":
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="complete",
            )
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="cosine",
                linkage="complete",
            )
        return clusterer.fit_predict(sample_vectors)

    raise ValueError("clustering_method must be 'kmeans' or 'agglomerative'.")


def refine_labels(
    labels: Iterable[object],
    spatial_coords: np.ndarray,
    alpha: float = 2.0,
    consensus_threshold: float = 0.6,
) -> np.ndarray:
    """Refine labels by adaptive spatial-majority voting.

    The neighborhood radius is inferred from the median nearest-neighbor
    distance and multiplied by ``alpha``. A spot is relabeled only if a single
    neighboring label reaches ``consensus_threshold``.
    """

    labels_array = np.asarray(labels).copy()
    spatial = np.asarray(spatial_coords, dtype=float)
    if len(labels_array) != spatial.shape[0]:
        raise ValueError("labels and spatial_coords must have the same length.")

    nearest = NearestNeighbors(n_neighbors=2).fit(spatial)
    distances, _ = nearest.kneighbors(spatial)
    dynamic_radius = alpha * float(np.median(distances[:, 1]))

    radius_model = NearestNeighbors(radius=dynamic_radius).fit(spatial)
    _, radius_indices = radius_model.radius_neighbors(spatial)

    refined = labels_array.copy()
    for i, neighbors in enumerate(radius_indices):
        neighbors = neighbors[neighbors != i]
        if len(neighbors) == 0:
            continue
        neighbor_labels = labels_array[neighbors]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] / len(neighbors) >= consensus_threshold:
            refined[i] = unique_labels[best]

    return refined


def run_agree(
    labels_list: Iterable[Iterable[object]],
    n_clusters: int,
    spatial_coords: np.ndarray | None = None,
    config: AGREEConfig | None = None,
    return_details: bool = False,
    verbose: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Run AGREE consensus clustering.

    Parameters
    ----------
    labels_list
        Base partitions. Each element is a vector with one label per spot/cell.
    n_clusters
        Target number of consensus clusters.
    spatial_coords
        Optional ``n_samples x 2`` spatial coordinate matrix.
    config
        Optional :class:`AGREEConfig`.
    return_details
        If ``True``, return intermediate weights and embeddings metadata.
    verbose
        Print partition weights and graph size.
    """

    cfg = config or AGREEConfig()
    fix_seed(cfg.seed)
    labels = _prepare_labels(labels_list)
    partition_weights = compute_partition_weights(labels)

    if verbose:
        print("Partition weights:")
        for idx, weight in enumerate(partition_weights):
            print(f"  partition_{idx}: {weight:.4f}")

    graph, membership, n_cluster_nodes = build_cluster_graph(
        labels,
        partition_weights=partition_weights,
        spatial_coords=spatial_coords,
        spatial_k=cfg.spatial_k,
        beta=cfg.spatial_beta,
        edge_threshold=cfg.graph_edge_threshold,
    )
    walks = _generate_weighted_walks(
        graph,
        n_cluster_nodes=n_cluster_nodes,
        num_walks=cfg.num_walks,
        walk_length=cfg.walk_length,
        seed=cfg.seed,
    )
    cluster_embeddings = _run_deepwalk_skipgram(
        walks,
        n_cluster_nodes=n_cluster_nodes,
        embedding_dim=cfg.embedding_dim,
        window_size=cfg.window_size,
        seed=cfg.seed,
        epochs=cfg.word2vec_epochs,
    )
    sample_vectors = _compute_sample_vectors(
        membership,
        cluster_embeddings,
        method=cfg.clustering_method,
    )
    consensus = _run_clustering_in_vector_space(
        sample_vectors,
        n_clusters=n_clusters,
        seed=cfg.seed,
        method=cfg.clustering_method,
    )

    if cfg.refine and spatial_coords is not None:
        consensus = refine_labels(
            consensus,
            spatial_coords=spatial_coords,
            alpha=cfg.refine_alpha,
            consensus_threshold=cfg.refine_consensus_threshold,
        )

    if not return_details:
        return consensus

    details = {
        "partition_weights": partition_weights,
        "n_cluster_nodes": n_cluster_nodes,
        "n_graph_edges": graph.number_of_edges(),
        "n_walks": len(walks),
        "config": cfg,
    }
    return consensus, details


def run_deepwalk_ec(
    labels_list: Iterable[Iterable[object]],
    K_CLUSTERS: int,
    spatial_coords: np.ndarray | None = None,
    spatial_beta: float = 0.3,
    seed: int = 42,
    d: int = 32,
    L: int = 20,
    w: int = 2,
    method: ClusteringMethod = "agglomerative",
    refine: bool | str = True,
) -> np.ndarray:
    """Backward-compatible wrapper for the original notebook entry point."""

    refine_flag = refine if isinstance(refine, bool) else str(refine).lower() == "true"
    config = AGREEConfig(
        seed=seed,
        embedding_dim=d,
        walk_length=L,
        window_size=w,
        spatial_beta=spatial_beta,
        clustering_method=method,
        refine=refine_flag,
    )
    return run_agree(
        labels_list=labels_list,
        n_clusters=K_CLUSTERS,
        spatial_coords=spatial_coords,
        config=config,
    )
