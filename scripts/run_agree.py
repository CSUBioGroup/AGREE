"""Run AGREE from CSV/TSV or h5ad inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from AGREE import AGREEConfig, run_agree
from AGREE.io import read_h5ad_partitions, read_partition_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGREE consensus clustering.")
    parser.add_argument("--input", required=True, help="Input .h5ad, .csv, or .tsv file.")
    parser.add_argument("--methods", required=True, help="Comma-separated base partition columns.")
    parser.add_argument("--n-clusters", type=int, required=True, help="Target number of clusters.")
    parser.add_argument("--output", required=True, help="Output .csv or .h5ad path.")
    parser.add_argument("--x-col", default=None, help="Spatial x column for CSV/TSV input.")
    parser.add_argument("--y-col", default=None, help="Spatial y column for CSV/TSV input.")
    parser.add_argument("--spatial-key", default="spatial", help="AnnData obsm key for coordinates.")
    parser.add_argument("--result-column", default="AGREE", help="Name of output label column.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--num-walks", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--spatial-k", type=int, default=6)
    parser.add_argument("--spatial-beta", type=float, default=0.3)
    parser.add_argument("--clustering-method", choices=["agglomerative", "kmeans"], default="agglomerative")
    parser.add_argument("--no-refine", action="store_true", help="Disable spatial label refinement.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    if input_path.suffix.lower() == ".h5ad":
        labels, spatial, container = read_h5ad_partitions(input_path, methods, args.spatial_key)
    else:
        labels, spatial, container = read_partition_table(input_path, methods, args.x_col, args.y_col)

    config = AGREEConfig(
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window_size=args.window_size,
        spatial_k=args.spatial_k,
        spatial_beta=args.spatial_beta,
        clustering_method=args.clustering_method,
        refine=not args.no_refine,
    )
    result = run_agree(
        labels,
        n_clusters=args.n_clusters,
        spatial_coords=spatial,
        config=config,
        verbose=args.verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".h5ad" and output_path.suffix.lower() == ".h5ad":
        container.obs[args.result_column] = pd.Categorical(result.astype(str))
        container.write_h5ad(output_path)
    else:
        if hasattr(container, "obs"):
            table = container.obs.copy()
        else:
            table = container.copy()
        table[args.result_column] = result
        table.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
