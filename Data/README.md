# Data

Place input datasets here, or download them from the public repository that will
be listed in the final manuscript.

Expected formats:

- `.h5ad` files with base clustering labels stored in `adata.obs`.
- CSV/TSV tables with one row per spot/cell and one column per base partition.
- Spatial coordinates stored in `adata.obsm["spatial"]` for `.h5ad` files, or
  two coordinate columns for tabular input.

Large data files are intentionally ignored by Git.
