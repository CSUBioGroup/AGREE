# Experiments

This directory is reserved for paper-level experiment launchers and result
summaries.

Recommended organization:

- `configs/`: dataset-specific method columns, target cluster numbers, and paths.
- `run_*.py`: reproducible experiment entry points.
- `summaries/`: small CSV files used to assemble manuscript tables.

Large intermediate outputs should be written to `results/` or an external data
directory and should not be committed to Git.
