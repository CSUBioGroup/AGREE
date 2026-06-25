# AGREE

AGREE is a spatial transcriptomics ensemble consensus clustering model. It
integrates multiple base spatial-domain partitions by constructing a weighted
cluster association graph, learning cluster embeddings with DeepWalk, projecting
cluster evidence back to spots/cells, and optionally refining labels with local
spatial consensus.

This repository is organized as the companion code for the AGREE manuscript.
Items such as the final paper title, public data DOI, and formal citation will
be updated after release.

## Installation

```bash
git clone https://github.com/CSUBioGroup/AGREE.git
cd AGREE
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

An equivalent Conda environment template is provided in `environment.yml`.

## Quick Start

```python
from AGREE import run_agree

base_labels = [
    adata.obs["BASS"].to_numpy(),
    adata.obs["SpatialPCA"].to_numpy(),
    adata.obs["GraphST"].to_numpy(),
    adata.obs["STAGATE"].to_numpy(),
    adata.obs["conST"].to_numpy(),
]

adata.obs["AGREE"] = run_agree(
    base_labels,
    n_clusters=7,
    spatial_coords=adata.obsm["spatial"],
)
```

## Command Line

Run AGREE on an AnnData file:

```bash
python scripts/run_agree.py \
  --input Data/example.h5ad \
  --methods BASS,SpatialPCA,GraphST,STAGATE,conST \
  --n-clusters 7 \
  --output results/example_agree.h5ad
```

Run AGREE on a CSV table:

```bash
python scripts/run_agree.py \
  --input Data/example_partitions.csv \
  --methods BASS,SpatialPCA,GraphST,STAGATE,conST \
  --x-col x \
  --y-col y \
  --n-clusters 7 \
  --output results/example_agree.csv
```

## Repository Layout

```text
AGREE/              Core Python package
scripts/            Command-line entry points
examples/           Minimal runnable examples
experiments/        Experiment organization notes
notebooks/          Notebook organization notes
docs/               Documentation scaffold
Data/               Data placement instructions
results/            Local output placeholder
tests/              Smoke tests
```

## Compared Tools

The paper experiments will compare AGREE with representative spatial domain
identification and ensemble consensus methods. The final list and links will be
updated with the manuscript.

Current base partitions used in internal experiments include:

- BASS
- SpatialPCA
- GraphST
- STAGATE
- conST

Additional experimental columns from previous notebooks can be used as long as
they are supplied as one label vector per spot/cell.

## Data

Large datasets are not stored in this repository. Please place `.h5ad` or
CSV/TSV input files under `Data/` locally. The final public data DOI will be
added after manuscript release.

## Cite

```bibtex
@article{AGREE2026,
  title   = {AGREE: spatial transcriptomics ensemble consensus clustering},
  author  = {CSUBioGroup},
  journal = {To be announced},
  year    = {2026}
}
```

## Contact

For questions, please open an issue on GitHub or contact the CSUBioGroup
maintainers.
