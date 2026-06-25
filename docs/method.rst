Method Overview
===============

AGREE takes multiple base partitions for the same spots or cells as input and
returns one consensus partition.

The current implementation follows five steps:

1. Encode each base partition into integer labels.
2. Estimate partition reliability using average normalized mutual information.
3. Build a weighted graph whose nodes are clusters from all base partitions.
4. Learn cluster-node embeddings with weighted DeepWalk and project them back to
   sample-level vectors.
5. Cluster sample-level vectors and optionally refine labels by local spatial
   majority support.

The legacy function name ``run_deepwalk_ec`` is kept for compatibility with
earlier notebooks. New code should prefer ``run_agree``.
