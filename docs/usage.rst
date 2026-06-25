Usage
=====

Python API
----------

.. code-block:: python

   from AGREE import run_agree

   labels = [bass_labels, spatialpca_labels, graphst_labels, stagate_labels, const_labels]
   pred = run_agree(labels, n_clusters=7, spatial_coords=adata.obsm["spatial"])

Command line
------------

.. code-block:: bash

   python scripts/run_agree.py \
     --input Data/example.h5ad \
     --methods BASS,SpatialPCA,GraphST,STAGATE,conST \
     --n-clusters 7 \
     --output results/example_agree.h5ad
