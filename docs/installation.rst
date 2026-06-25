Installation
============

Create a fresh environment and install the repository dependencies:

.. code-block:: bash

   conda create -n agree python=3.10
   conda activate agree
   pip install -r requirements.txt

For development:

.. code-block:: bash

   pip install -e .

Alternatively, use the provided Conda environment file:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate agree
