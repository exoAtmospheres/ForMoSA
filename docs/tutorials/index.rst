.. _tutorials:

Tutorials
=========

End-to-end worked examples showing ForMoSA in action.

The notebooks below cover two of the most common use cases.
The remaining tutorials are under active development — see the
:ref:`whats_new` page for the latest roadmap.

Spectroscopic: AB Pic b (SINFONI K-band)
-----------------------------------------

Medium-resolution K-band spectrum of the planetary-mass companion AB Pic b,
fitted with BT-Settl using PyMultiNest.

.. toctree::
   :maxdepth: 1

   sinfoni/abpicb/end_to_end_abpicb.ipynb

Photometric: VHS 1256 b
------------------------

Broadband photometry of VHS 1256 b, fitted with BT-Settl using Nestle.

.. toctree::
   :maxdepth: 1

   photo/vhs1256b/end_to_end_vhs1256b.ipynb

Planned tutorials
-----------------

.. todo::

   The following tutorials are in preparation:

   - **HCHR mode** — AF Lep b with VLT/HiRISE and Exo-REM cloudless
     (parallelisation with PyMultiNest)
   - **MOSAIC mode** — Beta Pic b combining spectroscopy and photometry
   - **Advanced plotting** — modifying default plots, computing BIC, χ²_red,
     molecular absorption, best-fit quantiles, interactive matplotlib views
   - **Cluster deployment** — running ForMoSA on an HPC cluster using MPI
