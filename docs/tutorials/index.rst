.. _tutorials:

Tutorials
=========

End-to-end worked examples showing ForMoSA v2.0 in action.
Each notebook is self-contained: it checks for required files and
downloads them automatically if they are missing.

.. note::

   All notebooks use the **dataclass API** (v2.0) as the primary
   approach. An INI-file alternative is shown at the end of each
   notebook for reference.

Tutorial 1 — Photometry: VHS 1256 b
-------------------------------------

Multi-instrument broadband photometry fitted with ExoREM using Nestle.
Covers SVO filter retrieval, physical and analytical scaling.

.. toctree::
   :maxdepth: 1

   photo/vhs1256b/tutorial_photometry.ipynb

Tutorial 2 — Spectroscopy: AB Pic b
--------------------------------------

VLT/SINFONI K-band medium-resolution spectrum fitted with BT-Settl using PyMultiNest.
Covers resolution adaptation, wavelength windowing, and radial velocity.

.. toctree::
   :maxdepth: 1

   spectroscopy/abpicb/tutorial_spectroscopy.ipynb

.. Tutorial 3 — HCHR mode: AF Lep b
.. -----------------------------------

.. VLT/HiRISE high-contrast high-resolution data fitted with Exo-REM.
.. Covers the ``STAR_FLUX`` extension and high-contrast modeling.

.. 
   .. toctree::
..    :maxdepth: 1

..    hchr/aflepb/tutorial_hchr.ipynb

.. Tutorial 4 — MOSAIC mode: HIP 64892 b
.. ------------------------------------

.. Combining spectroscopy and photometry in a single simultaneous fit.
.. Covers MOSAIC meta-likelihood and per-instrument intercalibration.
..
   .. toctree::
..    :maxdepth: 1

..    mosaic/hip64892b/tutorial_mosaic.ipynb

Tutorial 3 — Advanced plotting
--------------------------------

Customising every ForMoSA plot, computing χ²_red and BIC, and
saving publication-quality figures. Runs without fitting anything —
uses pre-computed results from Tutorial 2.

.. toctree::
   :maxdepth: 1

   plotting/tutorial_advanced_plotting_custom.ipynb

Tutorial 4 — Statistical tests
---------------------------------

Model comparison and goodness-of-fit statistics: χ²_red, BIC, and
Bayes factors. Uses pre-computed results and requires no new fits.

.. toctree::
   :maxdepth: 1

   plotting/tutorial_statistical_tests.ipynb

Tutorial 5 — Cluster / MPI deployment
----------------------------------------

Running ForMoSA on an HPC cluster with PyMultiNest and MPI.
Covers ``nohup`` pattern.

.. toctree::
   :maxdepth: 1

   cluster/tutorial_cluster.md
