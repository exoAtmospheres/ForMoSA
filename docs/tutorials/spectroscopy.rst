.. _tutorials-spectroscopy:

Spectroscopy
============

The repository includes one maintained spectroscopy notebook:

* :doc:`../demos/sinfoni/abpicb/end_to_end_abpicb`

What This Example Covers
------------------------

The AB Pic b notebook demonstrates a medium-resolution spectroscopic workflow
using:

* a spectroscopic FITS file with wavelength, flux, uncertainty, and resolution;
* ``ConfigGenerator`` and ``ConfigLoader`` for the INI workflow;
* :class:`ForMoSA.Analysis` for the adaptation and inversion stages.

Data Preparation Notes
----------------------

For spectroscopic data, the verified minimum requirements are:

* ``WAVELENGTH``
* ``FLUX``
* ``RESOLUTION``
* either ``ERROR`` or ``COVARIANCE``

``FACILITY`` and ``INSTRUMENT`` are recommended, especially if you want the
saved plots and output labels to remain informative.

Planned Additions
-----------------

* [A notebook dedicated to preparing spectroscopy with varying resolution.]
* [A refreshed AB Pic b example using the preferred current model grid.]
