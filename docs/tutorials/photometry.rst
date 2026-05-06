.. _tutorials-photometry:

Photometry
==========

The repository includes one maintained photometry-only notebook:

* :doc:`../demos/photo/vhs1256b/end_to_end_vhs1256b`

What This Example Covers
------------------------

The VHS 1256 b example demonstrates a photometric workflow built around:

* a photometric FITS file with ``WAVELENGTH``, ``FLUX``, ``ERROR``,
  ``FACILITY``, ``INSTRUMENT``, and ``FILTER_ID`` information;
* ``ConfigGenerator`` and ``ConfigLoader`` for building and editing a
  ``config.ini`` file;
* the v2.0.0 :class:`ForMoSA.Analysis` class for adaptation, nested sampling,
  and plotting.

Practical Notes
---------------

* Photometric filters are resolved through the SVO filter service using the
  facility, instrument, and filter identifiers stored in the observation file.
* Radius is a first-class parameter in v2.0.0 through ``ConfigParameters.r``.
* The notebook is currently the best source of worked photometry examples in
  this repository.

Planned Additions
-----------------

* [A shorter notebook focused only on preparing photometric FITS inputs.]
* [A refreshed VHS 1256 b notebook that explicitly highlights the radius prior
  choices and backend comparisons.]
