.. _grid:

Model Grids
===========

Model Grid
++++++++++

Core grid class that wraps an ``xarray.Dataset`` and exposes wavelength,
resolution, parameter coordinates, and interpolation helpers.

.. automodule:: ForMoSA.grid.model_grid
	:members:
	:undoc-members:
	:show-inheritance:

Grid Loader
+++++++++++

Responsible for loading and validating NetCDF grid files.

.. automodule:: ForMoSA.grid.grid_loader
	:members:
	:undoc-members:
	:show-inheritance:

SubGrid Base
++++++++++++

Abstract base class for spectroscopic and photometric subgrids.

.. inheritance-diagram:: ForMoSA.grid.subgrid_spectroscopy.SubGridSpectroscopy ForMoSA.grid.subgrid_photometry.SubGridPhotometry
   :parts: 1

.. automodule:: ForMoSA.grid.subgrid_base
	:members:
	:undoc-members:
	:show-inheritance:

SubGrid Set
+++++++++++

Container for a collection of adapted subgrids.

.. automodule:: ForMoSA.grid.subgrid_set
	:members:
	:undoc-members:
	:show-inheritance:

Spectroscopic SubGrid
+++++++++++++++++++++

.. automodule:: ForMoSA.grid.subgrid_spectroscopy
	:members:
	:undoc-members:
	:show-inheritance:

Photometric SubGrid
+++++++++++++++++++

.. automodule:: ForMoSA.grid.subgrid_photometry
	:members:
	:undoc-members:
	:show-inheritance:
