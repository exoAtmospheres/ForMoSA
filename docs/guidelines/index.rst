.. _guidelines:

Guidelines
==========

Good practices
--------------

ForMoSA is a code that handles multiple inputs to produce multiple outputs, considering various models and algorithms.
That is why we recommend organizing the working environment according to the following architecture:

.. code-block:: bash

   ~/YOUR/PATH/formosa_desk/
   ├── atm_grids/
   ├── project/      
   │   └── homo_sample/
   │      ├── adapted_grid/
   │      └── targetname/      
   │         ├── data/     
   │         └── results/
   ├── (ForMoSA/)
   ├── (PyMultiNest/)
   └── (MultiNest/)

* ``atm_grids`` contains the native grids formatted with xarray that can be downloaded below or generated upon request.
* ``project`` contains the characterization you want to perform in the context of a project that includes different observations or tests.
* ``homo_sample`` contains the analysis of a homogeneous library of observations (same wavelength sampling). This allows you to use the same adapted grid for several homogeneous datasets.
* ``adapted_grid`` contains the grids of models adapted to a homogeneous library of observations (same wavelength sampling).
* ``targetname`` contains the analysis of a single dataset.
* ``data`` contains the dataset.
* ``results`` contains the results of the fit.

* ``(ForMoSA/)`` is the location of ForMoSA in case of an installation with GitHub.
* ``(PyMultiNest/)`` is the location of PyMultiNest in case of an installation with GitHub.
* ``(MultiNest/)`` is the location of MultiNest in case of an installation with GitHub.


Inputs
------

Observation(s)
++++++++++++++

The data need to be in a ``.fits`` file. It should have (at least) the following extensions:

* **'WAV'** The wavelengths sampling
* **'WAVE_UNIT'** The wavelengths unit
* **'FLX'** The corresponding flux 
* **'ERR'** or **'COV'** The corresponding flux error (or covariance matrix) 
* **'RES'** The corresponding spectral resolution given for each wavelength 
* **'FAC'** The observatory where the data come from (need to be consistent with `VOSA <https://svo2.cab.inta-csic.es/theory/fps/>`_ for photometric points).
* **'INS'** The instrument used to obtain the data (need to be consistent with `VOSA <https://svo2.cab.inta-csic.es/theory/fps/>`_ for photometric points).
* **'FILT'** The photometric filter used in case of photometric point (need to be consistent with `VOSA <https://svo2.cab.inta-csic.es/theory/fps/>`_)

.. note::
    All of these extensions need to be numpy arrays of the same length.



Learn more about how to format your observation(s):

.. toctree::
   :maxdepth: 1

   input_format/obs_format.ipynb
   
ForMoSA enable you to invert multiple datasets observed with different instruments. 
In that case, each dataset should be defined separately, in different files (``data_1.fits``, ``data_2.fits``, ect...), and should be saved in the subdirectory ``data``


Atmospheric grids
+++++++++++++++++

You now need an atmospheric grid on which to run your inversion.

This is the list of the publically available grids which we have formated for ForMoSA. 
Download the grid you want to use by clicking over it's name. Ideally, save it inside the ``atm_grids/`` subdirectory. 
If you need to generate a grid which is not in this list, please contact us. We can generate custom grids under request.

* `ATMO <https://drive.google.com/file/d/1S1dcBD7UiuUCZIcNBNnJi6LMymrnkagM/view?usp=share_link>`_ from `M.W. Phillips et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_


* `BT-Settl <https://drive.google.com/file/d/1wvf4A-DupdVnYIpK_HmHE-fobqnYtvEz/view?usp=share_link>`_ from `Allard et al. 2013 <https://ui.adsabs.harvard.edu/abs/2013MSAIS..24..128A/abstract>`_


* `ExoREM <https://drive.google.com/file/d/1k9SQjHLnMCwmGOHtraRnhCgiZ1-4J3Wk/view?usp=share_link>`_ from `B. Charnay et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..172C/abstract>`_


Learn more about the format of the grids used in ForMoSA:

.. toctree::
   :maxdepth: 1

   input_format/model_format.ipynb


Configuration file
++++++++++++++++++

Finally, you need to prepare a configuration file.

This file (``config.ini``) allows you to communicate with ForMoSA. 

Learn how to set it up in various cases:

.. toctree::
   :maxdepth: 1

   input_format/config_format.ipynb
