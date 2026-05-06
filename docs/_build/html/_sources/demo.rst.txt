.. _demo:

Tutorials
=========

There are several ways of using ForMoSA.

To get started, we recomend you to keep the following structure locally.

.. code-block:: bash

   ~/YOUR/PATH/formosa_desk/
   ├── atm_grids/
   ├── inversion_targetname/      
   │   ├── inputs/      
   │   ├── config.ini      
   │   ├── adapted_grid/ 
   │   └── outputs/
   ├── (ForMoSA/)
   ├── (PyMultiNest/)
   └── (MultiNest/)

Depending on the way you installed ForMoSA, the ForMoSA, PyMultiNest, and MultiNest subfolders need to be cloned from GitHub. 
Follow the :doc:`installation` 


Observation(s)
++++++++++++++

First, you need to format the observation you wish to invert in a ``.fits`` file. It should have (at least) the following extensions:

* **'WAV'**
* **'FLX'** 
* **'ERR'** or **'COV'** 
* **'RES'** 
* **'INS'** 

If you wish to invert on multiple observations, we recommend that you define separate files (``data_1.fits``, ``data_2.fits``, ect...)

Ideally, save it/them inside the ``inputs/`` subdirectory.

Learn more about how to format your observation(s):

.. toctree::
   :maxdepth: 1

   tutorials/format_obs


Atmospheric grids
+++++++++++++++++

You now need an atmospheric grid on which to run your inversion.

This is the list of the publically available grids which we have formated for ForMoSA. 

Download the grid you want to use by clicking over it's name. Ideally, save it inside the ``atm_grids/`` subdirectory.

* `ATMO <https://drive.google.com/file/d/1S1dcBD7UiuUCZIcNBNnJi6LMymrnkagM/view?usp=share_link>`_ from `M.W. Phillips et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_


* `BT-Settl <https://drive.google.com/file/d/1wvf4A-DupdVnYIpK_HmHE-fobqnYtvEz/view?usp=share_link>`_ from `Allard et al. 2013 <https://ui.adsabs.harvard.edu/abs/2013MSAIS..24..128A/abstract>`_


* `ExoREM <https://drive.google.com/file/d/1k9SQjHLnMCwmGOHtraRnhCgiZ1-4J3Wk/view?usp=share_link>`_ from `B. Charnay et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..172C/abstract>`_


Learn more about:

.. toctree::
   :maxdepth: 1

   tutorials/exorem_info.ipynb


Configuration file
++++++++++++++++++

Finally, you need to prepare a configuration file.

This file (``config.ini``) allows you to communicate with ForMoSA. 

Learn how to set it up in various cases:

.. toctree::
   :maxdepth: 1

   tutorials/config_file.ipynb


Demos
+++++

.. toctree::
   :maxdepth: 1

   tutorials/demoabpic.ipynb
   tutorials/demobetapic.ipynb