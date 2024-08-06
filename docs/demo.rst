.. _demo:

Tutorials
=========

There are several ways of using ForMoSA.

To get started, we recomend you to keep the following structure locally.

.. code-block:: bash

   ~/YOUR/PATH/formosa_desk/
   ├── atm_grids/
   ├── demo_ABPicb/      
   │   ├── data.fits       
   │   ├── config.ini      
   │   ├── adapted_grid/ 
   │   └── outputs/
   ├── (ForMoSA/)
   ├── (PyMultiNest/)
   └── (MultiNest/)

Depending on the way you installed ForMoSA, the ForMoSA, PyMultiNest, and MultiNest subfolders need to be cloned from GitHub. 
Follow the :doc:`installation` 

Atmospheric grids
+++++++++++++++++

This is the list of the publically available atmospheric grids which we have formated for ForMoSA. 

Download the grid you want to use by clicking over it's name. Ideally, save it inside the ``atm_grids/`` subdirectory.

* `ATMO native <https://drive.google.com/file/d/1S1dcBD7UiuUCZIcNBNnJi6LMymrnkagM/view?usp=share_link>`_ from `M.W. Phillips et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_


* `BT-Settl native <https://drive.google.com/file/d/1wvf4A-DupdVnYIpK_HmHE-fobqnYtvEz/view?usp=share_link>`_ from `Allard et al. 2013 <https://ui.adsabs.harvard.edu/abs/2013MSAIS..24..128A/abstract>`_


* `ExoREM native <https://drive.google.com/file/d/1k9SQjHLnMCwmGOHtraRnhCgiZ1-4J3Wk/view?usp=share_link>`_ from `B. Charnay et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..172C/abstract>`_


Learn more at:

.. toctree::
   :maxdepth: 1

   atm_grids




List of tutorials:
++++++++++++++++++


.. toctree::
   :maxdepth: 1

   tutorials/demoabpic
..
   demo_mosaic.rst
   demo_pymultinest.rst