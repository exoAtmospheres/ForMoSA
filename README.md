<p align="left"><img src="docs/source/ForMoSA.png" alt="ForMoSA" width="250"/></p>


***
Installation
===

We strongly recommend using a ``conda`` environment ([learn mode here](https://conda.io/docs/user-guide/tasks/manage-environments.html)).

To install and use our package, proceed with the following sequence of commands:
	
    $ conda create -n formosa_env python=3.7

    $ conda activate formosa_env

    $ pip install astropy==4.1 configobj corner extinction nestle matplotlib numpy==1.21.6 PyAstronomy==0.18.0 scipy==1.7.3 spectres xarray==0.20.1 pyyaml

The following two lines help overcome known Python dependency issues.
    
    $ conda install xarray dask netCDF4 bottleneck

    $ pip install importlib-metadata==4.13.0

Please clone the main branch from our GitHub repository. Move to the desired local location and clone this repository. (This will clone the activ_dev branch)

    $ git clone https://github.com/exoAtmospheres/ForMoSA.git


***
Issues?
===

If you encounter any other problem, please create an issue on `GitHub <https://github.com/exoAtmospheres/ForMoSA/issues>`_.

***

[![Documentation Status](https://readthedocs.org/projects/formosa/badge/?version=latest)](https://formosa.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/formosa.svg)](https://badge.fury.io/py/formosa)
[![PyPI downloads](https://img.shields.io/pypi/dm/formosa.svg)](https://pypistats.org/packages/formosa)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)
