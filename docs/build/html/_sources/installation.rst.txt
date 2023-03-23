.. _installation:

Installation
============

For all Users
+++++++++++++

We strongly recommend using a ``conda`` environment (learn mode 
`here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_).
To install and use our package follow the following sequence of commands.

.. code-block:: bash
	
    $ conda create -n formosa_env python=3.7

    $ conda activate formosa_env

    $ pip install ForMoSA

    $ pip install importlib-metadata==4.13.0

    $ conda install xarray dask netCDF4 bottleneck

Issues?
+++++++

If you run into any problem, please create an issue on `GitHub <https://github.com/exoAtmospheres/ForMoSA/issues>`_.
