.. _installation:

Installation
============

We recommend a dedicated ``conda`` environment to keep ForMoSA's dependencies
isolated from the rest of your Python stack.


Setting Up a Conda Environment
-------------------------------

For all users:

.. code-block:: console

   $ conda create -n env_formosa python=3.12
   $ conda activate env_formosa

For macOS users with Apple Silicon (M1/M2/M3):

.. code-block:: console

   $ CONDA_SUBDIR=osx-arm64 conda create -n env_formosa python=3.12 numpy -c conda-forge
   $ conda activate env_formosa
   $ conda config --env --set subdir osx-arm64

Learn more about conda environments in the
`conda documentation <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.


A. Install from PyPI
---------------------

The quickest route. ``pip`` handles almost all dependencies automatically.

.. code-block:: console

   $ pip install ForMoSA
   $ conda install dask netCDF4 bottleneck

That's it — you're ready to run your first analysis.


B. Install from Source
-----------------------

For development or to track the latest changes:

.. code-block:: console

   $ git clone https://github.com/exoAtmospheres/ForMoSA.git
   $ cd ForMoSA
   $ pip install -e .
   $ conda install dask netCDF4 bottleneck

The ``-e`` flag installs in editable mode, so any local changes to the source
are reflected immediately without reinstalling.


PyMultiNest (Optional, Recommended for Large Fits)
----------------------------------------------------

`PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_ wraps the Fortran
`MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ library and is the
recommended back-end for fits with more than three free parameters.

First, ensure your system has a C++ compiler, a Fortran compiler, CMake, and
Open MPI. On macOS with Homebrew:

.. code-block:: console

   $ brew install cmake gcc open-mpi

Then clone and build MultiNest:

.. code-block:: console

   $ git clone https://github.com/JohannesBuchner/MultiNest
   $ cd MultiNest/build
   $ cmake ..
   $ make

.. note::
   If ``cmake ..`` complains about the policy version, run
   ``cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5`` instead.

Copy the compiled libraries into your conda environment (replace
``/YOUR_PATH`` with the actual path):

.. code-block:: console

   $ cp -v MultiNest/lib/* /YOUR_PATH/opt/anaconda3/envs/env_formosa/lib/

Finally, install the Python wrapper and MPI support:

.. code-block:: console

   $ pip install mpi4py
   $ git clone https://github.com/JohannesBuchner/PyMultiNest/
   $ cd PyMultiNest && python setup.py install

Verify the installation:

.. code-block:: python

   import pymultinest   # should import without error
