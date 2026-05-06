.. _installation:

Installation
============

The installation procedure for ForMoSA remains close to the v1.1.6 workflow.
The main difference in v2.0.0 is the public API, not the dependency story.

Create An Environment
---------------------

We strongly recommend a dedicated environment.

.. code-block:: console

   $ conda create -n env_formosa python=3.11
   $ conda activate env_formosa

On Apple Silicon, users who plan to rely on ``pymultinest`` should make sure the
environment is created under the ``osx-arm64`` architecture.

.. code-block:: console

   $ CONDA_SUBDIR=osx-arm64 conda create -n env_formosa python=3.11 numpy -c conda-forge
   $ conda activate env_formosa
   $ conda config --env --set subdir osx-arm64

Install From PyPI
-----------------

.. code-block:: console

   $ pip install ForMoSA
   $ conda install dask netCDF4 bottleneck

Install From Source
-------------------

.. code-block:: console

   $ git clone https://github.com/exoAtmospheres/ForMoSA.git
   $ cd ForMoSA
   $ pip install -e .
   $ conda install dask netCDF4 bottleneck

Optional Nested-Sampling Backends
---------------------------------

ForMoSA supports three nested-sampling backends in the current code:

* ``nestle``
* ``pymultinest``
* ``ultranest``

``nestle`` and ``ultranest`` are Python packages. ``pymultinest`` additionally
requires a working ``MultiNest`` installation.

PyMultiNest Users
-----------------

If you want to use ``pymultinest``, follow the upstream installation
instructions from `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/install.html>`_.
The repository's current installation workflow is:

.. code-block:: console

   $ git clone https://github.com/JohannesBuchner/PyMultiNest/
   $ cd PyMultiNest
   $ python setup.py install

Make sure your system provides a C/C++ compiler, a Fortran compiler, and MPI.
One documented path on macOS is:

.. code-block:: console

   $ brew install cmake
   $ brew install gcc
   $ brew install open-mpi
   $ pip install mpi4py

Then install ``MultiNest`` itself:

.. code-block:: console

   $ git clone https://github.com/JohannesBuchner/MultiNest
   $ cd MultiNest/build
   $ cmake ..
   $ make

.. note::

   If ``cmake ..`` fails with a policy warning, the current docs tree records
   the workaround ``cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5``.

Finally, copy the built libraries into the active environment if required by
your local ``PyMultiNest`` setup.

Torch Users
-----------

Torch is used for emulator-related workflows.

For CPU usage:

.. code-block:: console

   $ conda install torch torchvision torchaudio torchnmf

For GPU usage, follow the installation guidance from
`PyTorch <https://pytorch.org/>`_ for your CUDA setup.
