# ForMoSA Documentation v2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the outdated v1.1.6 ForMoSA documentation with a clean, accurate v2.0.0 set of pages covering welcome, installation, getting started, scaling, and a tutorials stub — without touching any docstrings or `whats_new.rst`.

**Architecture:** Option 2 (structure first, content second). The skeleton is wired and builds cleanly before any prose is written. All new content pages use MyST Markdown (`.md`); index/toctree files stay in RST (`.rst`). The `guidelines/` directory is renamed to `getting_started/`; `demos/` is renamed to `tutorials/`. A new top-level `scaling/` section is created.

**Tech Stack:** Sphinx (`sphinx_book_theme`, `myst_parser`, `nbsphinx`), Python 3.12, conda env `env_formosa`. Build: `cd docs && conda run -n env_formosa make html`.

**Spec:** `docs/superpowers/specs/2026-05-06-formosa-docs-v2-design.md`

**Constraints:**
- Do NOT modify any file inside `ForMoSA/` (source code)
- Do NOT modify `docs/whats_new.rst`
- Do NOT modify API `.rst` files in `docs/api/` (except verifying the module list)
- Tone: clean, clear, concise, professional with occasional wit — not casual, not robotic

---

## File Map

| Action | Path |
|--------|------|
| Modify | `docs/conf.py` |
| Copy | `paper/schema_ForMoSA.png` → `docs/_static/schema_ForMoSA.png` |
| Modify | `docs/index.rst` |
| Modify | `docs/installation.rst` |
| Rename dir | `docs/guidelines/` → `docs/getting_started/` |
| Create | `docs/getting_started/index.rst` |
| Create | `docs/getting_started/folder_structure.md` |
| Create | `docs/getting_started/data_formatting.md` |
| Create | `docs/getting_started/model_grid.md` |
| Create | `docs/getting_started/config_file.md` |
| Delete | `docs/guidelines/input_format/obs_format.ipynb` |
| Delete | `docs/guidelines/input_format/model_format.ipynb` |
| Delete | `docs/guidelines/input_format/config_format.ipynb` |
| Rename dir | `docs/demos/` → `docs/tutorials/` |
| Create | `docs/tutorials/index.rst` |
| Create | `docs/scaling/index.rst` |
| Create | `docs/scaling/analytical_vs_physical.md` |
| Create | `docs/scaling/mosaic_best_practices.md` |
| Verify | `docs/api/index.rst` |

---

## Task 1: Version bump and static asset

**Files:**
- Modify: `docs/conf.py:23`
- Copy: `paper/schema_ForMoSA.png` → `docs/_static/schema_ForMoSA.png`

- [ ] **Step 1: Update release version in `conf.py`**

Open `docs/conf.py` and change line 23:
```python
# Before
release = '1.1.6'

# After
release = '2.0.0'
```

- [ ] **Step 2: Copy workflow diagram to `_static/`**

```bash
cp /Users/rajpoot/Karmabhumi/Packages/ForMoSA/paper/schema_ForMoSA.png \
   /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs/_static/schema_ForMoSA.png
```

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/conf.py docs/_static/schema_ForMoSA.png
git commit -m "docs: bump release to 2.0.0 and add workflow diagram"
```

---

## Task 2: Directory skeleton

Rename `guidelines/` → `getting_started/`, rename `demos/` → `tutorials/`, create `scaling/`. Wire the toctrees. Verify the build passes before writing any prose.

**Files:**
- Rename: `docs/guidelines/` → `docs/getting_started/`
- Rename: `docs/demos/` → `docs/tutorials/`
- Create: `docs/getting_started/index.rst` (stub)
- Create: `docs/tutorials/index.rst` (stub)
- Create: `docs/scaling/index.rst` (stub)
- Create: `docs/scaling/analytical_vs_physical.md` (stub)
- Create: `docs/scaling/mosaic_best_practices.md` (stub)
- Create stubs: `docs/getting_started/folder_structure.md`, `data_formatting.md`, `model_grid.md`, `config_file.md`
- Modify: `docs/index.rst` — update toctree entries

- [ ] **Step 1: Rename `guidelines/` to `getting_started/`**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git mv docs/guidelines docs/getting_started
```

- [ ] **Step 2: Rename `demos/` to `tutorials/`**

```bash
git mv docs/demos docs/tutorials
```

- [ ] **Step 3: Create stub content pages in `getting_started/`**

Create `docs/getting_started/folder_structure.md`:
```markdown
# Folder Structure

*Coming soon.*
```

Create `docs/getting_started/data_formatting.md`:
```markdown
# Data Formatting

*Coming soon.*
```

Create `docs/getting_started/model_grid.md`:
```markdown
# Knowing Your Model Grid

*Coming soon.*
```

Create `docs/getting_started/config_file.md`:
```markdown
# Knowing the Config

*Coming soon.*
```

- [ ] **Step 4: Create `getting_started/index.rst` stub**

Replace the old `guidelines/index.rst` with:

```rst
.. _getting_started:

Getting Started
===============

.. toctree::
   :maxdepth: 1

   folder_structure
   data_formatting
   model_grid
   config_file
```

- [ ] **Step 5: Create `tutorials/index.rst` stub**

```rst
.. _tutorials:

Tutorials
=========

.. todo::

   Tutorials are under active development. Planned:

   - Photometry — VHS 1256 b with BT-Settl (+ Nestle)
   - Spectroscopy — AB Pic b SINFONI K-band (+ PyMultiNest)
   - HCHR mode — AF Lep b HiRISE with Exo-REM cloudless
   - MOSAIC mode — Beta Pic b (spectro + photo)
   - Advanced plotting and outputs
   - Deploying ForMoSA on a cluster with MPI
```

- [ ] **Step 6: Create `scaling/` directory with stubs**

Create `docs/scaling/index.rst`:
```rst
.. _scaling:

Scaling
=======

.. toctree::
   :maxdepth: 1

   analytical_vs_physical
   mosaic_best_practices
```

Create `docs/scaling/analytical_vs_physical.md`:
```markdown
# Analytical vs Physical Scaling

*Coming soon.*
```

Create `docs/scaling/mosaic_best_practices.md`:
```markdown
# MOSAIC Best Practices

*Coming soon.*
```

- [ ] **Step 7: Update `docs/index.rst` toctree**

Replace the toctree block in `docs/index.rst` so it references the new paths:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   getting_started/index
   tutorials/index
   scaling/index
   api/index
   whats_new
```

- [ ] **Step 8: Verify the build passes**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | tail -30
```

Expected: build completes, warnings may appear for the old ipynb files (no longer referenced — that is fine), no "unknown document" errors for any toctree entry.

If you see `WARNING: document isn't included in any toctree` for files inside the old `guidelines/input_format/` that are now inside `getting_started/`, delete them:

```bash
rm /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs/getting_started/input_format/obs_format.ipynb
rm /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs/getting_started/input_format/model_format.ipynb
rm /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs/getting_started/input_format/config_format.ipynb
rmdir /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs/getting_started/input_format
```

Rebuild and confirm clean:
```bash
conda run -n env_formosa make html 2>&1 | grep -E "ERROR|WARNING" | head -20
```

- [ ] **Step 9: Commit skeleton**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/
git commit -m "docs: rename guidelines→getting_started, demos→tutorials; add scaling skeleton"
```

---

## Task 3: Welcome page (`index.rst`)

**Files:**
- Modify: `docs/index.rst`

- [ ] **Step 1: Replace `docs/index.rst` with the full welcome page**

```rst
.. ForMoSA documentation master file

.. |br| raw:: html

   <br />

Forward Modeling Tool for Spectral Analysis (ForMoSA)
=====================================================

ForMoSA is an open-source Python package for modeling exoplanetary atmospheres
using a forward modeling approach. It compares observed spectra and photometry
against grids of atmospheric models via nested sampling to derive posterior
distributions on physical parameters.

**Quick links:**
:doc:`getting_started/index` |
:doc:`tutorials/index` |
:doc:`scaling/index` |
:doc:`api/index` |
:doc:`whats_new`


Features
--------

- **Class-based API** — a single :class:`~ForMoSA.Analysis` entry point manages the full lifecycle: grid adaptation, nested sampling, and plotting.
- **Multi-instrument support (MOSAIC)** — fit spectroscopic and photometric data simultaneously from multiple instruments with per-instrument intercalibration.
- **Three nested-sampling back-ends** — `nestle <http://kylebarbary.com/nestle/>`_, `PyMultiNest <https://github.com/JohannesBuchner/PyMultiNest>`_, and `UltraNest <https://johannesbuchner.github.io/UltraNest/>`_.
- **High-contrast mode** — model stellar speckles and systematics alongside the companion signal.
- **Automatic photometry filters** — filter curves retrieved and cached on the fly from the `SVO Filter Profile Service <https://svo2.cab.inta-csic.es/theory/fps/>`_.
- **Configurable priors** — uniform, log-uniform, Gaussian, and constant priors for every fitted parameter.
- **Comprehensive plotting** — corner plots, chain diagnostics, radar diagrams, best-fit spectra, CCFs, and RV–v sin i maps.
- **Flexible configuration** — Python dataclasses or INI files; both load into the same objects.


Statement of Need
-----------------

Recent advances in ground- and space-based observatories now enable routine,
high-quality observations of exoplanet atmospheres across a wide range of
wavelengths and resolutions. ForMoSA was developed to bridge the gap between
these observations and atmospheric models by providing a Bayesian framework to
robustly compare the two.

Unlike retrieval approaches that generate spectra on the fly using highly
parameterized models, ForMoSA adopts a **forward-modeling** approach based on
pre-computed, self-consistent atmospheric model grids — enabling a direct,
model-driven comparison with physically motivated theoretical predictions. The
framework supports simultaneous analysis of heterogeneous datasets (different
instruments, resolutions, and epochs) within a single generalized statistical
framework, making it well-suited for the next generation of high-contrast,
high-resolution instruments.


How Does ForMoSA Work?
----------------------

At its core, ForMoSA interpolates a pre-computed atmospheric model grid at each
nested-sampling iteration, applies physical transformations (Doppler shift,
rotational broadening, extinction, scaling), and evaluates a log-likelihood
against the observations. The posterior distributions on the fitted parameters
emerge naturally from the nested-sampling evidence accumulation.

.. figure:: _static/schema_ForMoSA.png
   :alt: ForMoSA workflow diagram
   :width: 100%

   **ForMoSA workflow.** The right-hand shaded area shows the core modules
   required for nested sampling; the left dark-grey area contains utility and
   support functions. See the :doc:`api/index` for full module documentation.

.. todo::

   Add an interactive code-dependency graph here once the graph tooling is
   integrated into the build.


Example Results
---------------

.. figure:: _static/priors_teff_plot.png
   :alt: Example posterior on effective temperature
   :width: 80%

   Example posterior distribution on effective temperature from a
   nested-sampling run with ForMoSA.

.. figure:: _static/rv.png
   :alt: RV map example
   :width: 80%

   Radial-velocity log-likelihood map produced by ``analysis.plot_rv_vsini_map()``.


.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   installation
   getting_started/index
   tutorials/index
   scaling/index
   api/index
   whats_new


Attribution
-----------

If you use ForMoSA in your research, please cite
`Petrus et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P/abstract>`_.


Version Track
-------------

- ``2.0.0`` Complete rewrite with a class-based API (``Analysis``), Python dataclass
  configuration, restructured package layout, automatic photometry filter retrieval,
  CCF / RV–v sin i analysis, structured logging, and typed error handling.
  **Not backwards-compatible with v1.x.** See :ref:`whats_new` for the full details.

- ``1.1.6`` Addition of high-contrast models, UltraNest, and automatically generated
  config files.

- ``1.1.2`` Multi-instrument (MOSAIC) and high-spectral-resolution support.

- ``1.0.13`` First distributed version, presented at
  `Cloud Academy 3 <https://alienearths.space/cloud-academy-3/>`_.

- ``1.0.5`` First operational release.


Acknowledgements
----------------

The authors express their sincere thanks to the
`Code/Astro Workshop <https://semaphorep.github.io/codeastro/>`_, which provided
the foundational training necessary to transform ForMoSA into a professional,
open-source Python package.

We gratefully acknowledge the funding and support for the ForM-X workshops held
in Nice (2023), Heidelberg (2024/2025), and Grenoble (2025), which were
instrumental in the development and refinement of the code. We also thank the
various laboratories and institutions — especially IPAG, Lagrange, and MPIA —
for their continued support.

This work has been supported by the French National Research Agency (ANR) through
the MIRAGES project (PI: A. Vigan, ANR-20-CE31-0017).


Issues
------

Run into something unexpected? Please open an issue on
`GitHub <https://github.com/exoAtmospheres/ForMoSA/issues>`_.
```

- [ ] **Step 2: Build and check welcome page renders**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR|WARNING" | head -20
```

Expected: no ERRORs. Any WARNING about `sphinx.ext.doctest` or intersphinx fetching is acceptable.

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/index.rst
git commit -m "docs: rewrite welcome page for v2.0.0"
```

---

## Task 4: Installation page (`installation.rst`)

**Files:**
- Modify: `docs/installation.rst`

- [ ] **Step 1: Replace `docs/installation.rst` with the simplified v2.0 content**

```rst
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
```

- [ ] **Step 2: Build and verify**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

Expected: no ERRORs.

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/installation.rst
git commit -m "docs: simplify installation page, remove torch section"
```

---

## Task 5: Getting Started hub (`getting_started/index.rst`)

**Files:**
- Modify: `docs/getting_started/index.rst`

- [ ] **Step 1: Write the hub page**

```rst
.. _getting_started:

Getting Started
===============

Everything you need before running your first ForMoSA analysis — folder layout,
data format, model grids, and the configuration API.

.. toctree::
   :maxdepth: 1

   folder_structure
   data_formatting
   model_grid
   config_file
```

- [ ] **Step 2: Build and verify no broken links**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/getting_started/index.rst
git commit -m "docs: write Getting Started hub page"
```

---

## Task 6: Folder structure page

**Files:**
- Modify: `docs/getting_started/folder_structure.md`

- [ ] **Step 1: Write full content**

```markdown
# Folder Structure

ForMoSA touches a lot of files — model grids, adapted sub-grids, observation
files, results, and plots. Keeping them organised from the start saves a
surprising amount of grief later.

## Recommended layout

```bash
~/formosa_desk/
├── atm_grids/              # Native model grids (.nc files)
├── project/
│   └── sample_name/        # A homogeneous set of observations (same wavelength grid)
│       ├── adapted_grid/   # Adapted sub-grids (written by analysis.adapt())
│       └── target_name/    # One target
│           ├── data/       # Input .fits observation files
│           └── results/    # NS results, plots, and saved state
├── (ForMoSA/)              # Source installation only
├── (PyMultiNest/)          # Source installation only
└── (MultiNest/)            # Source installation only
```

## What goes where

| Directory | Contents |
|-----------|----------|
| `atm_grids/` | Downloaded model grid `.nc` files (BT-Settl, Exo-REM, ATMO, …) |
| `sample_name/adapted_grid/` | Sub-grids written by `analysis.adapt()`. Re-usable across all targets observed with the same instrument setup, so you only pay the adaptation cost once. |
| `target_name/data/` | Your `.fits` observation files. One file per instrument in MOSAIC mode. |
| `target_name/results/` | Everything ForMoSA writes: `ns_results.json`, `.npz` sub-grids, corner plots, chain diagnostics. |

## Mapping the layout to `ConfigPath`

```python
from ForMoSA.config.global_config import ConfigPath

config_path = ConfigPath(
    model_path        = "~/formosa_desk/atm_grids/BT-Settl.nc",
    observation_path  = ["~/formosa_desk/project/sample_name/target_name/data/obs.fits"],
    adapt_store_path  = "~/formosa_desk/project/sample_name/adapted_grid/",
    result_path       = "~/formosa_desk/project/sample_name/target_name/results/",
)
```

```{note}
`adapt_store_path` is intentionally shared between targets in the same sample.
If two targets were observed with the same instrument and wavelength grid,
the adapted sub-grid is identical — no need to recompute it.
```
```

- [ ] **Step 2: Build and verify**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/getting_started/folder_structure.md
git commit -m "docs: write folder structure getting-started page"
```

---

## Task 7: Data formatting page

**Files:**
- Modify: `docs/getting_started/data_formatting.md`

- [ ] **Step 1: Write full content**

````markdown
# Data Formatting

ForMoSA reads observations from **FITS table files** (`.fits`). Each extension
holds a NumPy array; all arrays in the same file must be the same length. This
page explains what extensions are required for each observation type and shows
how to verify your file before running an analysis.

## Required extensions

| Extension | Aliases accepted | Description | Spectroscopic | Photometric | HCHR |
|-----------|-----------------|-------------|:---:|:---:|:---:|
| `WAV` | `WAVELENGTH`, `WAVE`, `LAMBDA` | Wavelength array | Yes | Yes | Yes |
| `WAVE_UNIT` | — | Wavelength unit string (e.g. `"µm"`) | Yes | Yes | Yes |
| `FLX` | `FLUX` | Flux array | Yes | Yes | Yes |
| `ERR` | `ERROR`, `SIGMA` | 1-D flux uncertainty (use instead of `COV`) | Yes | Yes | Yes |
| `COV` | — | Full covariance matrix — shape `(N, N)` — alternative to `ERR` | Yes | No | Yes |
| `RES` | `RESOLUTION` | Spectral resolution λ/Δλ per wavelength point | Yes | No | Yes |
| `FAC` | `FACILITY` | Observatory identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | No | Yes | No |
| `INS` | `INSTRUMENT` | Instrument identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | No | Yes | No |
| `FILT` | `FILTER`, `FILTER_ID` | Filter identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | No | Yes | No |
| `STAR_FLUX` | — | Stellar speckle reference spectrum (high-contrast mode only) | No | No | Yes |

```{note}
ForMoSA uses the **first matching alias** it finds. If your file uses `WAVE`
instead of `WAV`, that is fine — both are accepted.
```

```{important}
For photometric observations, `FAC`, `INS`, and `FILT` must be consistent with
the [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/)
naming convention. ForMoSA uses these strings to automatically download and
cache the filter transmission curve.
```

## Creating a FITS file with astropy

```python
import numpy as np
from astropy.io import fits

# Example: K-band spectroscopic observation
wav  = np.linspace(2.0, 2.5, 500)      # µm
flx  = np.random.normal(1.0, 0.05, 500)
err  = np.full(500, 0.05)
res  = np.full(500, 4000.0)            # R ~ 4000

hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(wav,  name="WAV"),
    fits.ImageHDU(flx,  name="FLX"),
    fits.ImageHDU(err,  name="ERR"),
    fits.ImageHDU(res,  name="RES"),
])
# WAVE_UNIT stored as a header keyword on the WAV extension
hdul["WAV"].header["BUNIT"] = "µm"

hdul.writeto("my_observation.fits", overwrite=True)
```

## Inspecting and plotting your file

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open("my_observation.fits") as hdul:
    hdul.info()          # prints all extensions and their shapes
    wav = hdul["WAV"].data
    flx = hdul["FLX"].data
    err = hdul["ERR"].data

plt.figure(figsize=(10, 4))
plt.plot(wav, flx, label="flux")
plt.fill_between(wav, flx - err, flx + err, alpha=0.3, label="1σ")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux")
plt.legend()
plt.tight_layout()
plt.show()
```

## MOSAIC mode: multiple files

When combining multiple instruments, provide one `.fits` file per instrument and
list them all in `ConfigPath.observation_path`:

```python
from ForMoSA.config.global_config import ConfigPath

config_path = ConfigPath(
    observation_path=[
        "data/sphere_yjh.fits",    # SPHERE YJH low-res spectroscopy
        "data/gravity_k.fits",     # GRAVITY K-band spectroscopy
        "data/nircam_photo.fits",  # JWST NIRCam photometry
    ],
    adapt_store_path="adapted_grid/",
    result_path="results/",
    model_path="atm_grids/ExoREM.nc",
)
```

ForMoSA will assign each file an index (0, 1, 2, …) that you can use to set
per-instrument parameters such as `rv_0`, `rv_1`, etc. in `ConfigParameters`.
````

- [ ] **Step 2: Build and verify**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/getting_started/data_formatting.md
git commit -m "docs: write data formatting getting-started page"
```

---

## Task 8: Model grid page

**Files:**
- Modify: `docs/getting_started/model_grid.md`

- [ ] **Step 1: Write full content**

````markdown
# Knowing Your Model Grid

ForMoSA compares observations against **pre-computed, self-consistent atmospheric
model grids** stored as [xarray](https://docs.xarray.dev/en/stable/) `.nc` files.
This page explains what a grid file contains, which grids are available, and
how to inspect one before starting an analysis.

## What is a model grid?

A model grid is a multi-dimensional array of synthetic spectra computed by an
atmospheric code (e.g. BT-Settl, Exo-REM, ATMO) over a regular grid of physical
parameters such as effective temperature (T_eff), surface gravity (log g),
metallicity ([M/H]), and C/O ratio.

ForMoSA uses `xarray.Dataset` to represent grids. The dataset has:

- **Coordinates** — one per physical parameter (e.g. `par1` = T_eff, `par2` = log g).
  The names `par1`–`par4` are fixed; what they *mean* depends on the grid.
- **Data variable `flux`** — shape `(N_par1, N_par2, …, N_wavelength)`.
- **Data variable `res`** — the native spectral resolution at each wavelength point,
  shape `(N_wavelength,)`.
- **Coordinate `wavelength`** — the wavelength axis in microns.

## Available grids

The following grids have been pre-formatted for ForMoSA and are available for
download. Save them inside your `atm_grids/` directory.

| Grid | Reference | Download |
|------|-----------|----------|
| BT-Settl | `Allard et al. 2013 <https://ui.adsabs.harvard.edu/abs/2013MSAIS..24..128A/abstract>`_ | `Download <https://drive.google.com/file/d/1wvf4A-DupdVnYIpK_HmHE-fobqnYtvEz/view?usp=share_link>`_ |
| Exo-REM | `Charnay et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..172C/abstract>`_ | `Download <https://drive.google.com/file/d/1k9SQjHLnMCwmGOHtraRnhCgiZ1-4J3Wk/view?usp=share_link>`_ |
| ATMO | `Phillips et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_ | `Download <https://drive.google.com/file/d/1S1dcBD7UiuUCZIcNBNnJi6LMymrnkagM/view?usp=share_link>`_ |

```{note}
Need a grid that isn't listed here? The ForMoSA team can generate custom
formatted grids on request — open an issue on
[GitHub](https://github.com/exoAtmospheres/ForMoSA/issues).
```

## Inspecting a grid

```python
import xarray as xr

# Open the grid (lazy-loads by default — no RAM spike)
grid = xr.open_dataset("atm_grids/BT-Settl.nc")
print(grid)
# Shows dimensions, coordinates, and data variables

# What parameter axes are available?
print(grid.coords)

# Range of Teff (par1) and logg (par2)
print("Teff range:", float(grid.par1.min()), "–", float(grid.par1.max()), "K")
print("logg range:", float(grid.par2.min()), "–", float(grid.par2.max()))
```

## Plotting a single spectrum

```python
import xarray as xr
import matplotlib.pyplot as plt

grid = xr.open_dataset("atm_grids/BT-Settl.nc")

# Select the model closest to Teff=1600 K, logg=4.0
spectrum = grid.flux.sel(par1=1600, par2=4.0, method="nearest")

plt.figure(figsize=(10, 4))
plt.plot(grid.wavelength, spectrum, linewidth=0.8)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux (model units)")
plt.title("BT-Settl: Teff = 1600 K, log g = 4.0")
plt.tight_layout()
plt.show()
```

## Checking resolution coverage

Before running `analysis.adapt()`, it is worth verifying that the grid's native
spectral resolution is higher than your observation's resolution at the relevant
wavelengths — otherwise the adaptation step cannot degrade the resolution correctly.

```python
import numpy as np
import xarray as xr

grid = xr.open_dataset("atm_grids/BT-Settl.nc")

# Native resolution of the grid across wavelength
plt.figure(figsize=(10, 3))
plt.plot(grid.wavelength, grid.res)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Spectral resolution λ/Δλ")
plt.title("BT-Settl native resolution")
plt.tight_layout()
plt.show()
```
````

- [ ] **Step 2: Build and verify**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/getting_started/model_grid.md
git commit -m "docs: write model grid getting-started page"
```

---

## Task 9: Config file page

**Files:**
- Modify: `docs/getting_started/config_file.md`

- [ ] **Step 1: Write full content**

````markdown
# Knowing the Config

ForMoSA is controlled through a set of **Python dataclasses** — one for each
concern. You can either instantiate them directly in code (preferred) or load
them from an INI file.

## Two ways to configure

### Option A — Python dataclasses (recommended)

```python
from ForMoSA import Analysis
from ForMoSA.config.global_config import (
    ConfigPath, ConfigAdapt, ConfigInversion, ConfigParameters
)

config_path = ConfigPath(
    observation_path  = ["data/obs.fits"],
    adapt_store_path  = "adapted_grid/",
    result_path       = "results/",
    model_path        = "atm_grids/BT-Settl.nc",
)
config_adapt      = ConfigAdapt()
config_inversion  = ConfigInversion(ns_algo="pymultinest", npoints=200)
config_parameters = ConfigParameters(
    par1=["uniform", "500",  "3000"],   # Teff
    par2=["uniform", "2.5",  "5.5"],    # log g
    r   =["uniform", "0.5",  "3.0"],    # radius (R_Jup)
    d   =["constant", "50"],            # distance (pc)
)

analysis = Analysis(config_path)
analysis.adapt(config_adapt, config_inversion)
analysis.nested_sampling(config_parameters, config_adapt, config_inversion)
analysis.plot(analysis.ns.results)
```

### Option B — INI file

```ini
[ConfigPath]
observation_path = data/obs.fits
adapt_store_path = adapted_grid/
result_path      = results/
model_path       = atm_grids/BT-Settl.nc

[ConfigInversion]
ns_algo  = pymultinest
npoints  = 200

[ConfigParameters]
par1 = uniform, 500, 3000
par2 = uniform, 2.5, 5.5
r    = uniform, 0.5, 3.0
d    = constant, 50
```

Load it with:

```python
from ForMoSA.config.global_config import ConfigLoader

loader = ConfigLoader("config.ini")
config_path, config_adapt, config_inversion, config_parameters = loader.load()
```

Generate a template INI file with:

```python
from ForMoSA.config.global_config import ConfigGenerator
ConfigGenerator().save("config_template.ini")
```

## Which dataclass controls what

| Dataclass | Controls |
|-----------|----------|
| `ConfigPath` | File paths: observations, model grid, adapted sub-grids, results |
| `ConfigAdapt` | Grid adaptation: interpolation method, resolution targets, continuum removal, parallelisation |
| `ConfigInversion` | Nested sampling: algorithm choice, live points, likelihood type, fitting wavelength range |
| `ConfigParameters` | Prior distributions for each fitted parameter |

## Prior syntax

Every parameter in `ConfigParameters` accepts a list in the form:

```python
["prior_type", "arg1", "arg2"]
```

| Prior type | Syntax | Description |
|------------|--------|-------------|
| `uniform` | `["uniform", "min", "max"]` | Flat prior between min and max |
| `log_uniform` | `["log_uniform", "min", "max"]` | Flat in log-space |
| `gaussian` | `["gaussian", "mean", "std"]` | Normal distribution |
| `constant` | `["constant", "value"]` | Fixed value, not sampled |
| `NA` | `["NA"]` | Parameter disabled |

## Mode validity matrix

The table below shows which parameters are relevant in each analysis mode.
Parameters marked "No" will be ignored if provided but are not required.

| Parameter | Standard | MOSAIC | Photometry-only | HCHR |
|-----------|:---:|:---:|:---:|:---:|
| `par1`–`par4` | Yes | Yes | Yes | Yes |
| `r` | Yes | Yes | Yes | Yes |
| `d` | Yes | Yes | Yes | Yes |
| `rv` | Yes | Yes | No | Yes |
| `vsini` | Yes | Yes | No | Yes |
| `ld` | Yes | Yes | No | No |
| `alpha` | Yes | Yes | Yes | Yes |
| `bb_T` | Yes | Yes | No | No |

In MOSAIC mode, any parameter can be made **instrument-local** by appending
its observation index: `rv_0`, `rv_1`, `alpha_2`, etc. Global parameters
(without a suffix) are shared across all instruments.

## Parameter reference

### `ConfigPath`

**`observation_path`** *(list of str or Path)*
: Paths to your `.fits` observation files. Single observation: one-element list.
  MOSAIC mode: one entry per instrument.

**`adapt_store_path`** *(str or Path)*
: Directory where adapted sub-grids are saved. Shared across targets in the
  same sample (see [Folder Structure](folder_structure.md)).

**`result_path`** *(str or Path)*
: Directory where ForMoSA writes results: `ns_results.json`, plots, saved
  observation state.

**`model_path`** *(str or Path)*
: Path to the `.nc` model grid file.

---

### `ConfigAdapt`

**`method`** *(str, default `"linear"`)*
: Interpolation method used to resample the model grid onto the observation
  wavelength grid. `"linear"` is robust for most cases.

**`target_res_obs`** *(list, default `["obs"]`)*
: Target spectral resolution for each observation. `"obs"` uses the native
  observation resolution. Provide a float to force a specific resolution.
  One value per observation in MOSAIC mode (or a single value broadcast to all).

**`target_res_mod`** *(list, default `["obs"]`)*
: Target wavelength and resolution for the adapted sub-grid. `"obs"` uses
  the observation wavelength grid; `"mod"` keeps the native model grid.

**`wav_cont`** *(list, default `["NA"]`)*
: Wavelength ranges (in µm) used for continuum estimation and removal. Format:
  `["1.0, 1.3", "1.5, 1.8"]`. `"NA"` disables continuum removal.

**`res_cont`** *(list, default `["NA"]`)*
: Spectral resolution used for the continuum estimate. Must match
  `wav_cont` in length. `"NA"` uses the native grid resolution.

**`backend`** *(str, default `"loky"`)*
: joblib parallelisation backend for grid adaptation. Options: `"loky"`,
  `"multiprocessing"`, `"threading"`, `"sequential"`, `"dask"`, `"ray"`.
  Use `"sequential"` to disable parallelisation for debugging.

**`n_jobs`** *(int, default `-1`)*
: Number of parallel workers. `-1` uses all available CPUs.

---

### `ConfigInversion`

**`ns_algo`** *(str, default `"pymultinest"`)*
: Nested-sampling back-end. Options: `"pymultinest"`, `"nestle"`, `"ultranest"`.
  PyMultiNest is recommended for fits with more than three free parameters.

**`npoints`** *(int, default `50`)*
: Number of live points. More points → better posterior sampling and evidence
  estimate, but longer run time. Start with 50–100 for testing; use 300–500
  for publication-quality runs.

**`logL_type`** *(list, default `["chi2"]`)*
: Log-likelihood function. Options: `"chi2"`, `"chi2_covariance"`,
  `"chi2_noisescaling"`, `"chi2_noisescaling_covariance"`, `"CCF_Zucker"`,
  `"CCF_Brogi"`, `"CCF_custom"`.
  One value per observation in MOSAIC mode (or broadcast).

**`wav_fit`** *(list, default `["0.9, 5.0"]`)*
: Wavelength range (µm) used for the likelihood evaluation. Syntax:
  `["min, max"]`. Points outside this range are masked.
  One value per observation in MOSAIC mode.

**`hc_lower_bounds_lsq`** / **`hc_higher_bounds_lsq`** *(list, default `["NA"]`)*
: Lower and upper bounds for the least-squares optimisation in HCHR mode.
  `"NA"` means unbounded. These are only relevant when `hc_mode = True` in
  the observation file.

---

### `ConfigParameters` — fitted parameters

All parameters share the same prior syntax: `["prior_type", "arg1", "arg2"]`.
Set to `["NA"]` to disable.

**`par1`, `par2`, `par3`, `par4`** — grid parameters
: The physical parameters of the atmospheric model grid. What they represent
  depends on the grid (e.g. for BT-Settl: `par1` = T_eff in K, `par2` = log g).
  Check your grid's documentation or inspect the coordinate names with `xarray`.

**`r`** — radius (R_Jup)
: Companion radius. Used in the physical flux scaling: `flux_obs = flux_model × (r / d)²`.
  Requires `d` to be set. Prior example: `["uniform", "0.5", "3.0"]`.

**`d`** — distance (pc)
: Distance to the system. Usually fixed to the Gaia/Hipparcos value.
  Example: `["constant", "50"]`.

**`rv`** — radial velocity (km/s)
: Doppler shift applied to the model spectrum before comparison. Not applicable
  to photometry-only observations.

**`vsini`** — rotational broadening (km/s)
: Rotational broadening applied to the model via a convolution kernel.
  Requires specifying the kernel function as a fourth element:
  `["uniform", "0", "100", "PyAstronomy"]`. The only currently supported
  function is `"PyAstronomy"`.

**`ld`** — limb-darkening coefficient
: Linear limb-darkening coefficient applied to the model before scaling.
  Spectroscopic mode only.

**`alpha`** — analytical scaling factor
: Multiplies the model flux by a constant: `flux_obs = flux_model × α`.
  Use instead of `r`+`d` when you do not want to constrain the radius.
  See [Analytical vs Physical Scaling](../scaling/analytical_vs_physical.md).

**`bb_T`** — blackbody temperature (K)
: Adds a blackbody component at temperature `bb_T` to the model spectrum.
  Useful when modelling circumplanetary disk contributions or thermal excess.
````

- [ ] **Step 2: Build and verify**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

- [ ] **Step 3: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/getting_started/config_file.md
git commit -m "docs: write config file getting-started page"
```

---

## Task 10: Scaling pages

**Files:**
- Modify: `docs/scaling/index.rst`
- Modify: `docs/scaling/analytical_vs_physical.md`
- Modify: `docs/scaling/mosaic_best_practices.md`

- [ ] **Step 1: Write `scaling/index.rst`**

```rst
.. _scaling:

Scaling
=======

Choosing the right scaling strategy is one of the more consequential decisions
in a ForMoSA analysis. These pages explain the options and when to use each.

.. toctree::
   :maxdepth: 1

   analytical_vs_physical
   mosaic_best_practices
```

- [ ] **Step 2: Write `scaling/analytical_vs_physical.md`**

````markdown
# Analytical vs Physical Scaling

When ForMoSA evaluates the likelihood, it compares a transformed model spectrum
to the observed flux. Part of that transformation is a **scaling** step that
brings the model to the same flux level as the data. ForMoSA offers two
approaches.

## Physical scaling: `r` + `d`

Physical scaling applies the inverse-square law:

$$
F_\text{obs}(\lambda) = F_\text{model}(\lambda) \times \left(\frac{r}{d}\right)^2
$$

where `r` is the companion radius in Jupiter radii and `d` is the distance in
parsecs. This scaling is physically motivated and lets you retrieve the radius
as a free parameter.

```python
config_parameters = ConfigParameters(
    par1 = ["uniform", "800",  "2000"],   # Teff
    par2 = ["uniform", "3.0",  "5.5"],    # log g
    r    = ["uniform", "0.5",  "3.0"],    # radius (R_Jup) — free
    d    = ["constant", "27.7"],          # distance fixed to Gaia value (pc)
)
```

**Use physical scaling when:**
- Your flux calibration is reliable (flux-calibrated spectrum or photometry).
- You want to constrain the companion's physical radius.
- You can fix the distance (e.g. from Gaia parallax).

## Analytical scaling: `alpha`

Analytical scaling multiplies the model by a constant factor:

$$
F_\text{obs}(\lambda) = F_\text{model}(\lambda) \times \alpha
$$

This is a pure nuisance parameter: it absorbs any flux-level offset without
making any physical claim about the radius or distance.

```python
config_parameters = ConfigParameters(
    par1  = ["uniform", "800",  "2000"],
    par2  = ["uniform", "3.0",  "5.5"],
    alpha = ["uniform", "0.0",  "10.0"],  # free scaling factor
    # r and d are NOT set
)
```

**Use analytical scaling when:**
- The absolute flux calibration of your data is uncertain.
- You are fitting contrast spectra (e.g. from integral-field unit observations)
  where the flux level is not physically meaningful.
- You want a quick exploratory fit without committing to a radius prior.

## Side-by-side comparison

| | Physical (`r` + `d`) | Analytical (`alpha`) |
|---|---|---|
| Physical meaning | Yes — retrieves radius | No — nuisance parameter |
| Requires distance | Yes (can be fixed) | No |
| Requires flux calibration | Yes | No |
| Adds free parameter? | Yes (`r`), or fix `d` | Yes (`alpha`) |
| Recommended for | Photometry, flux-calibrated spectra | Contrast spectra, exploratory fits |

## Combining both

You can set both `r`+`d` and `alpha` simultaneously — in that case ForMoSA
applies the physical scaling first and then multiplies by `alpha`. This is
occasionally useful when testing for residual systematic offsets on top of a
physical model, but in most cases you should pick one or the other.
````

- [ ] **Step 3: Write `scaling/mosaic_best_practices.md`**

````markdown
# MOSAIC Best Practices

MOSAIC mode allows ForMoSA to fit **multiple datasets simultaneously**, each
with its own likelihood, while sharing a common set of physical parameters.
This page explains how MOSAIC works, when to use it, and how to avoid common
pitfalls.

## What MOSAIC does

In standard mode, ForMoSA evaluates a single log-likelihood against one
observation. In MOSAIC mode, it evaluates one log-likelihood per observation
and combines them into a **meta-likelihood**:

$$
\ln \mathcal{L}_\text{total} = \sum_{i=0}^{N-1} \ln \mathcal{L}_i
$$

Each observation can have its own wavelength range (`wav_fit_i`), likelihood
type (`logL_type_i`), and intercalibration factor (`alpha_i`). Physical
parameters like T_eff, log g, and radius are shared across all observations.

## Setting up MOSAIC

Provide one `.fits` file per instrument in `observation_path`:

```python
config_path = ConfigPath(
    observation_path = [
        "data/sphere_yjh.fits",   # index 0
        "data/gravity_k.fits",    # index 1
        "data/nircam_photo.fits", # index 2
    ],
    adapt_store_path = "adapted_grid/",
    result_path      = "results/",
    model_path       = "atm_grids/ExoREM.nc",
)
```

Use per-instrument suffixes for local parameters:

```python
config_parameters = ConfigParameters(
    par1    = ["uniform", "800",  "2000"],   # Teff — shared
    par2    = ["uniform", "3.0",  "5.5"],    # log g — shared
    r       = ["uniform", "0.5",  "3.0"],    # radius — shared
    d       = ["constant", "27.7"],           # distance — shared
    alpha_0 = ["uniform", "0.5", "2.0"],     # intercal. for SPHERE
    alpha_1 = ["uniform", "0.5", "2.0"],     # intercal. for GRAVITY
    alpha_2 = ["uniform", "0.5", "2.0"],     # intercal. for NIRCam
)
```

Per-instrument `logL_type` and `wav_fit` in `ConfigInversion`:

```python
config_inversion = ConfigInversion(
    ns_algo   = "pymultinest",
    npoints   = 300,
    logL_type = ["chi2", "chi2", "chi2"],          # one per observation
    wav_fit   = ["0.95, 1.65", "2.0, 2.45", "2.0, 5.0"],  # one per observation
)
```

## When to use MOSAIC

- **Heterogeneous datasets** — different instruments with different resolutions
  and calibration histories.
- **Broadband SED coverage** — combining optical photometry, near-IR
  spectroscopy, and mid-IR photometry.
- **Mitigating calibration biases** — per-instrument `alpha` parameters absorb
  systematic flux offsets without corrupting the shared physical posteriors.

```{note}
See [Ravet et al. (2025)](https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P)
for a detailed study of how MOSAIC handles biases in heterogeneous datasets for
β Pic b.
```

## Common pitfalls

**Too many `alpha` parameters**
: If every observation has a free `alpha`, the likelihood surface can become
  degenerate — especially when observations overlap in wavelength. As a rule
  of thumb, fix `alpha` for your most reliably calibrated dataset and let it
  float for the others.

**Resolution mismatch across datasets**
: The `target_res_mod` in `ConfigAdapt` must be set consistently. If one
  instrument sees the grid at R=4000 and another at R=100, the adapted
  sub-grids will have different wavelength samplings — which is correct —
  but make sure `wav_fit` for each observation only covers wavelengths where
  that instrument actually has data.

**Unequal dataset weights**
: A high-resolution spectrum with 2000 wavelength points will dominate the
  meta-likelihood over a 3-point photometric SED. Consider whether this is
  physically justified, or whether a noise-scaling likelihood
  (`chi2_noisescaling`) is more appropriate for the spectroscopic observation.

**MOSAIC vs independent runs**
: Running ForMoSA separately on each instrument and then comparing the
  posteriors is *not* equivalent to MOSAIC. MOSAIC enforces a shared physical
  model during sampling; independent runs cannot enforce that constraint.
````

- [ ] **Step 4: Build and verify**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

- [ ] **Step 5: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/scaling/
git commit -m "docs: write scaling section (analytical vs physical, MOSAIC best practices)"
```

---

## Task 11: Tutorials stub

**Files:**
- Modify: `docs/tutorials/index.rst`

The tutorial notebooks that already exist in `docs/tutorials/` (formerly `demos/`) are left untouched. This task ensures the index page is clean and the toctree points to the existing notebooks correctly.

- [ ] **Step 1: Check what notebooks are present**

```bash
find /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs/tutorials -name "*.ipynb"
```

Expected output (notebooks that were in `demos/`):
```
docs/tutorials/sinfoni/abpicb/end_to_end_abpicb.ipynb
docs/tutorials/photo/vhs1256b/end_to_end_vhs1256b.ipynb
```

- [ ] **Step 2: Write `tutorials/index.rst`**

```rst
.. _tutorials:

Tutorials
=========

End-to-end worked examples showing ForMoSA in action.

The notebooks below cover two of the most common use cases.
The remaining tutorials are under active development — see the
:ref:`whats_new` page for the latest roadmap.

Spectroscopic: AB Pic b (SINFONI K-band)
-----------------------------------------

Medium-resolution K-band spectrum of the planetary-mass companion AB Pic b,
fitted with BT-Settl using PyMultiNest.

.. toctree::
   :maxdepth: 1

   sinfoni/abpicb/end_to_end_abpicb.ipynb

Photometric: VHS 1256 b
------------------------

Broadband photometry of VHS 1256 b, fitted with BT-Settl using Nestle.

.. toctree::
   :maxdepth: 1

   photo/vhs1256b/end_to_end_vhs1256b.ipynb

Planned tutorials
-----------------

.. todo::

   The following tutorials are in preparation:

   - **HCHR mode** — AF Lep b with VLT/HiRISE and Exo-REM cloudless
     (parallelisation with PyMultiNest)
   - **MOSAIC mode** — Beta Pic b combining spectroscopy and photometry
   - **Advanced plotting** — modifying default plots, computing BIC, χ²_red,
     molecular absorption, best-fit quantiles, interactive matplotlib views
   - **Cluster deployment** — running ForMoSA on an HPC cluster using MPI
```

- [ ] **Step 3: Build and verify notebooks still render**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

Expected: no ERRORs. nbsphinx may emit warnings if notebook kernels are not available — this is acceptable.

- [ ] **Step 4: Commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/tutorials/index.rst
git commit -m "docs: write tutorials index with existing notebooks and planned stubs"
```

---

## Task 12: API index verification

**Files:**
- Read: `docs/api/index.rst`

- [ ] **Step 1: Verify all modules are listed**

Open `docs/api/index.rst` and confirm these entries are present:
`analysis`, `config`, `core`, `observation`, `grid`, `parameter`, `filter`,
`transform`, `nested_sampling`, `plotting`, `main_utilities`.

If `plotting` is missing (it is a separate `.rst` file in `api/`), add it:
```rst
.. toctree::
   :titlesonly:

   analysis
   config
   core
   observation
   grid
   parameter
   filter
   transform
   nested_sampling
   plotting
   main_utilities
```

- [ ] **Step 2: Build and check API pages resolve**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html 2>&1 | grep -E "ERROR" | head -10
```

- [ ] **Step 3: Commit if changed**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/api/index.rst
git commit -m "docs: verify API index lists all modules"
```

---

## Task 13: Final build and cleanup

- [ ] **Step 1: Full clean build**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make clean html 2>&1 | tail -40
```

- [ ] **Step 2: Check for remaining ERRORs and critical WARNINGs**

```bash
conda run -n env_formosa make html 2>&1 | grep -E "^.*(ERROR|WARNING).*$" | grep -v "intersphinx\|nbsphinx\|autodoc_typehints" | head -30
```

Fix any `ERROR` lines. `WARNING` lines from `intersphinx` (network lookups),
`nbsphinx` (notebook kernel not found), or `autodoc_typehints` (forward
references) are acceptable and can be left.

- [ ] **Step 3: Verify the toctree is complete**

Confirm `docs/index.rst` contains exactly:
```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   installation
   getting_started/index
   tutorials/index
   scaling/index
   api/index
   whats_new
```

- [ ] **Step 4: Final commit**

```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA
git add docs/
git commit -m "docs: final v2.0.0 documentation pass — clean build"
```
