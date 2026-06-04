<p align="left"><img src="docs/_static/ForMoSA.png" alt="ForMoSA" width="250"/></p>

# ForMoSA — Forward Modeling Tool for Spectral Analysis

[![PyPI version](https://badge.fury.io/py/formosa.svg)](https://badge.fury.io/py/formosa)
[![PyPI downloads](https://img.shields.io/pypi/dm/formosa.svg)](https://pypistats.org/packages/formosa)
[![Documentation Status](https://readthedocs.org/projects/formosa/badge/?version=latest)](https://formosa.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/exoAtmospheres/ForMoSA)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Made at Code/Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)

ForMoSA is an open-source Python package for modeling exoplanetary atmospheres using a forward modeling approach. It compares observed spectra and photometry against grids of atmospheric models via nested sampling to derive posterior distributions on physical parameters.

📖 **Full documentation:** [https://formosa.readthedocs.io](https://formosa.readthedocs.io)

---

## Features

- **Class-based API** centred around a single `Analysis` entry point
- **Multi-instrument support (MOSAIC)** — fit spectroscopic and photometric data simultaneously from multiple instruments
- **Three nested-sampling back-ends** — [nestle](http://kylebarbary.com/nestle/), [PyMultiNest](https://github.com/JohannesBuchner/PyMultiNest), and [UltraNest](https://johannesbuchner.github.io/UltraNest/)
- **High-contrast mode** — model stellar speckles and systematics alongside the companion signal
- **Photometry filter service** — automatic retrieval and caching of filter curves from the [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/)
- **Configurable prior distributions** — uniform, log-uniform, Gaussian, and constant priors
- **Comprehensive plotting** — corner plots, chain diagnostics, radar diagrams, best-fit spectra, CCFs, and RV–v sin i maps
- **Flexible configuration** — INI-based config files or direct Python dataclass instantiation

---

## Installation

### From PyPI

```bash
pip install ForMoSA
conda install dask netCDF4 bottleneck
```

### From source

```bash
git clone https://github.com/exoAtmospheres/ForMoSA.git
cd ForMoSA
pip install -e .
conda install dask netCDF4 bottleneck
```

### For developers

The optional `dev` extra adds the testing, packaging, and documentation-building
tools on top of the runtime dependencies:

```bash
pip install -e ".[dev]"
```

A plain `pip install ForMoSA` installs only what is needed to run analyses — no
test or documentation tooling.

ForMoSA requires **Python ≥ 3.10**. See the
[installation guide](https://formosa.readthedocs.io/en/latest/installation.html)
for PyMultiNest, GPU/torch, and macOS Apple Silicon instructions.

---

## Quick Start

```python
from ForMoSA import Analysis
from ForMoSA.config.global_config import (
    ConfigPath, ConfigAdapt, ConfigInversion, ConfigParameters
)

# 1. Define paths
config_path = ConfigPath(
    observation_path=["path/to/observation.fits"],
    adapt_store_path="path/to/adapted_grid/",
    result_path="path/to/results/",
    model_path="path/to/model_grid.nc",
)

# 2. Initialise the analysis
analysis = Analysis(config_path)

# 3. Configure adaptation & inversion
config_adapt = ConfigAdapt(method="linear")
config_inversion = ConfigInversion(ns_algo="pymultinest", npoints=100)
config_parameters = ConfigParameters(
    par1=["uniform", "500", "3000"],   # e.g. Teff
    par2=["uniform", "2.5", "5.5"],    # e.g. log(g)
    r=["uniform", "0.5", "3.0"],       # radius in R_Jup
    d=["constant", "50"],              # distance in pc
)

# 4. Adapt the model grid to the observations
analysis.adapt(config_adapt, config_inversion)

# 5. Run nested sampling
analysis.nested_sampling(config_parameters, config_adapt, config_inversion)

# 6. Plot results
analysis.plot(analysis.ns.results)
```

---

## Package Structure

```
ForMoSA/
├── analysis.py              # Main Analysis class
├── config/                  # Configuration dataclasses & file I/O
├── core/                    # Enums, errors, logging, plot configs
├── filter/                  # SVO photometry filter interface
├── grid/                    # Model grids, subgrids, adaptation
├── nested_sampling/         # NS engine, post-processing, results, plotting
├── observation/             # Observation classes (spectral, photometric, set)
├── parameter/               # Parameters, priors, parameter sets
├── transform/               # Physics & observational effect pipelines
└── utils/                   # Spectral tools, likelihood & prior functions
```

---

## What's New in v2.0.0

> **v2.0.0 is a complete rewrite and is not backwards-compatible with v1.x.**
> See the [full migration guide](https://formosa.readthedocs.io/en/latest/whats_new.html) in the docs.

**Project updates since the v2.0.0 rewrite:**

- **Automated testing** — every push and pull request is install- and import-checked and the test suite is run on Python 3.10, 3.11, and 3.12.
- **One-tag releases** — pushing a `v*` tag builds the package and publishes it to PyPI via a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/), with auto-generated GitHub release notes.
- **JOSS paper** — a software paper accompanying the v2.0.0 release is in preparation.

| Area | v1.x | v2.0.0 |
|---|---|---|
| Entry point | `main.py` script + `launch_adapt()` / `launch_nested_sampling()` | Single `Analysis` class |
| Configuration | `config.ini` + `GlobFile` | Python dataclasses (`ConfigPath`, `ConfigAdapt`, …) |
| Observations | Raw arrays from INI file paths | Typed `ObservationSet` loaded from `.fits` |
| Model grid handling | `adapt/` functions | `grid/` subgrid classes |
| Photometry filters | Bundled `.npz` files | Auto-downloaded from SVO and cached |
| CCF / RV–v sin i | ✗ | ✓ new via `analysis.plot_ccf()` / `analysis.plot_rv_vsini_map()` |
| Logging | `print` statements | Python `logging` module |
| Error handling | Generic exceptions | `ForMoSAError` |

### Upgrading from v1.x

Replace the old script-based calls:

```python
# v1.x (old)
from ForMoSA.global_file import GlobFile
from ForMoSA.adapt.adapt_obs_mod import launch_adapt
from ForMoSA.nested_sampling.nested_sampling import launch_nested_sampling

global_params = GlobFile("config.ini")
launch_adapt(global_params, adapt_model=True)
launch_nested_sampling(global_params)
```

with the new class-based API:

```python
# v2.0.0 (new)
from ForMoSA import Analysis
from ForMoSA.config.global_config import ConfigPath, ConfigAdapt, ConfigInversion, ConfigParameters

analysis = Analysis(ConfigPath(
    observation_path=["obs.fits"],
    adapt_store_path="adapted_grid/",
    result_path="results/",
    model_path="model_grid.nc",
))
analysis.adapt(ConfigAdapt(), ConfigInversion())
analysis.nested_sampling(ConfigParameters(par1=["uniform","500","3000"], r=["uniform","0.5","3.0"], d=["constant","50"]))
analysis.plot(analysis.ns.results)
```

If you need the old behaviour in the short term, pin to `formosa==1.1.6`.

---

## Attribution

If you use ForMoSA in your research, please cite [Petrus et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P/abstract).

---

## Contributing

Contributions are welcome — bug reports, fixes, new features, and documentation
improvements alike. To set up a development environment:

```bash
git clone https://github.com/exoAtmospheres/ForMoSA.git
cd ForMoSA
pip install -e ".[dev]"
conda install dask netCDF4 bottleneck
```

Then:

1. **Branch off `activ_dev`** (the integration branch), not `main`:
   ```bash
   git checkout activ_dev
   git checkout -b my-feature
   ```
2. **Run the test suite** before opening a pull request:
   ```bash
   pytest
   ```
   > **Note:** the `tests/` suite is currently being updated to the v2.0.0
   > module layout; some modules are temporarily skipped. Continuous integration
   > checks that the package installs and imports cleanly on Python 3.10–3.12.
3. **Build the documentation locally** if your change touches the docs:
   ```bash
   cd docs && make html
   ```
4. **Open a pull request** against `activ_dev` describing the change.

Releases are automated: once changes are merged and a maintainer pushes a
`v*` tag (e.g. `v2.0.0`), the package is built and published to PyPI and a
GitHub release is created automatically.

---

## Issues

If you encounter any problems, please open an issue on [GitHub](https://github.com/exoAtmospheres/ForMoSA/issues).

---

## Acknowledgments

Our sincere thanks to [Code/Astro](https://semaphorep.github.io/codeastro/).

