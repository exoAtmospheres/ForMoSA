<p align="left"><img src="docs/_static/ForMoSA.png" alt="ForMoSA" width="250"/></p>

# ForMoSA — Forward Modeling Tool for Spectral Analysis

[![PyPI version](https://badge.fury.io/py/formosa.svg)](https://badge.fury.io/py/formosa)
[![PyPI downloads](https://img.shields.io/pypi/dm/formosa.svg)](https://pypistats.org/packages/formosa)
[![Documentation Status](https://readthedocs.org/projects/formosa/badge/?version=latest)](https://formosa.readthedocs.io/en/latest/?badge=latest)
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

See the [installation guide](https://formosa.readthedocs.io/en/latest/installation.html) for PyMultiNest, GPU/torch, and macOS Apple Silicon instructions.

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

## Attribution

If you use ForMoSA in your research, please cite [Petrus et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P/abstract).

---

## Issues

If you encounter any problems, please open an issue on [GitHub](https://github.com/exoAtmospheres/ForMoSA/issues).

---

## Acknowledgments

Our sincere thanks to [Code/Astro](https://semaphorep.github.io/codeastro/).

