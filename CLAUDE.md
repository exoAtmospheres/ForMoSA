# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable)
pip install -e .
conda install dask netCDF4 bottleneck

# Run all tests
pytest

# Run a single test file
pytest tests/test_observation.py

# Run a single test by name
pytest -k "test_name"
```

## Architecture

ForMoSA is a forward-modeling package for exoplanet/brown-dwarf atmosphere retrieval. The workflow has three sequential stages — adapt, sample, plot — all driven through a single `Analysis` entry point in `analysis.py`.

### Execution flow

```
Analysis(ConfigPath) → .adapt(ConfigAdapt, ConfigInversion)
                     → .nested_sampling(ConfigParameters, ...)
                     → .plot(ns.results)
```

1. **`Analysis.__init__`** loads the model grid (`ModelGrid.from_file`) and observations (`ObservationSet.from_fits`). If `adapted=True` or `fitted=True`, it recovers previously saved state from disk instead.
2. **`Analysis.adapt`** resamples each subgrid to each observation's wavelength/resolution grid. Spectroscopic observations produce `SubGridSpectroscopy`; photometric produce `SubGridPhotometry` (which also downloads SVO filter curves). All subgrids are saved to `adapt_store_path`.
3. **`Analysis.nested_sampling`** builds a `ParameterSet`, instantiates `NestedSampling`, and runs the chosen back-end (nestle / PyMultiNest / UltraNest). The back-end calls `transform/observed.py` to apply physical effects (RV, vsini, limb darkening, extinction, dilution) to the model at each likelihood evaluation.
4. **`Analysis.plot`** generates corner plots, chain plots, radar diagrams, and best-fit spectrum figures via `nested_sampling/plotting.py`.

### Configuration system

Two equivalent ways to configure:

- **Python dataclasses** (preferred): `ConfigPath`, `ConfigAdapt`, `ConfigInversion`, `ConfigParameters`, `Config_NS` (which bundles `ConfigNestle`, `ConfigPyMultiNest`, `ConfigUltraNest`) — all in `config/global_config.py`.
- **INI files**: `ConfigLoader` parses a `.ini` file into the same dataclasses; `ConfigGenerator` writes a default `.ini`.

MOSAIC (multi-instrument) mode: list parameters in `ConfigAdapt`/`ConfigInversion` where marked `MOSAIC: Yes`. Single-element lists are auto-broadcast to all observations.

### Key design patterns

- **`ObservationSet`** is an iterable container of `Observation` subclasses (`SpectralObservation`, `PhotometryObservation`). The observation type is stored in `obs.ObsType` as `ObservationType.obstype` string and drives all dispatch logic.
- **`SubGridSet`** mirrors `ObservationSet` — one subgrid per observation, keyed by `obs.name`.
- **`ObservationKeys` enum** (`core/enums.py`) maps canonical FITS column names (e.g. `WAVELENGTH`) to accepted aliases (`WAVE`, `WAV`, `LAMBDA`). `ObservationLoader` uses this to read `.fits` files flexibly.
- **`ParameterKind` enum** distinguishes grid parameters (`par1`…`par4`) from physical parameters (`r`, `d`, `rv`, `vsini`, `ld`, `alpha`, `bb_T`). MOSAIC-local parameters use `name_obsindex` suffix (e.g. `rv_0`).
- **`LogLikelihoodType` enum** selects the likelihood: `chi2`, `chi2_covariance`, `chi2_noisescaling`, `CCF_Brogi`, `CCF_Zucker`, `CCF_custom`.
- **High-contrast mode**: triggered when `obs.hc_mode = True`. The `STAR_FLUX` FITS column provides stellar speckle reference; continuum removal is skipped for HC observations.
- All errors raise `ForMoSAError` (from `core/errors.py`); logging uses `colorlog`/`rich` via `core/loggings.py`.

### Saved artefacts

| File | Location | Content |
|---|---|---|
| `*.npz` subgrids | `adapt_store_path/` | Adapted model subgrid per observation |
| `observations_*.npz` | `result_path/` | Adapted observations (with continuum metadata) |
| `ns_results.json` | `result_path/` | Nested sampling results (JSON-serialised `NSResults`) |
| `corner.pdf`, `chains.pdf`, `radar.pdf`, `best_fit.pdf` | `result_path/` | Diagnostic plots |

## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.
