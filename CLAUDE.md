# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, runtime only — public profile)
pip install -e .
conda install dask netCDF4 bottleneck

# Developer profile (adds pytest, build, twine, and the Sphinx/nbsphinx/pandoc docs stack)
pip install -e ".[dev]"

# Run all tests
# NOTE: pytest also needs the conda-only runtime deps installed
# (pip install dask netCDF4 bottleneck). Full suite = 86 tests.
pytest

# Run a single test file
pytest tests/test_observation.py

# Run a single test by name
pytest -k "test_name"
```

## Packaging & Release

- **Python ≥ 3.10 required.** The code uses PEP 604 unions (`X | None`) both in import-time
  annotations and at runtime (e.g. `isinstance(x, str | os.PathLike)`), so it cannot import on
  3.9. Dev env and Read the Docs run 3.12.
- **PyPI project name is lowercase `formosa`** (the `ForMoSA` dist normalises to it). The 2.x
  line starts at **2.0.1** — `2.0.0` is permanently unpublishable (filename burned; see gotchas).
- **Version is single-sourced** from `ForMoSA/__init__.py.__version__`; `pyproject.toml` reads it
  dynamically and `docs/conf.py` reads it from installed metadata. To bump, edit only `__version__`.
- **Release = push a `v*` tag.** `.github/workflows/publish.yml` then builds, creates a GitHub
  Release (with the two tutorial FITS files attached as assets), and publishes to PyPI via Trusted
  Publisher OIDC (requires the GitHub `pypi` environment — already configured).
- **CI workflows:** `tests.yml` (install + public-API import + pytest on 3.10/3.11/3.12, with
  `dask netCDF4 bottleneck` pip-installed for the suite), `publish.yml` (tag-triggered),
  `codeql.yml`, `draft-pdf.yml` (JOSS paper, on `paper/**`).

### Release gotchas (all hit during the first 2.x releases)
- `build-backend` must be `setuptools.build_meta` (not `setuptools.backends.legacy:build`).
- License classifier must be a valid trove classifier: `License :: OSI Approved :: BSD License`.
- **PyPI permanently reserves deleted filenames** — a version uploaded then deleted can never be
  re-uploaded; and an existing version can't be re-uploaded either. Always bump `__version__`,
  and make sure the `v*` tag points at the commit that contains the bump.
- Verify before tagging: `python -m build` + `twine check dist/*`, and validate classifiers
  against the `trove-classifiers` package.

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
