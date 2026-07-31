# ForMoSA Documentation

This directory contains the Sphinx source files for the ForMoSA documentation,
built and hosted on [Read the Docs](https://formosa.readthedocs.io).

## Directory Structure

```
docs/
├── index.rst                   # Landing page and top-level toctree
├── installation.rst            # Installation instructions
├── whats_new.rst                # v2.0.0 migration guide (one-time narrative, not
│                                 # a rolling changelog -- see /CHANGELOG.md for that)
│
├── getting_started/              # Getting-started guides
│   ├── index.rst
│   ├── config_file.md
│   ├── data_formatting.md
│   ├── folder_structure.md
│   └── model_grid.md
│
├── good_practices/                # Modeling strategy / methodology notes
│   ├── index.rst
│   ├── analytical_vs_physical.md
│   ├── mosaic_howto.md
│   └── performances_accuracy.md
│
├── tutorials/                     # End-to-end demo notebooks
│   ├── index.rst
│   ├── photo/vhs1256b/            # Photometric demo (VHS 1256 b)
│   ├── spectroscopy/abpicb/       # Spectroscopic demo (AB Pic b)
│   ├── mosaic/hip64892b/          # Multi-instrument (MOSAIC) demo
│   ├── hchr/aflepb/               # High-contrast, high-resolution demo
│   ├── plotting/                  # Advanced plotting & statistical-test notebooks
│   └── cluster/                   # Running on a compute cluster
│
├── api/                           # API reference (auto-generated from docstrings)
│   ├── index.rst
│   ├── analysis.rst
│   ├── config.rst
│   ├── core.rst
│   ├── observation.rst
│   ├── grid.rst
│   ├── parameter.rst
│   ├── filter.rst
│   ├── transform.rst
│   ├── nested_sampling.rst
│   ├── plotting.rst
│   └── main_utilities.rst
│
├── _static/                       # Static assets (images, CSS)
├── _build/                        # Build output (git-ignored)
├── conf.py                        # Sphinx configuration
├── requirements.txt               # Doc build dependencies
├── Makefile                       # Unix build helper
└── make.bat                       # Windows build helper
```

## Building the Docs

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

You also need [Pandoc](https://pandoc.org/installing.html) installed on your system for notebook rendering.

### Build

From the `docs/` directory:

```bash
make html
```

The output will be in `_build/html/`. Open `_build/html/index.html` in a browser to preview.

To clean previous builds:

```bash
make clean
```

## Contributing to the Docs

### Adding a new API page

1. Create a new `.rst` file in `api/` (e.g. `api/my_module.rst`).
2. Use `automodule` directives to pull docstrings from the source code.
3. Add the filename (without `.rst`) to the toctree in `api/index.rst`.

### Adding a new tutorial

1. Place your Jupyter notebook under `tutorials/` in an appropriate subdirectory (e.g. `tutorials/instrument/target/`).
2. Include any required data files alongside the notebook.
3. Add the notebook path to the toctree in `tutorials/index.rst`.

### Adding a getting-started or good-practices page

1. Place the `.md` file in `getting_started/` or `good_practices/`.
2. Reference it from the appropriate section in that directory's `index.rst`.

### General conventions

- Use reStructuredText (`.rst`) for narrative docs and API references.
- Use Markdown (`.md`) for getting-started and good-practices pages, and Jupyter notebooks (`.ipynb`) for tutorials.
- API `.rst` files use `automodule` — docstrings in the Python source are the single source of truth.
- Static images go in `_static/`.
- Keep the `_build/` directory out of version control.
- Don't hand-edit `/CHANGELOG.md` (repo root) — it's auto-generated from commit history on every push. `whats_new.rst` is a separate, hand-written v2.0.0 migration narrative and isn't affected by that automation.
