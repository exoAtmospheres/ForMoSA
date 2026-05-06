# ForMoSA Documentation

This directory contains the Sphinx source files for the ForMoSA documentation.

## Directory Structure

```
docs/
├── index.rst                   # Landing page and top-level toctree
├── installation.rst            # Installation instructions
├── whats_new.rst               # Changelog / release notes
│
├── guidelines/                 # User guidelines
│   ├── index.rst               # Good practices, input formats
│   └── input_format/           # Jupyter notebooks for input formatting
│       ├── obs_format.ipynb
│       ├── model_format.ipynb
│       ├── config_format.ipynb
│       └── config.ini          # Example config file
│
├── demos/                      # End-to-end demo notebooks
│   ├── index.rst               # Demos toctree
│   ├── sinfoni/abpicb/         # Spectroscopic demo (AB Pic b)
│   └── photo/vhs1256b/         # Photometric demo (VHS 1256 b)
│
├── api/                        # API reference (auto-generated from docstrings)
│   ├── index.rst               # API toctree
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
├── _static/                    # Static assets (images, CSS)
├── _build/                     # Build output (git-ignored)
├── conf.py                     # Sphinx configuration
├── requirements.txt            # Doc build dependencies
├── Makefile                    # Unix build helper
└── make.bat                    # Windows build helper
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

### Adding a new demo

1. Place your Jupyter notebook under `demos/` in an appropriate subdirectory (e.g. `demos/instrument/target/`).
2. Include any required data files alongside the notebook.
3. Add the notebook path to the toctree in `demos/index.rst`.

### Adding a new guideline / input format notebook

1. Place the notebook in `guidelines/input_format/`.
2. Reference it from the appropriate section in `guidelines/index.rst`.

### General conventions

- Use reStructuredText (`.rst`) for narrative docs and API references.
- Use Jupyter notebooks (`.ipynb`) for tutorials, demos, and input format guides.
- API `.rst` files use `automodule` — docstrings in the Python source are the single source of truth.
- Static images go in `_static/`.
- Keep the `_build/` directory out of version control.
