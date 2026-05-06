# ForMoSA Documentation

This directory contains the Sphinx source for the ForMoSA 2.0.0 documentation.

## Structure

```text
docs/
├── index.rst                 # Landing page
├── installation.rst          # Installation guide
├── tutorials/                # Narrative workflow pages
├── guidelines/               # Practical input/layout guidance
├── demos/                    # Maintained example notebooks
├── api/                      # API reference generated from docstrings
├── scaling.rst               # Scaling notes
├── whats_new.rst             # v2 migration and release notes
├── _static/                  # Static assets
├── conf.py                   # Sphinx configuration
└── requirements.txt          # Documentation build dependencies
```

## Build

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Then build from the repository root:

```bash
make -C docs html
```

The rendered site will be written to `docs/_build/html/`.

## Conventions

- Use `.rst` for narrative pages.
- Keep API pages thin and let docstrings remain the source of truth.
- Use notebooks for worked examples that are meant to be executed.
- When a requested tutorial or figure is not yet present in the repository, use
  a square-bracket placeholder instead of inventing content.
