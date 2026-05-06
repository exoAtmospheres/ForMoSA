# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# -- Path setup -------------------------------------------------------
# Add the project root to sys.path so autodoc can find the package
base_dir = os.path.abspath("..")
sys.path.insert(0, base_dir)

# -- Project information -----------------------------------------------
project = 'ForMoSA'
copyright = '2024, Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet, Allan Denis, Bhavesh Rajpoot, Mickaël Bonnefoy and Gaël Chauvin'
author = 'Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet, Allan Denis, Bhavesh Rajpoot, Mickaël Bonnefoy and Gaël Chauvin'

# Version — keep in sync with pyproject.toml
release = '1.1.6'
root_doc = 'index'

language = 'en'

# -- General configuration ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    # Core Sphinx
    'sphinx.ext.autodoc',               # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',              # Support NumPy-style docstrings
    'sphinx.ext.viewcode',              # Add [source] links to API docs
    'sphinx.ext.intersphinx',           # Cross-reference external docs
    'sphinx.ext.mathjax',               # Render LaTeX math
    'sphinx.ext.todo',                  # Support .. todo:: directives
    'sphinx.ext.doctest',               # Test snippets  in the docs
    'sphinx.ext.inheritance_diagram',   # Class inheritance diagrams
    'sphinx.ext.graphviz',              # Graphviz extension for diagrams

    # Third-party
    'nbsphinx',                    # Source parser for .ipynb files
    'sphinx_autodoc_typehints',    # Render type hints in docs
    'sphinx_copybutton',           # Copy button on code blocks
    'myst_parser',                 # Write docs in Markdown (.md)
]



# File types Sphinx should parse
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Main page
master_doc = 'index'

# Templates
templates_path = []

# Disable notebook timeout
nbsphinx_timeout = -1

# Allow errors from notebooks
nbsphinx_allow_errors = True

# Show both class-level docstring and __init__ docstring in class documentation
autoclass_content = 'both'

# Patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/.ipynb_checkpoints/*', 'README.md']

# -- Napoleon (NumPy docstring) settings --------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = False

# -- Autodoc settings ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Don't actually import these optional packages during doc build
autodoc_mock_imports = []

# Show typehints in the description, not in the signature
autodoc_typehints = 'description'

# Suppress warnings from third-party type annotations (e.g. matplotlib internals)
suppress_warnings = ['sphinx_autodoc_typehints.forward_reference']

# -- Intersphinx (cross-referencing external docs) -----------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


pygments_style = 'sphinx'

html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme = 'sphinx_book_theme'
html_logo = "_static/ForMoSA.png"
html_favicon = '_static/favicon.ico'

html_theme_options = {
    'path_to_docs': 'docs/',
    'repository_url': 'https://github.com/exoAtmospheres/ForMoSA',
    'repository_branch': 'main',
    'use_edit_page_button': True,
    'use_issues_button': True,
    'use_repository_button': True,
    'use_download_button': True,
    'use_fullscreen_button': True,
    'home_page_in_toc': True,
    'show_navbar_depth': 2,
    'show_toc_level': 2,
}

# -- Todo extension ------------------------------------------------------
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Inheritance extension ------------------------------------------------------

inheritance_graph_attrs = dict(rankdir="TB", splines='polyline')
# Also remove minimum node dimensions, and increase line size a bit.
inheritance_node_attrs = dict(height=0.02, margin=0.055, penwidth=1, width=0.01)
inheritance_edge_attrs = dict(penwidth=1)

graphviz_output_format = 'png'
