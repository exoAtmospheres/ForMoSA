# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import ForMoSA

import os
import sys
sys.path.insert(0, os.path.abspath('./../../ForMoSA')) # location of orbitize files with docstrings

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon', # allows Google style-guide docs to render more prettily
    'sphinx.ext.viewcode',
    'nbsphinx'
    ]

# Disable notebook timeout
nbsphinx_timeout = -1

# Allow notebook errors
nbsphinx_allow_errors = True

# Add any paths that contain templates here, relative to this directory.

templates_path = ['_templates']
exclude_patterns = []

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'


project = 'ForMoSA'
copyright = '2023, S. Petrus, P. Palma-Bifani, M. Bonnefoy, G. Chauvin, et al.'
author = 'Simon Petrus, Paulina Palma-Bifani, Mickaël Bonnefoy, Gaël Chauvin, et al.'

version = ForMoSA.__version__


language = None

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'favicon.ico'
