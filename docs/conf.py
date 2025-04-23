# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
base_dir = os.path.abspath("..")
sys.path.insert(0, base_dir)


project = 'ForMoSA'
copyright = '2024, Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet, Allan Denis, Mickaël Bonnefoy and Gaël Chauvin'
author = 'Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet, Allan Denis, Mickaël Bonnefoy and Gaël Chauvin'
release = '1.1.5'
root_doc = 'index'

language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.mathjax',    
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'nbsphinx',
              ]


templates_path = []

# Disable notebook timeout
nbsphinx_timeout = -1

# Allow errors from notebooks
nbsphinx_allow_errors = True

autoclass_content = 'both'


exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/.ipynb_checkpoints/*']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

import sphinx_rtd_theme
import sphinx_rtd_theme

html_static_path = ['_static']

#html_theme = 'bizstyle'
html_theme = 'sphinx_rtd_theme'
html_logo = "_static/ForMoSA.png"
html_favicon = '_static/favicon.ico'
#html_theme = 'classic'

#html_theme_options = {
#    'path_to_docs': 'docs',
#    'repository_url': 'https://github.com/exoAtmospheres/ForMoSA',
#    'repository_branch': 'activ_dev',
#    'launch_buttons': {
#        'notebook_interface': 'jupyterlab',
#    },
#    'use_edit_page_button': True,
#    'use_issues_button': True,
#    'use_repository_button': True,
#    'use_download_button': True,
#}

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

