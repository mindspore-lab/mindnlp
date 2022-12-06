# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import mindnlp
sys.path.append(os.path.relpath('..'))


project = 'MindNLP'
copyright = 'MindSpore and CQU NLP Team'
author = 'MindSpore and CQU NLP Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.doctest', 
    'sphinx.ext.intersphinx', 
    'sphinx.ext.todo', 
    'sphinx.ext.coverage', 
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

master_doc = 'index'

# multi-language docs
locale_dirs = ['./locales/'] # the path of zh_CN files(.po files)