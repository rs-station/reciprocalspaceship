# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import inspect
sys.path.insert(0, os.path.abspath('../reciprocalspaceship'))
import reciprocalspaceship as rs

# -- Project information -----------------------------------------------------

project = 'reciprocalspaceship'
copyright = '2020, Jack B. Greisman, Kevin M. Dalton'
author = 'Jack B. Greisman, Kevin M. Dalton'

# The full version, including alpha/beta/rc tags
release = rs.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'nbsphinx',
    "sphinx.ext.linkcode",
    "sphinx_panels",
    "sphinxcontrib.autoprogram",
    "autodocsumm",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False

###
# swap the below commented lines into the prolog to have it point to specific version on the repo (denoted by github releases)
# the `v` is necessary to hardcode because the github tags have the v while the python releases do not.
# currently the default branch on the webpage is the latest branch so I pointed them to master.
###

#   <a class="reference external" href="https://github.com/Hekstra-Lab/reciprocalspaceship/blob/v{{ env.config.release|e }}/{{ docname|e }}">{{ docname|e }}</a>.
#   Interactive online version:
#   <span style="white-space: nowrap;">
#     <a href="https://mybinder.org/v2/gh/Hekstra-Lab/reciprocalspaceship/v{{ env.config.release|e }}?filepath={{ docname|e }}">
#     <img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.

# prolog taken nearly verbatim from https://github.com/spatialaudio/nbsphinx/blob/98005a9d6b331b7d6d14221539154df69f7ae51a/doc/conf.py#L38
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/Hekstra-Lab/reciprocalspaceship/blob/main/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;">
        <a href="https://mybinder.org/v2/gh/Hekstra-Lab/reciprocalspaceship/main?filepath={{ docname|e }}?urlpath=lab">
        <img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.
      </span>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

autoclass_content = "class"
autosummary_generate = True
autodoc_default_options = {
    'members':True,
    'show-inheritance':True,
}
# numpydoc_show_inherited_class_members = False
# numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The main toctree document.
main_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

def setup(app):
    app.add_css_file("custom.css")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add favicon
html_favicon = 'img/rs-favicon_32x32.png'

# Add 
html_logo = 'img/rs.png'

# -- Resolve links to source code --------------------------------------------

# based on pandas/doc/source/conf.py
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(rs.__file__))

    return f"https://github.com/Hekstra-Lab/reciprocalspaceship/blob/main/reciprocalspaceship/{fn}{linespec}"
