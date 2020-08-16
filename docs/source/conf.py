# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

from scikeras import __version__


project = "SciKeras"
copyright = "2020, SciKeras Developers"
author = "SciKeras Developers"
release = __version__


# -- General configuration ---------------------------------------------------

#  on_rtd is whether we are on readthedocs.org, this line of code grabbed
#  from docs.readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "alphabetical",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": True,
}
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_context = {
    "display_github": True,
    "github_user": "adriangb",
    "github_repo": "scikeras",
    "github_version": "master",
    "conf_py_path": "/source/",
}

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- GitHub source code link ----------------------------------------------

# Functionality to build github source URI, taken from sklearn.

import inspect
import subprocess

from functools import partial
from operator import attrgetter


REVISION_CMD = "git rev-parse --short HEAD"


def _get_git_revision():
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()
    except (subprocess.CalledProcessError, OSError):
        print("Failed to execute git to get revision")
        return None
    return revision.decode("utf-8")


def _linkcode_resolve(domain, info, package, url_fmt, revision):
    """Determine a link to online source for a class/method/function
    This is called by sphinx.ext.linkcode
    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='http://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'http://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """

    if revision is None:
        return
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    class_name = info["fullname"].split(".")[0]
    if type(class_name) != str:
        # Python 2 only
        class_name = class_name.encode("utf-8")
    module = __import__(info["module"], fromlist=[class_name])
    obj = attrgetter(info["fullname"])(module)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return

    fn = os.path.relpath(
        fn, start=os.path.dirname(__import__(package).__file__)
    )
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ""
    return url_fmt.format(
        revision=revision, package=package, path=fn, lineno=lineno
    )


def project_linkcode_resolve(domain, info):
    global _linkcode_git_revision
    return _linkcode_resolve(
        domain,
        info,
        package="scikeras",
        revision=_linkcode_git_revision,
        url_fmt="https://github.com/adriangb/scikeras/"
        "blob/{revision}/"
        "{package}/{path}#L{lineno}",
    )


_linkcode_git_revision = _get_git_revision()

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = project_linkcode_resolve
