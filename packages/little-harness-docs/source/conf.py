"""Sphinx configuration for little-harness documentation."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_SRC = REPO_ROOT / "packages"

for src_dir in PACKAGES_SRC.glob("*/src"):
    sys.path.insert(0, str(src_dir))

sys.path.insert(0, str(PACKAGES_SRC / "little-harness-core" / "src"))

project = "little-harness"
copyright = "2025, Gabriel Menezes"
author = "Gabriel Menezes"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

always_document_param_types = True
typehints_use_rtype = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "little-harness"
html_baseurl = "https://dmenezesgabriel.github.io/little-harness/"
html_copy_source = False
html_show_sourcelink = False

html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
}

pygments_style = "friendly"
pygments_dark_style = "monokai"
