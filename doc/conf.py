import faulthandler
import os
import sys

import pyvista
import mne
import mne_gui_addons

faulthandler.enable()
os.environ["_MNE_BROWSER_NO_BLOCK"] = "true"
os.environ["MNE_BROWSER_OVERVIEW_MODE"] = "hidden"
os.environ["MNE_BROWSER_THEME"] = "light"
os.environ["MNE_3D_OPTION_THEME"] = "light"

project = "MNE-GUI-Addons"
release = mne_gui_addons.__version__
version = ".".join(release.split(".")[:2])
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_trees = ["_build"]
default_role = "py:obj"
modindex_common_prefix = ["mne_gui_addons."]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "dipy": (
        "https://dipy.org/documentation/latest/",
        "https://dipy.org/documentation/latest/objects.inv/",
    ),
    "mne": ("https://mne.tools/stable", None),
}
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # MNE
    "SourceSpaces": "mne.SourceSpaces",
    "Info": "mne.Info",
    "Epochs": "mne.Epochs",
    "AverageTFR": "mne.time_frequency.AverageTFR",
    "EpochsTFR": "mne.time_frequency.EpochsTFR",
    "Transform": "mne.transforms.Transform",
    # MNE-GUI-Addons
    # 'IntracranialElectrodeLocator': 'mne_gui_addons.IntracranialElectrodeLocator',  # Many doc errors!
}
numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    # not documented
    "IntracranialElectrodeLocator",
    "VolSourceEstimateViewer",
}
numpydoc_validate = True
numpydoc_validation_checks = {
    "all",
    # These we do not live by:
    "GL01",  # Docstring should start in the line immediately after the quotes
    "EX01",
    "EX02",  # examples failed (we test them separately)
    "ES01",  # no extended summary
    "SA01",  # no see also
    "YD01",  # no yields section
    "SA04",  # no description in See Also
    "PR04",  # Parameter "shape (n_channels" has no type
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
}
numpydoc_validation_exclude = {  # set of regex
    r"mne\.utils\.deprecated",
}
pyvista.OFF_SCREEN = False
pyvista.BUILDING_GALLERY = True
sphinx_gallery_conf = {
    "doc_module": ("mne_gui_addons",),
    "reference_url": dict(mne_gui_addons=None),
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "thumbnail_size": (160, 112),
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "abort_on_example_error": False,
    "image_scrapers": ("matplotlib", mne.gui._GUIScraper(), "pyvista"),
    "show_memory": not sys.platform.startswith(("win", "darwin")),
    "line_numbers": False,  # messes with style
    "capture_repr": ("_repr_html_",),
    "junit": os.path.join("..", "test-results", "sphinx-gallery", "junit.xml"),
    "matplotlib_animations": True,
    "compress_images": ("images", "thumbnails"),
    "filename_pattern": "^((?!sgskip).)*$",
}

autosummary_generate = True
autodoc_default_options = {"inherited-members": None}
nitpicky = True
nitpick_ignore = []
nitpick_ignore_regex = []
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/mne-tools/mne-gui-addons",
            icon="fa-brands fa-square-github",
        ),
        dict(
            name="Mastodon",
            url="https://fosstodon.org/@mne",
            icon="fa-brands fa-mastodon",
            attributes=dict(rel="me"),
        ),
        dict(
            name="Twitter",
            url="https://twitter.com/mne_python",
            icon="fa-brands fa-square-twitter",
        ),
        dict(
            name="Discourse",
            url="https://mne.discourse.group/",
            icon="fa-brands fa-discourse",
        ),
        dict(
            name="Discord",
            url="https://discord.gg/rKfvxTuATa",
            icon="fa-brands fa-discord",
        ),
    ],
    "use_edit_page_button": False,
}
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False
htmlhelp_basename = "mne-gui-addons-doc"
