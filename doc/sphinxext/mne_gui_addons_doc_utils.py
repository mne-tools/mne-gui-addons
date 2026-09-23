"""Doc building utils."""

import warnings

import mne_gui_addons

gui_scraper = mne_gui_addons._GUIScraper()


def reset_warnings(gallery_conf, fname):
    """Ensure we are future compatible and ignore silly warnings."""
    # In principle, our examples should produce no warnings.
    # Here we cause warnings to become errors, with a few exceptions.

    # remove tweaks from other module imports or example runs
    warnings.resetwarnings()
    # restrict
    warnings.filterwarnings("error")
    # internal warnings
    warnings.filterwarnings("default", module="sphinx")
