from packaging.version import Version

import mne_gui_addons


def test_import():
    """Test that import works."""
    assert Version(mne_gui_addons.__version__) > Version("0.0.0")
