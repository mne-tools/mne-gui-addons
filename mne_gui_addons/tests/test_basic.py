import mne_gui_addons
from packaging.version import Version


def test_import():
    """Test that import works."""
    assert Version(mne_gui_addons.__version__) > Version("0.0.0")
