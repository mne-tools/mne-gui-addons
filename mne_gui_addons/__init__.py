from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mne_gui_addons")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover
