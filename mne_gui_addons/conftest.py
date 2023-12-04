# get all MNE fixtures and settings
from mne.conftest import *  # noqa: F403

import warnings


def pytest_configure(config):
    """Configure pytest options."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:*The `pyvista.plotting.plotting` module has been deprecated.*",
    )
