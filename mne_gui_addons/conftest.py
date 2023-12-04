# get all MNE fixtures and settings
from mne.conftest import *  # noqa: F403

import warnings

# ignore warnings
warnings.filterwarnings(
    "ignore",
    message="The `pyvista.plotting.plotting` module has been deprecated.*",
)
