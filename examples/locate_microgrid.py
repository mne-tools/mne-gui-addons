"""
.. _ex-ieeg-microgrid:

==================================
Locating an intracranial microgrid
==================================

For intracranial grids that are too small for even high-resolution
computed tomography (CT) images to detect, the contacts must be
aligned with an intraoperative image such as from a surgical
microscope as shown below.
"""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import numpy as np
import nibabel as nib
import mne
import mne_gui_addons as mne_gui

# path to sample sEEG
misc_path = mne.datasets.misc.data_path()
subjects_dir = misc_path / "ecog"

# GUI requires pyvista backend
mne.viz.set_3d_backend("pyvistaqt")

# we need two things:
# 1) The electrophysiology file which contains the channels names
# that we would like to associate with positions in the brain
# 2) A scope image of the grid
raw = mne.io.read_raw(misc_path / "ecog" / "sample_ecog_ieeg.fif")
trans = mne.coreg.estimate_head_mri_t("sample_ecog", subjects_dir)

raw.set_montage(None)  # remove macro channels

gui = mne_gui.locate_ieeg(raw.info, trans, subject="sample_ecog", subjects_dir=subjects_dir)
