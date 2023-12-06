# -*- coding: utf-8 -*-
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-clause

import numpy as np

import pytest

from mne.datasets import testing

data_path = testing.data_path(download=False)
subject = "sample"
subjects_dir = data_path / "subjects"


@testing.requires_testing_data
def test_segment_io(renderer_interactive_pyvistaqt):
    """Test the input/output of the slice browser GUI."""
    from mne_gui_addons._segment import VolumeSegmenter

    with pytest.warns(match="`pial` surface not found"):
        VolumeSegmenter(
            subject=subject,
            subjects_dir=subjects_dir,
        )


# TODO: For some reason this leaves some stuff un-closed, we should fix it
@pytest.mark.allow_unclosed
@testing.requires_testing_data
def test_segment_display(renderer_interactive_pyvistaqt):
    """Test that the slice browser GUI displays properly."""
    pytest.importorskip("nibabel")
    from mne_gui_addons._segment import VolumeSegmenter

    # test no seghead, fsaverage doesn't have seghead
    with pytest.warns(RuntimeWarning, match="`seghead` not found"):
        gui = VolumeSegmenter(
            subject="fsaverage", subjects_dir=subjects_dir, verbose=True
        )

    # test functions
    gui.set_RAS([25.37, 0.00, 34.18])

    # test mark
    gui._mark()
    assert abs(np.nansum(gui._vol_img) - 250) < 3

    # increase tolerance
    gui.set_tolerance(0.5)

    # check more voxels marked
    gui._mark()
    assert np.nansum(gui._vol_img) > 253

    # check smooth
    gui.set_smooth(0.7)

    gui.close()
