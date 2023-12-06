# -*- coding: utf-8 -*-
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-clause

import numpy as np
from numpy.testing import assert_allclose

import pytest

from mne.datasets import testing
from mne.utils import catch_logging, use_log_level
from mne.viz.utils import _fake_click

data_path = testing.data_path(download=False)
subject = "sample"
subjects_dir = data_path / "subjects"


@testing.requires_testing_data
def test_slice_browser_io(renderer_interactive_pyvistaqt):
    """Test the input/output of the slice browser GUI."""
    nib = pytest.importorskip("nibabel")
    from mne_gui_addons._core import SliceBrowser

    data = np.ones((96, 96, 96), dtype=np.float32)
    data[30:50, 30:50, 30:50] = 2

    with pytest.warns(match="`pial` surface not found"):
        SliceBrowser(
            nib.MGHImage(data, np.eye(4)),
            subject=subject,
            subjects_dir=subjects_dir,
        )


# TODO: For some reason this leaves some stuff un-closed, we should fix it
@pytest.mark.allow_unclosed
@testing.requires_testing_data
def test_slice_browser_display(renderer_interactive_pyvistaqt):
    """Test that the slice browser GUI displays properly."""
    pytest.importorskip("nibabel")
    from mne_gui_addons._core import SliceBrowser

    # test no seghead, fsaverage doesn't have seghead
    with pytest.warns(RuntimeWarning, match="`seghead` not found"):
        with catch_logging() as log:
            gui = SliceBrowser(
                subject="fsaverage", subjects_dir=subjects_dir, verbose=True
            )
    log = log.getvalue()
    assert "using marching cubes" in log
    gui.close()

    # test functions
    with pytest.warns(RuntimeWarning, match="`pial` surface not found"):
        gui = SliceBrowser(subject=subject, subjects_dir=subjects_dir)

    # test RAS
    gui._RAS_textbox.setText("10 10 10")
    gui._RAS_textbox.focusOutEvent(event=None)
    assert_allclose(gui._ras, [10, 10, 10], atol=1e-5)

    # test vox
    gui._VOX_textbox.setText("150, 150, 150")
    gui._VOX_textbox.focusOutEvent(event=None)
    assert_allclose(gui._ras, [-27.27362, 31.039082, -49.287958], atol=0.1)

    # test click
    with use_log_level("debug"):
        _fake_click(
            gui._figs[2], gui._figs[2].axes[0], [137, 140], xform="data", kind="release"
        )
    assert_allclose(gui._ras, [4.726382, 21.039079, -49.287956], atol=0.1)
    gui.close()
