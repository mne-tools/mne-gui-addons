"""Fixtures and settings for the mne-gui-addons test suite."""

import pytest

# get all MNE fixtures and settings
from mne.conftest import *  # noqa: F403
from mne.conftest import _test_passed
from refleak.testing import Snapshot, gc_collect_once


def pytest_configure(config):
    """Configure pytest options."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*The `pyvista.plotting.plotting` module has been deprecated.*",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item):
    """Flush pending Qt work, including deferred (deleteLater) deletions."""
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    # pyvistaqt >= 0.13 closes a plotter's window with deleteLater(), which
    # processEvents() alone never dispatches; MNE's qt_windows_closed only
    # drains those from 1.12 on, so do it here (before any fixture teardown)
    # for older MNE.
    for _ in range(2):
        app.processEvents()
        app.sendPostedEvents(None, QEvent.DeferredDelete)


_vtk_object_base = None


def _is_leaky(obj):
    """Match plotters and VTK objects, none of which may outlive a test."""
    global _vtk_object_base
    if _vtk_object_base is None:
        from vtkmodules.vtkCommonCore import vtkObjectBase

        _vtk_object_base = vtkObjectBase
    from pyvistaqt import QtInteractor

    return isinstance(obj, (QtInteractor, _vtk_object_base))


@pytest.fixture(autouse=True)
def check_gc(request):
    """Ensure that no plotter or VTK object created during a test survives it."""
    snap = Snapshot(_is_leaky, label="plotter/VTK", freeze=True)
    try:
        yield
        # Don't pile a second failure onto a test that already failed
        if _test_passed(request):
            gc_collect_once(request)
            snap.assert_no_new(f"teardown of {request.node.name}", request=request)
    finally:  # never leave the heap frozen, whatever happened above
        snap.thaw()
