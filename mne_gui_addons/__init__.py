"""Convenience functions for opening GUIs."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

from importlib.metadata import version, PackageNotFoundError

import numpy as np
from mne.utils import verbose as _verbose, _check_option

from ._utils import _fill_doc

try:
    __version__ = version("mne_gui_addons")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover


@_verbose
@_fill_doc
def locate_ieeg(
    info,
    trans,
    base_image,
    subject=None,
    subjects_dir=None,
    groups=None,
    show=True,
    block=False,
    verbose=None,
):
    """Locate intracranial electrode contacts.

    Parameters
    ----------
    %(info_not_none)s
    %(trans_not_none)s
    base_image : path-like | nibabel.spatialimages.SpatialImage
        The CT or MR image on which the electrode contacts can located. It
        must be aligned to the Freesurfer T1 if ``subject`` and
        ``subjects_dir`` are provided. Path-like inputs and nibabel image
        objects are supported.
    %(subject)s
    %(subjects_dir)s
    groups : dict | None
        A dictionary with channels as keys and their group index as values.
        If None, the groups will be inferred by the channel names. Channel
        names must have a format like ``LAMY 7`` where a string prefix
        like ``LAMY`` precedes a numeric index like ``7``. If the channels
        are formatted improperly, group plotting will work incorrectly.
        Group assignments can be adjusted in the GUI.
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    %(verbose)s

    Returns
    -------
    gui : instance of IntracranialElectrodeLocator
        The graphical user interface (GUI) window.
    """
    from mne.viz.backends._utils import _init_mne_qtapp, _qt_app_exec
    from ._ieeg_locate import IntracranialElectrodeLocator

    app = _init_mne_qtapp()

    gui = IntracranialElectrodeLocator(
        info,
        trans,
        base_image,
        subject=subject,
        subjects_dir=subjects_dir,
        groups=groups,
        show=show,
        verbose=verbose,
    )
    if block:
        _qt_app_exec(app)
    return gui


@_verbose
@_fill_doc
def view_vol_stc(
    stcs,
    freq_first=True,
    group=False,
    subject=None,
    subjects_dir=None,
    src=None,
    inst=None,
    use_int=True,
    show_topomap=True,
    tmin=None,
    tmax=None,
    show=True,
    block=False,
    verbose=None,
):
    """View a volume time and/or frequency source time course estimate.

    Parameters
    ----------
    stcs : list of list | generator
        The source estimates, the options are: 1) List of lists or
        generators for epochs and frequencies (i.e. using
        :func:`mne.minimum_norm.apply_inverse_tfr_epochs` or
        :func:`mne.beamformer.apply_dics_tfr_epochs`-- in this case
        use ``freq_first=False``), or 2) List of source estimates across
        frequencies (e.g. :func:`mne.beamformer.apply_dics_csd`),
        or 3) List of source estimates across epochs
        (e.g. :func:`mne.minimum_norm.apply_inverse_epochs` and
        :func:`mne.beamformer.apply_dics_epochs`--in this
        case use ``freq_first=False``), or 4) Single
        source estimates (e.g. :func:`mne.minimum_norm.apply_inverse`
        and :func:`mne.beamformer.apply_dics`, note ``freq_first``
        will not be used in this case), or 5) List of list of lists or
        generators for subjects and frequencies and epochs (e.g.
        :func:`mne.minimum_norm.apply_inverse_tfr_epochs` for each subject in
        a list; use ``group=True``), or 6) List or generator for subjects
        with ``stcs`` from evoked data (e.g.
        :func:`mne.minimum_norm.apply_inverse` or
        :func:`mne.beamformer.apply_dics_csd` for each subject in a
        list; use ``group=True``).
    freq_first : bool
        If frequencies are the outer list of ``stcs`` use ``True``.
    group : bool | str
        If data is from different subjects is, group should be ``True``.
        If data is in time-frequency, group should be ``'ITC'`` to show
        inter-trial coherence (power is shown by default).
    %(subject)s
    %(subjects_dir)s
    src : instance of SourceSpaces
        The volume source space for the ``stc``.
    inst : EpochsTFR | AverageTFR | None | list
        The time-frequency or data instances to use to plot topography.
        If group-level results are given (``group=True``), a list of
        instances should be provided.
    use_int : bool
        If ``True``, cast the data to integers to reduce memory use.
    show_topomap : bool
        Whether to show the sensor topomap in the GUI.
    %(tmin)s
    %(tmax)s
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    %(verbose)s

    Returns
    -------
    gui : instance of VolSourceEstimateViewer
        The graphical user interface (GUI) window.
    """
    from mne.viz.backends._utils import _init_mne_qtapp, _qt_app_exec
    from ._vol_stc import (
        VolSourceEstimateViewer,
        BASE_INT_DTYPE,
        COMPLEX_DTYPE,
        RANGE_VALUE,
    )

    _check_option("group", group, (True, False, "itc", "power"))

    app = _init_mne_qtapp()

    def itc(data):
        data = np.array(data)
        return (np.abs((data / np.abs(data)).mean(axis=0)) * (RANGE_VALUE - 1)).astype(
            BASE_INT_DTYPE
        )

    # cast to integers to lower memory usage, use custom complex data
    # type if necessary
    data = list()
    for group_stcs in stcs if group else [stcs]:
        # can be generator, compute using first stc object, just a general
        # rescaling of data, does not need to be precise
        scalar = None  # rescale per subject for better comparison
        outer_data = list()
        for inner_stcs in group_stcs if np.iterable(group_stcs) else [group_stcs]:
            inner_data = list()
            for stc in inner_stcs if np.iterable(inner_stcs) else [inner_stcs]:
                stc.crop(tmin=tmin, tmax=tmax)
                if use_int:
                    if np.iscomplexobj(stc.data) and not group:
                        if scalar is None:
                            # this is an order of magnitude approximation,
                            # if another stc is 10x larger than the first one,
                            # it will have some clipping
                            scalar = (RANGE_VALUE - 1) / stc.data.real.max() / 10
                        stc_data = np.zeros(stc.data.shape, COMPLEX_DTYPE)
                        stc_data["re"] = np.clip(
                            stc.data.real * scalar, -RANGE_VALUE, RANGE_VALUE - 1
                        )
                        stc_data["im"] = np.clip(
                            stc.data.imag * scalar, -RANGE_VALUE, RANGE_VALUE - 1
                        )
                        inner_data.append(stc_data)
                    else:
                        if group in (True, "power") and np.iscomplexobj(stc.data):
                            stc_data = (stc.data * stc.data.conj()).real
                        else:
                            stc_data = stc.data.copy()
                        if scalar is None:
                            scalar = (RANGE_VALUE - 1) / stc_data.max() / 5
                        # ignore group == 'itc' if not complex
                        use_itc = group == "itc" and np.iscomplexobj(stc.data)
                        inner_data.append(
                            stc_data
                            if use_itc
                            else np.clip(
                                stc_data * scalar, -RANGE_VALUE, RANGE_VALUE - 1
                            ).astype(BASE_INT_DTYPE)
                        )
                else:
                    inner_data.append(stc.data)
            # compute ITC here, need epochs
            if group == "itc" and np.iscomplexobj(stc.data) and freq_first:
                outer_data.append(itc(inner_data))
            else:
                outer_data.append(
                    np.mean(inner_data, axis=0).round().astype(BASE_INT_DTYPE)
                    if group and freq_first
                    else inner_data
                )

        # compute ITC here, need epochs
        if group == "itc" and np.iscomplexobj(stc.data) and not freq_first:
            data.append(itc(outer_data))
        else:
            data.append(
                np.mean(outer_data, axis=0).round().astype(BASE_INT_DTYPE)
                if group and not freq_first
                else outer_data
            )

    data = np.array(data)

    if not group:
        data = data[0]  # flatten group dimension

    if data.ndim == 4:  # scalar solution, add dimension at the end
        data = data[..., None]

    # move frequencies to penultimate
    data = data.transpose(
        (1, 2, 3, 0, 4) if freq_first and not group else (0, 2, 3, 1, 4)
    )

    # crop inst(s) to tmin and tmax
    for this_inst in inst if isinstance(inst, (list, tuple)) else [inst]:
        this_inst.crop(tmin=tmin, tmax=tmax)

    gui = VolSourceEstimateViewer(
        data,
        subject=subject,
        subjects_dir=subjects_dir,
        src=src,
        inst=inst,
        show_topomap=show_topomap,
        group=group,
        show=show,
        verbose=verbose,
    )
    if block:
        _qt_app_exec(app)
    return gui


class _GUIScraper(object):
    """Scrape GUI outputs."""

    def __repr__(self):
        return "<GUIScraper>"

    def __call__(self, block, block_vars, gallery_conf):
        from ._ieeg_locate import IntracranialElectrodeLocator
        from ._vol_stc import VolSourceEstimateViewer
        from sphinx_gallery.scrapers import figure_rst
        from qtpy import QtGui

        for gui in block_vars["example_globals"].values():
            if (
                isinstance(gui, (IntracranialElectrodeLocator, VolSourceEstimateViewer))
                and not getattr(gui, "_scraped", False)
                and gallery_conf["builder_name"] == "html"
            ):
                gui._scraped = True  # monkey-patch but it's easy enough
                img_fname = next(block_vars["image_path_iterator"])
                # TODO fix in window refactor
                window = gui if hasattr(gui, "grab") else gui._renderer._window
                # window is QWindow
                # https://doc.qt.io/qt-5/qwidget.html#grab
                pixmap = window.grab()
                if hasattr(gui, "_renderer"):  # if no renderer, no need
                    # Now the tricky part: we need to get the 3D renderer,
                    # extract the image from it, and put it in the correct
                    # place in the pixmap. The easiest way to do this is
                    # actually to save the 3D image first, then load it
                    # using QPixmap and Qt geometry.
                    plotter = gui._renderer.plotter
                    plotter.screenshot(img_fname)
                    sub_pixmap = QtGui.QPixmap(img_fname)
                    # https://doc.qt.io/qt-5/qwidget.html#mapTo
                    # https://doc.qt.io/qt-5/qpainter.html#drawPixmap-1
                    QtGui.QPainter(pixmap).drawPixmap(
                        plotter.mapTo(window, plotter.rect().topLeft()), sub_pixmap
                    )
                # https://doc.qt.io/qt-5/qpixmap.html#save
                pixmap.save(img_fname)
                try:  # for compatibility with both GUIs, will be refactored
                    gui._renderer.close()  # TODO should be triggered by close
                except Exception:
                    pass
                gui.close()
                return figure_rst([img_fname], gallery_conf["src_dir"], "GUI")
        return ""
