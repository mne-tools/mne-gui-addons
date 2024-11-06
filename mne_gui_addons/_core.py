"""Shared GUI classes and functions."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
from functools import partial

import numpy as np
from matplotlib import patheffects
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mne import read_freesurfer_lut
from mne.surface import _marching_cubes, _read_mri_surface
from mne.transforms import _frame_to_str, apply_trans
from mne.utils import (
    _check_fname,
    _import_nibabel,
    get_subjects_dir,
    logger,
    verbose,
    warn,
)
from mne.viz.backends._utils import _qt_safe_window
from mne.viz.backends.renderer import _get_renderer
from mne.viz.utils import safe_event
from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_IMG_LABELS = [["I", "P"], ["I", "L"], ["P", "L"]]
_ZOOM_STEP_SIZE = 5
_ZOOM_BORDER = 1 / 5

# 20 colors generated to be evenly spaced in a cube, worked better than
# matplotlib color cycle
_UNIQUE_COLORS = [
    (0.1, 0.42, 0.43),
    (0.9, 0.34, 0.62),
    (0.47, 0.51, 0.3),
    (0.47, 0.55, 0.99),
    (0.79, 0.68, 0.06),
    (0.34, 0.74, 0.05),
    (0.58, 0.87, 0.13),
    (0.86, 0.98, 0.4),
    (0.92, 0.91, 0.66),
    (0.77, 0.38, 0.34),
    (0.9, 0.37, 0.1),
    (0.2, 0.62, 0.9),
    (0.22, 0.65, 0.64),
    (0.14, 0.94, 0.8),
    (0.34, 0.31, 0.68),
    (0.59, 0.28, 0.74),
    (0.46, 0.19, 0.94),
    (0.37, 0.93, 0.7),
    (0.56, 0.86, 0.55),
    (0.67, 0.69, 0.44),
]
_N_COLORS = len(_UNIQUE_COLORS)
_CMAP = LinearSegmentedColormap.from_list("colors", _UNIQUE_COLORS, N=_N_COLORS)


def _get_volume_info(img):
    header = img.header
    version = header["version"]
    vol_info = dict(head=[20])
    if version == 1:
        version = f"{version}  # volume info valid"
        vol_info["valid"] = version
        vol_info["filename"] = img.get_filename()
        vol_info["volume"] = header["dims"][:3]
        vol_info["voxelsize"] = header["delta"]
        vol_info["xras"], vol_info["yras"], vol_info["zras"] = header["Mdc"]
        vol_info["cras"] = header["Pxyz_c"]
    return vol_info


@verbose
def _load_image(img, verbose=None):
    """Load data from a 3D image file (e.g. CT, MR)."""
    nib = _import_nibabel("use GUI")
    if not isinstance(img, nib.spatialimages.SpatialImage):
        logger.debug(f"Loading {img}")
        _check_fname(img, overwrite="read", must_exist=True)
        img = nib.load(img)
    # get data
    orig_data = np.array(img.dataobj).astype(np.float32)
    # reorient data to RAS
    ornt = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(img.affine)
    ).astype(int)
    ras_ornt = nib.orientations.axcodes2ornt("RAS")
    ornt_trans = nib.orientations.ornt_transform(ornt, ras_ornt)
    img_data = nib.orientations.apply_orientation(orig_data, ornt_trans)
    orig_mgh = nib.MGHImage(orig_data, img.affine)
    vox_scan_ras_t = orig_mgh.header.get_vox2ras()
    vox_mri_t = orig_mgh.header.get_vox2ras_tkr()
    aff_trans = nib.orientations.inv_ornt_aff(ornt_trans, img.shape)
    ras_vox_scan_ras_t = np.dot(vox_scan_ras_t, aff_trans)
    return (
        img_data,
        vox_mri_t,
        vox_scan_ras_t,
        ras_vox_scan_ras_t,
        _get_volume_info(orig_mgh),
    )


def _make_mpl_plot(
    width=4,
    height=4,
    dpi=300,
    tight=True,
    hide_axes=True,
    facecolor="black",
    invert=True,
):
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    if tight:
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_facecolor(facecolor)
    # clean up excess plot text, invert
    if invert:
        ax.invert_yaxis()
    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return canvas, fig


def make_label(name):
    label = QLabel(name)
    label.setAlignment(QtCore.Qt.AlignCenter)
    return label


class ComboBox(QComboBox):
    """Dropdown menu that emits a click when popped up."""

    clicked = Signal()

    def showPopup(self):
        """Override show popup method to emit click."""
        self.clicked.emit()
        super().showPopup()


class SliceBrowser(QMainWindow):
    """Navigate between slices of an MRI, CT, etc. image."""

    _xy_idx = (
        (1, 2),
        (0, 2),
        (0, 1),
    )

    @_qt_safe_window(splash="_renderer.figure.splash", window="")
    def __init__(
        self,
        base_image=None,
        subject=None,
        subjects_dir=None,
        verbose=None,
    ):
        """GUI for browsing slices of anatomical images."""
        # initialize QMainWindow class
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        atlas_ids, colors = read_freesurfer_lut()
        self._fs_lut = {atlas_id: colors[name] for name, atlas_id in atlas_ids.items()}
        self._atlas_ids = {val: key for key, val in atlas_ids.items()}

        self._verbose = verbose
        # if bad/None subject, will raise an informative error when loading MRI
        subject = os.environ.get("SUBJECT") if subject is None else subject
        subjects_dir = str(get_subjects_dir(subjects_dir, raise_error=False))
        self._subject_dir = (
            op.join(subjects_dir, subject) if subject and subjects_dir else None
        )
        self._load_image_data(base_image=base_image)

        # GUI design

        # Main plots: make one plot for each view; sagittal, coronal, axial
        self._plt_grid = QGridLayout()
        self._figs = list()
        for i in range(3):
            canvas, fig = _make_mpl_plot()
            self._plt_grid.addWidget(canvas, i // 2, i % 2)
            self._figs.append(fig)
        self._renderer = _get_renderer(
            name="Slice Browser", size=(400, 400), bgcolor="w"
        )
        self._plt_grid.addWidget(self._renderer.plotter, 1, 1)

        self._set_ras([0.0, 0.0, 0.0], update_plots=False)

        self._plot_images()

        self._configure_ui()

    def _configure_ui(self):
        toolbar = self._configure_toolbar()
        bottom_hbox = self._configure_status_bar()

        # Put everything together
        plot_ch_hbox = QHBoxLayout()
        plot_ch_hbox.addLayout(self._plt_grid)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(toolbar)
        main_vbox.addLayout(plot_ch_hbox)
        main_vbox.addLayout(bottom_hbox)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

    def _load_image_data(self, base_image=None):
        """Get image data to display and transforms to/from vox/RAS."""
        self._using_atlas = False
        if self._subject_dir is None:
            # if the recon-all is not finished or the CT is not
            # downsampled to the MRI, the MRI can not be used
            self._mr_data = self._head = self._lh = self._rh = None
            self._mr_scan_ras_ras_vox_t = None
        else:
            mr_base_fname = op.join(self._subject_dir, "mri", "{}.mgz")
            mr_fname = (
                mr_base_fname.format("brain")
                if op.isfile(mr_base_fname.format("brain"))
                else mr_base_fname.format("T1")
            )
            (
                self._mr_data,
                mr_vox_mri_t,
                mr_vox_scan_ras_t,
                mr_ras_vox_scan_ras_t,
                self._mr_vol_info,
            ) = _load_image(mr_fname)
            self._mr_scan_ras_ras_vox_t = np.linalg.inv(mr_ras_vox_scan_ras_t)

        # ready alternate base image if provided, otherwise use brain/T1
        self._base_mr_aligned = True
        if base_image is None:
            assert self._mr_data is not None
            self._base_data = self._mr_data
            self._vox_mri_t = mr_vox_mri_t
            self._vox_scan_ras_t = mr_vox_scan_ras_t
            self._ras_vox_scan_ras_t = mr_ras_vox_scan_ras_t
        else:
            (
                self._base_data,
                self._vox_mri_t,
                self._vox_scan_ras_t,
                self._ras_vox_scan_ras_t,
                self._vol_info,
            ) = _load_image(base_image)
            if self._mr_data is not None:
                if self._mr_data.shape != self._base_data.shape or not np.allclose(
                    self._vox_scan_ras_t, mr_vox_scan_ras_t, rtol=1e-6
                ):
                    self._base_mr_aligned = False

        self._mri_vox_t = np.linalg.inv(self._vox_mri_t)
        self._scan_ras_vox_t = np.linalg.inv(self._vox_scan_ras_t)
        self._scan_ras_ras_vox_t = np.linalg.inv(self._ras_vox_scan_ras_t)

        self._scan_ras_mri_t = np.dot(self._vox_mri_t, self._scan_ras_vox_t)
        self._mri_scan_ras_t = np.dot(self._vox_scan_ras_t, self._mri_vox_t)

        self._voxel_sizes = np.array(self._base_data.shape)
        self._voxel_ratios = self._voxel_sizes / self._voxel_sizes.min()

        # We need our extents to land the centers of each pixel on the voxel
        # number. This code assumes 1mm isotropic...
        img_delta = 0.5
        self._img_extents = list(
            [
                -img_delta,
                self._voxel_sizes[idx[0]] - img_delta,
                -img_delta,
                self._voxel_sizes[idx[1]] - img_delta,
            ]
            for idx in self._xy_idx
        )

        if self._subject_dir is not None:
            if op.exists(op.join(self._subject_dir, "surf", "lh.seghead")):
                self._head = _read_mri_surface(
                    op.join(self._subject_dir, "surf", "lh.seghead")
                )
                assert _frame_to_str[self._head["coord_frame"]] == "mri"
                # transform to scanner RAS
                self._head["rr"] = apply_trans(
                    self._mri_scan_ras_t, self._head["rr"] * 1000
                )
            else:
                warn(
                    "`seghead` not found, using marching cubes on base image "
                    "for head plot, use :ref:`mne.bem.make_scalp_surfaces` "
                    "to add the scalp surface instead"
                )
                self._head = None

        if self._subject_dir is not None:
            # allow ?h.pial.T1 if ?h.pial doesn't exist
            # end with '' for better file not found error
            for img in ("", ".T1", ".T2", ""):
                surf_fname = op.join(
                    self._subject_dir, "surf", "{hemi}" + f".pial{img}"
                )
                if op.isfile(surf_fname.format(hemi="lh")):
                    break
            if op.exists(surf_fname.format(hemi="lh")):
                self._lh = _read_mri_surface(surf_fname.format(hemi="lh"))
                assert _frame_to_str[self._lh["coord_frame"]] == "mri"
                # convert to scanner RAS
                self._lh["rr"] = apply_trans(
                    self._mri_scan_ras_t, self._lh["rr"] * 1000
                )
                self._rh = _read_mri_surface(surf_fname.format(hemi="rh"))
                assert _frame_to_str[self._rh["coord_frame"]] == "mri"
                # convert to scanner RAS
                self._rh["rr"] = apply_trans(
                    self._mri_scan_ras_t, self._rh["rr"] * 1000
                )
            else:
                warn(
                    "`pial` surface not found, skipping adding to 3D "
                    "plot. This indicates the Freesurfer recon-all "
                    "has not finished or has been modified and "
                    "these files have been deleted."
                )
                self._lh = self._rh = None

    def _plot_images(self):
        """Use the MRI or CT to make plots."""
        # Plot sagittal (0), coronal (1) or axial (2) view
        self._images = dict(
            base=list(), cursor_v=list(), cursor_h=list(), bounds=list()
        )
        img_min = np.nanmin(self._base_data)
        img_max = np.nanmax(self._base_data)
        text_kwargs = dict(
            fontsize="medium",
            weight="bold",
            color="#66CCEE",
            family="monospace",
            ha="center",
            va="center",
            path_effects=[
                patheffects.withStroke(linewidth=4, foreground="k", alpha=0.75)
            ],
        )
        xyz = apply_trans(self._scan_ras_vox_t, self._ras)
        for axis in range(3):
            plot_x_idx, plot_y_idx = self._xy_idx[axis]
            fig = self._figs[axis]
            ax = fig.axes[0]
            self._images["base"].append(
                ax.imshow(
                    self._base_data[
                        (slice(None),) * axis + (self._current_slice[axis],)
                    ].T,
                    cmap="gray",
                    aspect="auto",
                    zorder=1,
                    vmin=img_min,
                    vmax=img_max,
                )
            )
            img_extent = self._img_extents[axis]  # x0, x1, y0, y1
            w, h = np.diff(np.array(img_extent).reshape(2, 2), axis=1)[:, 0]
            self._images["bounds"].append(
                Rectangle(
                    img_extent[::2],
                    w,
                    h,
                    edgecolor="w",
                    facecolor="none",
                    alpha=0.25,
                    lw=0.5,
                    zorder=1.5,
                )
            )
            ax.add_patch(self._images["bounds"][-1])
            v_x = (xyz[plot_x_idx],) * 2
            v_y = img_extent[2:4]
            self._images["cursor_v"].append(
                ax.plot(v_x, v_y, color="lime", linewidth=0.5, alpha=0.5, zorder=8)[0]
            )
            h_y = (xyz[plot_y_idx],) * 2
            h_x = img_extent[0:2]
            self._images["cursor_h"].append(
                ax.plot(h_x, h_y, color="lime", linewidth=0.5, alpha=0.5, zorder=8)[0]
            )
            # label axes
            self._figs[axis].text(0.5, 0.075, _IMG_LABELS[axis][0], **text_kwargs)
            self._figs[axis].text(0.075, 0.5, _IMG_LABELS[axis][1], **text_kwargs)
            self._figs[axis].axes[0].axis(img_extent)
            self._figs[axis].canvas.mpl_connect("scroll_event", self._on_scroll)
            self._figs[axis].canvas.mpl_connect(
                "button_release_event", partial(self._on_click, axis=axis)
            )
        # add head and brain in mm (convert from m)
        if self._head is None or not self._base_mr_aligned:
            logger.debug(
                "Using marching cubes on the base image for the "
                "3D visualization panel"
            )
            # in this case, leave in voxel coordinates
            thresh = np.quantile(self._base_data, 0.95)
            if not (self._base_data < thresh).any():
                thresh = self._base_data.min()
            rr, tris = _marching_cubes(
                np.where(self._base_data <= thresh, 0, 1),
                [1],
            )[0]
            rr = apply_trans(self._ras_vox_scan_ras_t, rr)  # base image vox -> RAS
            self._mc_actor, _ = self._renderer.mesh(
                *rr.T,
                triangles=tris,
                color="gray",
                opacity=0.2,
                reset_camera=False,
                render=False,
            )
            self._head_actor = None
        else:
            self._head_actor, _ = self._renderer.mesh(
                *self._head["rr"].T,
                triangles=self._head["tris"],
                color="gray",
                opacity=0.2,
                reset_camera=False,
                render=False,
            )
            self._mc_actor = None
        if self._lh is not None and self._rh is not None and self._base_mr_aligned:
            self._lh_actor, _ = self._renderer.mesh(
                *self._lh["rr"].T,
                triangles=self._lh["tris"],
                color="white",
                opacity=0.2,
                reset_camera=False,
                render=False,
            )
            self._rh_actor, _ = self._renderer.mesh(
                *self._rh["rr"].T,
                triangles=self._rh["tris"],
                color="white",
                opacity=0.2,
                reset_camera=False,
                render=False,
            )
        else:
            self._lh_actor = self._rh_actor = None
        self._renderer.set_camera(
            azimuth=90, elevation=90, distance=300, focalpoint=tuple(self._ras)
        )
        # update plots
        self._draw()
        self._renderer._update()

    def _configure_toolbar(self, hbox=None):
        """Make a bar at the top with tools on it."""
        hbox = QHBoxLayout() if hbox is None else hbox

        help_button = QPushButton("Help")
        help_button.released.connect(self._show_help)
        hbox.addWidget(help_button)

        hbox.addStretch(6)

        self._toggle_show_selector = ComboBox()

        # add title, not selectable
        self._toggle_show_selector.addItem("Show/Hide")
        model = self._toggle_show_selector.model()
        model.itemFromIndex(model.index(0, 0)).setSelectable(False)
        # color differently
        color = QtGui.QColor("gray")
        brush = QtGui.QBrush(color)
        brush.setStyle(QtCore.Qt.SolidPattern)
        model.setData(model.index(0, 0), brush, QtCore.Qt.BackgroundRole)

        if self._base_mr_aligned and hasattr(self, "_toggle_show_brain"):
            self._toggle_show_selector.addItem("Show brain slices")
            self._toggle_show_selector.addItem("Show atlas slices")

        if hasattr(self, "_toggle_show_mip"):
            self._toggle_show_selector.addItem("Show max intensity proj")

        if hasattr(self, "_toggle_show_max"):
            self._toggle_show_selector.addItem("Show local maxima")

        if self._head_actor is not None:
            self._toggle_show_selector.addItem("Hide 3D head")

        if self._lh_actor is not None and self._rh_actor is not None:
            self._toggle_show_selector.addItem("Hide 3D brain")

        if self._mc_actor is not None:
            self._toggle_show_selector.addItem("Hide 3D rendering")

        self._toggle_show_selector.currentIndexChanged.connect(self._toggle_show)
        hbox.addWidget(self._toggle_show_selector)
        return hbox

    def _configure_status_bar(self, hbox=None):
        """Make a bar at the bottom with information in it."""
        hbox = QHBoxLayout() if hbox is None else hbox

        self._intensity_label = QLabel("")  # update later
        hbox.addWidget(self._intensity_label)

        VOX_label = QLabel("VOX =")
        self._VOX_textbox = QLineEdit("")  # update later
        self._VOX_textbox.setMaximumHeight(25)
        self._VOX_textbox.setMinimumWidth(75)
        self._VOX_textbox.focusOutEvent = self._update_VOX
        hbox.addWidget(VOX_label)
        hbox.addWidget(self._VOX_textbox)

        RAS_label = QLabel("RAS =")
        self._RAS_textbox = QLineEdit("")  # update later
        self._RAS_textbox.setMaximumHeight(25)
        self._RAS_textbox.setMinimumWidth(150)
        self._RAS_textbox.focusOutEvent = self._update_RAS
        hbox.addWidget(RAS_label)
        hbox.addWidget(self._RAS_textbox)
        self._update_moved()  # update text now
        return hbox

    def _update_camera(self, render=False):
        """Update the camera position."""
        self._renderer.set_camera(focalpoint=tuple(self._ras), distance="auto")
        if render:
            self._renderer._update()

    def _on_scroll(self, event):
        """Process mouse scroll wheel event to zoom."""
        self._zoom(np.sign(event.step), draw=True)

    def _zoom(self, sign=1, draw=False):
        """Zoom in on the image."""
        delta = _ZOOM_STEP_SIZE * sign
        for axis, fig in enumerate(self._figs):
            xcur = self._images["cursor_v"][axis].get_xdata()[0]
            ycur = self._images["cursor_h"][axis].get_ydata()[0]
            rx, ry = (self._voxel_ratios[idx] for idx in self._xy_idx[axis])
            xmin, xmax = fig.axes[0].get_xlim()
            ymin, ymax = fig.axes[0].get_ylim()

            xwidth = (xmax - xmin) / 2 - delta * rx
            ywidth = (ymax - ymin) / 2 - delta * ry
            if xwidth <= 0 or ywidth <= 0:
                return

            xmid = (xmin + xmax) / 2
            ymid = (ymin + ymax) / 2
            if sign >= 0:  # may need to shift if zooming in or clicking
                xedge = min([xmid + xwidth - xcur, xcur - xmid + xwidth])
                if xedge < 2 * xwidth * _ZOOM_BORDER:
                    xmid += np.sign(xcur - xmid) * (2 * xwidth * _ZOOM_BORDER - xedge)
                yedge = min([ymid + ywidth - ycur, ycur - ymid + ywidth])
                if yedge < 2 * ywidth * _ZOOM_BORDER:
                    ymid += np.sign(ycur - ymid) * (2 * ywidth * _ZOOM_BORDER - yedge)

            fig.axes[0].set_xlim(xmid - xwidth, xmid + xwidth)
            fig.axes[0].set_ylim(ymid - ywidth, ymid + ywidth)
            if draw:
                fig.canvas.draw()

    @Slot()
    def _update_RAS(self, event):
        """Interpret user input to the RAS textbox."""
        ras = self._convert_text(self._RAS_textbox.text(), "ras")
        if ras is not None:
            self._set_ras(ras)

    @Slot()
    def _update_VOX(self, event):
        """Interpret user input to the RAS textbox."""
        ras = self._convert_text(self._VOX_textbox.text(), "vox")
        if ras is not None:
            self._set_ras(ras)

    def _toggle_show(self):
        """Show or hide objects in the 3D rendering."""
        text = self._toggle_show_selector.currentText()
        if text == "Show/Hide":
            return
        idx = self._toggle_show_selector.currentIndex()
        show_hide, item = text.split(" ")[0], " ".join(text.split(" ")[1:])
        show_hide_opp = "Show" if show_hide == "Hide" else "Hide"
        if "slices" in item:
            # atlas shown and brain already on or brain already on and atlas shown
            if show_hide == "Show" and "mri" in self._images:
                idx2, item2 = (2, "atlas") if self._using_atlas else (1, "brain")
                self._toggle_show_selector.setItemText(idx2, f"Show {item2} slices")
                self._toggle_show_brain()
            mr_base_fname = op.join(self._subject_dir, "mri", "{}.mgz")
            if show_hide == "Show" and "atlas" in item and not self._using_atlas:
                if op.isfile(mr_base_fname.format("wmparc")):
                    self._mr_data = _load_image(mr_base_fname.format("wmparc"))[0]
                else:
                    self._mr_data = _load_image(mr_base_fname.format("aparc+aseg"))[0]
                self._using_atlas = True
            if show_hide == "Show" and "brain" in item and self._using_atlas:
                if op.isfile(mr_base_fname.format("brain")):
                    self._mr_data = _load_image(mr_base_fname.format("brain"))[0]
                else:
                    self._mr_data = _load_image(mr_base_fname.format("T1"))[0]
                self._using_atlas = False
            self._toggle_show_brain()
            self._update_moved()
        elif item == "max intensity proj":
            self._toggle_show_mip()
        elif item == "local maxima":
            self._toggle_show_max()
        else:
            actors = {
                "3D head": [self._head_actor],
                "3D brain": [self._lh_actor, self._rh_actor],
                "3D rendering": [self._mc_actor],
            }[item]
            for actor in actors:
                actor.SetVisibility(show_hide == "Show")
            self._renderer._update()
        self._toggle_show_selector.setItemText(idx, f"{show_hide_opp} {item}")
        self._toggle_show_selector.setCurrentIndex(0)  # back to title

    def _convert_text(self, text, text_kind):
        text = text.replace("\n", "")
        vals = text.split(",")
        if len(vals) != 3:
            vals = text.split(" ")  # spaces also okay as in freesurfer
        vals = [var.lstrip().rstrip() for var in vals]
        try:
            vals = np.array([float(var) for var in vals]).reshape(3)
        except Exception:
            self._update_moved()  # resets RAS label
            return
        if text_kind == "vox":
            vox = vals
            ras = apply_trans(self._vox_scan_ras_t, vox)
        else:
            assert text_kind == "ras"
            ras = vals
            vox = apply_trans(self._scan_ras_vox_t, ras)
        wrong_size = any(
            var < 0 or var > n - 1 for var, n in zip(vox, self._voxel_sizes)
        )
        if wrong_size:
            self._update_moved()  # resets RAS label
            return
        return ras

    @property
    def _ras(self):
        return self._ras_safe

    def set_RAS(self, ras):
        """Set the crosshairs to a given RAS.

        Parameters
        ----------
        ras : array-like
            The right-anterior-superior scanner RAS coordinate.
        """
        self._set_ras(ras)

    def _set_ras(self, ras, update_plots=True):
        ras = np.asarray(ras, dtype=float)
        assert ras.shape == (3,)
        msg = ", ".join(f"{x:0.2f}" for x in ras)
        logger.debug(f"Trying RAS:  ({msg}) mm")
        # clip to valid
        vox = apply_trans(self._scan_ras_vox_t, ras)
        vox = np.array(
            [np.clip(d, 0, self._voxel_sizes[ii] - 1) for ii, d in enumerate(vox)]
        )
        # transform back, make write-only
        self._ras_safe = apply_trans(self._vox_scan_ras_t, vox)
        self._ras_safe.flags["WRITEABLE"] = False
        msg = ", ".join(f"{x:0.2f}" for x in self._ras_safe)
        logger.debug(f"Setting RAS: ({msg}) mm")
        if update_plots:
            self._move_cursors_to_pos()
        self.setFocus()  # focus back to main

    def set_vox(self, vox):
        """Set the crosshairs to a given voxel coordinate.

        Parameters
        ----------
        vox : array-like
            The voxel coordinate.
        """
        self._set_ras(apply_trans(self._vox_scan_ras_t, vox))

    @property
    def _vox(self):
        return apply_trans(self._scan_ras_vox_t, self._ras)

    @property
    def _current_slice(self):
        return apply_trans(self._scan_ras_ras_vox_t, self._ras).round().astype(int)

    def _draw(self, axis=None):
        """Update the figures with a draw call."""
        for axis in range(3) if axis is None else [axis]:
            self._figs[axis].canvas.draw()

    def _update_base_images(self, axis=None, draw=False):
        """Update the base images."""
        for axis in range(3) if axis is None else [axis]:
            self._images["base"][axis].set_data(
                self._base_data[(slice(None),) * axis + (self._current_slice[axis],)].T
            )
            if draw:
                self._draw(axis)

    def _update_images(self, axis=None, draw=True):
        """Update CT and channel images when general changes happen."""
        self._update_base_images(axis=axis)
        if draw:
            self._draw(axis)

    def _move_cursors_to_pos(self):
        """Move the cursors to a position."""
        ras_vox = apply_trans(self._scan_ras_ras_vox_t, self._ras)
        for axis in range(3):
            x, y = ras_vox[list(self._xy_idx[axis])]
            self._images["cursor_v"][axis].set_xdata([x, x])
            self._images["cursor_h"][axis].set_ydata([y, y])
        self._update_images(draw=True)
        self._update_moved()

    def _show_help(self):
        """Show the help menu."""
        QMessageBox.information(
            self,
            "Help",
            "Help:\n"
            "'+'/'-': zoom\nleft/right arrow: left/right\n"
            "up/down arrow: superior/inferior\n"
            "left angle bracket/right angle bracket: anterior/posterior",
        )

    def keyPressEvent(self, event):
        """Execute functions when the user presses a key."""
        if event.key() == "escape":
            self.close()

        elif event.key() == QtCore.Qt.Key_Return:
            for widget in (self._RAS_textbox, self._VOX_textbox):
                if widget.hasFocus():
                    widget.clearFocus()
                    self.setFocus()  # removing focus calls focus out event

        elif event.text() == "h":
            self._show_help()

        elif event.text() in ("=", "+", "-"):
            self._zoom(sign=-2 * (event.text() == "-") + 1, draw=True)

        # Changing slices
        elif event.key() in (
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Comma,
            QtCore.Qt.Key_Period,
            QtCore.Qt.Key_PageUp,
            QtCore.Qt.Key_PageDown,
        ):
            ras = np.array(self._ras)
            if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                ras[2] += 2 * (event.key() == QtCore.Qt.Key_Up) - 1
            elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                ras[0] += 2 * (event.key() == QtCore.Qt.Key_Right) - 1
            else:
                ras[1] += (
                    2
                    * (
                        event.key() == QtCore.Qt.Key_PageUp
                        or event.key() == QtCore.Qt.Key_Period
                    )
                    - 1
                )
            self._set_ras(ras)

    def _on_click(self, event, axis):
        """Move to view on MRI and CT on click."""
        if event.inaxes is self._figs[axis].axes[0]:
            # Data coordinates are voxel coordinates
            pos = (event.xdata, event.ydata)
            logger.debug(f'Clicked {"XYZ"[axis]} ({axis}) axis at pos {pos}')
            xyz = apply_trans(self._scan_ras_ras_vox_t, self._ras)
            xyz[list(self._xy_idx[axis])] = pos
            logger.debug(f"Using RAS voxel  {list(xyz)}")
            ras = apply_trans(self._ras_vox_scan_ras_t, xyz)
            self._set_ras(ras)
            self._zoom(sign=0, draw=True)

    def _update_moved(self):
        """Update when cursor position changes."""
        self._RAS_textbox.setText("{:.2f}, {:.2f}, {:.2f}".format(*self._ras))
        self._VOX_textbox.setText(
            "{:3d}, {:3d}, {:3d}".format(*self._vox.round().astype(int))
        )
        intensity_text = (
            f"intensity = {self._base_data[tuple(self._current_slice)]:.2f}"
        )
        if self._using_atlas:
            vox = (
                apply_trans(self._mr_scan_ras_ras_vox_t, self._ras).round().astype(int)
            )
            label = self._atlas_ids[int(self._mr_data[tuple(vox)])]
            intensity_text += f" ({label})"
        self._intensity_label.setText(intensity_text)

    @safe_event
    def closeEvent(self, event):
        """Clean up upon closing the window."""
        try:
            self._renderer.plotter.close()
        except AttributeError:
            pass
        self.close()
