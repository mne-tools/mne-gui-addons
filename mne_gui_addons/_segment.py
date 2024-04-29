# -*- coding: utf-8 -*-
"""Tissue Segmentation GUI for finding making 3D volumes."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from mne.surface import _marching_cubes, write_surface
from mne.transforms import apply_trans

from ._core import SliceBrowser, _CMAP, _N_COLORS, make_label, _load_image

from qtpy import QtCore
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSlider,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSpinBox,
)


def _get_neighbors(loc, image, voxels, val, tol):
    """Find all the neighbors above a threshold near a voxel."""
    neighbors = set()
    for axis in range(len(loc)):
        for i in (-1, 1):
            next_loc = np.array(loc)
            next_loc[axis] += i
            next_loc = tuple(next_loc)
            if abs(image[next_loc] - val) / val < tol:
                neighbors.add(next_loc)
    return neighbors


def _voxel_neighbors(seed, image, tol, max_n_voxels=200):
    """Find voxels contiguous with a seed location within a tolerance

    Parameters
    ----------
    seed : tuple | ndarray
        The location in image coordinated to seed the algorithm.
    image : ndarray
        The image to search.
    tol : float
        The tolerance as a percentage.

    Returns
    -------
    voxels : set
        The set of locations including the ``seed`` voxel and
        surrounding that meet the criteria.
    """
    seed = np.array(seed).round().astype(int)
    val = image[tuple(seed)]
    voxels = neighbors = set([tuple(seed)])
    while neighbors:
        next_neighbors = set()
        for next_loc in neighbors:
            voxel_neighbors = _get_neighbors(next_loc, image, voxels, val, tol)
            # prevent looping back to already visited voxels
            voxel_neighbors = voxel_neighbors.difference(voxels)
            # add voxels not already visited to search next
            next_neighbors = next_neighbors.union(voxel_neighbors)
            # add new voxels that match the criteria to the overall set
            voxels = voxels.union(voxel_neighbors)
        neighbors = next_neighbors  # start again checking all new neighbors
        if len(voxels) > max_n_voxels:
            break
    return voxels


class VolumeSegmenter(SliceBrowser):
    """GUI for segmenting volumes e.g. tumors.

    Attributes
    ----------
    verts : ndarray
        The vertices of the marked volume.
    tris : ndarray
        The triangles connecting the vertices of the marked volume.
    """

    def __init__(
        self,
        base_image=None,
        subject=None,
        subjects_dir=None,
        show=True,
        verbose=None,
    ):
        self.verts = self.tris = None
        self._vol_coords = list()  # store list for undo
        self._alpha = 0.8
        self._vol_actor = None

        super(VolumeSegmenter, self).__init__(
            base_image=base_image,
            subject=subject,
            subjects_dir=subjects_dir,
        )

        self._vol_img = np.zeros(self._base_data.shape) * np.nan
        self._plot_vol_images()

        if show:
            self.show()

    def _configure_ui(self):
        toolbar = self._configure_toolbar()
        slider_bar = self._configure_sliders()
        status_bar = self._configure_status_bar()

        plot_layout = QHBoxLayout()
        plot_layout.addLayout(self._plt_grid)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(toolbar)
        main_vbox.addLayout(slider_bar)
        main_vbox.addLayout(plot_layout)
        main_vbox.addLayout(status_bar)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

    def _configure_sliders(self):
        """Make a bar with sliders on it."""

        def make_slider(smin, smax, sval, sfun=None):
            slider = QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(int(round(smin)))
            slider.setMaximum(int(round(smax)))
            slider.setValue(int(round(sval)))
            slider.setTracking(False)  # only update on release
            if sfun is not None:
                slider.valueChanged.connect(sfun)
            slider.keyPressEvent = self.keyPressEvent
            return slider

        slider_hbox = QHBoxLayout()

        ch_vbox = QVBoxLayout()
        ch_vbox.addWidget(make_label("alpha"))
        ch_vbox.addWidget(make_label("tolerance"))
        ch_vbox.addWidget(make_label("smooth"))
        slider_hbox.addLayout(ch_vbox)

        slider_vbox = QVBoxLayout()
        self._alpha_slider = make_slider(0, 100, self._alpha * 100, self._update_alpha)
        slider_vbox.addWidget(self._alpha_slider)
        # no callback needed, will only be used when marked
        self._tol_slider = make_slider(0, 100, 10, None)
        slider_vbox.addWidget(self._tol_slider)
        self._smooth_slider = make_slider(
            0, 100, 0, lambda x: self._plot_3d(render=True)
        )
        slider_vbox.addWidget(self._smooth_slider)

        slider_hbox.addLayout(slider_vbox)

        img_vbox = QVBoxLayout()
        img_vbox.addWidget(make_label("Image Min"))
        img_vbox.addWidget(make_label("Image Max"))
        img_vbox.addWidget(make_label("Brain Alpha"))
        slider_hbox.addLayout(img_vbox)

        img_slider_vbox = QVBoxLayout()
        img_min = int(round(np.nanmin(self._base_data)))
        img_max = int(round(np.nanmax(self._base_data)))
        self._img_min_slider = make_slider(
            img_min, img_max, img_min, self._update_img_scale
        )
        img_slider_vbox.addWidget(self._img_min_slider)
        self._img_max_slider = make_slider(
            img_min, img_max, img_max, self._update_img_scale
        )
        img_slider_vbox.addWidget(self._img_max_slider)

        self._brain_alpha_slider = make_slider(0, 100, 20, self._update_brain_alpha)
        img_slider_vbox.addWidget(self._brain_alpha_slider)

        slider_hbox.addLayout(img_slider_vbox)

        button_vbox = QVBoxLayout()

        self._undo_button = QPushButton("Undo")
        self._undo_button.setEnabled(False)
        self._undo_button.released.connect(self._undo)
        button_vbox.addWidget(self._undo_button)

        mark_button = QPushButton("Mark")
        mark_button.released.connect(self._mark)
        button_vbox.addWidget(mark_button)

        mark_all_button = QPushButton("Mark All")
        mark_all_button.released.connect(self._mark_all)
        button_vbox.addWidget(mark_all_button)

        slider_hbox.addLayout(button_vbox)

        return slider_hbox

    def _configure_status_bar(self):
        """Configure the status bar."""
        hbox = QHBoxLayout()

        self._export_button = QPushButton("Export")
        self._export_button.released.connect(self._export_surface)
        self._export_button.setEnabled(False)
        hbox.addWidget(self._export_button)

        hbox.addWidget(make_label("\t"))  # small break
        hbox.addWidget(make_label("Max # Voxels"))
        self._max_n_voxels_spin_box = QSpinBox()
        self._max_n_voxels_spin_box.setRange(0, 10000)
        self._max_n_voxels_spin_box.setValue(200)
        self._max_n_voxels_spin_box.setSingleStep(10)
        hbox.addWidget(self._max_n_voxels_spin_box)

        hbox.addWidget(make_label("\t"))  # small break
        brainmask_button = QPushButton("Add Brainmask")
        brainmask_button.released.connect(self._apply_brainmask)
        hbox.addWidget(brainmask_button)

        hbox.addStretch(1)

        super()._configure_status_bar(hbox=hbox)
        return hbox

    def _apply_brainmask(self):
        """Mask the volume using the brainmask"""
        if self._subject_dir is None or not op.isfile(
            op.join(self._subject_dir, "mri", "brainmask.mgz")
        ):
            QMessageBox.information(
                self,
                "Recon-all Not Computed",
                "The brainmask was not found, please pass the 'subject' "
                "and 'subjects_dir' arguments for a completed recon-all",
            )
            return
        QMessageBox.information(
            self,
            "Applying Brainmask",
            "Applying the brainmask, this will take ~30 seconds",
        )
        img_data, _, _, ras_vox_scan_ras_t, _ = _load_image(
            op.join(self._subject_dir, "mri", "brainmask.mgz")
        )
        idxs = np.indices(self._base_data.shape)
        idxs = np.transpose(idxs, [1, 2, 3, 0])  # (*image_data.shape, 3)
        idxs = idxs.reshape(-1, 3)  # (n_voxels, 3)
        idxs = apply_trans(self._ras_vox_scan_ras_t, idxs)  # vox -> scanner RAS
        idxs = apply_trans(
            np.linalg.inv(ras_vox_scan_ras_t), idxs
        )  # scanner RAS -> mri vox
        idxs = idxs.round().astype(int)  # round to nearest voxel
        brain = set([(x, y, z) for x, y, z in np.array(np.where(img_data > 0)).T])
        mask = np.array([tuple(idx) not in brain for idx in idxs])
        self._base_data[mask.reshape(self._base_data.shape)] = 0
        self._update_images()

    def _update_brain_alpha(self):
        """Change the alpha level of the brain."""
        alpha = self._brain_alpha_slider.value() / 100
        for actor in (self._lh_actor, self._rh_actor):
            if actor is not None:
                actor.GetProperty().SetOpacity(alpha)
        self._renderer._update()

    def _export_surface(self):
        """Export the surface to a file."""
        fname, _ = QFileDialog.getSaveFileName(self, "Export Filename")
        if not fname:
            return
        write_surface(
            fname, self.verts, self.tris, volume_info=self._vol_info, overwrite=True
        )

    def set_clim(self, vmin=None, vmax=None):
        """Set the color limits of the image.

        Parameters
        ----------
        vmin : float [0, 1]
            The minimum percentage.
        vmax : float [0, 1]
            The maximum percentage.
        """
        if vmin is not None:
            self._img_min_slider.setValue(vmin)
        if vmax is not None:
            self._img_max_slider.setValue(vmax)

    def set_smooth(self, smooth):
        """Set the smoothness of the 3D rendering of the segmented volume.

        Parameters
        ----------
        smooth : float [0, 1]
            The smoothness of the 3D rendering.
        """
        self._smooth_slider.setValue(int(round(smooth * 100)))

    def _update_img_scale(self):
        """Update base image slider values."""
        new_min = self._img_min_slider.value()
        new_max = self._img_max_slider.value()
        # handle inversions
        self._img_min_slider.setValue(min([new_min, new_max]))
        self._img_max_slider.setValue(max([new_min, new_max]))
        self._update_base_images(draw=True)

    def _update_base_images(self, axis=None, draw=False):
        """Update the CT image(s)."""
        for axis in range(3) if axis is None else [axis]:
            img_data = np.take(self._base_data, self._current_slice[axis], axis=axis).T
            img_data[img_data < self._img_min_slider.value()] = np.nan
            img_data[img_data > self._img_max_slider.value()] = np.nan
            self._images["base"][axis].set_data(img_data)
            self._images["base"][axis].set_clim(
                (self._img_min_slider.value(), self._img_max_slider.value())
            )
            if draw:
                self._draw(axis)

    def _plot_vol_images(self):
        self._images["vol"] = list()
        for axis in range(3):
            fig = self._figs[axis]
            ax = fig.axes[0]
            vol_data = np.take(self._vol_img, self._current_slice[axis], axis=axis).T
            self._images["vol"].append(
                ax.imshow(
                    vol_data,
                    aspect="auto",
                    zorder=3,
                    cmap=_CMAP,
                    alpha=self._alpha,
                    vmin=0,
                    vmax=_N_COLORS,
                )
            )

    def set_tolerance(self, tol):
        """Set the tolerance for how different than the seed to mark the volume.

        Parameters
        ----------
        tol : float [0, 1]
            The tolerance from the seed voxel.
        """
        self._tol_slider.setValue(int(round(tol * 100)))

    def set_alpha(self, alpha):
        """Change the opacity on the slice plots and 3D rendering.

        Parameters
        ----------
        alpha : float [0, 1]
            The opacity value.
        """
        self._alpha_slider.setValue(int(round(alpha * 100)))

    def _update_alpha(self):
        """Update volume plot alpha."""
        self._alpha = self._alpha_slider.value() / 100
        for axis in range(3):
            self._images["vol"][axis].set_alpha(self._alpha)
        self._draw()
        if self._vol_actor is not None:
            self._vol_actor.GetProperty().SetOpacity(self._alpha)
        self._renderer._update()
        self.setFocus()  # remove focus from 3d plotter

    def _undo(self):
        """Undo last change to voxels."""
        self._vol_coords.pop()
        if not self._vol_coords:
            self._undo_button.setEnabled(False)
            self._export_button.setEnabled(False)
        voxels = self._vol_coords[-1] if self._vol_coords else set()
        self._vol_img = np.zeros(self._base_data.shape) * np.nan
        for voxel in voxels:
            self._vol_img[voxel] = 1
        self._update_vol_images(draw=True)
        self._plot_3d(render=True)
        self.setFocus()

    def _mark(self):
        """Mark the volume with the current tolerance and location."""
        self._undo_button.setEnabled(True)
        self._export_button.setEnabled(True)
        voxels = _voxel_neighbors(
            self._vox,
            self._base_data,
            self._tol_slider.value() / 100,
            self._max_n_voxels_spin_box.value(),
        )
        if self._vol_coords:
            voxels = voxels.union(self._vol_coords[-1])
        self._vol_coords.append(voxels)
        for voxel in voxels:
            self._vol_img[voxel] = 1
        self._update_vol_images(draw=True)
        self._plot_3d(render=True)
        self.setFocus()

    def _mark_all(self):
        """Mark the volume globally with the current tolerance and location."""
        self._undo_button.setEnabled(True)
        self._export_button.setEnabled(True)
        val = self._base_data[tuple(self._vox.round().astype(int))]
        tol = self._tol_slider.value() / 100
        voxels = set(
            [
                (x, y, z)
                for x, y, z in np.array(
                    np.where(abs(self._base_data - val) / val <= tol)
                ).T
            ]
        )
        if self._vol_coords:
            voxels = voxels.union(self._vol_coords[-1])
        self._vol_coords.append(voxels)
        for voxel in voxels:
            self._vol_img[voxel] = 1
        self._update_vol_images(draw=True)
        self._plot_3d(render=True)
        self.setFocus()

    def _update_vol_images(self, axis=None, draw=False):
        """Update the volume image(s)."""
        for axis in range(3) if axis is None else [axis]:
            vol_data = np.take(self._vol_img, self._current_slice[axis], axis=axis).T
            self._images["vol"][axis].set_data(vol_data)
            if draw:
                self._draw(axis)

    def _plot_3d(self, render=False):
        """Plot the volume in 3D."""
        if self._vol_actor is not None:
            self._renderer.plotter.remove_actor(self._vol_actor, render=False)
        if self._vol_coords:
            smooth = self._smooth_slider.value() / 100
            verts, tris = _marching_cubes(self._vol_img, [1], smooth=smooth)[0]
            verts = apply_trans(self._ras_vox_scan_ras_t, verts)  # vox -> scanner RAS
            self._vol_actor = self._renderer.mesh(
                *verts.T,
                tris,
                color=_CMAP(0)[:3],
                opacity=self._alpha,
            )[0]
            self.verts = verts
            self.tris = tris
        if render:
            self._renderer._update()

    def _update_images(self, axis=None, draw=True):
        """Update images when general changes happen."""
        self._update_base_images(axis=axis)
        self._update_vol_images(axis=axis)
        if draw:
            self._draw(axis)
