from __future__ import annotations

import numpy as np


def _unit_to_hz_scale(unit: str) -> float:
    unit = unit.strip().lower()
    if unit == "hz":
        return 1.0
    if unit == "khz":
        return 1e3
    if unit == "mhz":
        return 1e6
    if unit == "ghz":
        return 1e9
    return 1e9


# Length-unit conversion factors from meters → target unit.
_LENGTH_UNIT_FACTORS = {
    "m": 1.0,
    "in": 1.0 / 0.0254,
    "ft": 1.0 / 0.3048,
}


def _length_unit(name: str | None) -> tuple[str, float]:
    key = (name or "m").strip().lower()
    return (key, _LENGTH_UNIT_FACTORS[key]) if key in _LENGTH_UNIT_FACTORS else ("m", 1.0)


def _split_into_bands(indices: list[int]) -> list[list[int]]:
    """Split a sorted list of indices into contiguous runs."""
    if not indices:
        return []
    bands: list[list[int]] = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            bands.append(current)
            current = [idx]
    bands.append(current)
    return bands


def _compute_band(
    self,
    band_az_indices: list[int],
    freq_indices_sorted: list[int],
    elev_idx: int,
    pol_idx: int,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
):
    """Compute ISAR image, range, and cross-range axes for one azimuth band.

    Returns dict with keys: az_values, isar_display, x_range, y_range
    on success, or a string error message on failure.
    """
    band_az_values = self.active_dataset.azimuths[band_az_indices]
    order = np.argsort(band_az_values)
    sorted_band_indices = [band_az_indices[i] for i in order]
    az_values = band_az_values[order].astype(float)
    theta_rad = np.deg2rad(az_values)

    if not np.all(np.isfinite(theta_rad)) or np.any(np.diff(theta_rad) <= 0):
        return "Azimuth samples must be strictly increasing within a band."
    dtheta = float(np.mean(np.diff(theta_rad)))
    if dtheta <= 0.0:
        return "ISAR imaging requires increasing azimuth samples."

    rcs_slice = self.active_dataset.rcs[
        np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
    ][:, 0, :, 0]
    phase_slice = self.active_dataset.rcs_phase[
        np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
    ][:, 0, :, 0]
    if not np.any(np.isfinite(phase_slice)):
        return "ISAR imaging requires phase-aware samples; selected data has no finite rcs_phase."
    rcs_slice = np.where(np.isfinite(rcs_slice), rcs_slice, 0.0)

    win_az = self._isar_window(theta_rad.size)
    win_freq = self._isar_window(freq_hz.size)
    rcs_windowed = rcs_slice * np.outer(win_az, win_freq)

    # Double IFFT (matched sign convention): IFFT over frequency → range,
    # IFFT over azimuth → cross-range. Using fft on the second axis would
    # mirror the cross-range image left↔right.
    range_az_fft = np.fft.ifft(rcs_windowed, axis=1)
    isar_complex = np.fft.ifft(range_az_fft, axis=0)
    isar_complex = np.fft.fftshift(isar_complex, axes=(0, 1))

    magnitude = np.abs(isar_complex)
    if self._plot_scale_is_linear():
        isar_display = magnitude
    else:
        isar_display = self.active_dataset.rcs_to_dbsm(magnitude)

    c0 = 299_792_458.0
    y_range = np.fft.fftshift(np.fft.fftfreq(freq_hz.size, d=df)) * (c0 / 2.0) * unit_scale
    cross_freq = np.fft.fftshift(np.fft.fftfreq(theta_rad.size, d=dtheta))
    center_freq_hz = float(np.mean(freq_hz))
    x_range = cross_freq * (c0 / (2.0 * max(center_freq_hz, 1.0))) * unit_scale

    return {
        "az_values": az_values,
        "isar_display": isar_display,
        "x_range": x_range,
        "y_range": y_range,
    }


def render(self) -> None:
    self.last_plot_mode = "isar_image"
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    if not az_indices:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_indices = sorted(self._selected_indices(self.list_freq))
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for ISAR imaging.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return

    bands = _split_into_bands(az_indices)
    bands = [b for b in bands if len(b) >= 2]
    if not bands:
        self.status.showMessage(
            "Each azimuth band needs at least 2 contiguous samples for ISAR imaging."
        )
        return

    # Validate frequency axis (shared across all bands).
    freq_values_full = self.active_dataset.frequencies[freq_indices]
    freq_order = np.argsort(freq_values_full)
    freq_indices_sorted = [freq_indices[i] for i in freq_order]
    freq_values = freq_values_full[freq_order].astype(float)
    if np.any(np.diff(freq_values) <= 0) or not np.all(np.isfinite(freq_values)):
        self.status.showMessage(
            "Frequency samples must be finite and strictly increasing for ISAR imaging."
        )
        return

    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
    freq_hz = freq_values * _unit_to_hz_scale(freq_unit)
    df = float(np.mean(np.diff(freq_hz)))
    if df <= 0.0:
        self.status.showMessage("ISAR imaging requires increasing frequency samples.")
        return

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = _length_unit(units_combo.currentText() if units_combo else "m")

    band_results = []
    for band_az_indices in bands:
        result = _compute_band(
            self,
            band_az_indices,
            freq_indices_sorted,
            elev_idx,
            pol_idx,
            freq_hz,
            df,
            unit_scale,
        )
        if isinstance(result, str):
            self.status.showMessage(result)
            return
        band_results.append(result)

    n_bands = len(band_results)

    # Build (or rebuild) axes layout. We always rebuild on a fresh render so
    # the panel count matches the band count; "hold" is not meaningful when
    # the layout itself depends on the selection.
    self._remove_colorbar()
    self.plot_figure.clear()
    if n_bands == 1:
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        active_axes = [self.plot_ax]
    else:
        ax_array = self.plot_figure.subplots(1, n_bands, sharey=True)
        if not isinstance(ax_array, np.ndarray):
            ax_array = np.array([ax_array])
        active_axes = list(ax_array.ravel())
        self.plot_axes = active_axes
        self.plot_ax = active_axes[0]
    self._style_plot_axes()

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax

    last_mesh = None
    overall_x_min = float("inf")
    overall_x_max = float("-inf")
    overall_y_min = float("inf")
    overall_y_max = float("-inf")
    for ax, br in zip(active_axes, band_results):
        mesh = ax.pcolormesh(
            br["x_range"],
            br["y_range"],
            br["isar_display"].T,
            shading="auto",
            cmap=cmap,
            vmin=zmin if use_clamp else None,
            vmax=zmax if use_clamp else None,
        )
        last_mesh = mesh
        overall_x_min = min(overall_x_min, float(br["x_range"].min()))
        overall_x_max = max(overall_x_max, float(br["x_range"].max()))
        overall_y_min = min(overall_y_min, float(br["y_range"].min()))
        overall_y_max = max(overall_y_max, float(br["y_range"].max()))
        if n_bands > 1:
            ax.set_title(
                f"{float(br['az_values'][0]):g}°–{float(br['az_values'][-1]):g}°",
                color=self._current_plot_text(),
            )

    elev_value = self.active_dataset.elevations[elev_idx]
    pol_value = self.active_dataset.polarizations[pol_idx]
    fig_title = f"ISAR Image | Elevation {elev_value} deg | Pol {pol_value}"
    if n_bands > 1:
        self.plot_figure.suptitle(fig_title, color=self._current_plot_text())
    else:
        active_axes[0].set_title(fig_title, color=self._current_plot_text())

    for ax in active_axes:
        ax.set_xlabel(f"Cross-Range ({unit_name})")
    active_axes[0].set_ylabel(f"Range ({unit_name})")

    if self.chk_colorbar.isChecked() and last_mesh is not None:
        colorbar = self.plot_figure.colorbar(last_mesh, ax=active_axes)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label("RCS (Linear)", color=self._current_plot_text())
        else:
            colorbar.set_label("RCS (dBsm)", color=self._current_plot_text())
        colorbar.ax.tick_params(colors=self._current_plot_text())
        for label in colorbar.ax.get_yticklabels():
            label.set_color(self._current_plot_text())

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(overall_x_min)
    self.spin_plot_xmax.setValue(overall_x_max)
    self.spin_plot_ymin.setValue(overall_y_min)
    self.spin_plot_ymax.setValue(overall_y_max)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    self._apply_plot_limits()
    if n_bands == 1:
        self.status.showMessage("ISAR image updated.")
    else:
        self.status.showMessage(f"ISAR image updated ({n_bands} bands).")
