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


_LENGTH_UNIT_FACTORS = {
    "m": 1.0,
    "in": 1.0 / 0.0254,
    "ft": 1.0 / 0.3048,
}


def _length_unit(name: str | None) -> tuple[str, float]:
    key = (name or "m").strip().lower()
    return (key, _LENGTH_UNIT_FACTORS[key]) if key in _LENGTH_UNIT_FACTORS else ("m", 1.0)


def _resample_complex_uniform(
    values: np.ndarray,
    samples: np.ndarray,
    axis: int,
    rel_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Linearly resample complex `samples` onto a uniform grid along `axis`.

    Returns (uniform_values, resampled_samples, non_uniformity).
    `non_uniformity` is the relative spread of the original spacings —
    `(max_diff - min_diff) / median_diff`. When this is below `rel_tol`,
    the inputs are returned unchanged (no work done) and the value is
    reported so callers can warn the user.

    Real and imaginary components are interpolated separately — that's
    what np.interp expects, and for ISAR it's what you want anyway since
    interpolating |z| or arg(z) introduces phase-wrap artefacts.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values, samples, 0.0
    diffs = np.diff(values)
    median_diff = float(np.median(diffs))
    if median_diff <= 0.0:
        return values, samples, 0.0
    non_uniformity = float(np.max(diffs) - np.min(diffs)) / median_diff
    if non_uniformity < rel_tol:
        return values, samples, non_uniformity

    target = np.linspace(values[0], values[-1], values.size)
    samples = np.moveaxis(samples, axis, -1)
    flat_in = samples.reshape(-1, samples.shape[-1])
    flat_out = np.empty_like(flat_in)
    for i in range(flat_in.shape[0]):
        flat_out[i, :] = (
            np.interp(target, values, flat_in[i, :].real)
            + 1j * np.interp(target, values, flat_in[i, :].imag)
        )
    out = flat_out.reshape(samples.shape)
    out = np.moveaxis(out, -1, axis)
    return target, out, non_uniformity


def _split_into_bands(indices: list[int]) -> list[list[int]]:
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


def _next_pow_two(n: int) -> int:
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def _resolve_pad(
    name: str | None,
    n_az: int,
    n_freq: int,
    *,
    algorithm: str = "decoupled fft",
    theta: np.ndarray | None = None,
    freq_hz: np.ndarray | None = None,
) -> int:
    """Resolve the cross-range pad selection into an integer N_kx target.

    For decoupled FFT, "Auto" and "None" both mean N_kx = N_az (no pad).
    For PFA, "Auto" computes the N_kx needed to give the same scene extent
    that decoupled FFT would produce on the same data; "None" leaves N_kx
    at N_az and the resulting PFA image will be cramped to the polar k-space
    bounding box.
    """
    label = (name or "Auto").strip().lower()
    if label == "match range":
        return max(n_az, n_freq)
    if label.startswith("next power"):
        return max(_next_pow_two(n_az), n_az)
    if label == "none":
        return n_az
    # "Auto" — sane default per algorithm
    if algorithm == "polar format" and theta is not None and freq_hz is not None and theta.size >= 2:
        theta_arr = np.asarray(theta, dtype=float)
        theta_sorted = np.sort(theta_arr)
        dtheta = float(np.mean(np.diff(theta_sorted)))
        f_c = float(np.mean(freq_hz))
        if dtheta > 0.0 and f_c > 0.0:
            # Match decoupled-FFT scene extent: c/(2·f_c·dθ) = 2π·N_kx/kx_span.
            # Compute kx_span from the actual sample positions so this works
            # for symmetric and one-sided apertures alike.
            c0 = 299_792_458.0
            k_max = 2.0 * np.pi * float(np.max(freq_hz)) / c0
            k_min = 2.0 * np.pi * float(np.min(freq_hz)) / c0
            sin_t = np.sin(theta_arr)
            kx_candidates = np.concatenate((2.0 * k_min * sin_t, 2.0 * k_max * sin_t))
            kx_span = float(kx_candidates.max() - kx_candidates.min())
            if kx_span > 0.0:
                decoupled_full_extent = c0 / (2.0 * f_c * dtheta)
                n_auto = int(np.ceil(decoupled_full_extent * kx_span / (2.0 * np.pi)))
                return max(min(n_auto, 8192), n_az)
    return n_az


def _pfa_polar_to_cart(
    rcs_polar: np.ndarray,
    theta: np.ndarray,
    k: np.ndarray,
    n_kx: int,
    n_ky: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage 1-D interpolation from polar (theta, k) to Cartesian (kx, ky).

    Args:
        rcs_polar: complex (N_az, N_freq) — sample at angle theta[j] and wavenumber k[i].
        theta: (N_az,) radians, ascending.
        k: (N_freq,) wavenumbers (rad/m), ascending.
        n_kx, n_ky: target Cartesian grid dimensions.

    Returns:
        cart: complex (n_ky, n_kx) on a uniform Cartesian (kx, ky) grid.
        kx_grid: (n_kx,) ascending.
        ky_grid: (n_ky,) ascending.
    """
    n_az, n_freq = rcs_polar.shape
    if n_az < 2 or n_freq < 2:
        raise ValueError("PFA needs at least 2 angles and 2 frequencies")

    # Actual bounding box of the polar arcs in Cartesian k-space.
    # Each (θ_j, f_i) sample sits at (2·k_i·sin θ_j, 2·k_i·cos θ_j); the
    # corners of the box come from f_min and f_max times the extreme sines
    # and cosines, which differs for symmetric, one-sided positive, and
    # one-sided negative apertures.
    th_max_abs = float(np.max(np.abs(theta)))
    kmax = float(k.max())
    kmin = float(k.min())
    sin_t = np.sin(theta)
    kx_candidates = np.concatenate((2.0 * kmin * sin_t, 2.0 * kmax * sin_t))
    kx_min = float(kx_candidates.min())
    kx_max = float(kx_candidates.max())
    ky_min = 2.0 * kmin * np.cos(th_max_abs)
    ky_max = 2.0 * kmax  # at theta=0, cos=1

    kx_grid = np.linspace(kx_min, kx_max, n_kx)
    ky_grid = np.linspace(ky_min, ky_max, n_ky)

    # Stage 1: For each azimuth row, resample along ky onto the common ky_grid.
    # Native ky for row j is 2·k·cos(theta[j]) (uniform, since k is uniform).
    intermediate = np.zeros((n_az, n_ky), dtype=np.complex128)
    for j in range(n_az):
        ky_native = 2.0 * k * np.cos(theta[j])
        if ky_native[0] > ky_native[-1]:
            ky_native = ky_native[::-1]
            row = rcs_polar[j, ::-1]
        else:
            row = rcs_polar[j, :]
        intermediate[j, :] = (
            np.interp(ky_grid, ky_native, row.real, left=0.0, right=0.0)
            + 1j * np.interp(ky_grid, ky_native, row.imag, left=0.0, right=0.0)
        )

    # Stage 2: For each ky_grid[q], resample along kx onto the common kx_grid.
    # Native kx at row j (after stage 1) is ky_grid[q] · tan(theta[j]).
    cart = np.zeros((n_ky, n_kx), dtype=np.complex128)
    tan_theta = np.tan(theta)
    for q in range(n_ky):
        kx_native = ky_grid[q] * tan_theta
        if kx_native[0] > kx_native[-1]:
            kx_native = kx_native[::-1]
            col = intermediate[::-1, q]
        else:
            col = intermediate[:, q]
        cart[q, :] = (
            np.interp(kx_grid, kx_native, col.real, left=0.0, right=0.0)
            + 1j * np.interp(kx_grid, kx_native, col.imag, left=0.0, right=0.0)
        )

    return cart, kx_grid, ky_grid


def _compute_band_decoupled(
    self,
    rcs_polar: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    df: float,
    n_kx: int,
    unit_scale: float,
):
    """Classical decoupled-FFT ISAR for one band, with optional cross-range pad."""
    n_az = theta.size
    n_freq = freq_hz.size

    win_az_native = self._isar_window(n_az)
    win_freq = self._isar_window(n_freq)
    rcs_windowed = rcs_polar * np.outer(win_az_native, win_freq)

    # Optional zero-pad along azimuth to oversample cross-range.
    if n_kx > n_az:
        pad_total = n_kx - n_az
        pad_lead = pad_total // 2
        pad_trail = pad_total - pad_lead
        rcs_windowed = np.pad(
            rcs_windowed, ((pad_lead, pad_trail), (0, 0)), mode="constant"
        )
    else:
        n_kx = n_az

    # Double IFFT: ifft over freq → range, ifft over az → cross-range.
    range_az = np.fft.ifft(rcs_windowed, axis=1)
    isar_complex = np.fft.ifft(range_az, axis=0)
    isar_complex = np.fft.fftshift(isar_complex, axes=(0, 1))

    c0 = 299_792_458.0

    # Native dθ from the original azimuth samples (padding only adds zeros,
    # doesn't change physical sample spacing).
    dtheta = float(np.mean(np.diff(theta)))

    y_range = np.fft.fftshift(np.fft.fftfreq(n_freq, d=df)) * (c0 / 2.0) * unit_scale
    cross_freq_grid_d = (np.arange(n_kx) - n_kx // 2) / (n_az * dtheta)
    f_c = float(np.mean(freq_hz))
    x_range = cross_freq_grid_d * (c0 / (2.0 * max(f_c, 1.0))) * unit_scale

    return isar_complex, x_range, y_range


def _compute_band_pfa(
    self,
    rcs_polar: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    n_kx: int,
    unit_scale: float,
):
    """Polar-Format Algorithm: polar→Cartesian remap, window, 2-D IFFT."""
    n_az = theta.size
    n_freq = freq_hz.size

    c0 = 299_792_458.0
    k = 2.0 * np.pi * freq_hz / c0  # (n_freq,) wavenumbers

    n_ky = n_freq
    n_kx_eff = max(n_kx, n_az)

    cart, kx_grid, ky_grid = _pfa_polar_to_cart(rcs_polar, theta, k, n_kx_eff, n_ky)

    # Window in rectangular k-space (after resampling).
    win_kx = self._isar_window(n_kx_eff)
    win_ky = self._isar_window(n_ky)
    cart_windowed = cart * np.outer(win_ky, win_kx)

    # 2-D IFFT. cart has shape (n_ky, n_kx_eff); IFFT preserves shape.
    image = np.fft.ifft2(np.fft.ifftshift(cart_windowed))
    image = np.fft.fftshift(image)

    # Transpose to (n_kx_eff, n_ky) so callers see (cross-range, range), matching
    # the shape convention used by _compute_band_decoupled.
    image = image.T

    dkx = kx_grid[1] - kx_grid[0]
    dky = ky_grid[1] - ky_grid[0]
    dx = 2.0 * np.pi / (n_kx_eff * dkx)
    dy = 2.0 * np.pi / (n_ky * dky)
    x_range = (np.arange(n_kx_eff) - n_kx_eff // 2) * dx * unit_scale
    y_range = (np.arange(n_ky) - n_ky // 2) * dy * unit_scale

    return image, x_range, y_range


def _compute_band(
    self,
    band_az_indices: list[int],
    freq_indices_sorted: list[int],
    elev_idx: int,
    pol_idx: int,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
    algorithm: str,
    pad_choice: str,
):
    band_az_values = self.active_dataset.azimuths[band_az_indices]
    order = np.argsort(band_az_values)
    sorted_band_indices = [band_az_indices[i] for i in order]
    az_values = band_az_values[order].astype(float)
    theta = np.deg2rad(az_values)

    if not np.all(np.isfinite(theta)) or np.any(np.diff(theta) <= 0):
        return "Azimuth samples must be strictly increasing within a band."

    rcs_slice = self.active_dataset.rcs[
        np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
    ][:, 0, :, 0]
    phase_slice = self.active_dataset.rcs_phase[
        np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
    ][:, 0, :, 0]
    if not np.any(np.isfinite(phase_slice)):
        return "ISAR imaging requires phase-aware samples; selected data has no finite rcs_phase."
    rcs_slice = np.where(np.isfinite(rcs_slice), rcs_slice, 0.0)

    # Resample non-uniform input onto uniform grids. Decoupled FFT *requires*
    # uniform sampling (fftfreq math); even PFA's linear interpolation gives
    # noticeably crisper edges when input is regularised first.
    az_uniform, rcs_slice, az_nonuniformity = _resample_complex_uniform(
        az_values, rcs_slice, axis=0
    )
    freq_uniform, rcs_slice, fr_nonuniformity = _resample_complex_uniform(
        freq_hz, rcs_slice, axis=1
    )
    theta = np.deg2rad(az_uniform)
    if freq_uniform.size >= 2:
        df_eff = float(np.mean(np.diff(freq_uniform)))
    else:
        df_eff = df
    az_values = az_uniform

    pad_target = _resolve_pad(
        pad_choice,
        theta.size,
        len(freq_indices_sorted),
        algorithm=algorithm,
        theta=theta,
        freq_hz=freq_uniform,
    )
    n_kx = max(pad_target, theta.size)

    if algorithm == "polar format":
        complex_image, x_range, y_range = _compute_band_pfa(
            self, rcs_slice, theta, freq_uniform, n_kx, unit_scale
        )
    else:
        complex_image, x_range, y_range = _compute_band_decoupled(
            self, rcs_slice, theta, freq_uniform, df_eff, n_kx, unit_scale
        )

    # Sanity-check the computed scene extent. If it's NaN/inf or absurdly
    # large (>10 km), something is degenerate — bail with a useful message
    # rather than feeding nonsense to the spinboxes and pcolormesh.
    if (
        not np.all(np.isfinite(x_range))
        or not np.all(np.isfinite(y_range))
        or float(np.max(np.abs(x_range))) > 1.0e4
        or float(np.max(np.abs(y_range))) > 1.0e4
    ):
        x_max_abs = float(np.max(np.abs(x_range))) if np.all(np.isfinite(x_range)) else float("inf")
        y_max_abs = float(np.max(np.abs(y_range))) if np.all(np.isfinite(y_range)) else float("inf")
        th_max_deg = float(np.rad2deg(np.max(np.abs(theta))))
        dth_deg = float(np.rad2deg(np.mean(np.diff(theta)))) if theta.size > 1 else 0.0
        f_min_ghz = float(np.min(freq_uniform)) / 1e9
        f_max_ghz = float(np.max(freq_uniform)) / 1e9
        return (
            f"ISAR ({algorithm}) produced a degenerate scene extent: "
            f"x≈±{x_max_abs:.1e}m, y≈±{y_max_abs:.1e}m. "
            f"Inputs: θ_max={th_max_deg:.4f}°, dθ={dth_deg:.6f}°, "
            f"f∈[{f_min_ghz:.3f}, {f_max_ghz:.3f}] GHz, n_kx={n_kx}. "
            f"Likely a too-narrow azimuth selection or unit mismatch."
        )

    magnitude = np.abs(complex_image)
    if self._plot_scale_is_linear():
        isar_display = magnitude
    else:
        isar_display = self.active_dataset.rcs_to_dbsm(magnitude)

    return {
        "az_values": az_values,
        "isar_display": isar_display,
        "x_range": x_range,
        "y_range": y_range,
        "az_nonuniformity": az_nonuniformity,
        "freq_nonuniformity": fr_nonuniformity,
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

    algo_combo = getattr(self, "combo_isar_algorithm", None)
    algorithm = (algo_combo.currentText() if algo_combo else "Decoupled FFT").strip().lower()

    pad_combo = getattr(self, "combo_isar_pad", None)
    pad_choice = pad_combo.currentText() if pad_combo else "Auto"

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
            algorithm,
            pad_choice,
        )
        if isinstance(result, str):
            self.status.showMessage(result)
            return
        band_results.append(result)

    n_bands = len(band_results)

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
        x_min = float(br["x_range"].min())
        x_max = float(br["x_range"].max())
        y_min = float(br["y_range"].min())
        y_max = float(br["y_range"].max())
        # imshow on a uniform grid is several times faster than pcolormesh
        # for big arrays (1601-frequency datasets feel laggy with pcolormesh).
        mesh = ax.imshow(
            br["isar_display"].T,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=zmin if use_clamp else None,
            vmax=zmax if use_clamp else None,
        )
        last_mesh = mesh
        overall_x_min = min(overall_x_min, x_min)
        overall_x_max = max(overall_x_max, x_max)
        overall_y_min = min(overall_y_min, y_min)
        overall_y_max = max(overall_y_max, y_max)
        if n_bands > 1:
            ax.set_title(
                f"{float(br['az_values'][0]):g}°–{float(br['az_values'][-1]):g}°",
                color=self._current_plot_text(),
            )

    elev_value = self.active_dataset.elevations[elev_idx]
    pol_value = self.active_dataset.polarizations[pol_idx]
    algo_label = "PFA" if algorithm == "polar format" else "Decoupled FFT"
    fig_title = f"ISAR Image | Elevation {elev_value} deg | Pol {pol_value} | {algo_label}"
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

    # Surface any resampling that happened so the user knows their input
    # wasn't on a uniform grid. The number is the relative spread of native
    # spacings ((max-min)/median); anything > ~0.001 was actually resampled.
    az_max = max(br.get("az_nonuniformity", 0.0) for br in band_results)
    fr_max = max(br.get("freq_nonuniformity", 0.0) for br in band_results)
    parts = [f"ISAR image updated ({algo_label}"]
    if n_bands > 1:
        parts.append(f", {n_bands} bands")
    parts.append(")")
    notes = []
    if az_max >= 1e-3:
        notes.append(f"resampled azimuth (Δ-spread {az_max*100:.1f}%)")
    if fr_max >= 1e-3:
        notes.append(f"resampled frequency (Δ-spread {fr_max*100:.1f}%)")
    if notes:
        parts.append(" — " + ", ".join(notes))
    self.status.showMessage("".join(parts))
