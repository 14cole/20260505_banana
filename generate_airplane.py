"""Generate a synthetic .pio file with an airplane silhouette of point
scatterers. Run this once, then load the resulting `airplane.pio` in both
SABER and GRIM to compare.

Each row below is one ideal point scatterer at (x, y) in meters with a
complex amplitude. The .pio data is the coherent sum
    S(f, θ) = Σ_p A_p · exp(-j · 2k · (xp·sinθ + yp·cosθ))
which is the textbook far-field monostatic backscatter response. Any
correct ISAR processor should reconstruct the points at exactly these
coordinates.

Output:
    airplane.pio

Sweep:
    Azimuth:   -30° to +30°, 0.2° step  (301 angles, ±2.15 m Nyquist
                                          at 10 GHz — comfortable for
                                          ±2.0 m wingtips)
    Frequency: 2 to 18 GHz, 0.025 GHz step  (641 freqs, ±3.0 m range
                                              Nyquist — comfortable for
                                              ±2.3 m fore/aft extent)
    Single elevation (0°), single polarization (HH).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grim_dataset import RcsGrid


# ─── Airplane scatterer geometry ────────────────────────────────────────
# x = cross-range (wing direction, +right wing, -left wing)
# y = range       (nose-to-tail, +nose, -tail)
# A = complex amplitude (relative; 1.0 is the strongest scatterer)
#
# Top-down silhouette of a generic single-engine fighter, 4 m long, 4 m wingspan.
SCATTERERS: list[tuple[float, float, complex]] = [
    # Nose (single dominant tip scatterer)
    (+0.00, +2.00, 1.00 + 0j),
    # Cockpit canopy edge
    (+0.00, +1.20, 0.50 + 0j),
    # Engine inlets (left/right of fuselage just behind cockpit)
    (-0.35, +0.50, 0.85 + 0j),
    (+0.35, +0.50, 0.85 + 0j),
    # Wing roots (where wings join fuselage — strong dihedral returns)
    (-0.50, +0.00, 0.75 + 0j),
    (+0.50, +0.00, 0.75 + 0j),
    # Wing leading edges (mid-span)
    (-1.20, -0.10, 0.40 + 0j),
    (+1.20, -0.10, 0.40 + 0j),
    # Wing tips (edge-diffraction returns)
    (-2.00, -0.40, 0.50 + 0j),
    (+2.00, -0.40, 0.50 + 0j),
    # Wing trailing edge mid-span
    (-1.20, -0.45, 0.30 + 0j),
    (+1.20, -0.45, 0.30 + 0j),
    # Engine exhaust (single hot scatterer near tail centerline)
    (+0.00, -1.40, 0.70 + 0j),
    # Vertical tail base + tip
    (+0.00, -1.70, 0.45 + 0j),
    (+0.00, -2.10, 0.55 + 0j),
    # Horizontal stabilizers
    (-0.80, -1.90, 0.40 + 0j),
    (+0.80, -1.90, 0.40 + 0j),
]


# ─── Sweep parameters ───────────────────────────────────────────────────
AZIMUTHS_DEG = np.arange(-30.0, 30.0 + 1e-9, 0.2)         # 301 angles
FREQS_GHZ = np.arange(2.0, 18.0 + 1e-9, 0.025)             # 641 freqs


def synthesize() -> np.ndarray:
    """Compute S(az, freq) as an (N_az, N_freq) complex64 array."""
    c0 = 299_792_458.0
    theta = np.deg2rad(AZIMUTHS_DEG).astype(np.float64)     # (N_az,)
    f_hz = (FREQS_GHZ * 1e9).astype(np.float64)              # (N_freq,)
    k = 2.0 * np.pi * f_hz / c0                              # (N_freq,)

    sin_t = np.sin(theta)[:, None]                           # (N_az, 1)
    cos_t = np.cos(theta)[:, None]                           # (N_az, 1)
    k_b = k[None, :]                                         # (1, N_freq)

    samples = np.zeros((theta.size, k.size), dtype=np.complex128)
    for x, y, amp in SCATTERERS:
        # Phase = -2k(x·sinθ + y·cosθ); amplitude is constant across (f,θ)
        phase = -2.0 * k_b * (x * sin_t + y * cos_t)
        samples += amp * np.exp(1j * phase)

    return samples.astype(np.complex64)


def main() -> None:
    samples = synthesize()
    print(
        f"Synthesized {samples.shape[0]} az × {samples.shape[1]} freq "
        f"= {samples.size:,} complex samples from "
        f"{len(SCATTERERS)} point scatterers."
    )

    azimuths = AZIMUTHS_DEG.astype(float)
    frequencies = FREQS_GHZ.astype(float)
    elevations = np.array([0.0], dtype=float)
    polarizations = np.array(["HH"], dtype=object)

    # RcsGrid expects shape (az, el, freq, pol); samples is (az, freq).
    rcs = samples[:, np.newaxis, :, np.newaxis]

    grid = RcsGrid(
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=rcs,
        rcs_domain="complex_amplitude",
        history="Synthetic airplane silhouette (generate_airplane.py)",
        units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airplane.pio")
    written = grid.save_pio(out_path)
    print(f"Wrote {written} ({os.path.getsize(written) / 1024:.0f} KB)")
    print()
    print("Expected ISAR image from a correct processor:")
    for x, y, _ in SCATTERERS:
        print(f"    bright peak at  x={x:+.2f} m,  y={y:+.2f} m")


if __name__ == "__main__":
    main()
