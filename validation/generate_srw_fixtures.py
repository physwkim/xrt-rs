#!/usr/bin/env python3
"""Generate SRW cross-comparison fixtures.

Run: /Users/stevek/mamba/envs/xrt/bin/python3 validation/generate_srw_fixtures.py
"""

import json
import math
import os
import sys
from array import array

sys.path.insert(0, "/Users/stevek/codes/beamalignment/SRW/env/python")
from srwpy.srwlib import *

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIXTURES, exist_ok=True)


def _write(name, data):
    path = os.path.join(FIXTURES, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


def gen_undulator_spectrum():
    """Compute undulator radiation spectrum on-axis using SRW."""
    print("[SRW] undulator spectrum")

    # Undulator: period=20mm, nPer=100, K=1.5
    B_y = 1.5 / (0.934 * 2.0)  # K = 0.934 * B[T] * period[cm]
    harm1 = SRWLMagFldH(1, 'v', B_y, 0, 1)
    und = SRWLMagFldU([harm1], 0.02, 100)  # period=20mm=0.02m, nPer=100
    magFld = SRWLMagFldC([und], [0], [0], [0])

    # Electron beam: 6 GeV, 200 mA
    elBeam = SRWLPartBeam()
    elBeam.Iavg = 0.2
    elBeam.partStatMom1.gamma = 6.0 / 0.51099890221e-03
    elBeam.partStatMom1.z = -0.5 * 0.02 * 100  # start at center of undulator

    # Wavefront: on-axis spectrum
    wfr = SRWLWfr()
    wfr.allocate(500, 1, 1)  # 500 energy points, 1x1 spatial
    wfr.mesh.zStart = 30.0  # 30m observation distance
    wfr.mesh.eStart = 2000.0
    wfr.mesh.eFin = 30000.0
    wfr.mesh.xStart = 0.0
    wfr.mesh.xFin = 0.0
    wfr.mesh.yStart = 0.0
    wfr.mesh.yFin = 0.0
    wfr.partBeam = elBeam

    # Calculate SR
    arPrecPar = [1, 0.01, 0, 0, 50000, 1, 0]
    srwl.CalcElecFieldSR(wfr, 0, magFld, arPrecPar)

    # Extract on-axis spectrum (intensity vs energy)
    arI = array('f', [0] * wfr.mesh.ne)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 0, wfr.mesh.eStart, 0, 0)

    energies = [wfr.mesh.eStart + i * (wfr.mesh.eFin - wfr.mesh.eStart) / (wfr.mesh.ne - 1) for i in range(wfr.mesh.ne)]
    intensities = list(arI)

    # Find peaks
    peaks = []
    for i in range(1, len(intensities) - 1):
        if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1] and intensities[i] > max(intensities) * 0.01:
            peaks.append({"energy_ev": energies[i], "intensity": intensities[i]})

    # Fundamental energy from formula
    gamma = elBeam.partStatMom1.gamma
    K = 1.5
    e1_ev = 2 * gamma**2 * 12398.4e-10 / (0.02 * (1 + K**2/2))  # hc/lambda_u formula

    data = {
        "generator": "srw",
        "domain": "undulator_spectrum",
        "undulator": {
            "period_m": 0.02,
            "n_periods": 100,
            "K": 1.5,
            "B_y": B_y,
        },
        "electron": {
            "energy_gev": 6.0,
            "current_a": 0.2,
            "gamma": gamma,
        },
        "observation": {
            "distance_m": 30.0,
            "n_energy_points": 500,
            "e_min_ev": 2000.0,
            "e_max_ev": 30000.0,
        },
        "result": {
            "fundamental_energy_ev": e1_ev,
            "n_peaks": len(peaks),
            "peaks": peaks[:5],  # first 5 peaks
            "max_intensity": max(intensities),
            "energies_sample": energies[::50],  # every 50th point
            "intensities_sample": intensities[::50],
        },
    }
    _write("srw_undulator_spectrum.json", data)


def gen_gaussian_beam():
    """Compute Gaussian beam propagation with SRW."""
    print("[SRW] Gaussian beam propagation")

    # Gaussian beam at 10 keV
    energy_ev = 10000.0
    wavelength_m = 12398.4e-10 / energy_ev  # 1.24 Å
    sigma_x = 50e-6  # 50 µm
    sigma_y = 50e-6

    wfr = SRWLWfr()
    wfr.allocate(1, 101, 101)
    wfr.mesh.zStart = 0
    wfr.mesh.eStart = energy_ev
    wfr.mesh.eFin = energy_ev
    wfr.mesh.xStart = -0.0005  # ±0.5mm
    wfr.mesh.xFin = 0.0005
    wfr.mesh.yStart = -0.0005
    wfr.mesh.yFin = 0.0005

    # Create Gaussian beam
    # Parameters: [waist_x, waist_y, pulse_duration, photon_energy_eV,
    #              polarization, mx, my, sigT_or_flag, ...]
    GsnBm = SRWLGsnBm()
    GsnBm.x = 0
    GsnBm.y = 0
    GsnBm.z = 0
    GsnBm.xp = 0
    GsnBm.yp = 0
    GsnBm.avgPhotEn = energy_ev
    GsnBm.pulseEn = 1e-3
    GsnBm.repRate = 1
    GsnBm.polar = 1  # linear horizontal
    GsnBm.sigX = sigma_x
    GsnBm.sigY = sigma_y
    GsnBm.sigT = 1e-12
    GsnBm.mx = 0
    GsnBm.my = 0
    srwl.CalcElecFieldGaussian(wfr, GsnBm, [0])

    # Extract initial intensity
    arI0 = array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)
    srwl.CalcIntFromElecField(arI0, wfr, 6, 0, 3, energy_ev, 0, 0)

    # Propagate 10m drift
    drift = SRWLOptD(10.0)
    pp = [0, 0, 1, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
    optBL = SRWLOptC([drift], [pp])
    srwl.PropagElecField(wfr, optBL)

    # Extract propagated intensity
    arI1 = array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, energy_ev, 0, 0)

    data = {
        "generator": "srw",
        "domain": "gaussian_beam_propagation",
        "beam": {
            "energy_ev": energy_ev,
            "sigma_x_m": sigma_x,
            "sigma_y_m": sigma_y,
        },
        "propagation": {
            "distance_m": 10.0,
        },
        "result": {
            "initial_peak_intensity": max(arI0),
            "propagated_peak_intensity": max(arI1),
            "intensity_ratio": max(arI1) / max(arI0) if max(arI0) > 0 else 0,
        },
    }
    _write("srw_gaussian_beam.json", data)


def gen_thin_lens():
    """Gaussian beam through a thin lens — verify focusing."""
    print("[SRW] thin lens focusing")

    energy_ev = 10000.0
    sigma = 100e-6  # 100 µm

    wfr = SRWLWfr()
    wfr.allocate(1, 101, 101)
    wfr.mesh.zStart = 0
    wfr.mesh.eStart = energy_ev
    wfr.mesh.eFin = energy_ev
    wfr.mesh.xStart = -0.001
    wfr.mesh.xFin = 0.001
    wfr.mesh.yStart = -0.001
    wfr.mesh.yFin = 0.001

    GsnBm = SRWLGsnBm()
    GsnBm.x = 0; GsnBm.y = 0; GsnBm.z = 0
    GsnBm.xp = 0; GsnBm.yp = 0
    GsnBm.avgPhotEn = energy_ev
    GsnBm.pulseEn = 1e-3; GsnBm.repRate = 1
    GsnBm.polar = 1
    GsnBm.sigX = sigma; GsnBm.sigY = sigma
    GsnBm.sigT = 1e-12; GsnBm.mx = 0; GsnBm.my = 0
    srwl.CalcElecFieldGaussian(wfr, GsnBm, [0])

    # Thin lens f=5m + drift 5m
    lens = SRWLOptL(5.0, 5.0)
    drift = SRWLOptD(5.0)
    pp_lens = [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    pp_drift = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
    optBL = SRWLOptC([lens, drift], [pp_lens, pp_drift])
    srwl.PropagElecField(wfr, optBL)

    arI = array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, energy_ev, 0, 0)

    data = {
        "generator": "srw",
        "domain": "thin_lens_focusing",
        "beam": {"energy_ev": energy_ev, "sigma_m": sigma},
        "optics": {"focal_length_m": 5.0, "drift_m": 5.0},
        "result": {
            "peak_intensity": max(arI),
            "focused": max(arI) > 0,
        },
    }
    _write("srw_thin_lens.json", data)


if __name__ == "__main__":
    print("Generating SRW cross-comparison fixtures")
    gen_undulator_spectrum()
    gen_gaussian_beam()
    gen_thin_lens()
    print("\nDone.")
