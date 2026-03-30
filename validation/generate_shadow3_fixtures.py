#!/usr/bin/env python3
"""Generate shadow3 cross-comparison fixtures.

Run: /Users/stevek/mamba/envs/shadow3/bin/python3 validation/generate_shadow3_fixtures.py
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, "/Users/stevek/codes/beamalignment/shadow3")
os.chdir("/Users/stevek/codes/beamalignment/shadow3")  # shadow3 writes temp files

import Shadow

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIXTURES, exist_ok=True)


def _write(name, data):
    path = os.path.join(FIXTURES, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


def gen_flat_mirror():
    """Trace rays through a flat mirror and record before/after positions."""
    print("[shadow3] flat mirror reflection")

    beam = Shadow.Beam()
    src = Shadow.Source()
    src.NPOINT = 100
    src.FDISTR = 1  # uniform
    src.HDIV1 = 0.0001
    src.HDIV2 = 0.0001
    src.VDIV1 = 0.00005
    src.VDIV2 = 0.00005
    src.set_energy_monochromatic(10000.0)
    beam.genSource(src)

    # Flat mirror at 3 mrad grazing
    oe = Shadow.OE()
    oe.FMIRR = 5  # plane
    oe.T_SOURCE = 10000.0  # 100 m in cm (shadow uses cm)
    oe.T_IMAGE = 5000.0    # 50 m
    oe.T_INCIDENCE = 89.828  # 90 - 0.172 deg ≈ 3 mrad grazing
    oe.T_REFLECTION = 89.828
    oe.DUMMY = 1.0  # cm
    oe.F_REFLEC = 0  # no reflectivity

    beam.traceOE(oe, 1)

    # Extract good rays
    good = beam.rays[:, 9] > 0
    x = beam.rays[good, 0].tolist()
    y = beam.rays[good, 1].tolist()
    z = beam.rays[good, 2].tolist()
    xp = beam.rays[good, 3].tolist()
    yp = beam.rays[good, 4].tolist()
    zp = beam.rays[good, 5].tolist()
    energy = beam.rays[good, 10].tolist()

    data = {
        "generator": "shadow3",
        "domain": "flat_mirror_reflection",
        "n_good": int(good.sum()),
        "n_total": int(len(beam.rays)),
        "source": {
            "npoint": 100,
            "energy_ev": 10000.0,
            "hdiv_rad": 0.0001,
            "vdiv_rad": 0.00005,
        },
        "oe": {
            "surface": "flat",
            "t_source_cm": 10000.0,
            "t_image_cm": 5000.0,
            "grazing_mrad": 3.0,
        },
        "result": {
            "x": x[:20],  # first 20 rays
            "y": y[:20],
            "z": z[:20],
            "xp": xp[:20],
            "yp": yp[:20],
            "zp": zp[:20],
            "energy": energy[:20],
        },
    }
    _write("shadow3_flat_mirror.json", data)


def gen_spherical_mirror():
    """Spherical mirror focusing test."""
    print("[shadow3] spherical mirror")

    beam = Shadow.Beam()
    src = Shadow.Source()
    src.NPOINT = 200
    src.FDISTR = 3
    src.SIGDIX = 5e-5
    src.SIGDIZ = 5e-5
    src.set_energy_monochromatic(10000.0)
    beam.genSource(src)

    oe = Shadow.OE()
    oe.FMIRR = 1  # spherical
    oe.RMIRR = 500000.0  # R = 5 km in cm
    oe.T_SOURCE = 10000.0
    oe.T_IMAGE = 10000.0
    oe.T_INCIDENCE = 89.828
    oe.T_REFLECTION = 89.828
    oe.DUMMY = 1.0
    oe.F_REFLEC = 0

    beam.traceOE(oe, 1)

    good = beam.rays[:, 9] > 0
    x_std = float(np.std(beam.rays[good, 0]))
    z_std = float(np.std(beam.rays[good, 2]))
    n_good = int(good.sum())

    data = {
        "generator": "shadow3",
        "domain": "spherical_mirror",
        "n_good": n_good,
        "oe": {
            "surface": "spherical",
            "radius_cm": 500000.0,
            "grazing_mrad": 3.0,
        },
        "result": {
            "x_std_cm": x_std,
            "z_std_cm": z_std,
        },
    }
    _write("shadow3_spherical_mirror.json", data)


def gen_conic_surface():
    """Conic coefficient surface (FMIRR=10) test."""
    print("[shadow3] conic coefficient surface")

    # Create a sphere using CCC coefficients
    R = 500000.0  # 5 km in cm
    beam = Shadow.Beam()
    src = Shadow.Source()
    src.NPOINT = 100
    src.FDISTR = 3
    src.SIGDIX = 5e-5
    src.SIGDIZ = 5e-5
    src.set_energy_monochromatic(10000.0)
    beam.genSource(src)

    oe = Shadow.OE()
    oe.FMIRR = 10  # conic coefficients
    oe.CCC = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0*R, 0.0])
    oe.T_SOURCE = 10000.0
    oe.T_IMAGE = 10000.0
    oe.T_INCIDENCE = 89.828
    oe.T_REFLECTION = 89.828
    oe.DUMMY = 1.0
    oe.F_REFLEC = 0

    beam.traceOE(oe, 1)

    good = beam.rays[:, 9] > 0
    x_std = float(np.std(beam.rays[good, 0]))
    z_std = float(np.std(beam.rays[good, 2]))

    data = {
        "generator": "shadow3",
        "domain": "conic_surface",
        "oe": {
            "ccc": oe.CCC.tolist(),
            "equivalent_sphere_radius_cm": R,
        },
        "result": {
            "n_good": int(good.sum()),
            "x_std_cm": x_std,
            "z_std_cm": z_std,
        },
    }
    _write("shadow3_conic_surface.json", data)


def gen_source_comparison():
    """Compare shadow3 source statistics with known values."""
    print("[shadow3] source statistics")

    beam = Shadow.Beam()
    src = Shadow.Source()
    src.NPOINT = 50000
    src.FDISTR = 3  # Gaussian
    src.FSOUR = 3   # Gaussian spatial
    src.SIGMAX = 0.01  # 0.1 mm = 0.01 cm
    src.SIGMAZ = 0.005 # 0.05 mm = 0.005 cm
    src.SIGDIX = 1e-4
    src.SIGDIZ = 5e-5
    src.set_energy_monochromatic(10000.0)
    beam.genSource(src)

    good = beam.rays[:, 9] > 0
    rays = beam.rays[good]

    data = {
        "generator": "shadow3",
        "domain": "source_statistics",
        "source": {
            "npoint": 50000,
            "energy_ev": 10000.0,
            "sigmax_cm": 0.01,
            "sigmaz_cm": 0.005,
            "sigdix_rad": 1e-4,
            "sigdiz_rad": 5e-5,
        },
        "result": {
            "n_good": int(good.sum()),
            "mean_x_cm": float(np.mean(rays[:, 0])),
            "std_x_cm": float(np.std(rays[:, 0])),
            "mean_z_cm": float(np.mean(rays[:, 2])),
            "std_z_cm": float(np.std(rays[:, 2])),
            "mean_xp": float(np.mean(rays[:, 3])),
            "std_xp": float(np.std(rays[:, 3])),
            "mean_zp": float(np.mean(rays[:, 5])),
            "std_zp": float(np.std(rays[:, 5])),
            "mean_energy_ev": float(np.mean(rays[:, 10])),
            # Direction cosine norm check
            "mean_dir_norm": float(np.mean(np.sqrt(rays[:, 3]**2 + rays[:, 4]**2 + rays[:, 5]**2))),
        },
    }
    _write("shadow3_source.json", data)


if __name__ == "__main__":
    print("Generating shadow3 cross-comparison fixtures")
    gen_flat_mirror()
    gen_spherical_mirror()
    gen_conic_surface()
    gen_source_comparison()
    print("\nDone.")
