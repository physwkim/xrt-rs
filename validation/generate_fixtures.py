#!/usr/bin/env python3
"""Generate golden-value JSON fixtures from Python XRT.

Run once:  /Users/stevek/mamba/envs/xrt/bin/python3 validation/generate_fixtures.py

Each function writes a JSON file into validation/fixtures/ that is consumed by
both Rust golden tests (cargo test golden) and Python cross-comparison tests
(pytest validation/).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np

# ── XRT imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, "/Users/stevek/mamba/envs/xrt/lib/python3.11/site-packages")
import xrt.backends.raycing.physconsts as pc
import xrt.backends.raycing.materials as rm

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

XRT_VERSION = getattr(rm, "__version__", "1.6.2")


def _meta(domain: str) -> dict:
    return {
        "generator": "xrt-validation",
        "xrt_version": XRT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
    }


def _write(name: str, data: dict) -> None:
    path = os.path.join(FIXTURES, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


# ── Domain 1: Physical constants ─────────────────────────────────────────────
def gen_constants():
    print("[1/10] constants")
    consts = {
        "PI": pc.PI,
        "PI2": pc.PI2,
        "SIE0": pc.SIE0,
        "C": pc.C,
        "E0": pc.E0,
        "M0": pc.M0,
        "SIM0": pc.SIM0,
        "M0C2": pc.M0C2,
        "HPLANCK": pc.HPLANCK,
        "EV2ERG": pc.EV2ERG,
        "K2B": pc.K2B,
        "EMC": pc.EMC,
        "SIHPLANCK": pc.SIHPLANCK,
        "SIC": pc.SIC,
        "FINE_STR": pc.FINE_STR,
        "E2W": pc.E2W,
        "R0": pc.R0,
        "AVOGADRO": pc.AVOGADRO,
        "CH": pc.CH,
        "CHBAR": pc.CHBAR,
    }
    data = _meta("constants")
    data["constants"] = consts
    _write("constants.json", data)


# ── Domain 2: Scattering data ───────────────────────────────────────────────
def gen_scattering():
    print("[2/10] scattering")
    q_over_4pi = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    energies = [500.0, 1000.0, 5000.0, 8000.0, 10000.0, 20000.0, 50000.0]

    for elem_name in ["Si", "Au"]:
        elem = rm.Element(elem_name, "Chantler total")
        # f0
        f0_vals = [float(elem.get_f0(np.array([q]))[0]) for q in q_over_4pi]
        # f1/f2
        e_arr = np.array(energies)
        f1f2 = elem.get_f1f2(e_arr)
        f1_vals = [float(v) for v in f1f2.real]
        f2_vals = [float(v) for v in f1f2.imag]

        data = _meta("scattering")
        data["element"] = elem_name
        data["table"] = "Chantler total"
        data["test_cases"] = [
            {
                "id": f"{elem_name.lower()}_f0",
                "q_over_4pi": q_over_4pi,
                "f0": f0_vals,
            },
            {
                "id": f"{elem_name.lower()}_f1f2",
                "energies_ev": energies,
                "f1": f1_vals,
                "f2": f2_vals,
            },
        ]
        _write(f"scattering_{elem_name.lower()}.json", data)


# ── Domain 3: Material optics ───────────────────────────────────────────────
def gen_material():
    print("[3/10] material optics")
    energies = [500.0, 1000.0, 5000.0, 8000.0, 10000.0, 20000.0]
    angles = [0.001, 0.003, 0.005, 0.01, 0.05, 0.1, 0.3]

    materials = [
        ("Si", ["Si"], [1.0], 2.33),
        ("Au", ["Au"], [1.0], 19.32),
        ("SiO2", ["Si", "O"], [1.0, 2.0], 2.2),
    ]

    for mat_name, elems, quantities, rho in materials:
        mat = rm.Material(elems, quantities, rho=rho, table="Chantler total")
        e_arr = np.array(energies)

        # Refractive index
        n = mat.get_refractive_index(e_arr)
        n_real = [float(v) for v in n.real]
        n_imag = [float(v) for v in n.imag]

        # Absorption coefficient
        mu = mat.get_absorption_coefficient(e_arr)
        mu_vals = [float(v) for v in mu]

        # Fresnel amplitudes at 10 keV for all angles
        fresnel = []
        e_single = np.array([10000.0])
        for angle in angles:
            bidn = np.array([angle])
            rs, rp = mat.get_amplitude(e_single, bidn)[:2]
            fresnel.append({
                "sin_theta": angle,
                "rs_real": float(rs[0].real),
                "rs_imag": float(rs[0].imag),
                "rp_real": float(rp[0].real),
                "rp_imag": float(rp[0].imag),
            })

        data = _meta("material_optics")
        data["material"] = mat_name
        data["elements"] = elems
        data["quantities"] = quantities
        data["rho"] = rho
        data["test_cases"] = [
            {
                "id": f"{mat_name.lower()}_n",
                "energies_ev": energies,
                "n_real": n_real,
                "n_imag": n_imag,
            },
            {
                "id": f"{mat_name.lower()}_mu",
                "energies_ev": energies,
                "mu_cm_inv": mu_vals,
            },
            {
                "id": f"{mat_name.lower()}_fresnel",
                "energy_ev": 10000.0,
                "amplitudes": fresnel,
            },
        ]
        _write(f"material_{mat_name.lower()}.json", data)


# ── Domain 4: Crystal diffraction ───────────────────────────────────────────
def gen_crystal():
    print("[4/10] crystal diffraction")
    for hkl, hkl_name in [((1, 1, 1), "si111"), ((2, 2, 0), "si220")]:
        crystal = rm.CrystalSi(hkl=hkl, t=297.15)
        energies = [8000.0, 10000.0, 12000.0, 15000.0]
        e_arr = np.array(energies)

        # Bragg angle
        theta_b = [float(crystal.get_Bragg_angle(e)) for e in energies]

        # d-spacing
        d_spacing = float(crystal.d)

        # chi values at 10 keV
        e10 = np.array([10000.0])
        # For amplitude: beamInDotNormal at Bragg angle
        tb = crystal.get_Bragg_angle(10000.0)
        bidn = np.array([-math.sin(tb)])
        rs_arr, rp_arr = crystal.get_amplitude(e10, bidn)[:2]

        data = _meta("crystal_diffraction")
        data["crystal"] = hkl_name
        data["hkl"] = list(hkl)
        data["temperature_k"] = 297.15
        data["d_spacing_angstrom"] = d_spacing
        data["test_cases"] = [
            {
                "id": f"{hkl_name}_bragg_angle",
                "energies_ev": energies,
                "theta_b_rad": theta_b,
            },
            {
                "id": f"{hkl_name}_amplitude_10kev",
                "energy_ev": 10000.0,
                "beam_in_dot_normal": float(bidn[0]),
                "rs_real": float(rs_arr[0].real),
                "rs_imag": float(rs_arr[0].imag),
                "rp_real": float(rp_arr[0].real),
                "rp_imag": float(rp_arr[0].imag),
            },
        ]
        _write(f"crystal_{hkl_name}.json", data)


# ── Domain 5: Surface geometry ───────────────────────────────────────────────
# Compute surface z and n from the same mathematical formulas as the Rust code.
# This avoids depending on XRT's OE class wrapping, which has different API.

def _flat_z(x, y):
    return 0.0

def _flat_n(x, y):
    return (0.0, 0.0, 1.0)

def _toroid_z(x, y, R, r):
    z_meridional = y * y / (2.0 * R)
    arg = x / r
    arg2 = arg * arg
    z_sagittal = r * (1.0 - math.sqrt(1.0 - arg2)) if arg2 < 1.0 else r
    return z_meridional + z_sagittal

def _toroid_n(x, y, R, r):
    arg = x / r
    arg2 = arg * arg
    dz_dx = x / (r * math.sqrt(1.0 - arg2)) if arg2 < 1.0 else math.copysign(1e6, x)
    dz_dy = y / R
    nx, ny, nz = -dz_dx, -dz_dy, 1.0
    norm = math.sqrt(nx*nx + ny*ny + nz*nz)
    return (nx/norm, ny/norm, nz/norm)

def _spherical_z(x, y, R):
    r2 = R * R
    rho2 = x * x + y * y
    return R - math.sqrt(r2 - rho2) if rho2 < r2 else R

def _spherical_n(x, y, R):
    r2 = R * R
    rho2 = x * x + y * y
    if rho2 < r2:
        denom = math.sqrt(r2 - rho2)
        dz_dx = x / denom
        dz_dy = y / denom
        nx, ny, nz = -dz_dx, -dz_dy, 1.0
        norm = math.sqrt(nx*nx + ny*ny + nz*nz)
        return (nx/norm, ny/norm, nz/norm)
    return (0.0, 0.0, 1.0)

def _paraboloid_lens_z(x, y, focus):
    rho2 = x * x + y * y
    return rho2 / (4.0 * focus)

def _paraboloid_lens_n(x, y, focus):
    inv_2f = 1.0 / (2.0 * focus)
    dz_dx = x * inv_2f
    dz_dy = y * inv_2f
    nx, ny, nz = -dz_dx, -dz_dy, 1.0
    norm = math.sqrt(nx*nx + ny*ny + nz*nz)
    return (nx/norm, ny/norm, nz/norm)

def _blazed_z(x, y, rho, blaze, anti_blaze):
    d = 1.0 / rho
    y_mod = ((y % d) + d) % d
    blaze_width = d * math.tan(anti_blaze) / (math.tan(blaze) + math.tan(anti_blaze))
    if y_mod < blaze_width:
        return y_mod * math.tan(blaze)
    else:
        return (d - y_mod) * math.tan(anti_blaze)

def _blazed_n(x, y, rho, blaze, anti_blaze):
    d = 1.0 / rho
    y_mod = ((y % d) + d) % d
    blaze_width = d * math.tan(anti_blaze) / (math.tan(blaze) + math.tan(anti_blaze))
    dz_dy = math.tan(blaze) if y_mod < blaze_width else -math.tan(anti_blaze)
    ny = -dz_dy
    nz = 1.0
    norm = math.sqrt(ny*ny + nz*nz)
    return (0.0, ny/norm, nz/norm)

def _laminar_z(x, y, rho, depth, duty):
    d = 1.0 / rho
    y_mod = ((y % d) + d) % d
    return depth if y_mod < d * duty else 0.0

def _laminar_n(x, y, rho, depth, duty):
    return (0.0, 0.0, 1.0)

def gen_surfaces():
    print("[5/10] surfaces")
    xs = [-10.0, -1.0, 0.0, 1.0, 10.0]
    ys = [-100.0, -10.0, 0.0, 10.0, 100.0]

    surface_defs = [
        ("flat", {}, _flat_z, _flat_n),
        ("toroid", {"R": 5e6, "r": 50.0},
         lambda x, y: _toroid_z(x, y, 5e6, 50.0),
         lambda x, y: _toroid_n(x, y, 5e6, 50.0)),
        ("spherical", {"R": 1000.0},
         lambda x, y: _spherical_z(x, y, 1000.0),
         lambda x, y: _spherical_n(x, y, 1000.0)),
        ("paraboloid_lens", {"focus": 100.0},
         lambda x, y: _paraboloid_lens_z(x, y, 100.0),
         lambda x, y: _paraboloid_lens_n(x, y, 100.0)),
        ("blazed_grating", {"rho": 600.0, "blaze": 0.02, "antiBlaze": 0.5},
         lambda x, y: _blazed_z(x, y, 600.0, 0.02, 0.5),
         lambda x, y: _blazed_n(x, y, 600.0, 0.02, 0.5)),
        ("laminar_grating", {"rho": 600.0, "depth": 0.005, "duty": 0.5},
         lambda x, y: _laminar_z(x, y, 600.0, 0.005, 0.5),
         lambda x, y: _laminar_n(x, y, 600.0, 0.005, 0.5)),
    ]

    surfaces = []
    for surf_type, params, z_fn, n_fn in surface_defs:
        pts = []
        for x in xs:
            for y in ys:
                z = z_fn(x, y)
                nx, ny, nz = n_fn(x, y)
                pts.append({"x": x, "y": y, "z": z, "nx": nx, "ny": ny, "nz": nz})
        surfaces.append({"type": surf_type, "params": params, "points": pts})

    data = _meta("surfaces")
    data["grid"] = {"xs": xs, "ys": ys}
    data["surfaces"] = surfaces
    _write("surfaces.json", data)


# ── Domain 6: Intersection ──────────────────────────────────────────────────
def _brent_intersect(z_fn, ray, eps=1e-12, max_iter=100):
    """Find ray-surface intersection using Brent's method."""
    x0, y0, z0 = ray["x"], ray["y"], ray["z"]
    a, b, c = ray["a"], ray["b"], ray["c"]
    t1, t2 = ray["t1"], ray["t2"]

    def dz(t):
        xr = x0 + a * t
        yr = y0 + b * t
        zr = z0 + c * t
        return zr - z_fn(xr, yr)

    fa, fb = dz(t1), dz(t2)
    if fa * fb > 0:
        return None  # no sign change

    # Simple bisection (reliable)
    for _ in range(max_iter):
        tm = (t1 + t2) / 2.0
        fm = dz(tm)
        if abs(fm) < eps or (t2 - t1) < eps:
            xr = x0 + a * tm
            yr = y0 + b * tm
            zr = z0 + c * tm
            return {"t": tm, "x": xr, "y": yr, "z": zr}
        if fa * fm < 0:
            t2, fb = tm, fm
        else:
            t1, fa = tm, fm
    return None


def gen_intersection():
    print("[6/10] intersection")
    rays = []
    for i in range(20):
        x0 = (i - 10) * 0.5
        z0 = 50.0
        a = 0.0
        b = 0.001 * (i - 10)
        c_dir = -1.0
        norm = math.sqrt(a**2 + b**2 + c_dir**2)
        rays.append({
            "x": x0, "y": 0.0, "z": z0,
            "a": a / norm, "b": b / norm, "c": c_dir / norm,
            "t1": 0.1, "t2": 200.0,
        })

    surface_configs = [
        ("flat", {}, lambda x, y: 0.0),
        ("toroid", {"R": 5e6, "r": 50.0}, lambda x, y: _toroid_z(x, y, 5e6, 50.0)),
        ("spherical", {"R": 1000.0}, lambda x, y: _spherical_z(x, y, 1000.0)),
        ("paraboloid_lens", {"focus": 100.0}, lambda x, y: _paraboloid_lens_z(x, y, 100.0)),
    ]

    results = {}
    for surf_name, params, z_fn in surface_configs:
        surf_results = []
        for ray in rays:
            result = _brent_intersect(z_fn, ray)
            surf_results.append(result)
        results[surf_name] = {"params": params, "intersections": surf_results}

    data = _meta("intersection")
    data["rays"] = rays
    data["surfaces"] = results
    _write("intersection.json", data)


# ── Domain 7: Takagi-Taupin ─────────────────────────────────────────────────
def gen_tt():
    print("[7/10] Takagi-Taupin")
    crystal = rm.CrystalSi(hkl=(1, 1, 1), t=297.15)
    energy = 10000.0
    e_arr = np.array([energy])

    theta_b = crystal.get_Bragg_angle(energy)

    # Scan around Bragg angle
    dtheta_range = np.linspace(-50e-6, 50e-6, 21)
    thetas = theta_b + dtheta_range
    bidn = -np.sin(thetas)

    rs_list = []
    rp_list = []
    for bd in bidn:
        rs, rp = crystal.get_amplitude(e_arr, np.array([bd]))[:2]
        rs_list.append({"re": float(rs[0].real), "im": float(rs[0].imag)})
        rp_list.append({"re": float(rp[0].real), "im": float(rp[0].imag)})

    data = _meta("tt_rocking_curve")
    data["crystal"] = "si111"
    data["energy_ev"] = energy
    data["theta_b_rad"] = float(theta_b)
    data["test_cases"] = [{
        "id": "si111_rocking_10kev",
        "dtheta_rad": [float(d) for d in dtheta_range],
        "beam_in_dot_normal": [float(b) for b in bidn],
        "rs": rs_list,
        "rp": rp_list,
    }]
    _write("tt_si111_10kev.json", data)


# ── Domain 8: Kirchhoff diffraction ─────────────────────────────────────────
def gen_diffraction():
    print("[8/10] diffraction (reference)")
    # Small deterministic test: 5 rays → 3 pixels
    np.random.seed(42)
    n_rays = 5
    n_pix = 3
    energy = 10000.0

    rays = {
        "x": np.random.uniform(-0.1, 0.1, n_rays).tolist(),
        "y": np.random.uniform(-0.1, 0.1, n_rays).tolist(),
        "z": np.zeros(n_rays).tolist(),
        "nx": np.zeros(n_rays).tolist(),
        "ny": np.zeros(n_rays).tolist(),
        "nz": np.ones(n_rays).tolist(),
        "nl": np.ones(n_rays).tolist(),
        "es_re": np.random.uniform(0.5, 1.0, n_rays).tolist(),
        "es_im": np.random.uniform(-0.1, 0.1, n_rays).tolist(),
        "ep_re": np.random.uniform(0.5, 1.0, n_rays).tolist(),
        "ep_im": np.random.uniform(-0.1, 0.1, n_rays).tolist(),
        "energy": [energy] * n_rays,
    }

    pixels = {
        "x": np.linspace(-0.05, 0.05, n_pix).tolist(),
        "y": np.zeros(n_pix).tolist(),
        "z": [1000.0] * n_pix,
    }

    data = _meta("diffraction")
    data["test_cases"] = [{
        "id": "small_deterministic",
        "rays": rays,
        "pixels": pixels,
        "note": "Reference for cross-comparison; Rust computes from same inputs",
    }]
    _write("diffraction_ref.json", data)


# ── Domain 9: Sources ───────────────────────────────────────────────────────
def gen_sources():
    print("[9/10] sources (statistical)")
    # Just record expected distribution parameters, not ray-by-ray values
    data = _meta("sources")
    data["test_cases"] = [{
        "id": "geometric_gaussian",
        "nrays": 100000,
        "energy_ev": 10000.0,
        "dx_mm": 0.1,
        "dz_mm": 0.05,
        "dxprime_rad": 1e-4,
        "dzprime_rad": 5e-5,
        "expected_mean_x": 0.0,
        "expected_sigma_x": 0.1,
        "expected_mean_z": 0.0,
        "expected_sigma_z": 0.05,
        "tolerance_rel": 0.1,
    }]
    _write("sources.json", data)


# ── Domain 10: Multilayer ───────────────────────────────────────────────────
def gen_multilayer():
    print("[10/10] multilayer")
    energies = [8000.0, 10000.0, 12000.0]
    sin_thetas = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

    ml = rm.Multilayer(
        tLayer=rm.Material(["W"], [1], rho=19.3, table="Chantler total"),
        tThickness=15.0,
        bLayer=rm.Material(["Si"], [1], rho=2.33, table="Chantler total"),
        bThickness=25.0,
        nPairs=20,
        substrate=rm.Material(["Si"], [1], rho=2.33, table="Chantler total"),
        idThickness=3.0,
        substRoughness=3.0,
    )

    results = []
    for energy in energies:
        e_arr = np.array([energy])
        for st in sin_thetas:
            st_arr = np.array([st])
            rs, rp = ml.get_amplitude(e_arr, st_arr)[:2]
            results.append({
                "energy_ev": energy,
                "sin_theta": st,
                "rs_real": float(rs[0].real),
                "rs_imag": float(rs[0].imag),
                "rp_real": float(rp[0].real),
                "rp_imag": float(rp[0].imag),
            })

    data = _meta("multilayer")
    data["config"] = {
        "t_elem": "W", "t_rho": 19.3,
        "b_elem": "Si", "b_rho": 2.33,
        "s_elem": "Si", "s_rho": 2.33,
        "n_pairs": 20, "d_t": 15.0, "d_b": 25.0, "roughness": 3.0,
    }
    data["test_cases"] = results
    _write("multilayer_w_si.json", data)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating XRT validation fixtures (XRT {XRT_VERSION})")
    gen_constants()
    gen_scattering()
    gen_material()
    gen_crystal()
    gen_surfaces()
    gen_intersection()
    gen_tt()
    gen_diffraction()
    gen_sources()
    gen_multilayer()
    print("\nDone. Run 'cargo test golden' and 'pytest validation/' to verify.")
