#!/usr/bin/env python3
"""Generate golden-value JSON fixtures from Python XRT.

Run once:  /Users/stevek/mamba/envs/xrt/bin/python3 validation/generate_fixtures.py

Each function writes a JSON file into validation/fixtures/ that is consumed by
both Rust golden tests (cargo test golden) and Python cross-comparison tests
(pytest validation/).
"""

import json
import cmath
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

    # Edge-adjacent energies per element (Gap 5)
    edge_energies = {
        "Si": ("K", 1839, [1829.0, 1838.0, 1840.0, 1849.0]),
        "Au": ("L3", 11919, [11909.0, 11918.0, 11920.0, 11929.0]),
        "W":  ("L3", 10207, [10197.0, 10206.0, 10208.0, 10217.0]),
        "O":  ("K", 543, [533.0, 542.0, 544.0, 553.0]),
    }

    for elem_name in ["Si", "Au", "W", "O"]:
        elem = rm.Element(elem_name, "Chantler total")
        # f0
        f0_vals = [float(elem.get_f0(np.array([q]))[0]) for q in q_over_4pi]
        # f1/f2
        e_arr = np.array(energies)
        f1f2 = elem.get_f1f2(e_arr)
        f1_vals = [float(v) for v in f1f2.real]
        f2_vals = [float(v) for v in f1f2.imag]

        test_cases = [
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

        # Edge f1/f2 test case
        if elem_name in edge_energies:
            edge_name, edge_ev, edge_pts = edge_energies[elem_name]
            e_edge = np.array(edge_pts)
            f1f2_edge = elem.get_f1f2(e_edge)
            test_cases.append({
                "id": f"{elem_name.lower()}_edge_{edge_name}",
                "edge_name": edge_name,
                "edge_ev": edge_ev,
                "energies_ev": edge_pts,
                "f1": [float(v) for v in f1f2_edge.real],
                "f2": [float(v) for v in f1f2_edge.imag],
            })

        data = _meta("scattering")
        data["element"] = elem_name
        data["table"] = "Chantler total"
        data["test_cases"] = test_cases
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

        # Edge-adjacent test cases (Gap 5)
        edge_map = {
            "Si": ("K", 1839, [1829, 1838, 1840, 1849]),
            "Au": ("L3", 11919, [11909, 11918, 11920, 11929]),
            "SiO2": ("O_K", 543, [533, 542, 544, 553]),
        }
        edge_cases = []
        if mat_name in edge_map:
            edge_name, edge_ev, edge_pts = edge_map[mat_name]
            e_edge = np.array(edge_pts, dtype=float)
            n_edge = mat.get_refractive_index(e_edge)
            mu_edge = mat.get_absorption_coefficient(e_edge)
            edge_cases.append({
                "id": f"{mat_name.lower()}_edge_{edge_name}",
                "edge_name": edge_name,
                "edge_ev": edge_ev,
                "energies_ev": edge_pts,
                "n_real": [float(v) for v in n_edge.real],
                "n_imag": [float(v) for v in n_edge.imag],
                "mu_cm_inv": [float(v) for v in mu_edge],
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
        ] + edge_cases
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

        # chi values at multiple energies (Gap 4)
        chi_cases = []
        for e_val in energies:
            tb_e = crystal.get_Bragg_angle(e_val)
            wavelength = pc.CH / e_val  # Å
            stol = math.sin(tb_e) / wavelength  # sin(θ)/λ in Å⁻¹
            result = crystal.get_F_chi(np.array([e_val]), np.array([stol]))
            # result = (F0, Fhkl, Fhkl_, chi0, chih, chih_bar)
            chi_cases.append({
                "energy_ev": e_val,
                "theta_b_rad": tb_e,
                "stol": stol,
                "chi0_re": float(result[3][0].real),
                "chi0_im": float(result[3][0].imag),
                "chih_re": float(result[4][0].real),
                "chih_im": float(result[4][0].imag),
                "chih_bar_re": float(result[5][0].real),
                "chih_bar_im": float(result[5][0].imag),
            })

        # Mini rocking curve: 5 angles around Bragg (Gap 4)
        dtheta_mini = [-20e-6, -10e-6, 0.0, 10e-6, 20e-6]
        rocking_thetas = [tb + dt for dt in dtheta_mini]
        rocking_bidn = [-math.sin(th) for th in rocking_thetas]
        rocking_rs = []
        rocking_rp = []
        for bd in rocking_bidn:
            rs_r, rp_r = crystal.get_amplitude(e10, np.array([bd]))[:2]
            rocking_rs.append({"re": float(rs_r[0].real), "im": float(rs_r[0].imag)})
            rocking_rp.append({"re": float(rp_r[0].real), "im": float(rp_r[0].imag)})

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
            {
                "id": f"{hkl_name}_chi",
                "chi_values": chi_cases,
            },
            {
                "id": f"{hkl_name}_rocking_mini",
                "energy_ev": 10000.0,
                "dtheta_rad": dtheta_mini,
                "beam_in_dot_normal": rocking_bidn,
                "rs": rocking_rs,
                "rp": rocking_rp,
            },
            {
                "id": f"{hkl_name}_darwin_width",
                "energy_ev": 10000.0,
                "b": -1.0,
                "darwin_width_s_rad": float(
                    2.0 * abs(cmath.sqrt(
                        complex(chi_cases[1]["chih_re"], chi_cases[1]["chih_im"])
                        * complex(chi_cases[1]["chih_bar_re"], chi_cases[1]["chih_bar_im"])
                    )) / math.sin(2.0 * chi_cases[1]["theta_b_rad"])
                ),
                "darwin_width_p_rad": float(
                    2.0 * abs(cmath.sqrt(
                        complex(chi_cases[1]["chih_re"], chi_cases[1]["chih_im"])
                        * complex(chi_cases[1]["chih_bar_re"], chi_cases[1]["chih_bar_im"])
                    )) * abs(math.cos(2.0 * chi_cases[1]["theta_b_rad"]))
                    / math.sin(2.0 * chi_cases[1]["theta_b_rad"])
                ),
            },
        ]

        # Darwin width from XRT get_Darwin_width
        try:
            dw_s_xrt = crystal.get_Darwin_width(10000.0, 1., 's')
            dw_p_xrt = crystal.get_Darwin_width(10000.0, 1., 'p')
            # Update the darwin_width test case with XRT reference
            for tc_item in data["test_cases"]:
                if tc_item["id"] == f"{hkl_name}_darwin_width":
                    tc_item["xrt_darwin_s_rad"] = float(dw_s_xrt)
                    tc_item["xrt_darwin_p_rad"] = float(dw_p_xrt)
                    print(f"      Darwin width: s={float(dw_s_xrt):.4e}, p={float(dw_p_xrt):.4e}")
        except Exception as e:
            print(f"      Darwin width XRT: {e}")

        # Multi-geometry cross-comparison (Laue, Bragg transmitted)
        for geom_name, geom_str, bidn_sign in [
            ("bragg_transmitted", "Bragg transmitted", -1.0),
            ("laue_reflected", "Laue reflected", 1.0),
        ]:
            try:
                cr_geom = rm.CrystalSi(hkl=hkl, geom=geom_str)
                # Set crystal thickness for Laue/transmitted cases
                cr_geom.t = 0.1  # mm — may set thickness or temperature depending on version
                tb_g = cr_geom.get_Bragg_angle(10000.0)
                bidn_g = np.array([bidn_sign * math.sin(tb_g)])
                rs_g, rp_g = cr_geom.get_amplitude(e10, bidn_g)[:2]
                if np.isfinite(rs_g[0]) and np.isfinite(rp_g[0]):
                    data["test_cases"].append({
                        "id": f"{hkl_name}_{geom_name}_10kev",
                        "energy_ev": 10000.0,
                        "geometry": geom_name,
                        "thickness_mm": 0.1,
                        "beam_in_dot_normal": float(bidn_g[0]),
                        "rs_real": float(rs_g[0].real),
                        "rs_imag": float(rs_g[0].imag),
                        "rp_real": float(rp_g[0].real),
                        "rp_imag": float(rp_g[0].imag),
                    })
                    print(f"      {geom_name}: |Rs|={abs(rs_g[0]):.4f}")
                else:
                    print(f"      {geom_name}: NaN, skipped")
            except Exception as e:
                print(f"      {geom_name}: {e}")

        _write(f"crystal_{hkl_name}.json", data)

    # Ge(111) crystal — CrystalDiamond variant
    print("    Ge(111)")
    ge = rm.CrystalDiamond(hkl=(1,1,1), d=None, elements='Ge',
                            a=5.6579, rho=5.323, t=297.15)
    ge_energies = [8000.0, 10000.0, 12000.0, 15000.0]
    ge_theta = [float(ge.get_Bragg_angle(e)) for e in ge_energies]
    ge_data = _meta("crystal_diffraction")
    ge_data["crystal"] = "ge111"
    ge_data["hkl"] = [1, 1, 1]
    ge_data["element"] = "Ge"
    ge_data["lattice_a"] = 5.6579
    ge_data["rho"] = 5.323
    ge_data["d_spacing_angstrom"] = float(ge.d)
    ge_data["test_cases"] = [{
        "id": "ge111_bragg_angle",
        "energies_ev": ge_energies,
        "theta_b_rad": ge_theta,
    }]
    _write("crystal_ge111.json", ge_data)

    # Si high-order reflections
    for hkl, hkl_name in [((3, 3, 3), "si333"), ((4, 4, 4), "si444")]:
        print(f"    {hkl_name}")
        cr = rm.CrystalSi(hkl=hkl, t=297.15)
        energies = [10000.0, 15000.0, 20000.0, 30000.0]
        theta_list = [float(cr.get_Bragg_angle(e)) for e in energies]
        hi_data = _meta("crystal_diffraction")
        hi_data["crystal"] = hkl_name
        hi_data["hkl"] = list(hkl)
        hi_data["temperature_k"] = 297.15
        hi_data["d_spacing_angstrom"] = float(cr.d)
        hi_data["test_cases"] = [{
            "id": f"{hkl_name}_bragg_angle",
            "energies_ev": energies,
            "theta_b_rad": theta_list,
        }]
        _write(f"crystal_{hkl_name}.json", hi_data)


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

    # XRT OE class independent verification (Gap 2)
    import xrt.backends.raycing.oes as oes
    xrt_oe_data = {}
    oe_configs = [
        ("flat", oes.OE(bl=None, name="flat"), {}),
        ("toroid", oes.ToroidMirror(bl=None, name="t", R=5e6, r=50.0),
         {"R": 5e6, "r": 50.0}),
        ("paraboloid_lens", oes.ParaboloidFlatLens(bl=None, name="p", focus=100.0),
         {"focus": 100.0}),
        ("blazed_grating", oes.BlazedGrating(bl=None, name="b", rho=600.0,
                                              blaze=0.02, antiblaze=0.5),
         {"rho": 600.0, "blaze": 0.02, "antiBlaze": 0.5}),
    ]
    for surf_name, oe_obj, params in oe_configs:
        pts = []
        for x in xs:
            for y in ys:
                x_arr = np.array([x])
                y_arr = np.array([y])
                z_raw = oe_obj.local_z(x_arr, y_arr)
                z_val = float(np.atleast_1d(z_raw)[0])
                n_result = oe_obj.local_n(x_arr, y_arr)
                nx_val = float(np.atleast_1d(n_result[0])[0])
                ny_val = float(np.atleast_1d(n_result[1])[0])
                nz_val = float(np.atleast_1d(n_result[2])[0])
                pts.append({
                    "x": x, "y": y,
                    "z": z_val, "nx": nx_val, "ny": ny_val, "nz": nz_val,
                })
        xrt_oe_data[surf_name] = {"params": params, "points": pts}

    data = _meta("surfaces")
    data["grid"] = {"xs": xs, "ys": ys}
    data["surfaces"] = surfaces
    data["xrt_oe"] = xrt_oe_data

    # Supplementary blazed grating points at non-boundary y values
    blazed_extra_ys = [0.0003, 0.0007, 0.0012]  # within one grating period
    blazed_extra_pts = []
    for y in blazed_extra_ys:
        z = _blazed_z(0.0, y, 600.0, 0.02, 0.5)
        nx, ny, nz = _blazed_n(0.0, y, 600.0, 0.02, 0.5)
        blazed_extra_pts.append({"x": 0.0, "y": y, "z": z, "nx": nx, "ny": ny, "nz": nz})
    data["blazed_non_boundary"] = blazed_extra_pts

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


# ── Diffraction helper ──────────────────────────────────────────────────────
def _compute_diffraction(rays, pixels):
    """Compute Kirchhoff diffraction integral (same formula as Rust diffraction.rs)."""
    n_pix = len(pixels["x"])
    n_rays = len(rays["x"])
    results = []
    for p in range(n_pix):
        es_sum = complex(0, 0)
        ep_sum = complex(0, 0)
        for r in range(n_rays):
            dx = pixels["x"][p] - rays["x"][r]
            dy = pixels["y"][p] - rays["y"][r]
            dz = pixels["z"][p] - rays["z"][r]
            path = math.sqrt(dx * dx + dy * dy + dz * dz)
            if path < 1e-30:
                continue
            ns = (rays["nx"][r] * dx + rays["ny"][r] * dy + rays["nz"][r] * dz) / path
            k = rays["energy"][r] / pc.CHBAR * 1e7  # mm⁻¹
            phase = k * path
            exp_ikr = complex(math.cos(phase), math.sin(phase))
            obliquity = rays["nl"][r] + ns
            amplitude_factor = k / (4.0 * math.pi) * obliquity / path
            u = complex(0, 1) * amplitude_factor * exp_ikr
            es = complex(rays["es_re"][r], rays["es_im"][r])
            ep = complex(rays["ep_re"][r], rays["ep_im"][r])
            es_sum += es * u
            ep_sum += ep * u
        results.append({
            "es_re": es_sum.real, "es_im": es_sum.imag,
            "ep_re": ep_sum.real, "ep_im": ep_sum.imag,
        })
    return results


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

    expected = _compute_diffraction(rays, pixels)

    data = _meta("diffraction")
    data["test_cases"] = [{
        "id": "small_deterministic",
        "rays": rays,
        "pixels": pixels,
        "expected": expected,
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
        "expected_unit_vector": True,
        "expected_dxprime_sigma_rad": 1e-4,
        "expected_dzprime_sigma_rad": 5e-5,
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

    # Roughness variation test (Gap: roughness > 0)
    roughness_tests = []
    for sigma in [0.0, 1.0, 3.0, 5.0]:
        ml_r = rm.Multilayer(
            tLayer=rm.Material(["W"], [1], rho=19.3, table="Chantler total"),
            tThickness=15.0,
            bLayer=rm.Material(["Si"], [1], rho=2.33, table="Chantler total"),
            bThickness=25.0,
            nPairs=20,
            substrate=rm.Material(["Si"], [1], rho=2.33, table="Chantler total"),
            substRoughness=sigma,
        )
        e_test = np.array([10000.0])
        st_test = np.array([0.02])
        rs_r, rp_r = ml_r.get_amplitude(e_test, st_test)[:2]
        roughness_tests.append({
            "roughness": sigma,
            "energy_ev": 10000.0,
            "sin_theta": 0.02,
            "rs_abs": float(abs(rs_r[0])),
            "rp_abs": float(abs(rp_r[0])),
        })
    data["roughness_variation"] = roughness_tests

    _write("multilayer_w_si.json", data)


# ── Domain 11: Synchrotron sources ─────────────────────────────────────────
def gen_synchrotron():
    print("[11] synchrotron sources")
    # Critical energy: Ec(keV) = 0.665 * E(GeV)^2 * B(T)
    bm_e_gev = 3.0
    bm_current = 0.3
    bm_b0 = 1.0
    bm_ec_kev = 0.665 * bm_e_gev**2 * bm_b0

    # Wiggler: same critical energy formula, K = 0.934 * B(T) * period(cm)
    wig_e_gev = 3.0
    wig_current = 0.3
    wig_k = 10.0
    wig_period = 80.0  # mm
    wig_n_periods = 10
    wig_b0 = wig_k / (0.0934 * wig_period)  # T
    wig_ec_kev = 0.665 * wig_e_gev**2 * wig_b0

    # Undulator: fundamental energy E1(eV) = 950 * E(GeV)^2 / (period(mm) * (1 + K^2/2))
    und_e_gev = 6.0
    und_current = 0.2
    und_kx = 0.0
    und_ky = 1.5
    und_period = 20.0  # mm
    und_n_periods = 100
    # E₁(eV) = 949.6 × E(GeV)² / (λ_u(cm) × (1 + K²/2))
    und_period_cm = und_period / 10.0  # mm → cm
    und_e1_ev = 949.6 * und_e_gev**2 / (und_period_cm * (1 + und_ky**2 / 2.0))

    data = _meta("synchrotron_sources")
    data["test_cases"] = [
        {
            "id": "bending_magnet",
            "electron_energy_gev": bm_e_gev,
            "beam_current": bm_current,
            "b_field": bm_b0,
            "nrays": 10000,
            "e_min": 5000.0,
            "e_max": 15000.0,
            "theta_max": 1e-3,
            "psi_max": 1e-3,
            "expected_ec_ev": bm_ec_kev * 1000,
        },
        {
            "id": "wiggler",
            "electron_energy_gev": wig_e_gev,
            "beam_current": wig_current,
            "k_param": wig_k,
            "period_mm": wig_period,
            "n_periods": wig_n_periods,
            "nrays": 10000,
            "e_min": 5000.0,
            "e_max": 50000.0,
            "theta_max": 1e-3,
            "psi_max": 1e-3,
            "expected_ec_ev": wig_ec_kev * 1000,
        },
        {
            "id": "undulator",
            "electron_energy_gev": und_e_gev,
            "beam_current": und_current,
            "kx": und_kx,
            "ky": und_ky,
            "period_mm": und_period,
            "n_periods": und_n_periods,
            "nrays": 10000,
            "e_min": und_e1_ev * 0.5,
            "e_max": und_e1_ev * 1.5,
            "theta_max": 1e-4,
            "psi_max": 1e-4,
            "expected_e1_ev": und_e1_ev,
        },
    ]
    # XRT reference: generate actual rays from Python XRT sources
    # Key fix: XRT sources need a BeamLine object (for sinAzimuth/cosAzimuth)
    import xrt.backends.raycing as raycing
    import xrt.backends.raycing.sources as rsources
    import os
    os.environ['XRT_CL'] = 'none'  # disable OpenCL (not available on Metal)

    bl = raycing.BeamLine()

    # BendingMagnet reference
    try:
        bm = rsources.BendingMagnet(
            bl=bl, name='BM',
            eE=3.0, eI=0.3, B0=1.0,
            eMin=5000.0, eMax=15000.0,
            nrays=10000,
        )
        bm_beam = bm.shine()
        bm_e = np.array(bm_beam.E)
        good = np.isfinite(bm_e) & (bm_e > 0)
        bm_mean_e = float(np.mean(bm_e[good]))
        bm_std_e = float(np.std(bm_e[good]))
        for tc in data["test_cases"]:
            if tc["id"] == "bending_magnet":
                tc["xrt_mean_energy"] = bm_mean_e
                tc["xrt_std_energy"] = bm_std_e
                print(f"    BM XRT: mean_E={bm_mean_e:.0f}, std_E={bm_std_e:.0f}")
    except Exception as e:
        print(f"    BM XRT: {e}")

    # Undulator reference (use same E range as fixture)
    try:
        und_tc = next(t for t in data["test_cases"] if t["id"] == "undulator")
        und_xrt = rsources.Undulator(
            bl=bl, name='Und',
            eE=und_e_gev, eI=und_current,
            period=und_period, n=und_n_periods, K=und_ky,
            eMin=und_tc["e_min"], eMax=und_tc["e_max"],
            nrays=5000,
            targetOpenCL=None,
        )
        und_beam = und_xrt.shine()
        und_e = np.array(und_beam.E)
        good_u = np.isfinite(und_e) & (und_e > 0)
        und_mean_e = float(np.mean(und_e[good_u]))
        for tc in data["test_cases"]:
            if tc["id"] == "undulator":
                tc["xrt_mean_energy"] = und_mean_e
                print(f"    Und XRT: mean_E={und_mean_e:.0f}")
    except Exception as e:
        print(f"    Und XRT: {e}")

    _write("synchrotron_sources.json", data)


# ── Domain 12: Parametric surfaces ─────────────────────────────────────────
def gen_parametric():
    print("[12] parametric surfaces")
    # Test round-trip: xyz → param → xyz for known surface shapes
    surfaces = {
        "elliptical": {"a": 5000.0, "b": 50.0, "y0": 0.0},
        "parabolical": {"p": 500.0, "y0": 0.0},
        "hyperbolic": {"a": 5000.0, "b": 50.0, "y0": 0.0},
    }

    data = _meta("parametric_surfaces")
    data["surfaces"] = surfaces
    _write("parametric_surfaces.json", data)


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
    gen_synchrotron()
    gen_parametric()
    print("\nDone. Run 'cargo test golden' and 'pytest validation/' to verify.")
