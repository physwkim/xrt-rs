"""Domain 10: Multilayer Parratt reflectivity cross-comparison."""

import numpy as np
import pytest
from conftest import load_fixture, assert_complex_close


class TestMultilayerGolden:
    """Verify Python XRT multilayer against fixtures."""

    def test_w_si_multilayer(self, xrt):
        _, rm = xrt
        data = load_fixture("multilayer_w_si.json")
        cfg = data["config"]

        ml = rm.Multilayer(
            tLayer=rm.Material([cfg["t_elem"]], [1], rho=cfg["t_rho"],
                               table="Chantler total"),
            bLayer=rm.Material([cfg["b_elem"]], [1], rho=cfg["b_rho"],
                               table="Chantler total"),
            substrate=rm.Material([cfg["s_elem"]], [1], rho=cfg["s_rho"],
                                  table="Chantler total"),
            nPairs=cfg["n_pairs"],
            tThickness=cfg["d_t"], bThickness=cfg["d_b"],
            idThickness=cfg["roughness"],
            substRoughness=cfg["roughness"],
        )

        for tc in data["test_cases"]:
            e_arr = np.array([tc["energy_ev"]])
            st_arr = np.array([tc["sin_theta"]])
            rs, rp = ml.get_amplitude(e_arr, st_arr)[:2]
            label = f"E={tc['energy_ev']},st={tc['sin_theta']}"
            assert_complex_close(
                float(rs[0].real), float(rs[0].imag),
                tc["rs_real"], tc["rs_imag"],
                1e-8, f"rs({label})")
            assert_complex_close(
                float(rp[0].real), float(rp[0].imag),
                tc["rp_real"], tc["rp_imag"],
                1e-8, f"rp({label})")


class TestMultilayerCross:
    """Cross-compare Python XRT and xrt-rs multilayer."""

    def test_w_si_cross(self, xrt, xrt_rs):
        _, rm = xrt
        rs_mod = xrt_rs
        data = load_fixture("multilayer_w_si.json")
        cfg = data["config"]

        # Python
        ml = rm.Multilayer(
            tLayer=rm.Material([cfg["t_elem"]], [1], rho=cfg["t_rho"],
                               table="Chantler total"),
            bLayer=rm.Material([cfg["b_elem"]], [1], rho=cfg["b_rho"],
                               table="Chantler total"),
            substrate=rm.Material([cfg["s_elem"]], [1], rho=cfg["s_rho"],
                                  table="Chantler total"),
            nPairs=cfg["n_pairs"],
            tThickness=cfg["d_t"], bThickness=cfg["d_b"],
            idThickness=cfg["roughness"],
            substRoughness=cfg["roughness"],
        )

        for tc in data["test_cases"]:
            e_arr = np.array([tc["energy_ev"]])
            st_arr = np.array([tc["sin_theta"]])

            # Python
            py_rs, py_rp = ml.get_amplitude(e_arr, st_arr)[:2]

            # Rust
            rs_abs, rp_abs = rs_mod.multilayer_amplitude_rs(
                cfg["t_elem"], cfg["t_rho"],
                cfg["b_elem"], cfg["b_rho"],
                cfg["s_elem"], cfg["s_rho"],
                cfg["n_pairs"], cfg["d_t"], cfg["d_b"],
                cfg["roughness"],
                [tc["energy_ev"]], [tc["sin_theta"]],
            )

            py_rs_abs = abs(py_rs[0])
            label = f"E={tc['energy_ev']},st={tc['sin_theta']}"
            # Compare magnitudes (Rust binding returns abs values)
            diff = abs(py_rs_abs - rs_abs[0])
            assert diff < 1e-8, (
                f"|rs| mismatch at {label}: py={py_rs_abs:.10e}, "
                f"rs={rs_abs[0]:.10e}, diff={diff:.2e}"
            )
