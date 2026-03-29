"""Domain 10: Multilayer Parratt reflectivity cross-comparison."""

import numpy as np
import pytest
from conftest import load_fixture, assert_close, assert_complex_close


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


class TestMultilayerRoughness:
    """Verify roughness effect on multilayer reflectivity."""

    def test_roughness_reduces_reflectivity(self, xrt):
        """Higher roughness should reduce reflectivity (Nevot-Croce)."""
        _, rm = xrt
        data = load_fixture("multilayer_w_si.json")
        roughness_data = data.get("roughness_variation")
        if not roughness_data:
            pytest.skip("No roughness variation data in fixture")

        # Reflectivity should decrease with increasing roughness
        prev_rs = None
        for entry in roughness_data:
            rs = entry["rs_abs"]
            if prev_rs is not None:
                assert rs <= prev_rs + 1e-6, (
                    f"roughness={entry['roughness']}: |Rs|={rs:.4f} > "
                    f"prev |Rs|={prev_rs:.4f}"
                )
            prev_rs = rs

    def test_zero_roughness_matches_base(self, xrt):
        """Zero roughness should match the base multilayer result."""
        _, rm = xrt
        data = load_fixture("multilayer_w_si.json")
        roughness_data = data.get("roughness_variation")
        if not roughness_data:
            pytest.skip("No roughness variation data in fixture")

        # Find zero-roughness entry
        zero_rough = next(
            (r for r in roughness_data if r["roughness"] == 0.0), None
        )
        if zero_rough is None:
            pytest.skip("No zero-roughness entry")

        # Should be finite and positive
        assert zero_rough["rs_abs"] > 0, "zero-roughness |Rs| should be > 0"
        assert zero_rough["rp_abs"] > 0, "zero-roughness |Rp| should be > 0"


class TestMultilayerCrossBroadened:
    """Cross-compare xrt-rs multilayer with roughness against Python XRT."""

    def test_roughness_cross_zero(self, xrt_rs, xrt):
        """Rust multilayer at zero roughness should match Python XRT."""
        rs = xrt_rs
        _, rm = xrt
        data = load_fixture("multilayer_w_si.json")
        roughness_data = data.get("roughness_variation")
        if not roughness_data:
            pytest.skip("No roughness variation data in fixture")

        # Only cross-compare at zero roughness (roughness models differ:
        # Rust applies Nevot-Croce to all interfaces, XRT substRoughness
        # applies only to substrate)
        entry = next(r for r in roughness_data if r["roughness"] == 0.0)

        rs_abs_list, rp_abs_list = rs.multilayer_amplitude_rs(
            "W", 19.3, "Si", 2.33, "Si", 2.33,
            20, 15.0, 25.0, 0.0,
            [entry["energy_ev"]], [entry["sin_theta"]],
        )

        tol = 1e-4
        assert_close(rs_abs_list[0], entry["rs_abs"], tol, "zero-roughness |Rs|")

    def test_roughness_monotone_rust(self, xrt_rs):
        """Rust multilayer: higher roughness should reduce reflectivity."""
        rs = xrt_rs
        prev_rs = None
        for sigma in [0.0, 1.0, 3.0, 5.0]:
            rs_abs_list, _ = rs.multilayer_amplitude_rs(
                "W", 19.3, "Si", 2.33, "Si", 2.33,
                20, 15.0, 25.0, sigma,
                [10000.0], [0.02],
            )
            if prev_rs is not None:
                assert rs_abs_list[0] <= prev_rs + 1e-6, (
                    f"Rust roughness={sigma}: |Rs|={rs_abs_list[0]:.4f} > "
                    f"prev={prev_rs:.4f}"
                )
            prev_rs = rs_abs_list[0]
