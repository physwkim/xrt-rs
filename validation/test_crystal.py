"""Domain 4: Crystal diffraction — Bragg angle, chi, amplitude."""

import math

import numpy as np
import pytest
from conftest import load_fixture, assert_close, assert_complex_close


class TestCrystalGolden:
    """Verify Python XRT crystal values match fixtures."""

    @pytest.mark.parametrize("crystal", ["si111", "si220"])
    def test_bragg_angle(self, xrt, crystal):
        _, rm = xrt
        data = load_fixture(f"crystal_{crystal}.json")
        tc = next(t for t in data["test_cases"] if "bragg_angle" in t["id"])

        hkl = tuple(data["hkl"])
        cr = rm.CrystalSi(hkl=hkl, t=data["temperature_k"])

        for energy, expected in zip(tc["energies_ev"], tc["theta_b_rad"]):
            actual = float(cr.get_Bragg_angle(energy))
            assert_close(actual, expected, 1e-12,
                         f"theta_B(E={energy})")

    @pytest.mark.parametrize("crystal", ["si111", "si220"])
    def test_amplitude(self, xrt, crystal):
        _, rm = xrt
        data = load_fixture(f"crystal_{crystal}.json")
        tc = next(t for t in data["test_cases"] if "amplitude" in t["id"])

        hkl = tuple(data["hkl"])
        cr = rm.CrystalSi(hkl=hkl, t=data["temperature_k"])

        e_arr = np.array([tc["energy_ev"]])
        bidn = np.array([tc["beam_in_dot_normal"]])
        rs, rp = cr.get_amplitude(e_arr, bidn)[:2]

        assert_complex_close(
            float(rs[0].real), float(rs[0].imag),
            tc["rs_real"], tc["rs_imag"],
            1e-6, "rs")
        assert_complex_close(
            float(rp[0].real), float(rp[0].imag),
            tc["rp_real"], tc["rp_imag"],
            1e-6, "rp")


class TestCrystalChi:
    """Verify Python XRT crystal chi values match fixtures."""

    @pytest.mark.parametrize("crystal", ["si111", "si220"])
    def test_chi(self, xrt, crystal):
        _, rm = xrt
        import xrt.backends.raycing.physconsts as pc_mod
        data = load_fixture(f"crystal_{crystal}.json")
        tc = next(t for t in data["test_cases"] if t["id"].endswith("_chi"))

        hkl = tuple(data["hkl"])
        cr = rm.CrystalSi(hkl=hkl, t=data["temperature_k"])

        tol = 1e-10
        for chi_val in tc["chi_values"]:
            energy = chi_val["energy_ev"]
            stol = chi_val["stol"]
            result = cr.get_F_chi(np.array([energy]), np.array([stol]))
            # result = (F0, Fhkl, Fhkl_, chi0, chih, chih_bar)
            assert_complex_close(
                float(result[3][0].real), float(result[3][0].imag),
                chi_val["chi0_re"], chi_val["chi0_im"],
                tol, f"chi0(E={energy})")
            assert_complex_close(
                float(result[4][0].real), float(result[4][0].imag),
                chi_val["chih_re"], chi_val["chih_im"],
                tol, f"chih(E={energy})")
            assert_complex_close(
                float(result[5][0].real), float(result[5][0].imag),
                chi_val["chih_bar_re"], chi_val["chih_bar_im"],
                tol, f"chih_bar(E={energy})")


class TestCrystalDarwinWidth:
    """Verify Darwin width analytical formula consistency."""

    @pytest.mark.parametrize("crystal", ["si111", "si220"])
    def test_darwin_width_s_vs_p(self, xrt, crystal):
        _, rm = xrt
        data = load_fixture(f"crystal_{crystal}.json")
        tc = next(t for t in data["test_cases"] if "darwin_width" in t["id"])

        # S-polarization width should be >= P-polarization width
        assert tc["darwin_width_s_rad"] >= tc["darwin_width_p_rad"], (
            f"dw_s ({tc['darwin_width_s_rad']:.6e}) < dw_p ({tc['darwin_width_p_rad']:.6e})"
        )

        # Both should be positive and in reasonable range (1-100 µrad for Si at 10 keV)
        assert 1e-7 < tc["darwin_width_s_rad"] < 1e-4, (
            f"dw_s = {tc['darwin_width_s_rad']:.6e} outside expected range"
        )
        assert 1e-7 < tc["darwin_width_p_rad"] < 1e-4, (
            f"dw_p = {tc['darwin_width_p_rad']:.6e} outside expected range"
        )


class TestCrystalDarwinWidthXRT:
    """Cross-compare Darwin width with XRT reference."""

    @pytest.mark.parametrize("crystal", ["si111", "si220"])
    def test_darwin_width_vs_xrt(self, xrt, crystal):
        _, rm = xrt
        data = load_fixture(f"crystal_{crystal}.json")
        tc = next(
            (t for t in data["test_cases"] if "darwin_width" in t["id"]),
            None,
        )
        if tc is None or "xrt_darwin_s_rad" not in tc:
            pytest.skip("No XRT Darwin width reference")

        # Verify XRT reference is self-consistent
        assert tc["xrt_darwin_s_rad"] > tc["xrt_darwin_p_rad"], (
            f"XRT Darwin S ({tc['xrt_darwin_s_rad']:.4e}) should be > P ({tc['xrt_darwin_p_rad']:.4e})"
        )

        # Verify fixture analytical formula matches XRT
        assert_close(
            tc["darwin_width_s_rad"],
            tc["xrt_darwin_s_rad"],
            tc["xrt_darwin_s_rad"] * 0.1,  # 10% tolerance
            "analytical vs XRT Darwin S",
        )
