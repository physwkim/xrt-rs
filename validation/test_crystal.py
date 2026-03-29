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
