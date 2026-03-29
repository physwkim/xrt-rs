"""Domain 3: Material optics — n(E), μ(E), Fresnel Rs/Rp."""

import numpy as np
import pytest
from conftest import load_fixture, assert_close, assert_complex_close


class TestMaterialGolden:
    """Verify Python XRT material values match fixtures."""

    @pytest.mark.parametrize("mat", ["si", "au", "sio2"])
    def test_refractive_index(self, xrt, mat):
        _, rm = xrt
        data = load_fixture(f"material_{mat}.json")
        tc = next(t for t in data["test_cases"] if t["id"].endswith("_n"))

        m = rm.Material(data["elements"], data["quantities"],
                        rho=data["rho"], table="Chantler total")
        e_arr = np.array(tc["energies_ev"])
        n = m.get_refractive_index(e_arr)

        for i, energy in enumerate(tc["energies_ev"]):
            assert_close(float(n[i].real), tc["n_real"][i], 1e-10,
                         f"n.re(E={energy})")
            assert_close(float(n[i].imag), tc["n_imag"][i], 1e-10,
                         f"n.im(E={energy})")

    @pytest.mark.parametrize("mat", ["si", "au", "sio2"])
    def test_absorption_coefficient(self, xrt, mat):
        _, rm = xrt
        data = load_fixture(f"material_{mat}.json")
        tc = next(t for t in data["test_cases"] if t["id"].endswith("_mu"))

        m = rm.Material(data["elements"], data["quantities"],
                        rho=data["rho"], table="Chantler total")
        e_arr = np.array(tc["energies_ev"])
        mu = m.get_absorption_coefficient(e_arr)

        for i, energy in enumerate(tc["energies_ev"]):
            assert_close(float(mu[i]), tc["mu_cm_inv"][i], 1e-6,
                         f"mu(E={energy})")

    @pytest.mark.parametrize("mat", ["si", "au", "sio2"])
    def test_fresnel(self, xrt, mat):
        _, rm = xrt
        data = load_fixture(f"material_{mat}.json")
        tc = next(t for t in data["test_cases"] if t["id"].endswith("_fresnel"))

        m = rm.Material(data["elements"], data["quantities"],
                        rho=data["rho"], table="Chantler total")
        e_arr = np.array([tc["energy_ev"]])

        for amp in tc["amplitudes"]:
            bidn = np.array([amp["sin_theta"]])
            rs, rp = m.get_amplitude(e_arr, bidn)[:2]
            assert_complex_close(
                float(rs[0].real), float(rs[0].imag),
                amp["rs_real"], amp["rs_imag"],
                1e-8, f"rs(θ={amp['sin_theta']})")
            assert_complex_close(
                float(rp[0].real), float(rp[0].imag),
                amp["rp_real"], amp["rp_imag"],
                1e-8, f"rp(θ={amp['sin_theta']})")
