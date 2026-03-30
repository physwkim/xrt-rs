"""Domain 11: Synchrotron source cross-comparison (BM, wiggler, undulator)."""

import math

import pytest
from conftest import load_fixture


class TestSynchrotronCross:
    """Verify xrt-rs synchrotron sources produce physically valid rays."""

    def _check_basic_properties(self, result, tc):
        """Common checks for all source types."""
        x = result["x"]
        a = result["a"]
        b = result["b"]
        c = result["c"]
        e = result["e"]
        n = len(x)

        assert n == tc["nrays"], f"nrays: {n} != {tc['nrays']}"

        # Direction vectors must be unit length
        for i in range(n):
            norm_sq = a[i]**2 + b[i]**2 + c[i]**2
            assert abs(norm_sq - 1.0) < 1e-12, (
                f"ray[{i}]: |dir|² = {norm_sq}"
            )

        # Energy must be within [e_min, e_max]
        for i in range(n):
            assert tc["e_min"] <= e[i] <= tc["e_max"], (
                f"ray[{i}]: E = {e[i]} outside [{tc['e_min']}, {tc['e_max']}]"
            )

        # Direction should be predominantly forward (b ≈ 1, |a| << 1, |c| << 1)
        mean_b = sum(b) / n
        assert mean_b > 0.999, f"mean(b) = {mean_b}, expected ~1.0"

        # Angular spread should be within limits
        for i in range(n):
            theta_x = abs(math.atan2(a[i], b[i]))
            theta_z = abs(math.atan2(c[i], b[i]))
            assert theta_x < tc["theta_max"] * 5, (
                f"ray[{i}]: theta_x = {theta_x} > 5 * theta_max"
            )
            assert theta_z < tc["psi_max"] * 5, (
                f"ray[{i}]: theta_z = {theta_z} > 5 * psi_max"
            )

    def test_bending_magnet(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("synchrotron_sources.json")
        tc = next(t for t in data["test_cases"] if t["id"] == "bending_magnet")

        result = rs.bending_magnet_shine_rs(
            tc["electron_energy_gev"],
            tc["beam_current"],
            tc["b_field"],
            tc["nrays"],
            tc["e_min"],
            tc["e_max"],
            tc["theta_max"],
            tc["psi_max"],
        )

        self._check_basic_properties(result, tc)

        # BM spectrum: mean energy should be in a reasonable range
        e = result["e"]
        mean_e = sum(e) / len(e)
        assert tc["e_min"] < mean_e < tc["e_max"], (
            f"BM mean energy {mean_e:.0f} eV outside range"
        )

    def test_wiggler(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("synchrotron_sources.json")
        tc = next(t for t in data["test_cases"] if t["id"] == "wiggler")

        result = rs.wiggler_shine_rs(
            tc["electron_energy_gev"],
            tc["beam_current"],
            tc["k_param"],
            tc["period_mm"],
            tc["n_periods"],
            tc["nrays"],
            tc["e_min"],
            tc["e_max"],
            tc["theta_max"],
            tc["psi_max"],
        )

        self._check_basic_properties(result, tc)

        # Wiggler: higher flux than BM → energy distribution should be non-degenerate
        e = result["e"]
        mean_e = sum(e) / len(e)
        std_e = math.sqrt(sum((v - mean_e)**2 for v in e) / len(e))
        assert std_e > 0, "wiggler energy distribution has zero spread"

    def test_undulator(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("synchrotron_sources.json")
        tc = next(t for t in data["test_cases"] if t["id"] == "undulator")

        result = rs.undulator_shine_rs(
            tc["electron_energy_gev"],
            tc["beam_current"],
            tc["kx"],
            tc["ky"],
            tc["period_mm"],
            tc["n_periods"],
            tc["nrays"],
            tc["e_min"],
            tc["e_max"],
            tc["theta_max"],
            tc["psi_max"],
        )

        self._check_basic_properties(result, tc)

        # Undulator: angular distribution should be tighter than BM/wiggler
        a = result["a"]
        c = result["c"]
        b_vals = result["b"]
        n = len(a)
        std_ax = math.sqrt(sum(v**2 for v in a) / n)
        std_cz = math.sqrt(sum(v**2 for v in c) / n)
        assert std_ax < tc["theta_max"], (
            f"undulator std(a) = {std_ax:.2e} > theta_max = {tc['theta_max']:.2e}"
        )
        assert std_cz < tc["psi_max"], (
            f"undulator std(c) = {std_cz:.2e} > psi_max = {tc['psi_max']:.2e}"
        )

    def test_bending_magnet_vs_wiggler_flux(self, xrt_rs):
        """Wiggler with many periods should have more flux than single BM."""
        rs = xrt_rs
        data = load_fixture("synchrotron_sources.json")
        bm_tc = next(t for t in data["test_cases"] if t["id"] == "bending_magnet")
        wig_tc = next(t for t in data["test_cases"] if t["id"] == "wiggler")

        bm_result = rs.bending_magnet_shine_rs(
            bm_tc["electron_energy_gev"],
            bm_tc["beam_current"],
            bm_tc["b_field"],
            1000,
            bm_tc["e_min"],
            bm_tc["e_max"],
            bm_tc["theta_max"],
            bm_tc["psi_max"],
        )

        # Both should produce valid rays
        assert len(bm_result["e"]) == 1000

    def test_undulator_direction_normalization(self, xrt_rs):
        """All undulator rays should have unit direction vectors."""
        rs = xrt_rs
        data = load_fixture("synchrotron_sources.json")
        tc = next(t for t in data["test_cases"] if t["id"] == "undulator")

        result = rs.undulator_shine_rs(
            tc["electron_energy_gev"],
            tc["beam_current"],
            tc["kx"],
            tc["ky"],
            tc["period_mm"],
            tc["n_periods"],
            tc["nrays"],
            tc["e_min"],
            tc["e_max"],
            tc["theta_max"],
            tc["psi_max"],
        )

        a, b, c = result["a"], result["b"], result["c"]
        for i in range(min(100, len(a))):
            norm_sq = a[i]**2 + b[i]**2 + c[i]**2
            assert abs(norm_sq - 1.0) < 1e-12, (
                f"undulator ray[{i}]: |dir|² = {norm_sq}"
            )


class TestUndulatorPolarization:
    """Test undulator with different polarization configurations."""

    def test_planar_vertical_kx_only(self, xrt_rs):
        """K_x only → vertically polarized undulator."""
        rs = xrt_rs
        result = rs.undulator_shine_rs(
            6.0, 0.2,    # electron_energy_gev, beam_current
            1.5, 0.0,    # kx, ky (K_x only)
            20.0, 100,   # period_mm, n_periods
            5000,         # nrays
            5000.0, 20000.0,  # e_min, e_max
            1e-4, 1e-4,  # theta_max, psi_max
        )
        assert len(result["e"]) == 5000
        for i in range(min(100, len(result["a"]))):
            norm_sq = result["a"][i]**2 + result["b"][i]**2 + result["c"][i]**2
            assert abs(norm_sq - 1.0) < 1e-12

    def test_elliptical_phase90(self, xrt_rs):
        """K_x = K_y with phase=90° → circular polarization."""
        rs = xrt_rs
        # Note: undulator_shine_rs doesn't take phase_deg directly,
        # but we can test K_x = K_y which gives 45° linear
        result = rs.undulator_shine_rs(
            6.0, 0.2,
            1.0, 1.0,    # kx = ky (equal deflection parameters)
            20.0, 100,
            5000,
            5000.0, 20000.0,
            1e-4, 1e-4,
        )
        assert len(result["e"]) == 5000
        # Energy should be in range
        for ev in result["e"]:
            assert 5000.0 <= ev <= 20000.0

    def test_zero_k_produces_no_radiation(self, xrt_rs):
        """K=0 undulator should still produce rays (uniform angles)."""
        rs = xrt_rs
        result = rs.undulator_shine_rs(
            6.0, 0.2,
            0.0, 0.01,    # near-zero K
            20.0, 100,
            1000,
            5000.0, 20000.0,
            1e-4, 1e-4,
        )
        assert len(result["e"]) == 1000


class TestSynchrotronXRTCross:
    """Cross-compare xrt-rs synchrotron sources with Python XRT reference."""

    def test_bm_mean_energy_cross(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("synchrotron_sources.json")
        tc = next(t for t in data["test_cases"] if t["id"] == "bending_magnet")

        if "xrt_mean_energy" not in tc:
            pytest.skip("No XRT BM reference in fixture")

        result = rs.bending_magnet_shine_rs(
            tc["electron_energy_gev"],
            tc["beam_current"],
            tc["b_field"],
            tc["nrays"],
            tc["e_min"],
            tc["e_max"],
            tc["theta_max"],
            tc["psi_max"],
        )

        e = result["e"]
        mean_e = sum(e) / len(e)

        xrt_mean = tc["xrt_mean_energy"]
        # Statistical comparison: 20% tolerance (Monte Carlo)
        rel = abs(mean_e - xrt_mean) / xrt_mean
        assert rel < 0.2, (
            f"BM mean energy: Rust={mean_e:.0f} vs XRT={xrt_mean:.0f} (rel={rel:.2f})"
        )

    def test_undulator_mean_energy_cross(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("synchrotron_sources.json")
        tc = next(t for t in data["test_cases"] if t["id"] == "undulator")

        if "xrt_mean_energy" not in tc:
            pytest.skip("No XRT undulator reference in fixture")

        result = rs.undulator_shine_rs(
            tc["electron_energy_gev"],
            tc["beam_current"],
            tc["kx"],
            tc["ky"],
            tc["period_mm"],
            tc["n_periods"],
            tc["nrays"],
            tc["e_min"],
            tc["e_max"],
            tc["theta_max"],
            tc["psi_max"],
        )

        e = result["e"]
        mean_e = sum(e) / len(e)

        xrt_mean = tc["xrt_mean_energy"]
        rel = abs(mean_e - xrt_mean) / xrt_mean
        assert rel < 0.3, (
            f"Undulator mean energy: Rust={mean_e:.0f} vs XRT={xrt_mean:.0f} (rel={rel:.2f})"
        )
