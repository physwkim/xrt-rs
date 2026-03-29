"""Domain 9: Source distributions (statistical comparison)."""

import math

import pytest
from conftest import load_fixture


class TestSourcesCross:
    """Verify xrt-rs source generates reasonable distributions."""

    def test_geometric_source_statistics(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("sources.json")
        tc = data["test_cases"][0]

        result = rs.geometric_source_shine(
            tc["nrays"],
            tc["energy_ev"],
            tc["dx_mm"],
            tc["dz_mm"],
            tc["dxprime_rad"],
            tc["dzprime_rad"],
        )

        x = result["x"]
        z = result["z"]
        e = result["e"]

        n = len(x)
        assert n == tc["nrays"]

        # Mean should be near 0
        mean_x = sum(x) / n
        mean_z = sum(z) / n
        tol = tc["tolerance_rel"]

        assert abs(mean_x) < tc["expected_sigma_x"] * tol * 10, (
            f"mean_x = {mean_x}, expected ~0"
        )
        assert abs(mean_z) < tc["expected_sigma_z"] * tol * 10, (
            f"mean_z = {mean_z}, expected ~0"
        )

        # Std should be close to expected
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_z = math.sqrt(sum((zi - mean_z) ** 2 for zi in z) / n)

        assert abs(std_x - tc["expected_sigma_x"]) / tc["expected_sigma_x"] < tol, (
            f"std_x = {std_x}, expected {tc['expected_sigma_x']}"
        )
        assert abs(std_z - tc["expected_sigma_z"]) / tc["expected_sigma_z"] < tol, (
            f"std_z = {std_z}, expected {tc['expected_sigma_z']}"
        )

        # All energies should be the requested value
        for ev in e:
            assert abs(ev - tc["energy_ev"]) < 1e-6

    def test_direction_unit_vectors(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("sources.json")
        tc = data["test_cases"][0]

        result = rs.geometric_source_shine(
            tc["nrays"],
            tc["energy_ev"],
            tc["dx_mm"],
            tc["dz_mm"],
            tc["dxprime_rad"],
            tc["dzprime_rad"],
        )

        a = result["a"]
        b = result["b"]
        c = result["c"]

        tol = 1e-12
        for i in range(len(a)):
            norm_sq = a[i] ** 2 + b[i] ** 2 + c[i] ** 2
            assert abs(norm_sq - 1.0) < tol, (
                f"ray[{i}]: |dir|² = {norm_sq}, expected 1.0"
            )

    def test_divergence_sigma(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("sources.json")
        tc = data["test_cases"][0]

        result = rs.geometric_source_shine(
            tc["nrays"],
            tc["energy_ev"],
            tc["dx_mm"],
            tc["dz_mm"],
            tc["dxprime_rad"],
            tc["dzprime_rad"],
        )

        a = result["a"]
        b = result["b"]
        c = result["c"]
        n = len(a)

        # Compute angular divergence: atan2(a, b) ≈ a/b for small angles
        dx_angles = [math.atan2(a[i], b[i]) for i in range(n)]
        dz_angles = [math.atan2(c[i], b[i]) for i in range(n)]

        mean_dx = sum(dx_angles) / n
        mean_dz = sum(dz_angles) / n
        std_dx = math.sqrt(sum((v - mean_dx) ** 2 for v in dx_angles) / n)
        std_dz = math.sqrt(sum((v - mean_dz) ** 2 for v in dz_angles) / n)

        tol_rel = 0.1  # 10% tolerance for statistical test
        exp_dx = tc["expected_dxprime_sigma_rad"]
        exp_dz = tc["expected_dzprime_sigma_rad"]

        assert abs(std_dx - exp_dx) / exp_dx < tol_rel, (
            f"std(dx_angle) = {std_dx:.6e}, expected {exp_dx:.6e}"
        )
        assert abs(std_dz - exp_dz) / exp_dz < tol_rel, (
            f"std(dz_angle) = {std_dz:.6e}, expected {exp_dz:.6e}"
        )
