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
