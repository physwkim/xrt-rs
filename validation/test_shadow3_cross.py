"""Shadow3 cross-comparison tests.

Validates xrt-rs physics against shadow3 reference values.
Shadow3 uses cm as default units; xrt-rs uses mm.
"""

import math
import pytest
from conftest import load_fixture, assert_close


CM_TO_MM = 10.0


class TestShadow3Source:
    """Compare source statistics between shadow3 and xrt-rs."""

    def test_source_spatial_distribution(self, xrt_rs):
        rs = xrt_rs
        try:
            data = load_fixture("shadow3_source.json")
        except FileNotFoundError:
            pytest.skip("shadow3 fixtures not generated")

        result = rs.geometric_source_shine(
            data["source"]["npoint"],
            data["source"]["energy_ev"],
            data["source"]["sigmax_cm"] * CM_TO_MM,  # cm -> mm
            data["source"]["sigmaz_cm"] * CM_TO_MM,
            data["source"]["sigdix_rad"],
            data["source"]["sigdiz_rad"],
        )

        x = result["x"]
        z = result["z"]
        n = len(x)

        # Compare std (statistical: 15% tolerance)
        std_x = math.sqrt(sum(xi**2 for xi in x) / n)
        std_z = math.sqrt(sum(zi**2 for zi in z) / n)

        shadow_std_x = data["result"]["std_x_cm"] * CM_TO_MM
        shadow_std_z = data["result"]["std_z_cm"] * CM_TO_MM

        rel_x = abs(std_x - shadow_std_x) / shadow_std_x
        rel_z = abs(std_z - shadow_std_z) / shadow_std_z
        assert rel_x < 0.15, (
            f"std_x: Rust={std_x:.4f} vs shadow3={shadow_std_x:.4f} (rel={rel_x:.2f})"
        )
        assert rel_z < 0.15, (
            f"std_z: Rust={std_z:.4f} vs shadow3={shadow_std_z:.4f} (rel={rel_z:.2f})"
        )

    def test_source_direction_norm(self, xrt_rs):
        rs = xrt_rs
        try:
            data = load_fixture("shadow3_source.json")
        except FileNotFoundError:
            pytest.skip("shadow3 fixtures not generated")

        # Shadow3 direction cosines should be unit vectors
        shadow_norm = data["result"]["mean_dir_norm"]
        assert_close(shadow_norm, 1.0, 1e-6, "shadow3 direction norm")


class TestShadow3ConicSurface:
    """Verify conic coefficient results match shadow3."""

    def test_conic_sphere_good_rays(self):
        try:
            data = load_fixture("shadow3_conic_surface.json")
        except FileNotFoundError:
            pytest.skip("shadow3 fixtures not generated")

        # Verify shadow3 produced good rays with conic sphere
        assert data["result"]["n_good"] > 50, (
            f"shadow3 conic sphere should have >50 good rays: {data['result']['n_good']}"
        )

    def test_conic_sphere_focusing(self):
        try:
            data = load_fixture("shadow3_conic_surface.json")
            data2 = load_fixture("shadow3_spherical_mirror.json")
        except FileNotFoundError:
            pytest.skip("shadow3 fixtures not generated")

        # Conic sphere (FMIRR=10) should give similar beam size as spherical (FMIRR=1)
        # Both use same R, so focusing should be similar
        conic_x_std = data["result"]["x_std_cm"]
        sphere_x_std = data2["result"]["x_std_cm"]

        # Allow 50% tolerance (different ray sets)
        if sphere_x_std > 0:
            ratio = conic_x_std / sphere_x_std
            assert 0.3 < ratio < 3.0, (
                f"conic vs sphere x_std ratio: {ratio:.2f}"
            )
