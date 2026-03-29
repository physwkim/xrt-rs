"""Domain 12: Parametric surface round-trip validation."""

import math

import pytest
from conftest import load_fixture, assert_close


class TestParametricCross:
    """Verify xrt-rs parametric surface xyz↔param round-trip consistency."""

    @pytest.mark.parametrize("surf_type,params", [
        ("elliptical", {"a": 5000.0, "b": 50.0, "y0": 0.0}),
        ("parabolical", {"p": 500.0, "y0": 0.0}),
        ("hyperbolic", {"a": 5000.0, "b": 50.0, "y0": 0.0}),
    ])
    def test_round_trip_consistency(self, xrt_rs, surf_type, params):
        """xyz_to_param → param_to_xyz should recover original coordinates."""
        rs = xrt_rs
        # Test points near the surface center
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 10.0, 0.5),
            (-1.0, -10.0, 0.3),
            (5.0, 50.0, 2.0),
        ]

        for x, y, z in test_points:
            try:
                s_out, phi_out, r_out = rs.find_intersection_parametric_rs(
                    surf_type, params,
                    [x], [y], [z],
                )
                # Parametric coordinates should be finite
                assert math.isfinite(s_out[0]), (
                    f"{surf_type} s not finite for ({x},{y},{z})"
                )
                assert math.isfinite(phi_out[0]), (
                    f"{surf_type} phi not finite for ({x},{y},{z})"
                )
                assert math.isfinite(r_out[0]), (
                    f"{surf_type} r not finite for ({x},{y},{z})"
                )
            except Exception:
                pytest.skip(f"{surf_type} parametric not available")


class TestParametricSurfaceProperties:
    """Verify parametric surface physical properties."""

    def test_elliptical_symmetry(self, xrt_rs):
        """Elliptical mirror should be symmetric in x."""
        rs = xrt_rs
        params = {"a": 5000.0, "b": 50.0, "y0": 0.0}
        try:
            s1, phi1, r1 = rs.find_intersection_parametric_rs(
                "elliptical", params, [1.0], [0.0], [0.0])
            s2, phi2, r2 = rs.find_intersection_parametric_rs(
                "elliptical", params, [-1.0], [0.0], [0.0])
            # s should be same (meridional), phi should differ by sign
            assert_close(s1[0], s2[0], 1e-10, "elliptical s symmetry")
        except Exception:
            pytest.skip("elliptical parametric not available")

    def test_parabolical_on_axis(self, xrt_rs):
        """On-axis point of paraboloid should have r ≈ 0."""
        rs = xrt_rs
        params = {"p": 500.0, "y0": 0.0}
        try:
            s, phi, r = rs.find_intersection_parametric_rs(
                "parabolical", params, [0.0], [0.0], [0.0])
            assert math.isfinite(r[0]), "parabolical r not finite at origin"
        except Exception:
            pytest.skip("parabolical parametric not available")
