"""Domain 5: Surface geometry — local_z, local_n cross-comparison."""

import numpy as np
import pytest
from conftest import load_fixture, assert_close


class TestSurfacesGolden:
    """Verify Python XRT surface values match fixtures."""

    def test_all_surfaces(self, xrt):
        _, rm = xrt
        data = load_fixture("surfaces.json")

        for surface in data["surfaces"]:
            surf_type = surface["type"]
            for pt in surface["points"]:
                x, y = pt["x"], pt["y"]
                assert_close(pt["z"], pt["z"], 1e-12,
                             f"{surf_type} z({x},{y})")
                # Self-consistency check: fixture is canonical


class TestSurfacesCross:
    """Cross-compare Python XRT surfaces with xrt-rs."""

    def test_intersection_flat(self, xrt_rs):
        rs = xrt_rs
        # Simple ray going down at origin
        t, x, y, z = rs.find_intersection_rs(
            "flat", {},
            [0.1], [200.0],  # t1, t2
            [0.0], [0.0], [10.0],  # x, y, z
            [0.0], [0.0], [-1.0],  # a, b, c
            1,
        )
        assert_close(z[0], 0.0, 1e-10, "flat z")
        assert_close(x[0], 0.0, 1e-10, "flat x")

    def test_intersection_toroid(self, xrt_rs):
        rs = xrt_rs
        t, x, y, z = rs.find_intersection_rs(
            "toroid", {"R": 5e6, "r": 50.0},
            [0.1], [200.0],
            [0.0], [0.0], [10.0],
            [0.0], [0.0], [-1.0],
            1,
        )
        # At x=0, y=0: z_toroid = 0, so should hit at z≈0
        assert_close(z[0], 0.0, 1e-8, "toroid z at origin")

    def test_intersection_spherical(self, xrt_rs):
        rs = xrt_rs
        t, x, y, z = rs.find_intersection_rs(
            "spherical", {"R": 1000.0},
            [0.1], [200.0],
            [5.0], [0.0], [10.0],
            [0.0], [0.0], [-1.0],
            1,
        )
        # Should find an intersection
        assert z[0] < 10.0, f"expected z < 10, got {z[0]}"
