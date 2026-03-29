"""Domain 6: Ray-surface intersection cross-comparison."""

import pytest
from conftest import load_fixture, assert_close


class TestIntersectionCross:
    """Cross-compare Python and Rust intersection results."""

    @staticmethod
    def _extract_rays(data):
        rays = data["rays"]
        return {
            "xs": [r["x"] for r in rays],
            "ys": [r["y"] for r in rays],
            "zs": [r["z"] for r in rays],
            "als": [r["a"] for r in rays],
            "bs": [r["b"] for r in rays],
            "cs": [r["c"] for r in rays],
            "t1s": [r["t1"] for r in rays],
            "t2s": [r["t2"] for r in rays],
        }

    def test_flat_intersection(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("intersection.json")
        r = self._extract_rays(data)

        t_out, x_out, y_out, z_out = rs.find_intersection_rs(
            "flat", {}, r["t1s"], r["t2s"],
            r["xs"], r["ys"], r["zs"], r["als"], r["bs"], r["cs"], 1)

        for i in range(len(data["rays"])):
            assert_close(z_out[i], 0.0, 1e-10, f"ray{i} z")

    def test_toroid_intersection(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("intersection.json")
        r = self._extract_rays(data)

        t_out, x_out, y_out, z_out = rs.find_intersection_rs(
            "toroid", {"R": 5e6, "r": 50.0},
            r["t1s"], r["t2s"],
            r["xs"], r["ys"], r["zs"], r["als"], r["bs"], r["cs"], 1)

        for i in range(len(data["rays"])):
            assert z_out[i] < 50.0 + 1.0, f"ray{i} z={z_out[i]} unreasonable"

    @pytest.mark.parametrize("surf_name,surf_type,params", [
        ("flat", "flat", {}),
        ("toroid", "toroid", {"R": 5e6, "r": 50.0}),
        ("spherical", "spherical", {"R": 1000.0}),
        ("paraboloid_lens", "paraboloid_lens", {"focus": 100.0}),
    ])
    def test_intersection_numerical(self, xrt_rs, surf_name, surf_type, params):
        rs = xrt_rs
        data = load_fixture("intersection.json")
        r = self._extract_rays(data)
        intersections = data["surfaces"][surf_name]["intersections"]

        t_out, x_out, y_out, z_out = rs.find_intersection_rs(
            surf_type, params, r["t1s"], r["t2s"],
            r["xs"], r["ys"], r["zs"], r["als"], r["bs"], r["cs"], 1)

        tol = 1e-8
        for i, expected in enumerate(intersections):
            if expected is None:
                continue
            assert_close(t_out[i], expected["t"], tol,
                         f"{surf_name} ray{i} t")
            assert_close(x_out[i], expected["x"], tol,
                         f"{surf_name} ray{i} x")
            assert_close(y_out[i], expected["y"], tol,
                         f"{surf_name} ray{i} y")
            assert_close(z_out[i], expected["z"], tol,
                         f"{surf_name} ray{i} z")
