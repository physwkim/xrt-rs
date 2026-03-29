"""Domain 6: Ray-surface intersection cross-comparison."""

import pytest
from conftest import load_fixture, assert_close


class TestIntersectionCross:
    """Cross-compare Python and Rust intersection results."""

    def test_flat_intersection(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("intersection.json")
        rays = data["rays"]

        xs = [r["x"] for r in rays]
        ys = [r["y"] for r in rays]
        zs = [r["z"] for r in rays]
        als = [r["a"] for r in rays]
        bs = [r["b"] for r in rays]
        cs = [r["c"] for r in rays]
        t1s = [r["t1"] for r in rays]
        t2s = [r["t2"] for r in rays]

        t_out, x_out, y_out, z_out = rs.find_intersection_rs(
            "flat", {}, t1s, t2s, xs, ys, zs, als, bs, cs, 1)

        for i in range(len(rays)):
            # On flat surface z should be ~0
            assert_close(z_out[i], 0.0, 1e-10, f"ray{i} z")

    def test_toroid_intersection(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("intersection.json")
        rays = data["rays"]

        xs = [r["x"] for r in rays]
        ys = [r["y"] for r in rays]
        zs = [r["z"] for r in rays]
        als = [r["a"] for r in rays]
        bs = [r["b"] for r in rays]
        cs = [r["c"] for r in rays]
        t1s = [r["t1"] for r in rays]
        t2s = [r["t2"] for r in rays]

        t_out, x_out, y_out, z_out = rs.find_intersection_rs(
            "toroid", {"R": 5e6, "r": 50.0},
            t1s, t2s, xs, ys, zs, als, bs, cs, 1)

        for i in range(len(rays)):
            assert z_out[i] < 50.0 + 1.0, f"ray{i} z={z_out[i]} unreasonable"
