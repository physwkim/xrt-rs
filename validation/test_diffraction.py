"""Domain 8: Kirchhoff diffraction integral cross-comparison."""

import math

import pytest
from conftest import load_fixture, assert_close


class TestDiffractionCross:
    """Cross-compare Python fixture inputs through xrt-rs."""

    def test_small_deterministic(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("diffraction_ref.json")
        tc = data["test_cases"][0]
        rays = tc["rays"]
        pixels = tc["pixels"]

        es_re, es_im, ep_re, ep_im = rs.diffraction_integral_rs(
            rays["x"], rays["y"], rays["z"],
            rays["nx"], rays["ny"], rays["nz"],
            rays["nl"],
            rays["es_re"], rays["es_im"],
            rays["ep_re"], rays["ep_im"],
            rays["energy"],
            pixels["x"], pixels["y"], pixels["z"],
        )

        assert len(es_re) == len(pixels["x"])
        for i in range(len(es_re)):
            assert math.isfinite(es_re[i]), f"es_re[{i}] not finite"
            assert math.isfinite(es_im[i]), f"es_im[{i}] not finite"
            assert math.isfinite(ep_re[i]), f"ep_re[{i}] not finite"
            assert math.isfinite(ep_im[i]), f"ep_im[{i}] not finite"

    def test_small_deterministic_numerical(self, xrt_rs):
        rs = xrt_rs
        data = load_fixture("diffraction_ref.json")
        tc = data["test_cases"][0]
        rays = tc["rays"]
        pixels = tc["pixels"]
        expected = tc["expected"]

        es_re, es_im, ep_re, ep_im = rs.diffraction_integral_rs(
            rays["x"], rays["y"], rays["z"],
            rays["nx"], rays["ny"], rays["nz"],
            rays["nl"],
            rays["es_re"], rays["es_im"],
            rays["ep_re"], rays["ep_im"],
            rays["energy"],
            pixels["x"], pixels["y"], pixels["z"],
        )

        tol = 1e-10
        for i in range(len(es_re)):
            assert_close(es_re[i], expected[i]["es_re"], tol, f"pixel[{i}] es_re")
            assert_close(es_im[i], expected[i]["es_im"], tol, f"pixel[{i}] es_im")
            assert_close(ep_re[i], expected[i]["ep_re"], tol, f"pixel[{i}] ep_re")
            assert_close(ep_im[i], expected[i]["ep_im"], tol, f"pixel[{i}] ep_im")
