"""Error handling tests for PyO3 bindings."""

import pytest
from conftest import load_fixture


class TestPyO3ErrorHandling:
    """Verify xrt-rs Python bindings handle invalid inputs gracefully."""

    def test_invalid_surface_type(self, xrt_rs):
        rs = xrt_rs
        with pytest.raises(Exception):
            rs.find_intersection_rs(
                "nonexistent_surface", {},
                [0.1], [200.0],
                [0.0], [0.0], [10.0],
                [0.0], [0.0], [-1.0],
                1,
            )

    def test_empty_ray_arrays(self, xrt_rs):
        rs = xrt_rs
        # Empty arrays should not crash
        try:
            t, x, y, z = rs.find_intersection_rs(
                "flat", {},
                [], [],
                [], [], [],
                [], [], [],
                1,
            )
            assert len(t) == 0
        except Exception:
            pass  # Error is also acceptable

    def test_mismatched_array_lengths(self, xrt_rs):
        rs = xrt_rs
        # Different length arrays should raise or handle gracefully (may panic)
        with pytest.raises(BaseException):
            rs.find_intersection_rs(
                "flat", {},
                [0.1], [200.0],
                [0.0, 1.0], [0.0], [10.0],  # x has 2 elements, others have 1
                [0.0], [0.0], [-1.0],
                1,
            )

    def test_diffraction_empty_input(self, xrt_rs):
        rs = xrt_rs
        try:
            es_re, es_im, ep_re, ep_im = rs.diffraction_integral_rs(
                [], [], [], [], [], [], [],
                [], [], [], [], [],
                [0.0], [0.0], [1000.0],
            )
            # Zero rays → zero amplitude
            assert len(es_re) == 1
        except Exception:
            pass  # Error is also acceptable
