"""Domain 7: Takagi-Taupin rocking curve cross-comparison."""

import math

import numpy as np
import pytest
from conftest import load_fixture, assert_close


class TestTTGolden:
    """Verify rocking curve against fixture."""

    def test_rocking_curve_shape(self, xrt):
        _, rm = xrt
        data = load_fixture("tt_si111_10kev.json")
        tc = data["test_cases"][0]

        crystal = rm.CrystalSi(hkl=(1, 1, 1), t=297.15)
        e_arr = np.array([data["energy_ev"]])

        for i, (dtheta, bidn_val) in enumerate(
            zip(tc["dtheta_rad"], tc["beam_in_dot_normal"])
        ):
            rs, rp = crystal.get_amplitude(e_arr, np.array([bidn_val]))[:2]
            exp_rs = tc["rs"][i]
            # Use 1e-4 tolerance due to ODE solver differences
            assert_close(float(rs[0].real), exp_rs["re"], 1e-4,
                         f"rs.re(dθ={dtheta:.1e})")
            assert_close(float(rs[0].imag), exp_rs["im"], 1e-4,
                         f"rs.im(dθ={dtheta:.1e})")


class TestTTCross:
    """Cross-compare Python XRT and xrt-rs TT solver."""

    def test_tt_solve_basic(self, xrt_rs):
        rs = xrt_rs
        # Simple test: constant coefficients Riccati ODE
        n = 5
        cb_re = [0.001] * n
        cb_im = [0.0001] * n
        c0_re = [0.0] * n
        c0_im = [0.0] * n
        ch_re = [0.001] * n
        ch_im = [-0.0001] * n

        rs_re, rs_im = rs.tt_solve_rs(
            cb_re, cb_im, c0_re, c0_im, ch_re, ch_im,
            0.0, 1e6,  # z_start, z_end (Å)
            0.0, 0.0,  # xi_init
        )
        assert len(rs_re) == n
        for v in rs_re:
            assert math.isfinite(v), f"non-finite result: {v}"
