"""Domain 1: Physical constants — Python XRT vs xrt-rs PyO3."""

import pytest
from conftest import load_fixture, assert_close_rel


class TestConstantsGolden:
    """Verify Python XRT constants match fixture values."""

    def test_constants_match_fixture(self, xrt):
        pc, _ = xrt
        data = load_fixture("constants.json")
        consts = data["constants"]
        tol = 1e-15
        for name, expected in consts.items():
            actual = getattr(pc, name)
            assert_close_rel(actual, expected, tol, name)


class TestConstantsCross:
    """Cross-compare Python XRT constants with xrt-rs."""

    CONST_NAMES = [
        "CH", "CHBAR", "R0", "AVOGADRO", "FINE_STR", "E2W",
        "HPLANCK", "SIE0", "C", "M0", "M0C2",
    ]

    def test_constants_cross(self, xrt, xrt_rs):
        pc, _ = xrt
        # xrt_rs doesn't export constants directly, so we skip
        # This is a placeholder for when constants are added to the binding
        pytest.skip("xrt_rs does not expose constants yet")
