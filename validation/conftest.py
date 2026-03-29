"""Shared fixtures and helpers for XRT ↔ xrt-rs validation tests."""

import json
import math
import os

import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path) as f:
        return json.load(f)


def assert_close(actual, expected, tol, label=""):
    """Assert actual ≈ expected within absolute tolerance."""
    diff = abs(actual - expected)
    assert diff <= tol, (
        f"{label}: {actual} != {expected} (diff={diff:.2e}, tol={tol:.2e})"
    )


def assert_close_rel(actual, expected, tol, label=""):
    """Assert actual ≈ expected within relative tolerance."""
    if expected == 0.0:
        assert abs(actual) <= tol, f"{label}: {actual} != 0 (tol={tol:.2e})"
        return
    rel = abs(actual - expected) / abs(expected)
    assert rel <= tol, (
        f"{label}: {actual} != {expected} (rel={rel:.2e}, tol={tol:.2e})"
    )


def assert_complex_close(actual_re, actual_im, exp_re, exp_im, tol, label=""):
    """Assert complex values are close (component-wise)."""
    assert_close(actual_re, exp_re, tol, f"{label}.re")
    assert_close(actual_im, exp_im, tol, f"{label}.im")


@pytest.fixture
def xrt():
    """Import Python xrt modules."""
    import xrt.backends.raycing.physconsts as pc
    import xrt.backends.raycing.materials as rm
    return pc, rm


@pytest.fixture
def xrt_rs():
    """Import the Rust xrt_rs PyO3 module."""
    try:
        import xrt_rs
        return xrt_rs
    except ImportError:
        pytest.skip("xrt_rs not built; run: cd crates/xrt-python && maturin develop --release")
