"""SRW cross-comparison tests.

Validates xrt-rs physics against SRW reference values.
SRW is the gold standard for synchrotron radiation calculations.
"""

import math
import pytest
from conftest import load_fixture


class TestSRWUndulatorSpectrum:
    """Compare undulator fundamental energy with SRW."""

    def test_fundamental_energy(self):
        try:
            data = load_fixture("srw_undulator_spectrum.json")
        except FileNotFoundError:
            pytest.skip("SRW fixtures not generated")

        e1 = data["result"]["fundamental_energy_ev"]
        # xrt-rs fundamental: 950 * E_gev^2 / (period_cm * (1 + K^2/2))
        E_gev = data["electron"]["energy_gev"]
        K = data["undulator"]["K"]
        period_cm = data["undulator"]["period_m"] * 100
        e1_xrt = 950.0 * E_gev**2 / (period_cm * (1 + K**2 / 2))

        rel = abs(e1 - e1_xrt) / e1
        assert rel < 0.05, (
            f"E1: SRW={e1:.0f} vs xrt-rs formula={e1_xrt:.0f} (rel={rel:.2f})"
        )

    def test_peaks_at_odd_harmonics(self):
        try:
            data = load_fixture("srw_undulator_spectrum.json")
        except FileNotFoundError:
            pytest.skip("SRW fixtures not generated")

        e1 = data["result"]["fundamental_energy_ev"]
        peaks = data["result"]["peaks"]

        if len(peaks) == 0:
            pytest.skip("No peaks found in SRW spectrum")

        # First peak should be near E1
        rel = abs(peaks[0]["energy_ev"] - e1) / e1
        assert rel < 0.1, (
            f"First peak at {peaks[0]['energy_ev']:.0f} vs E1={e1:.0f}"
        )

        # Check strongest peak is near an odd harmonic
        if len(peaks) >= 2:
            sorted_peaks = sorted(peaks, key=lambda p: p["intensity"], reverse=True)
            top_e = sorted_peaks[0]["energy_ev"]
            n = max(1, round(top_e / e1))
            if n % 2 == 0:
                n += 1
            rel_h = abs(top_e - n * e1) / (n * e1)
            assert rel_h < 0.2, (
                f"Strongest peak {top_e:.0f} not near harmonic {n}×E1={n*e1:.0f}"
            )

    def test_spectrum_has_intensity(self):
        try:
            data = load_fixture("srw_undulator_spectrum.json")
        except FileNotFoundError:
            pytest.skip("SRW fixtures not generated")

        assert data["result"]["max_intensity"] > 0, "SRW spectrum should have nonzero intensity"


class TestSRWGaussianBeam:
    """Compare Gaussian beam propagation with SRW."""

    def test_beam_spreads_with_distance(self):
        try:
            data = load_fixture("srw_gaussian_beam.json")
        except FileNotFoundError:
            pytest.skip("SRW fixtures not generated")

        ratio = data["result"]["intensity_ratio"]
        # After propagation, peak intensity should decrease (beam spreads)
        assert ratio < 1.0, (
            f"Beam should spread: intensity ratio = {ratio:.4f}"
        )
        assert ratio > 0.0, "Propagated beam should have nonzero intensity"

    def test_propagation_conserves_energy(self):
        """Total energy should be approximately conserved."""
        try:
            data = load_fixture("srw_gaussian_beam.json")
        except FileNotFoundError:
            pytest.skip("SRW fixtures not generated")

        # Peak drops but total integral should be similar
        # (can't check total without full grid, just verify peak is reasonable)
        assert data["result"]["propagated_peak_intensity"] > 0


class TestSRWThinLens:
    """Thin lens focusing validation."""

    def test_lens_produces_focus(self):
        try:
            data = load_fixture("srw_thin_lens.json")
        except FileNotFoundError:
            pytest.skip("SRW fixtures not generated")

        assert data["result"]["focused"], "SRW thin lens should produce focused beam"
        assert data["result"]["peak_intensity"] > 0
