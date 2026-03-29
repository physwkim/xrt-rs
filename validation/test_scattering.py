"""Domain 2: Scattering factors — f0, f1/f2 cross-comparison."""

import numpy as np
import pytest
from conftest import load_fixture, assert_close


class TestScatteringGolden:
    """Verify Python XRT scattering values match fixtures."""

    @pytest.mark.parametrize("elem", ["si", "au", "w", "o"])
    def test_f0(self, xrt, elem):
        _, rm = xrt
        data = load_fixture(f"scattering_{elem}.json")
        tc = next(t for t in data["test_cases"] if t["id"].endswith("_f0"))

        el = rm.Element(data["element"], data["table"])
        for q, expected in zip(tc["q_over_4pi"], tc["f0"]):
            actual = float(el.get_f0(np.array([q]))[0])
            assert_close(actual, expected, 1e-14, f"f0(q={q})")

    @pytest.mark.parametrize("elem", ["si", "au", "w", "o"])
    def test_f1f2(self, xrt, elem):
        _, rm = xrt
        data = load_fixture(f"scattering_{elem}.json")
        tc = next(t for t in data["test_cases"] if t["id"].endswith("_f1f2"))

        el = rm.Element(data["element"], data["table"])
        e_arr = np.array(tc["energies_ev"])
        f1f2 = el.get_f1f2(e_arr)

        for i, energy in enumerate(tc["energies_ev"]):
            assert_close(float(f1f2[i].real), tc["f1"][i], 1e-12,
                         f"f1(E={energy})")
            assert_close(float(f1f2[i].imag), tc["f2"][i], 1e-12,
                         f"f2(E={energy})")


class TestScatteringEdge:
    """Verify f1/f2 at absorption edge energies."""

    @pytest.mark.parametrize("elem,edge_id", [
        ("si", "si_edge_K"),
        ("au", "au_edge_L3"),
        ("w", "w_edge_L3"),
        ("o", "o_edge_K"),
    ])
    def test_f1f2_edge(self, xrt, elem, edge_id):
        _, rm = xrt
        data = load_fixture(f"scattering_{elem}.json")
        tc = next(
            (t for t in data["test_cases"] if t["id"] == edge_id),
            None,
        )
        if tc is None:
            pytest.skip(f"No edge test case {edge_id} in scattering_{elem}.json")

        el = rm.Element(data["element"], data["table"])
        e_arr = np.array(tc["energies_ev"])
        f1f2 = el.get_f1f2(e_arr)

        for i, energy in enumerate(tc["energies_ev"]):
            assert_close(float(f1f2[i].real), tc["f1"][i], 1e-12,
                         f"edge f1(E={energy})")
            assert_close(float(f1f2[i].imag), tc["f2"][i], 1e-12,
                         f"edge f2(E={energy})")

        # Qualitative check: f2 should change significantly across edge
        f2_vals = [float(f1f2[i].imag) for i in range(len(tc["energies_ev"]))]
        f2_below = (f2_vals[0] + f2_vals[1]) / 2
        f2_above = (f2_vals[2] + f2_vals[3]) / 2
        assert f2_below != pytest.approx(f2_above, rel=0.01), (
            f"f2 should change across {tc['edge_name']} edge: "
            f"below={f2_below:.4e}, above={f2_above:.4e}"
        )
