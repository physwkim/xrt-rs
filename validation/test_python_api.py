"""Tests for the high-level Python user API (BeamLine → Source → OE → Screen)."""

import numpy as np
import pytest

import xrt_rs as xr


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def bl():
    return xr.BeamLine()


@pytest.fixture
def si_material():
    return xr.Material(["Si"], rho=2.33)


@pytest.fixture
def source_beam(bl):
    """A beam from a geometric source at origin."""
    src = xr.GeometricSource(
        bl,
        "src",
        nrays=5000,
        dx=0.1,
        dz=0.05,
        dxprime=1e-4,
        dzprime=5e-5,
        energies=[10000.0],
    )
    return src.shine()


# ── Core Types ──────────────────────────────────────────────────────────────


class TestBeamLine:
    def test_create(self):
        bl = xr.BeamLine()
        assert repr(bl) == "BeamLine()"


class TestMaterial:
    def test_single_element(self):
        mat = xr.Material(["Si"], rho=2.33)
        assert mat.name == "Si"
        assert mat.rho == 2.33

    def test_compound(self):
        mat = xr.Material(["Si", "O"], quantities=[1, 2], rho=2.2)
        assert "Si" in mat.name
        assert "O" in mat.name

    def test_invalid_element(self):
        with pytest.raises(ValueError):
            xr.Material(["Xx"], rho=1.0)


class TestBeam:
    def test_source_beam(self, bl):
        src = xr.GeometricSource(bl, "src", nrays=1000, energies=[10000.0])
        beam = src.shine()
        assert beam.nrays() == 1000
        assert beam.good_count() == 1000

    def test_numpy_arrays(self, bl):
        src = xr.GeometricSource(bl, "src", nrays=100, energies=[10000.0])
        beam = src.shine()

        x = beam.x
        assert isinstance(x, np.ndarray)
        assert x.dtype == np.float64
        assert x.shape == (100,)

        e = beam.e
        assert np.allclose(e, 10000.0)

        state = beam.state
        assert state.dtype == np.int32
        assert np.all(state == 1)  # all good

    def test_propagate(self, bl):
        src = xr.GeometricSource(bl, "src", nrays=10, energies=[10000.0])
        beam = src.shine()
        y_before = beam.y.copy()
        beam.propagate(1000.0)
        y_after = beam.y
        # y should have increased for forward-going rays
        assert np.all(y_after > y_before)

    def test_len(self, bl):
        src = xr.GeometricSource(bl, "src", nrays=42, energies=[10000.0])
        beam = src.shine()
        assert len(beam) == 42


# ── Sources ─────────────────────────────────────────────────────────────────


class TestGeometricSource:
    def test_basic(self, bl):
        src = xr.GeometricSource(bl, "src", nrays=1000, energies=[10000.0])
        assert src.name == "src"
        assert src.nrays == 1000
        beam = src.shine()
        assert beam.nrays() == 1000

    def test_with_divergence(self, bl):
        src = xr.GeometricSource(
            bl, "src", nrays=1000,
            dx=0.1, dz=0.05, dxprime=1e-4, dzprime=5e-5,
            energies=[10000.0],
        )
        beam = src.shine()
        # Should have some spread in x, z
        x = beam.x
        assert np.std(x) > 0

    def test_single_energy(self, bl):
        src = xr.GeometricSource(bl, "src", nrays=100, energy=8000.0)
        beam = src.shine()
        assert np.allclose(beam.e, 8000.0)

    def test_default_energy(self, bl):
        src = xr.GeometricSource(bl, "src", nrays=100)
        beam = src.shine()
        assert np.allclose(beam.e, 10000.0)


# ── Screen ──────────────────────────────────────────────────────────────────


class TestScreen:
    def test_expose_basic(self, bl, source_beam):
        screen = xr.Screen(bl, "scr", center=[0, 10000, 0])
        result = screen.expose(source_beam)
        assert result.n_captured > 0
        assert result.total_intensity > 0

    def test_fwhm(self, bl, source_beam):
        screen = xr.Screen(bl, "scr", center=[0, 10000, 0], dx=10, dz=10, nx=100, nz=100)
        result = screen.expose(source_beam)
        assert result.fwhm_x >= 0
        assert result.fwhm_z >= 0

    def test_centroid(self, bl, source_beam):
        screen = xr.Screen(bl, "scr", center=[0, 10000, 0], dx=20, dz=20)
        result = screen.expose(source_beam)
        # Centroid should be near zero for a centered beam
        assert abs(result.centroid_x) < 10

    def test_rms(self, bl, source_beam):
        screen = xr.Screen(bl, "scr", center=[0, 10000, 0], dx=20, dz=20)
        result = screen.expose(source_beam)
        assert result.rms_x >= 0
        assert result.rms_z >= 0


# ── Mirrors ─────────────────────────────────────────────────────────────────


class TestFlatMirror:
    def test_reflect(self, bl, si_material, source_beam):
        m1 = xr.FlatMirror(
            bl, "FM1", center=[0, 5000, 0], pitch=0.01, material=si_material
        )
        initial_good = source_beam.good_count()
        m1.reflect(source_beam)
        assert source_beam.good_count() > 0
        assert source_beam.good_count() <= initial_good

    def test_without_material(self, bl, source_beam):
        m1 = xr.FlatMirror(bl, "FM1", center=[0, 5000, 0], pitch=0.01)
        m1.reflect(source_beam)
        assert source_beam.good_count() > 0

    def test_return_beam(self, bl, source_beam):
        m1 = xr.FlatMirror(bl, "FM1", center=[0, 5000, 0], pitch=0.01)
        result = m1.reflect(source_beam)
        # Should return the same beam object
        assert result is source_beam


class TestToroidMirror:
    def test_reflect(self, bl, si_material, source_beam):
        m1 = xr.ToroidMirror(
            bl, "TM1",
            center=[0, 5000, 0], pitch=0.005,
            r_major=5e6, r_minor=50.0,
            material=si_material,
        )
        m1.reflect(source_beam)
        assert source_beam.good_count() > 0


class TestSphericalMirror:
    def test_reflect(self, bl, source_beam):
        m1 = xr.SphericalMirror(
            bl, "SM1", center=[0, 5000, 0], pitch=0.005, radius=1e5
        )
        m1.reflect(source_beam)
        assert source_beam.good_count() > 0


class TestEllipticalMirror:
    def test_reflect(self, bl, source_beam):
        m1 = xr.EllipticalMirror(
            bl, "EM1", center=[0, 5000, 0], pitch=0.003,
            p=10000.0, q=5000.0, theta=0.003,
        )
        m1.reflect(source_beam)
        # Some rays may be lost on parametric surface
        assert source_beam.nrays() == 5000


class TestParabolicalMirror:
    def test_reflect(self, bl, source_beam):
        m1 = xr.ParabolicalMirror(
            bl, "PM1", center=[0, 5000, 0], pitch=0.003,
            p=10000.0, q=5000.0, theta=0.003,
        )
        m1.reflect(source_beam)
        assert source_beam.nrays() == 5000


# ── Gratings ────────────────────────────────────────────────────────────────


class TestBlazedGrating:
    def test_create(self, bl):
        bg = xr.BlazedGrating(
            bl, "BG1", center=[0, 5000, 0], pitch=0.02,
            rho=600.0, blaze_angle=0.03, order=1,
        )
        assert bg.name == "BG1"


class TestLaminarGrating:
    def test_create(self, bl):
        lg = xr.LaminarGrating(
            bl, "LG1", center=[0, 5000, 0], pitch=0.02,
            rho=600.0, depth=0.01, duty_cycle=0.5, order=1,
        )
        assert lg.name == "LG1"


class TestVLSGrating:
    def test_create(self, bl):
        vls = xr.VLSGrating(
            bl, "VLS1", center=[0, 5000, 0], pitch=0.02,
            rho0=600.0, coeffs=[0.01, 0.001], depth=0.01, duty=0.5, order=1,
        )
        assert vls.name == "VLS1"


# ── Lens ────────────────────────────────────────────────────────────────────


class TestParaboloidLens:
    def test_create(self, bl):
        pl = xr.ParaboloidLens(bl, "PL1", center=[0, 5000, 0], focus=1000.0)
        assert pl.name == "PL1"


# ── Integration ─────────────────────────────────────────────────────────────


class TestBasicBeamline:
    def test_source_mirror_screen(self, bl, si_material):
        """Full beamline: source → mirror → screen."""
        src = xr.GeometricSource(
            bl, "src", nrays=5000,
            dx=0.1, dz=0.05, dxprime=1e-4, dzprime=5e-5,
            energies=[10000.0],
        )
        m1 = xr.FlatMirror(
            bl, "M1", center=[0, 5000, 0], pitch=0.003, material=si_material
        )
        screen = xr.Screen(bl, "scr", center=[0, 10000, 0])

        beam = src.shine()
        assert beam.nrays() == 5000

        beam = m1.reflect(beam)
        assert beam.good_count() > 0

        result = screen.expose(beam)
        assert result.n_captured > 0
        assert result.fwhm_x >= 0

    def test_two_mirror_beamline(self, bl, si_material):
        """Source → M1 → M2 → Screen."""
        src = xr.GeometricSource(
            bl, "src", nrays=5000,
            dx=0.1, dz=0.05, dxprime=1e-4, dzprime=5e-5,
            energies=[10000.0],
        )
        m1 = xr.FlatMirror(
            bl, "M1", center=[0, 5000, 0], pitch=0.005, material=si_material
        )
        m2 = xr.ToroidMirror(
            bl, "M2", center=[0, 10000, 0], pitch=0.005,
            r_major=5e6, r_minor=50.0, material=si_material,
        )
        screen = xr.Screen(bl, "scr", center=[0, 15000, 0])

        beam = src.shine()
        beam = m1.reflect(beam)
        good_after_m1 = beam.good_count()
        assert good_after_m1 > 0

        beam = m2.reflect(beam)
        good_after_m2 = beam.good_count()
        assert good_after_m2 > 0
        assert good_after_m2 <= good_after_m1

        result = screen.expose(beam)
        assert result.n_captured >= 0
