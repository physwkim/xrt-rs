//! Python optical element classes.
//!
//! Each OE type is a standalone `#[pyclass]` wrapping an `OeInner` enum
//! that dispatches to the correct Rust OE type for reflection.

#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

use xrt_core::beam::Beam;
use xrt_oes::grating_oe::GratingOpticalElement;
use xrt_oes::material_oe::MaterialOpticalElement;
use xrt_oes::oe::{OeParams, OpticalElement};
use xrt_oes::param_oe::ParametricOpticalElement;
use xrt_oes::reflect::DeflectionMode;

use xrt_oes::surfaces::bent_flat::BentFlatSurface;
use xrt_oes::surfaces::cylindrical::CylindricalSurface;
use xrt_oes::surfaces::elliptical::EllipticalSurface;
use xrt_oes::surfaces::flat::FlatSurface;
use xrt_oes::surfaces::grating::{BlazedGrating, LaminarGrating};
use xrt_oes::surfaces::lens::ParaboloidLensSurface;
use xrt_oes::surfaces::parabolical::ParabolicalSurface;
use xrt_oes::surfaces::spherical::SphericalSurface;
use xrt_oes::surfaces::toroid::ToroidSurface;
use xrt_oes::surfaces::vfm::VfmSurface;
use xrt_oes::surfaces::vls_grating::VlsGrating;

use crate::py_beam::PyBeam;
use crate::py_material::PyMaterial;
use crate::surface_enum::{ParametricSurfaceEnum, SurfaceEnum};

// ─── Internal dispatch enum ────────────────────────────────────────────────

pub(crate) enum OeInner {
    Material(MaterialOpticalElement<SurfaceEnum>),
    Bare(OpticalElement<SurfaceEnum>),
    Grating(GratingOpticalElement<SurfaceEnum>),
    Parametric(ParametricOpticalElement<ParametricSurfaceEnum>),
}

impl OeInner {
    pub fn reflect(&self, beam: &mut Beam) {
        match self {
            Self::Material(oe) => {
                oe.reflect(beam);
            }
            Self::Bare(oe) => {
                oe.reflect(beam);
            }
            Self::Grating(oe) => {
                oe.reflect(beam);
            }
            Self::Parametric(oe) => {
                oe.reflect(beam);
            }
        }
    }

    pub fn set_pitch(&mut self, pitch: f64) {
        match self {
            Self::Material(oe) => oe.params.pitch = pitch,
            Self::Bare(oe) => oe.params.pitch = pitch,
            Self::Grating(oe) => oe.params.pitch = pitch,
            Self::Parametric(oe) => oe.params.pitch = pitch,
        }
    }

    pub fn set_roll(&mut self, roll: f64) {
        match self {
            Self::Material(oe) => oe.params.roll = roll,
            Self::Bare(oe) => oe.params.roll = roll,
            Self::Grating(oe) => oe.params.roll = roll,
            Self::Parametric(oe) => oe.params.roll = roll,
        }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn parse_center(center: &[f64]) -> PyResult<[f64; 3]> {
    if center.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "center must have 3 elements [x, y, z]",
        ));
    }
    Ok([center[0], center[1], center[2]])
}

fn make_params(
    center: [f64; 3],
    pitch: f64,
    roll: f64,
    yaw: f64,
    mode: DeflectionMode,
) -> OeParams {
    OeParams {
        center,
        pitch,
        roll,
        yaw,
        mode,
        ..Default::default()
    }
}

fn make_mirror_oe(
    surface: SurfaceEnum,
    params: OeParams,
    material: Option<PyRef<'_, PyMaterial>>,
) -> OeInner {
    match material {
        Some(m) => OeInner::Material(MaterialOpticalElement::new(surface, params, m.inner.clone())),
        None => OeInner::Bare(OpticalElement::new(surface, params)),
    }
}

/// Shared reflect logic: borrows beam mutably, runs reflection, returns beam.
fn oe_reflect(oe: &OeInner, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
    {
        let bound = beam.bind(py);
        let mut guard = bound.borrow_mut();
        oe.reflect(&mut guard.inner);
    }
    Ok(beam)
}

// ─── FlatMirror ────────────────────────────────────────────────────────────

#[pyclass(name = "FlatMirror")]
pub struct PyFlatMirror {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyFlatMirror {
    #[new]
    #[pyo3(signature = (_bl, name, *, center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0, material=None))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface = SurfaceEnum::Flat(FlatSurface);
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = make_mirror_oe(surface, params, material);
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    /// Set pitch angle [rad] (for runtime adjustment, e.g. RL alignment).
    #[setter]
    fn set_pitch(&mut self, pitch: f64) {
        self.oe.set_pitch(pitch);
    }

    /// Set roll angle [rad].
    #[setter]
    fn set_roll(&mut self, roll: f64) {
        self.oe.set_roll(roll);
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("FlatMirror('{}')", self.name)
    }
}

// ─── ToroidMirror ──────────────────────────────────────────────────────────

/// Toroidal mirror with separate meridional (R) and sagittal (r) radii.
#[pyclass(name = "ToroidMirror")]
pub struct PyToroidMirror {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyToroidMirror {
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        r_major=1e6, r_minor=50.0,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        r_major: f64,
        r_minor: f64,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface = SurfaceEnum::Toroid(ToroidSurface::new(r_major, r_minor));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = make_mirror_oe(surface, params, material);
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("ToroidMirror('{}')", self.name)
    }
}

// ─── SphericalMirror ───────────────────────────────────────────────────────

#[pyclass(name = "SphericalMirror")]
pub struct PySphericalMirror {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PySphericalMirror {
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        radius=1e6,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        radius: f64,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface = SurfaceEnum::Spherical(SphericalSurface::new(radius));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = make_mirror_oe(surface, params, material);
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("SphericalMirror('{}')", self.name)
    }
}

// ─── CylindricalMirror ────────────────────────────────────────────────────

#[pyclass(name = "CylindricalMirror")]
pub struct PyCylindricalMirror {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyCylindricalMirror {
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        radius=1e6,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        radius: f64,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface = SurfaceEnum::Cylindrical(CylindricalSurface::new(radius));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = make_mirror_oe(surface, params, material);
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("CylindricalMirror('{}')", self.name)
    }
}

// ─── BentFlatMirror ────────────────────────────────────────────────────────

#[pyclass(name = "BentFlatMirror")]
pub struct PyBentFlatMirror {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyBentFlatMirror {
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        radius=1e6, y_min=-50.0,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        radius: f64,
        y_min: f64,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface = SurfaceEnum::BentFlat(BentFlatSurface::new(radius, y_min));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = make_mirror_oe(surface, params, material);
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("BentFlatMirror('{}')", self.name)
    }
}

// ─── VFM (Variable-radius Focusing Mirror) ─────────────────────────────────

#[pyclass(name = "VFM")]
pub struct PyVFM {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyVFM {
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        r_major=1e6, r_minor=50.0, y_min=-50.0,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        r_major: f64,
        r_minor: f64,
        y_min: f64,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface = SurfaceEnum::Vfm(VfmSurface::new(r_major, r_minor, y_min));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = make_mirror_oe(surface, params, material);
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("VFM('{}')", self.name)
    }
}

// ─── EllipticalMirror (parametric) ─────────────────────────────────────────

#[pyclass(name = "EllipticalMirror")]
pub struct PyEllipticalMirror {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyEllipticalMirror {
    /// Create an EllipticalMirror from conjugate distances p, q and grazing angle theta.
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        p=10000.0, q=5000.0, theta=0.003
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        p: f64,
        q: f64,
        theta: f64,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface =
            ParametricSurfaceEnum::Elliptical(EllipticalSurface::from_pq(p, q, theta));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = OeInner::Parametric(ParametricOpticalElement::new(surface, params));
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("EllipticalMirror('{}')", self.name)
    }
}

// ─── ParabolicalMirror (parametric) ────────────────────────────────────────

#[pyclass(name = "ParabolicalMirror")]
pub struct PyParabolicalMirror {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyParabolicalMirror {
    /// Create a ParabolicalMirror from conjugate distances p, q and grazing angle theta.
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        p=10000.0, q=5000.0, theta=0.003
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        p: f64,
        q: f64,
        theta: f64,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface =
            ParametricSurfaceEnum::Parabolical(ParabolicalSurface::from_pq(p, q, theta));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Reflect);
        let oe = OeInner::Parametric(ParametricOpticalElement::new(surface, params));
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("ParabolicalMirror('{}')", self.name)
    }
}

// ─── BlazedGrating ─────────────────────────────────────────────────────────

#[pyclass(name = "BlazedGrating")]
pub struct PyBlazedGrating {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyBlazedGrating {
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        rho=600.0, blaze_angle=0.03, anti_blaze_angle=0.06,
        order=1,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        rho: f64,
        blaze_angle: f64,
        anti_blaze_angle: f64,
        order: i32,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface =
            SurfaceEnum::BlazedGrating(BlazedGrating::new(rho, blaze_angle, anti_blaze_angle));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Grating { order });
        let mut oe = GratingOpticalElement::new(surface, params, order);
        if let Some(m) = material {
            oe = oe.with_material(m.inner.clone());
        }
        Ok(Self {
            name: name.to_string(),
            oe: OeInner::Grating(oe),
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("BlazedGrating('{}')", self.name)
    }
}

// ─── LaminarGrating ────────────────────────────────────────────────────────

#[pyclass(name = "LaminarGrating")]
pub struct PyLaminarGrating {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyLaminarGrating {
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        rho=600.0, depth=0.01, duty_cycle=0.5,
        order=1,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        rho: f64,
        depth: f64,
        duty_cycle: f64,
        order: i32,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface =
            SurfaceEnum::LaminarGrating(LaminarGrating::new(rho, depth, duty_cycle));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Grating { order });
        let mut oe = GratingOpticalElement::new(surface, params, order);
        if let Some(m) = material {
            oe = oe.with_material(m.inner.clone());
        }
        Ok(Self {
            name: name.to_string(),
            oe: OeInner::Grating(oe),
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("LaminarGrating('{}')", self.name)
    }
}

// ─── VLSGrating ────────────────────────────────────────────────────────────

#[pyclass(name = "VLSGrating")]
pub struct PyVLSGrating {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyVLSGrating {
    /// VLS (Variable Line Spacing) grating.
    ///
    /// rho0: base groove density [lines/mm]
    /// coeffs: VLS polynomial coefficients [c1, c2, ...]
    /// depth: groove depth [mm]
    /// duty: duty cycle (0..1)
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        rho0=600.0, coeffs=vec![], depth=0.01, duty=0.5,
        order=1,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        rho0: f64,
        coeffs: Vec<f64>,
        depth: f64,
        duty: f64,
        order: i32,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface =
            SurfaceEnum::VlsGrating(VlsGrating::new(rho0, coeffs, depth, duty));
        let params = make_params(c, pitch, roll, yaw, DeflectionMode::Grating { order });
        let mut oe = GratingOpticalElement::new(surface, params, order);
        if let Some(m) = material {
            oe = oe.with_material(m.inner.clone());
        }
        Ok(Self {
            name: name.to_string(),
            oe: OeInner::Grating(oe),
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("VLSGrating('{}')", self.name)
    }
}

// ─── ParaboloidLens ────────────────────────────────────────────────────────

/// Compound Refractive Lens element (single paraboloid surface).
#[pyclass(name = "ParaboloidLens")]
pub struct PyParaboloidLens {
    name: String,
    oe: OeInner,
}

#[pymethods]
impl PyParaboloidLens {
    /// Create a ParaboloidLens.
    ///
    /// Args:
    ///     focus: Focal length [mm]
    ///     z_max: Maximum sag / clipping height [mm] (optional)
    ///     n_ratio: Refractive index ratio n1/n2 for Snell's law
    ///     material: Lens material (required for proper refraction)
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0], pitch=0.0, roll=0.0, yaw=0.0,
        focus=1000.0, z_max=None,
        n_ratio=0.999995,
        material=None
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        pitch: f64,
        roll: f64,
        yaw: f64,
        focus: f64,
        z_max: Option<f64>,
        n_ratio: f64,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        let c = parse_center(&center)?;
        let surface = SurfaceEnum::ParaboloidLens(ParaboloidLensSurface::new(focus, z_max));
        let mode = DeflectionMode::Refract {
            n1_over_n2: n_ratio,
        };
        let params = make_params(c, pitch, roll, yaw, mode);
        let oe = match material {
            Some(m) => {
                OeInner::Material(MaterialOpticalElement::new(surface, params, m.inner.clone()))
            }
            None => OeInner::Bare(OpticalElement::new(surface, params)),
        };
        Ok(Self {
            name: name.to_string(),
            oe,
        })
    }

    fn reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        oe_reflect(&self.oe, beam, py)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("ParaboloidLens('{}')", self.name)
    }
}
