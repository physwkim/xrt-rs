//! PyCrystalSi + PyDCM: Python wrappers for crystal diffraction and DCM.
//!
//! Provides Si crystal material and a double-crystal monochromator (DCM)
//! compound element for XRT-compatible beamline simulation.

#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

use xrt_core::consts::CH;
use xrt_materials::crystal::{CrystalBase, CrystalGeometry};
use xrt_materials::crystal_variants::CrystalSi;
use xrt_materials::data::ScatteringTable;
use xrt_oes::crystal_oe::CrystalOpticalElement;
use xrt_oes::oe::OeParams;
use xrt_oes::reflect::DeflectionMode;
use xrt_oes::surfaces::flat::FlatSurface;

use crate::py_beam::PyBeam;

// ─── CrystalSi ───────────────────────────────────────────────────────────────

/// Silicon crystal with temperature-dependent lattice constant.
///
/// Used as the crystal material for DCM and crystal analyzer elements.
///
/// Example:
///     si = CrystalSi(hkl=(1,1,1))
///     bragg = si.get_bragg_angle(10000.0)  # 10 keV
#[pyclass(name = "CrystalSi")]
pub struct PyCrystalSi {
    pub(crate) inner: CrystalSi,
}

#[pymethods]
impl PyCrystalSi {
    /// Create a new Si crystal.
    ///
    /// Args:
    ///     hkl: Miller indices (default (1,1,1))
    ///     t_k: Temperature in Kelvin (default 297.15 = 24°C)
    #[new]
    #[pyo3(signature = (*, hkl=(1,1,1), t_k=297.15))]
    fn new(hkl: (i32, i32, i32), t_k: f64) -> PyResult<Self> {
        let inner = CrystalSi::new(
            [hkl.0, hkl.1, hkl.2],
            t_k,
            CrystalGeometry::BraggReflected,
            1.0,
            None,
            0.0,
            ScatteringTable::ChantlerTotal,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Interplanar d-spacing [Å].
    #[getter]
    fn d(&self) -> f64 {
        self.inner.base.d
    }

    /// Compute Bragg angle [rad] for a given photon energy [eV].
    fn get_bragg_angle(&self, energy: f64) -> f64 {
        (CH / (2.0 * self.inner.base.d * energy)).asin()
    }

    fn __repr__(&self) -> String {
        let hkl = &self.inner.base.hkl;
        format!(
            "CrystalSi(hkl=({},{},{}), d={:.4} Å)",
            hkl[0], hkl[1], hkl[2], self.inner.base.d
        )
    }
}

// ─── DCM (Double Crystal Monochromator) ──────────────────────────────────────

/// Double Crystal Monochromator: two Si crystal reflections in fixed-exit geometry.
///
/// Crystal 1 deflects the beam upward by 2θ_B (Bragg).
/// Crystal 2 deflects it back to horizontal.
/// `cryst2_fine_pitch` controls the angular mismatch → intensity via rocking curve.
///
/// Example:
///     si = CrystalSi(hkl=(1,1,1))
///     dcm = DCM(bl, 'DCM', center=[0, 19500, 0], crystal=si,
///               bragg=si.get_bragg_angle(10000.0), fixed_exit=20.0)
#[pyclass(name = "DCM")]
pub struct PyDCM {
    name: String,
    center: [f64; 3],
    bragg: f64,
    fine_pitch: f64,
    fixed_exit: f64,
    crystal_base: CrystalBase,
    sf: CrystalSi,
    crystal1: CrystalOpticalElement<FlatSurface>,
    crystal2: CrystalOpticalElement<FlatSurface>,
}

impl PyDCM {
    fn make_crystal2(
        center: [f64; 3],
        bragg: f64,
        fine_pitch: f64,
        fixed_exit: f64,
        crystal_base: &CrystalBase,
    ) -> CrystalOpticalElement<FlatSurface> {
        let gap_y = fixed_exit / (2.0 * bragg).tan();
        let params = OeParams {
            center: [center[0], center[1] + gap_y, center[2] + fixed_exit],
            pitch: bragg - fine_pitch,
            mode: DeflectionMode::Reflect,
            ..Default::default()
        };
        CrystalOpticalElement::new(FlatSurface, params, crystal_base.clone())
    }
}

#[pymethods]
impl PyDCM {
    /// Create a new DCM.
    ///
    /// Args:
    ///     bl: BeamLine (for API compatibility)
    ///     name: Element name
    ///     center: Position of crystal 1 [x, y, z] in mm
    ///     crystal: CrystalSi instance
    ///     bragg: Bragg angle [rad]
    ///     cryst2_fine_pitch: Fine pitch offset on crystal 2 [rad] (default 0)
    ///     fixed_exit: Vertical beam offset between in/out beams [mm] (default 20)
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        center=vec![0.0, 0.0, 0.0],
        crystal,
        bragg,
        cryst2_fine_pitch=0.0,
        fixed_exit=20.0,
    ))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        crystal: &PyCrystalSi,
        bragg: f64,
        cryst2_fine_pitch: f64,
        fixed_exit: f64,
    ) -> PyResult<Self> {
        if center.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "center must have 3 elements [x, y, z]",
            ));
        }
        let c = [center[0], center[1], center[2]];

        // Crystal 1: at center position, pitch = bragg
        let params1 = OeParams {
            center: c,
            pitch: bragg,
            mode: DeflectionMode::Reflect,
            ..Default::default()
        };
        let crystal1 = CrystalOpticalElement::new(FlatSurface, params1, crystal.inner.base.clone());

        // Crystal 2: offset by fixed_exit
        let crystal2 =
            Self::make_crystal2(c, bragg, cryst2_fine_pitch, fixed_exit, &crystal.inner.base);

        Ok(Self {
            name: name.to_string(),
            center: c,
            bragg,
            fine_pitch: cryst2_fine_pitch,
            fixed_exit,
            crystal_base: crystal.inner.base.clone(),
            sf: crystal.inner.clone(),
            crystal1,
            crystal2,
        })
    }

    /// Perform double Bragg reflection through both crystals.
    ///
    /// Returns the same beam object (mutated in place).
    fn double_reflect(&self, beam: Py<PyBeam>, py: Python<'_>) -> PyResult<Py<PyBeam>> {
        {
            let bound = beam.bind(py);
            let mut guard = bound.borrow_mut();
            self.crystal1.reflect(&mut guard.inner, &self.sf);
            self.crystal2.reflect(&mut guard.inner, &self.sf);
        }
        Ok(beam)
    }

    /// Set the fine pitch offset on crystal 2.
    #[setter]
    fn set_cryst2_fine_pitch(&mut self, value: f64) {
        self.fine_pitch = value;
        self.crystal2 = Self::make_crystal2(
            self.center,
            self.bragg,
            value,
            self.fixed_exit,
            &self.crystal_base,
        );
    }

    /// Get the fine pitch offset on crystal 2.
    #[getter]
    fn cryst2_fine_pitch(&self) -> f64 {
        self.fine_pitch
    }

    /// Bragg angle [rad].
    #[getter]
    fn bragg(&self) -> f64 {
        self.bragg
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!(
            "DCM('{}', bragg={:.4} rad, fine_pitch={:.2e} rad)",
            self.name, self.bragg, self.fine_pitch
        )
    }
}
