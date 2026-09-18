//! PyGeometricSource: Python wrapper for GeometricSource.

use pyo3::prelude::*;

use xrt_rs::sources::distributions::{EnergyDist, SpatialDist};
use xrt_rs::sources::geometric::GeometricSource;
use xrt_rs::sources::polarization::Polarization;

use crate::py_beam::PyBeam;

/// A geometric (parametric) X-ray source.
///
/// Generates rays with configurable position, angle, energy,
/// and polarization distributions.
#[pyclass(name = "GeometricSource")]
pub struct PyGeometricSource {
    inner: GeometricSource,
    name: String,
}

#[pymethods]
impl PyGeometricSource {
    /// Create a new GeometricSource.
    ///
    /// Args:
    ///     bl: BeamLine (for API compatibility)
    ///     name: Source name
    ///     nrays: Number of rays (default 10000)
    ///     dx: Horizontal source size sigma [mm]
    ///     dz: Vertical source size sigma [mm]
    ///     dxprime: Horizontal divergence sigma [rad]
    ///     dzprime: Vertical divergence sigma [rad]
    ///     energies: List of photon energies [eV]
    ///     energy: Single photon energy [eV] (alternative to energies)
    ///     polarization: 'horizontal', 'vertical', 'plus45', 'minus45', 'left', 'right'
    #[new]
    #[pyo3(signature = (
        _bl, name, *,
        nrays=10000,
        dx=0.0, dz=0.0,
        dxprime=0.0, dzprime=0.0,
        energies=None, energy=None,
        polarization="horizontal"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        nrays: usize,
        dx: f64,
        dz: f64,
        dxprime: f64,
        dzprime: f64,
        energies: Option<Vec<f64>>,
        energy: Option<f64>,
        polarization: &str,
    ) -> PyResult<Self> {
        let dist_e = match (energies, energy) {
            (Some(es), _) => EnergyDist::Lines(es, None),
            (None, Some(e)) => EnergyDist::Lines(vec![e], None),
            (None, None) => EnergyDist::Lines(vec![10000.0], None),
        };

        let pol = match polarization.to_lowercase().as_str() {
            "vertical" | "v" => Polarization::Vertical,
            "plus45" | "+45" => Polarization::Plus45,
            "minus45" | "-45" => Polarization::Minus45,
            "left" | "lcp" => Polarization::Left,
            "right" | "rcp" => Polarization::Right,
            _ => Polarization::Horizontal,
        };

        let inner = GeometricSource {
            nrays,
            dist_x: if dx > 0.0 {
                SpatialDist::Normal(dx)
            } else {
                SpatialDist::None
            },
            dist_z: if dz > 0.0 {
                SpatialDist::Normal(dz)
            } else {
                SpatialDist::None
            },
            dist_xprime: if dxprime > 0.0 {
                SpatialDist::Normal(dxprime)
            } else {
                SpatialDist::None
            },
            dist_zprime: if dzprime > 0.0 {
                SpatialDist::Normal(dzprime)
            } else {
                SpatialDist::None
            },
            dist_e,
            polarization: pol,
            ..Default::default()
        };

        Ok(Self {
            inner,
            name: name.to_string(),
        })
    }

    /// Generate a beam from this source.
    fn shine(&self) -> PyResult<PyBeam> {
        let beam = self.inner.shine();
        Ok(PyBeam { inner: beam })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn nrays(&self) -> usize {
        self.inner.nrays
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometricSource('{}', nrays={})",
            self.name, self.inner.nrays
        )
    }
}
