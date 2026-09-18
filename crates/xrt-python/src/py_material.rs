//! PyMaterial: Python wrapper for Material (Fresnel reflectivity).

use pyo3::prelude::*;

use xrt_rs::materials::data::ScatteringTable;
use xrt_rs::materials::material::{Material, MaterialKind};

#[pyclass(name = "Material")]
pub struct PyMaterial {
    pub(crate) inner: Material,
}

#[pymethods]
impl PyMaterial {
    /// Create a new Material.
    ///
    /// Args:
    ///     elements: Element symbols, e.g. ['Si'] or ['Si', 'O']
    ///     quantities: Stoichiometric coefficients, e.g. [1, 2] for SiO2
    ///     rho: Density in g/cm^3
    ///     kind: Material kind ('mirror', 'plate', 'lens', 'grating', 'fzp')
    ///     thickness: Thickness in mm (for thin mirrors)
    #[new]
    #[pyo3(signature = (elements, *, quantities=None, rho, kind="mirror", thickness=None))]
    fn new(
        elements: Vec<String>,
        quantities: Option<Vec<f64>>,
        rho: f64,
        kind: &str,
        thickness: Option<f64>,
    ) -> PyResult<Self> {
        let elem_refs: Vec<&str> = elements.iter().map(|s| s.as_str()).collect();
        let mat_kind = MaterialKind::from_str_xrt(kind);
        let inner = Material::new(
            &elem_refs,
            quantities.as_deref(),
            rho,
            mat_kind,
            thickness,
            ScatteringTable::ChantlerTotal,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Material formula name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Density [g/cm^3].
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }

    fn __repr__(&self) -> String {
        format!("Material('{}', rho={})", self.inner.name, self.inner.rho)
    }
}
