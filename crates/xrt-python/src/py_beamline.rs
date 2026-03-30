//! PyBeamLine: lightweight beamline container for XRT API compatibility.

use pyo3::prelude::*;

/// A beamline container.
///
/// In the current implementation this is a lightweight marker object
/// passed to OE/Source/Screen constructors for XRT API compatibility.
#[pyclass(name = "BeamLine")]
pub struct PyBeamLine {}

#[pymethods]
impl PyBeamLine {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn __repr__(&self) -> String {
        "BeamLine()".to_string()
    }
}
