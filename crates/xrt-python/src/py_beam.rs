//! PyBeam: Python wrapper for the Beam ray container.

use numpy::PyArray1;
use pyo3::prelude::*;

use xrt_core::beam::Beam;

#[pyclass(name = "Beam")]
pub struct PyBeam {
    pub(crate) inner: Beam,
}

#[pymethods]
impl PyBeam {
    /// Horizontal position array [mm] (copy).
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.x.to_vec())
    }

    /// Longitudinal position array [mm] (copy).
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.y.to_vec())
    }

    /// Vertical position array [mm] (copy).
    #[getter]
    fn z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.z.to_vec())
    }

    /// Horizontal direction cosine (copy).
    #[getter]
    fn a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.a.to_vec())
    }

    /// Longitudinal direction cosine (copy).
    #[getter]
    fn b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.b.to_vec())
    }

    /// Vertical direction cosine (copy).
    #[getter]
    fn c<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.c.to_vec())
    }

    /// Energy array [eV] (copy).
    #[getter]
    fn e<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.e.to_vec())
    }

    /// Ray state array (copy). 1=good, 2=out, 3=over, -1=dead, 0=undefined.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_vec(py, self.inner.state.to_vec())
    }

    /// s-polarization coherency (copy).
    #[getter]
    fn jss<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.jss.to_vec())
    }

    /// p-polarization coherency (copy).
    #[getter]
    fn jpp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.jpp.to_vec())
    }

    /// Total number of rays.
    fn nrays(&self) -> usize {
        self.inner.nrays()
    }

    /// Number of good (alive) rays.
    fn good_count(&self) -> usize {
        self.inner.good_indices().len()
    }

    /// Propagate good rays through free space by `distance` mm.
    fn propagate(&mut self, distance: f64) {
        self.inner.propagate(distance);
    }

    fn __repr__(&self) -> String {
        let n = self.inner.nrays();
        let g = self.inner.good_indices().len();
        format!("Beam(nrays={n}, good={g})")
    }

    fn __len__(&self) -> usize {
        self.inner.nrays()
    }
}
