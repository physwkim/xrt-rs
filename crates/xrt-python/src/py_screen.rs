//! PyScreen + PyScreenCapture: Python wrappers for Screen/detector.

use pyo3::prelude::*;

use xrt_oes::screen::{Screen, ScreenCapture};

use crate::py_beam::PyBeam;

/// A detector screen that captures beam footprint.
#[pyclass(name = "Screen")]
pub struct PyScreen {
    pub(crate) inner: Screen,
    pub(crate) name: String,
}

#[pymethods]
impl PyScreen {
    /// Create a new Screen.
    ///
    /// Args:
    ///     bl: BeamLine (for API compatibility)
    ///     name: Screen name
    ///     center: Position [x, y, z] in mm
    ///     dx: Half-width in x [mm] (default 50)
    ///     dz: Half-width in z [mm] (default 50)
    ///     nx: Number of bins in x (default 200)
    ///     nz: Number of bins in z (default 200)
    #[new]
    #[pyo3(signature = (_bl, name, *, center, dx=50.0, dz=50.0, nx=200, nz=200))]
    fn new(
        _bl: &Bound<'_, PyAny>,
        name: &str,
        center: Vec<f64>,
        dx: f64,
        dz: f64,
        nx: usize,
        nz: usize,
    ) -> PyResult<Self> {
        if center.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "center must have 3 elements [x, y, z]",
            ));
        }
        let inner = Screen::new([center[0], center[1], center[2]], dx, dz, nx, nz);
        Ok(Self {
            inner,
            name: name.to_string(),
        })
    }

    /// Capture beam footprint on this screen.
    ///
    /// Returns a ScreenCapture with intensity distribution and statistics.
    fn expose(&self, beam: PyRef<'_, PyBeam>) -> PyResult<PyScreenCapture> {
        let capture = self.inner.capture(&beam.inner);
        Ok(PyScreenCapture { inner: capture })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!(
            "Screen('{}', center=[{:.1}, {:.1}, {:.1}])",
            self.name, self.inner.center[0], self.inner.center[1], self.inner.center[2]
        )
    }
}

/// Result of capturing a beam on a screen.
#[pyclass(name = "ScreenCapture")]
pub struct PyScreenCapture {
    pub(crate) inner: ScreenCapture,
}

#[pymethods]
impl PyScreenCapture {
    /// FWHM of x-projection [mm].
    #[getter]
    fn fwhm_x(&self) -> f64 {
        self.inner.fwhm_x().unwrap_or(0.0)
    }

    /// FWHM of z-projection [mm].
    #[getter]
    fn fwhm_z(&self) -> f64 {
        self.inner.fwhm_z().unwrap_or(0.0)
    }

    /// Centroid x position [mm].
    #[getter]
    fn centroid_x(&self) -> f64 {
        self.inner.centroid()[0]
    }

    /// Centroid z position [mm].
    #[getter]
    fn centroid_z(&self) -> f64 {
        self.inner.centroid()[1]
    }

    /// RMS beam size in x [mm].
    #[getter]
    fn rms_x(&self) -> f64 {
        self.inner.rms_size()[0]
    }

    /// RMS beam size in z [mm].
    #[getter]
    fn rms_z(&self) -> f64 {
        self.inner.rms_size()[1]
    }

    /// Number of captured rays.
    #[getter]
    fn n_captured(&self) -> usize {
        self.inner.n_captured
    }

    /// Number of missed rays (outside screen).
    #[getter]
    fn n_missed(&self) -> usize {
        self.inner.n_missed
    }

    /// Total integrated intensity.
    #[getter]
    fn total_intensity(&self) -> f64 {
        self.inner.total_intensity()
    }

    /// Peak intensity.
    #[getter]
    fn peak_intensity(&self) -> f64 {
        self.inner.peak_intensity()
    }

    fn __repr__(&self) -> String {
        format!(
            "ScreenCapture(captured={}, fwhm_x={:.4}, fwhm_z={:.4})",
            self.inner.n_captured,
            self.inner.fwhm_x().unwrap_or(0.0),
            self.inner.fwhm_z().unwrap_or(0.0)
        )
    }
}
