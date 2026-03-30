#![allow(clippy::too_many_arguments, clippy::type_complexity)]
//! PyO3 bindings for xrt-rs.
//!
//! Provides both low-level functions (find_intersection_rs, etc.) and
//! high-level Python classes (BeamLine, GeometricSource, OE types, Screen)
//! for XRT-compatible beamline simulation.
//!
//! # High-level usage
//! ```python
//! import xrt_rs as xr
//!
//! bl = xr.BeamLine()
//! source = xr.GeometricSource(bl, 'source', nrays=10000, energies=[10000.0])
//! m1 = xr.ToroidMirror(bl, 'M1', center=[0, 5000, 0],
//!     pitch=0.005, R=5e6, r=50.0, material=xr.Material(['Si'], rho=2.33))
//! screen = xr.Screen(bl, 'screen', center=[0, 10000, 0])
//!
//! beam = source.shine()
//! m1.reflect(beam)
//! result = screen.expose(beam)
//! print(f"FWHM: {result.fwhm_x:.3f} x {result.fwhm_z:.3f} mm")
//! ```

mod py_beam;
mod py_beamline;
mod py_material;
mod py_oe;
mod py_screen;
mod py_source;
mod surface_enum;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use ndarray::Array1;
use num_complex::Complex64;

use xrt_math::rootfind::RootFindConfig;
use xrt_oes::intersection::find_intersection_surface;
use xrt_oes::surfaces::flat::FlatSurface;
use xrt_oes::surfaces::toroid::ToroidSurface;
use xrt_oes::surfaces::spherical::SphericalSurface;
use xrt_oes::surfaces::lens::ParaboloidLensSurface;
use xrt_oes::surfaces::grating::{BlazedGrating, LaminarGrating};
use xrt_oes::surfaces::fzp::FzpSurface;

/// Find ray-surface intersection for a batch of rays.
///
/// Supported surface types:
///   "flat", "toroid", "spherical", "paraboloid_lens",
///   "blazed_grating", "laminar_grating", "fzp"
///
/// Returns:
///     (t_out, x_out, y_out, z_out): intersection results
#[pyfunction]
fn find_intersection_rs(
    surface_type: &str,
    params: &Bound<'_, PyDict>,
    t1: Vec<f64>,
    t2: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    invert_normal: i32,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = x.len();
    let config = RootFindConfig::default();

    let mut t_out = vec![0.0; n];
    let mut x_out = vec![0.0; n];
    let mut y_out = vec![0.0; n];
    let mut z_out = vec![0.0; n];

    macro_rules! solve_surface {
        ($surface:expr) => {
            for i in 0..n {
                let result = find_intersection_surface(
                    &$surface,
                    t1[i], t2[i],
                    x[i], y[i], z[i],
                    a[i], b[i], c[i],
                    invert_normal, &config,
                );
                t_out[i] = result.t;
                x_out[i] = result.x;
                y_out[i] = result.y;
                z_out[i] = result.z;
            }
        };
    }

    match surface_type {
        "flat" => {
            solve_surface!(FlatSurface);
        }
        "toroid" => {
            let r_major: f64 = params
                .get_item("R")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'R'"))?
                .extract()?;
            let r_minor: f64 = params
                .get_item("r")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'r'"))?
                .extract()?;
            solve_surface!(ToroidSurface::new(r_major, r_minor));
        }
        "spherical" => {
            let r: f64 = params
                .get_item("R")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'R'"))?
                .extract()?;
            solve_surface!(SphericalSurface::new(r));
        }
        "paraboloid_lens" => {
            let focus: f64 = params
                .get_item("focus")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'focus'"))?
                .extract()?;
            let z_max: Option<f64> = params
                .get_item("zmax")?
                .map(|v| v.extract())
                .transpose()?;
            solve_surface!(ParaboloidLensSurface::new(focus, z_max));
        }
        "blazed_grating" => {
            let rho: f64 = params
                .get_item("rho")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'rho'"))?
                .extract()?;
            let blaze: f64 = params
                .get_item("blaze")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'blaze'"))?
                .extract()?;
            let anti_blaze: f64 = params
                .get_item("antiBlaze")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'antiBlaze'"))?
                .extract()?;
            solve_surface!(BlazedGrating::new(rho, blaze, anti_blaze));
        }
        "laminar_grating" => {
            let rho: f64 = params
                .get_item("rho")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'rho'"))?
                .extract()?;
            let depth: f64 = params
                .get_item("depth")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'depth'"))?
                .extract()?;
            let duty: f64 = params
                .get_item("duty")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'duty'"))?
                .extract()?;
            solve_surface!(LaminarGrating::new(rho, depth, duty));
        }
        "fzp" => {
            let focus: f64 = params
                .get_item("focus")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'focus'"))?
                .extract()?;
            let wavelength: f64 = params
                .get_item("wavelength")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'wavelength'"))?
                .extract()?;
            let n_zones: usize = params
                .get_item("nZones")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'nZones'"))?
                .extract()?;
            solve_surface!(FzpSurface::new(focus, wavelength, n_zones));
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown surface type: {}",
                surface_type
            )));
        }
    }

    Ok((t_out, x_out, y_out, z_out))
}

/// Find parametric ray-surface intersection for elliptical, parabolical, hyperbolic.
///
/// Returns (s_out, phi_out, r_out, converged).
#[pyfunction]
fn find_intersection_parametric_rs(
    surface_type: &str,
    params: &Bound<'_, PyDict>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    use xrt_oes::surfaces::elliptical::EllipticalSurface;
    use xrt_oes::surfaces::parabolical::ParabolicalSurface;
    use xrt_oes::surfaces::hyperbolic::HyperbolicSurface;
    use xrt_oes::surface::ParametricSurface;

    let n = x.len();
    let mut s_out = vec![0.0; n];
    let mut phi_out = vec![0.0; n];
    let mut r_out = vec![0.0; n];

    macro_rules! convert_parametric {
        ($surface:expr) => {
            for i in 0..n {
                let (s, phi, r) = $surface.xyz_to_param(x[i], y[i], z[i]);
                s_out[i] = s;
                phi_out[i] = phi;
                r_out[i] = r;
            }
        };
    }

    match surface_type {
        "elliptical" => {
            let a: f64 = params.get_item("a")?.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'a'"))?.extract()?;
            let b: f64 = params.get_item("b")?.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'b'"))?.extract()?;
            let y0: f64 = params.get_item("y0")?.map(|v| v.extract()).transpose()?.unwrap_or(0.0);
            convert_parametric!(EllipticalSurface::new(a, b, y0));
        }
        "parabolical" => {
            let p: f64 = params.get_item("p")?.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'p'"))?.extract()?;
            let y0: f64 = params.get_item("y0")?.map(|v| v.extract()).transpose()?.unwrap_or(0.0);
            convert_parametric!(ParabolicalSurface::new(p, y0));
        }
        "hyperbolic" => {
            let a: f64 = params.get_item("a")?.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'a'"))?.extract()?;
            let b: f64 = params.get_item("b")?.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing 'b'"))?.extract()?;
            let y0: f64 = params.get_item("y0")?.map(|v| v.extract()).transpose()?.unwrap_or(0.0);
            convert_parametric!(HyperbolicSurface::new(a, b, y0));
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown parametric surface type: {}",
                surface_type
            )));
        }
    }

    Ok((s_out, phi_out, r_out))
}

/// CPU Kirchhoff diffraction integral.
///
/// Args:
///     ray_x, ray_y, ray_z: ray positions on OE [mm]
///     ray_nx, ray_ny, ray_nz: surface normals
///     ray_nl: obliquity factors
///     ray_es_re, ray_es_im: s-pol amplitude (real, imag)
///     ray_ep_re, ray_ep_im: p-pol amplitude (real, imag)
///     ray_energy: energies [eV]
///     pix_x, pix_y, pix_z: pixel positions [mm]
///
/// Returns:
///     (es_re, es_im, ep_re, ep_im): field amplitudes at pixels
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn diffraction_integral_rs(
    ray_x: Vec<f64>, ray_y: Vec<f64>, ray_z: Vec<f64>,
    ray_nx: Vec<f64>, ray_ny: Vec<f64>, ray_nz: Vec<f64>,
    ray_nl: Vec<f64>,
    ray_es_re: Vec<f64>, ray_es_im: Vec<f64>,
    ray_ep_re: Vec<f64>, ray_ep_im: Vec<f64>,
    ray_energy: Vec<f64>,
    pix_x: Vec<f64>, pix_y: Vec<f64>, pix_z: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    use xrt_waves::diffraction::{DiffractionRay, PixelPoint, diffraction_integral};

    let n_rays = ray_x.len();
    let rays: Vec<DiffractionRay> = (0..n_rays)
        .map(|i| DiffractionRay {
            x: ray_x[i], y: ray_y[i], z: ray_z[i],
            nx: ray_nx[i], ny: ray_ny[i], nz: ray_nz[i],
            nl: ray_nl[i],
            es: Complex64::new(ray_es_re[i], ray_es_im[i]),
            ep: Complex64::new(ray_ep_re[i], ray_ep_im[i]),
            energy: ray_energy[i],
        })
        .collect();

    let n_pixels = pix_x.len();
    let pixels: Vec<PixelPoint> = (0..n_pixels)
        .map(|i| PixelPoint { x: pix_x[i], y: pix_y[i], z: pix_z[i] })
        .collect();

    let results = diffraction_integral(&rays, &pixels);

    let es_re: Vec<f64> = results.iter().map(|r| r.es.re).collect();
    let es_im: Vec<f64> = results.iter().map(|r| r.es.im).collect();
    let ep_re: Vec<f64> = results.iter().map(|r| r.ep.re).collect();
    let ep_im: Vec<f64> = results.iter().map(|r| r.ep.im).collect();

    Ok((es_re, es_im, ep_re, ep_im))
}

/// Solve Takagi-Taupin equations for crystal diffraction.
///
/// Args:
///     cb_re, cb_im: χ_h·C/b coefficients (list of complex, s-pol)
///     c0_re, c0_im: χ_0+β coefficients
///     ch_re, ch_im: χ_h̄·C coefficients
///     z_start, z_end: integration bounds [Å]
///     xi_re, xi_im: initial ξ value
///
/// Returns:
///     (rs_re, rs_im): reflectivity amplitudes
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tt_solve_rs(
    cb_re: Vec<f64>, cb_im: Vec<f64>,
    c0_re: Vec<f64>, c0_im: Vec<f64>,
    ch_re: Vec<f64>, ch_im: Vec<f64>,
    z_start: f64, z_end: f64,
    xi_re: f64, xi_im: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use xrt_pytte::solver::{BraggCoeffs, SolverConfig, solve_bragg_parallel};

    let n = cb_re.len();
    let coeffs: Vec<BraggCoeffs> = (0..n)
        .map(|i| BraggCoeffs {
            cb: Complex64::new(cb_re[i], cb_im[i]),
            c0: Complex64::new(c0_re[i], c0_im[i]),
            ch: Complex64::new(ch_re[i], ch_im[i]),
        })
        .collect();

    let xi_init = Complex64::new(xi_re, xi_im);
    let results = solve_bragg_parallel(&coeffs, z_start, z_end, xi_init, &SolverConfig::default());

    let rs_re: Vec<f64> = results.iter().map(|r| r.re).collect();
    let rs_im: Vec<f64> = results.iter().map(|r| r.im).collect();

    Ok((rs_re, rs_im))
}

/// Generate rays from a bending magnet source.
///
/// Returns:
///     dict with keys: x, y, z, a, b, c, e, state, jss, jpp
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bending_magnet_shine_rs(
    py: Python<'_>,
    electron_energy_gev: f64,
    beam_current: f64,
    b_field: f64,
    nrays: usize,
    e_min: f64,
    e_max: f64,
    theta_max: f64,
    psi_max: f64,
) -> PyResult<PyObject> {
    use xrt_sources::bending_magnet::BendingMagnet;

    let mut bm = BendingMagnet::new(
        electron_energy_gev, beam_current, b_field, nrays,
        e_min, e_max, theta_max, psi_max,
    );
    let beam = bm.shine();

    let dict = PyDict::new(py);
    dict.set_item("x", beam.x.to_vec())?;
    dict.set_item("y", beam.y.to_vec())?;
    dict.set_item("z", beam.z.to_vec())?;
    dict.set_item("a", beam.a.to_vec())?;
    dict.set_item("b", beam.b.to_vec())?;
    dict.set_item("c", beam.c.to_vec())?;
    dict.set_item("e", beam.e.to_vec())?;
    dict.set_item("state", (0..beam.nrays()).map(|i| beam.state[i]).collect::<Vec<i32>>())?;
    dict.set_item("jss", beam.jss.to_vec())?;
    dict.set_item("jpp", beam.jpp.to_vec())?;

    Ok(dict.into())
}

/// Compute multilayer reflectivity amplitudes.
///
/// Args:
///     t_elem, t_rho: top layer element name, density
///     b_elem, b_rho: bottom layer element name, density
///     s_elem, s_rho: substrate element name, density
///     n_pairs: number of bilayer pairs
///     d_t, d_b: layer thicknesses [Å]
///     roughness: interfacial roughness [Å]
///     energies: photon energies [eV]
///     sin_theta: sin(grazing angle) values
///
/// Returns:
///     (rs_abs, rp_abs): reflectivity magnitudes per energy point
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn multilayer_amplitude_rs(
    t_elem: &str, t_rho: f64,
    b_elem: &str, b_rho: f64,
    s_elem: &str, s_rho: f64,
    n_pairs: usize,
    d_t: f64, d_b: f64,
    roughness: f64,
    energies: Vec<f64>,
    sin_theta: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use xrt_materials::data::ScatteringTable;
    use xrt_materials::material::{Material, MaterialKind};
    use xrt_materials::multilayer::{Multilayer, MultilayerGeom};

    let table = ScatteringTable::ChantlerTotal;
    let t_mat = Material::new(&[t_elem], None, t_rho, MaterialKind::Mirror, None, table)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let b_mat = Material::new(&[b_elem], None, b_rho, MaterialKind::Mirror, None, table)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let s_mat = Material::new(&[s_elem], None, s_rho, MaterialKind::Mirror, None, table)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let ml = Multilayer::new(t_mat, b_mat, s_mat, n_pairs, d_t, d_b, roughness, MultilayerGeom::Reflected);

    let e_arr = Array1::from_vec(energies);
    let st_arr = Array1::from_vec(sin_theta);
    let result = ml.get_amplitude(&e_arr, &st_arr)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let rs_abs: Vec<f64> = result.rs.iter().map(|c| c.norm()).collect();
    let rp_abs: Vec<f64> = result.rp.iter().map(|c| c.norm()).collect();

    Ok((rs_abs, rp_abs))
}

/// Generate rays from a geometric source.
///
/// Returns:
///     dict with keys: x, y, z, a, b, c, e, state
#[pyfunction]
fn geometric_source_shine(
    py: Python<'_>,
    nrays: usize,
    energy: f64,
    dx: f64,
    dz: f64,
    dxprime: f64,
    dzprime: f64,
) -> PyResult<PyObject> {
    use xrt_sources::distributions::{EnergyDist, SpatialDist};
    use xrt_sources::geometric::GeometricSource;

    let source = GeometricSource {
        nrays,
        dist_x: SpatialDist::Normal(dx),
        dist_z: SpatialDist::Normal(dz),
        dist_xprime: SpatialDist::Normal(dxprime),
        dist_zprime: SpatialDist::Normal(dzprime),
        dist_e: EnergyDist::Lines(vec![energy], None),
        ..Default::default()
    };

    let beam = source.shine();
    let n = beam.nrays();

    let dict = PyDict::new(py);
    dict.set_item("x", beam.x.to_vec())?;
    dict.set_item("y", beam.y.to_vec())?;
    dict.set_item("z", beam.z.to_vec())?;
    dict.set_item("a", beam.a.to_vec())?;
    dict.set_item("b", beam.b.to_vec())?;
    dict.set_item("c", beam.c.to_vec())?;
    dict.set_item("e", beam.e.to_vec())?;
    dict.set_item(
        "state",
        (0..n).map(|i| beam.state[i]).collect::<Vec<i32>>(),
    )?;

    Ok(dict.into())
}

/// Generate rays from a wiggler source.
///
/// Returns:
///     dict with keys: x, y, z, a, b, c, e, state, jss, jpp
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wiggler_shine_rs(
    py: Python<'_>,
    electron_energy_gev: f64,
    beam_current: f64,
    k_param: f64,
    period_mm: f64,
    n_periods: usize,
    nrays: usize,
    e_min: f64,
    e_max: f64,
    theta_max: f64,
    psi_max: f64,
) -> PyResult<PyObject> {
    use xrt_sources::wiggler::Wiggler;

    let mut w = Wiggler::new(
        electron_energy_gev, beam_current, k_param, period_mm, n_periods,
        nrays, e_min, e_max, theta_max, psi_max,
    );
    let beam = w.shine();

    let dict = PyDict::new(py);
    dict.set_item("x", beam.x.to_vec())?;
    dict.set_item("y", beam.y.to_vec())?;
    dict.set_item("z", beam.z.to_vec())?;
    dict.set_item("a", beam.a.to_vec())?;
    dict.set_item("b", beam.b.to_vec())?;
    dict.set_item("c", beam.c.to_vec())?;
    dict.set_item("e", beam.e.to_vec())?;
    dict.set_item("state", (0..beam.nrays()).map(|i| beam.state[i]).collect::<Vec<i32>>())?;
    dict.set_item("jss", beam.jss.to_vec())?;
    dict.set_item("jpp", beam.jpp.to_vec())?;

    Ok(dict.into())
}

/// Generate rays from an undulator source.
///
/// Returns:
///     dict with keys: x, y, z, a, b, c, e, state, jss, jpp
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn undulator_shine_rs(
    py: Python<'_>,
    electron_energy_gev: f64,
    beam_current: f64,
    kx: f64,
    ky: f64,
    period_mm: f64,
    n_periods: usize,
    nrays: usize,
    e_min: f64,
    e_max: f64,
    theta_max: f64,
    psi_max: f64,
) -> PyResult<PyObject> {
    use xrt_sources::undulator::Undulator;

    let mut u = Undulator::new(
        electron_energy_gev, beam_current, kx, ky, period_mm, n_periods,
        nrays, e_min, e_max, theta_max, psi_max,
    );
    let beam = u.shine();

    let dict = PyDict::new(py);
    dict.set_item("x", beam.x.to_vec())?;
    dict.set_item("y", beam.y.to_vec())?;
    dict.set_item("z", beam.z.to_vec())?;
    dict.set_item("a", beam.a.to_vec())?;
    dict.set_item("b", beam.b.to_vec())?;
    dict.set_item("c", beam.c.to_vec())?;
    dict.set_item("e", beam.e.to_vec())?;
    dict.set_item("state", (0..beam.nrays()).map(|i| beam.state[i]).collect::<Vec<i32>>())?;
    dict.set_item("jss", beam.jss.to_vec())?;
    dict.set_item("jpp", beam.jpp.to_vec())?;

    Ok(dict.into())
}

/// Get the version of xrt-rs.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition.
#[pymodule]
fn xrt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Existing low-level functions ────────────────────────────────────
    m.add_function(wrap_pyfunction!(find_intersection_rs, m)?)?;
    m.add_function(wrap_pyfunction!(find_intersection_parametric_rs, m)?)?;
    m.add_function(wrap_pyfunction!(diffraction_integral_rs, m)?)?;
    m.add_function(wrap_pyfunction!(tt_solve_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bending_magnet_shine_rs, m)?)?;
    m.add_function(wrap_pyfunction!(wiggler_shine_rs, m)?)?;
    m.add_function(wrap_pyfunction!(undulator_shine_rs, m)?)?;
    m.add_function(wrap_pyfunction!(multilayer_amplitude_rs, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_source_shine, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // ── High-level classes ──────────────────────────────────────────────
    // Core types
    m.add_class::<py_beam::PyBeam>()?;
    m.add_class::<py_material::PyMaterial>()?;
    m.add_class::<py_beamline::PyBeamLine>()?;
    m.add_class::<py_screen::PyScreen>()?;
    m.add_class::<py_screen::PyScreenCapture>()?;

    // Sources
    m.add_class::<py_source::PyGeometricSource>()?;

    // Mirror OEs
    m.add_class::<py_oe::PyFlatMirror>()?;
    m.add_class::<py_oe::PyToroidMirror>()?;
    m.add_class::<py_oe::PySphericalMirror>()?;
    m.add_class::<py_oe::PyCylindricalMirror>()?;
    m.add_class::<py_oe::PyBentFlatMirror>()?;
    m.add_class::<py_oe::PyVFM>()?;

    // Parametric mirror OEs
    m.add_class::<py_oe::PyEllipticalMirror>()?;
    m.add_class::<py_oe::PyParabolicalMirror>()?;

    // Grating OEs
    m.add_class::<py_oe::PyBlazedGrating>()?;
    m.add_class::<py_oe::PyLaminarGrating>()?;
    m.add_class::<py_oe::PyVLSGrating>()?;

    // Lens OEs
    m.add_class::<py_oe::PyParaboloidLens>()?;

    Ok(())
}
