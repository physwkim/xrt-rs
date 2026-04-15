//! Core reflection pipeline: _reflect_local equivalent.
//!
//! Ported from oes_base.py:1763-1962.
//!
//! Pipeline per-ray:
//! 1. Bracketing → t_min, t_max
//! 2. Intersection → t, (x, y, z) on surface
//! 3. Aperture check → state
//! 4. Surface normal at intersection
//! 5. Deflection (reflect/refract/grating)
//! 6. Material amplitude → update coherency

use num_complex::Complex64;
use rayon::prelude::*;

use xrt_core::beam::{Beam, RayState};
use xrt_math::rootfind::RootFindConfig;

use crate::aperture::Aperture;
use crate::bracketing::bracket_ray;
use crate::deflection;
use crate::intersection::find_intersection_surface;
use crate::surface::Surface;

/// What to do after finding the intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeflectionMode {
    /// Specular reflection (mirror)
    Reflect,
    /// Snell's law refraction (plate/lens) with n1/n2 ratio
    Refract { n1_over_n2: f64 },
    /// Grating deflection with diffraction order
    Grating { order: i32 },
    /// Pass straight through (transmitted crystal)
    PassThrough,
}

/// Per-ray reflection result.
#[derive(Debug, Clone)]
pub struct RayResult {
    /// New state
    pub state: RayState,
    /// Intersection local coordinates
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// New direction
    pub a: f64,
    pub b: f64,
    pub c: f64,
    /// Path length added
    pub path_delta: f64,
    /// Surface normal at intersection
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
    /// Beam-in dot normal
    pub beam_in_dot_normal: f64,
}

/// Reflect a beam off a surface (the core pipeline).
///
/// Operates on the "good" rays in the beam. Uses rayon for parallel per-ray
/// intersection finding, then applies aperture, deflection, and material
/// effects.
///
/// Returns the number of good rays after reflection.
pub fn reflect_local<S: Surface>(
    surface: &S,
    beam: &mut Beam,
    good: &[usize],
    aperture: &Aperture,
    invert_normal: i32,
    mode: DeflectionMode,
    phys_x: Option<[f64; 2]>,
    phys_y: Option<[f64; 2]>,
    config: &RootFindConfig,
) -> Vec<RayResult> {
    // Phase 1: parallel intersection finding
    let results: Vec<RayResult> = good
        .par_iter()
        .map(|&i| {
            let x0 = beam.x[i];
            let y0 = beam.y[i];
            let z0 = beam.z[i];
            let a = beam.a[i];
            let b = beam.b[i];
            let c = beam.c[i];

            // Bracketing
            let (t_min, t_max) = bracket_ray(x0, y0, z0, a, b, c, phys_x, phys_y);

            // Intersection
            let isect = find_intersection_surface(
                surface,
                t_min,
                t_max,
                x0,
                y0,
                z0,
                a,
                b,
                c,
                invert_normal,
                config,
            );

            if !isect.converged {
                return RayResult {
                    state: RayState::Out,
                    x: x0,
                    y: y0,
                    z: z0,
                    a,
                    b,
                    c,
                    path_delta: 0.0,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                    beam_in_dot_normal: 0.0,
                };
            }

            let ix = isect.x;
            let iy = isect.y;
            let iz = isect.z;

            // Aperture check
            let state = aperture.classify_ray(ix, iy);
            if state != RayState::Good {
                return RayResult {
                    state,
                    x: ix,
                    y: iy,
                    z: iz,
                    a,
                    b,
                    c,
                    path_delta: isect.t,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                    beam_in_dot_normal: 0.0,
                };
            }

            // Surface normal
            let [nx, ny, nz] = surface.local_n(ix, iy);

            // beam_in · normal
            let bidn = a * nx + b * ny + c * nz;

            // Deflection
            let (a_out, b_out, c_out) = match mode {
                DeflectionMode::Reflect => deflection::reflect_specular(a, b, c, nx, ny, nz, bidn),
                DeflectionMode::Grating { order } => {
                    // Get grating vector from surface
                    if let Some([gx, gy, gz]) = surface.local_g(ix, iy) {
                        deflection::grating_deflection(
                            a, b, c, gx, gy, gz, nx, ny, nz, bidn, beam.e[i], order, None,
                        )
                    } else {
                        // No grating vector → fall back to specular
                        deflection::reflect_specular(a, b, c, nx, ny, nz, bidn)
                    }
                }
                DeflectionMode::Refract { n1_over_n2 } => {
                    deflection::refract_snell(a, b, c, nx, ny, nz, bidn, n1_over_n2)
                }
                DeflectionMode::PassThrough => (a, b, c),
            };

            RayResult {
                state: RayState::Good,
                x: ix,
                y: iy,
                z: iz,
                a: a_out,
                b: b_out,
                c: c_out,
                path_delta: isect.t,
                nx,
                ny,
                nz,
                beam_in_dot_normal: bidn,
            }
        })
        .collect();

    results
}

/// Apply reflection results back to the beam.
pub fn apply_results(beam: &mut Beam, good: &[usize], results: &[RayResult]) {
    for (&i, result) in good.iter().zip(results.iter()) {
        beam.x[i] = result.x;
        beam.y[i] = result.y;
        beam.z[i] = result.z;
        beam.a[i] = result.a;
        beam.b[i] = result.b;
        beam.c[i] = result.c;
        beam.state[i] = result.state as i32;
        beam.path[i] += result.path_delta;
    }
}

/// Apply material reflectivity to beam coherency matrix.
///
/// `rs`, `rp`: complex amplitude reflectivities per good ray.
/// `abs_coeff`: absorption coefficient [1/cm] per good ray.
pub fn apply_material_amplitude(
    beam: &mut Beam,
    good_n: &[usize],
    rs: &[Complex64],
    rp: &[Complex64],
) {
    for (idx, &i) in good_n.iter().enumerate() {
        let rs_abs2 = rs[idx].norm_sqr();
        let rp_abs2 = rp[idx].norm_sqr();

        beam.jss[i] *= rs_abs2;
        beam.jpp[i] *= rp_abs2;
        beam.jsp[i] *= rs[idx] * rp[idx].conj();

        if let Some(ref mut es) = beam.es {
            es[i] *= rs[idx];
        }
        if let Some(ref mut ep) = beam.ep {
            ep[i] *= rp[idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::flat::FlatSurface;

    fn make_test_beam(n: usize) -> Beam {
        let mut beam = Beam::new(n);
        // Rays from z=10, going at grazing angle toward surface
        let angle = 0.01_f64; // 10 mrad grazing angle
        for i in 0..n {
            beam.x[i] = (i as f64 - n as f64 / 2.0) * 0.1;
            beam.y[i] = -100.0;
            beam.z[i] = 10.0;
            beam.a[i] = 0.0;
            beam.b[i] = angle.cos();
            beam.c[i] = -angle.sin();
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
        }
        beam
    }

    #[test]
    fn reflect_flat_mirror() {
        let surface = FlatSurface;
        let mut beam = make_test_beam(10);
        let config = RootFindConfig::default();
        let aperture = Aperture::default();

        let good: Vec<usize> = (0..10).collect();

        let results = reflect_local(
            &surface,
            &mut beam,
            &good,
            &aperture,
            1,
            DeflectionMode::Reflect,
            None,
            None,
            &config,
        );

        // All rays should intersect and be reflected
        for r in &results {
            assert_eq!(r.state, RayState::Good, "ray state = {:?}", r.state);
            // Intersection z should be ~0 (flat surface)
            assert!(r.z.abs() < 1e-6, "z = {}", r.z);
            // Reflected c should be positive (going away from surface)
            assert!(r.c > 0.0, "c_out = {} should be > 0", r.c);
        }

        apply_results(&mut beam, &good, &results);

        // Verify beam was updated
        for i in 0..10 {
            assert!(beam.c[i] > 0.0);
            assert_eq!(beam.state[i], RayState::Good as i32);
        }
    }
}
