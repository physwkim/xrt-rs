//! Ray-surface intersection finding.
//!
//! Per-ray scalar functions designed for rayon parallel iteration.
//! Ported from oes_base.py:640-946 (find_dz, find_intersection).

use crate::math::rootfind::{self, IntersectionResult, RootFindConfig, SurfaceEval};

use crate::oes::surface::{ParametricSurface, Surface};

/// Adapter: makes a `Surface` work with the generic `SurfaceEval` trait.
pub struct SurfaceAdapter<'a, S: Surface> {
    surface: &'a S,
    invert_normal: f64,
}

impl<'a, S: Surface> SurfaceAdapter<'a, S> {
    pub fn new(surface: &'a S, invert_normal: i32) -> Self {
        Self {
            surface,
            invert_normal: invert_normal as f64,
        }
    }
}

impl<S: Surface> SurfaceEval for SurfaceAdapter<'_, S> {
    fn find_dz(
        &self,
        t: f64,
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
        _invert_normal: i32,
    ) -> (f64, f64, f64, f64) {
        let x = x0 + a * t;
        let y = y0 + b * t;
        let z = z0 + c * t;

        let mut surf_z = self.surface.local_z(x, y);
        if let Some(dz) = self.surface.local_z_distorted(x, y) {
            surf_z += dz;
        }

        // dz = (z - surf_z) * invertNormal
        // Positive dz means ray is above surface
        let dz = (z - surf_z) * self.invert_normal;
        (dz, x, y, z)
    }
}

/// Adapter: makes a `ParametricSurface` work with the generic `SurfaceEval` trait.
pub struct ParametricSurfaceAdapter<'a, P: ParametricSurface> {
    surface: &'a P,
    invert_normal: f64,
}

impl<'a, P: ParametricSurface> ParametricSurfaceAdapter<'a, P> {
    pub fn new(surface: &'a P, invert_normal: i32) -> Self {
        Self {
            surface,
            invert_normal: invert_normal as f64,
        }
    }
}

impl<P: ParametricSurface> SurfaceEval for ParametricSurfaceAdapter<'_, P> {
    fn find_dz(
        &self,
        t: f64,
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
        _invert_normal: i32,
    ) -> (f64, f64, f64, f64) {
        let x = x0 + a * t;
        let y = y0 + b * t;
        let z = z0 + c * t;

        // Convert to parametric coordinates
        let (s, phi, r) = self.surface.xyz_to_param(x, y, z);

        let mut surf_r = self.surface.local_r(s, phi);
        if let Some(dr) = self.surface.local_r_distorted(s, phi) {
            surf_r += dr;
        }

        // dz = (surf_r - r) * invertNormal (note: reversed sign for parametric)
        let dz = (surf_r - r) * (-self.invert_normal);
        (dz, s, phi, r)
    }
}

/// Find intersection of a single ray with a Surface.
///
/// Uses Brent's method for robust root finding.
/// Returns `IntersectionResult` with the ray parameter t and local coordinates.
pub fn find_intersection_surface<S: Surface>(
    surface: &S,
    t_min: f64,
    t_max: f64,
    x0: f64,
    y0: f64,
    z0: f64,
    a: f64,
    b: f64,
    c: f64,
    invert_normal: i32,
    config: &RootFindConfig,
) -> IntersectionResult {
    let adapter = SurfaceAdapter::new(surface, invert_normal);
    rootfind::brent_method(
        &adapter,
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
    )
}

/// Find intersection of a single ray with a ParametricSurface.
pub fn find_intersection_parametric<P: ParametricSurface>(
    surface: &P,
    t_min: f64,
    t_max: f64,
    x0: f64,
    y0: f64,
    z0: f64,
    a: f64,
    b: f64,
    c: f64,
    invert_normal: i32,
    config: &RootFindConfig,
) -> IntersectionResult {
    let adapter = ParametricSurfaceAdapter::new(surface, invert_normal);
    rootfind::brent_method(
        &adapter,
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
    )
}

/// Find intersections for multiple rays in parallel using rayon.
pub fn find_intersections_parallel<S: Surface>(
    surface: &S,
    t_min: &[f64],
    t_max: &[f64],
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    invert_normal: i32,
    config: &RootFindConfig,
) -> Vec<IntersectionResult> {
    use rayon::prelude::*;

    let n = x0.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            find_intersection_surface(
                surface,
                t_min[i],
                t_max[i],
                x0[i],
                y0[i],
                z0[i],
                a[i],
                b[i],
                c[i],
                invert_normal,
                config,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oes::surfaces::flat::FlatSurface;
    use crate::oes::surfaces::toroid::ToroidSurface;

    #[test]
    fn flat_mirror_intersection() {
        let surface = FlatSurface;
        let config = RootFindConfig::default();

        // Ray from (0, 0, 10) going towards -z at 45° in yz plane
        let result = find_intersection_surface(
            &surface,
            0.0,
            20.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.5_f64.sqrt(),
            -(0.5_f64.sqrt()),
            1,
            &config,
        );

        assert!(result.converged);
        // t = 10 / sin(45°) = 10 * sqrt(2) ≈ 14.14
        let expected_t = 10.0 * 2.0_f64.sqrt();
        assert!(
            (result.t - expected_t).abs() < 1e-8,
            "t = {}, expected {}",
            result.t,
            expected_t
        );
        assert!(result.z.abs() < 1e-10, "z = {}", result.z);
    }

    #[test]
    fn toroid_intersection() {
        let surface = ToroidSurface::new(5e6, 50.0);
        let config = RootFindConfig::default();

        // Ray from (0, 0, 10) going straight down (-z)
        let result = find_intersection_surface(
            &surface, 0.0, 20.0, 0.0, 0.0, 10.0, 0.0, 0.0, -1.0, 1, &config,
        );

        assert!(result.converged);
        // At x=0, y=0: surface_z = 0, so intersection at z=0, t=10
        assert!((result.t - 10.0).abs() < 1e-8, "t = {}", result.t);
    }

    #[test]
    fn parallel_flat_intersections() {
        let surface = FlatSurface;
        let config = RootFindConfig::default();
        let n = 100;

        let t_min = vec![0.0; n];
        let t_max = vec![20.0; n];
        let x0 = vec![0.0; n];
        let y0 = vec![0.0; n];
        let z0 = vec![10.0; n];
        let a = vec![0.0; n];
        let b = vec![0.0; n];
        let c = vec![-1.0; n];

        let results = find_intersections_parallel(
            &surface, &t_min, &t_max, &x0, &y0, &z0, &a, &b, &c, 1, &config,
        );

        assert_eq!(results.len(), n);
        for r in &results {
            assert!(r.converged);
            assert!((r.t - 10.0).abs() < 1e-8);
        }
    }
}
