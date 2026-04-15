//! Generic surface of revolution.
//!
//! Defined by a radial profile r(s) rotated around the y-axis.
//! This is the base for capillary optics and other axially symmetric surfaces.

use crate::surface::ParametricSurface;

/// A surface of revolution from a user-provided radial profile.
///
/// The profile is defined as a set of (s, r) points with linear interpolation.
#[derive(Debug, Clone)]
pub struct SurfaceOfRevolution {
    /// Tabulated s-positions [mm] (must be sorted ascending)
    pub s_table: Vec<f64>,
    /// Tabulated radii [mm] at each s-position
    pub r_table: Vec<f64>,
}

impl SurfaceOfRevolution {
    pub fn new(s_table: Vec<f64>, r_table: Vec<f64>) -> Self {
        assert_eq!(
            s_table.len(),
            r_table.len(),
            "s and r tables must have same length"
        );
        assert!(s_table.len() >= 2, "need at least 2 points");
        Self { s_table, r_table }
    }

    /// Interpolate radius at position s.
    fn interp_r(&self, s: f64) -> f64 {
        if s <= self.s_table[0] {
            return self.r_table[0];
        }
        if s >= *self.s_table.last().unwrap() {
            return *self.r_table.last().unwrap();
        }
        // Binary search for interval
        let idx = self.s_table.partition_point(|&v| v < s);
        if idx == 0 {
            return self.r_table[0];
        }
        let i = idx - 1;
        let t = (s - self.s_table[i]) / (self.s_table[i + 1] - self.s_table[i]);
        self.r_table[i] * (1.0 - t) + self.r_table[i + 1] * t
    }

    /// Interpolate dr/ds at position s (finite difference).
    fn interp_dr_ds(&self, s: f64) -> f64 {
        let eps = 1e-6;
        let r_plus = self.interp_r(s + eps);
        let r_minus = self.interp_r(s - eps);
        (r_plus - r_minus) / (2.0 * eps)
    }
}

impl ParametricSurface for SurfaceOfRevolution {
    fn local_r(&self, s: f64, _phi: f64) -> f64 {
        self.interp_r(s)
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let dr_ds = self.interp_dr_ds(s);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let nx = -sin_phi;
        let ny = dr_ds;
        let nz = -cos_phi;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }

    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r = (x * x + z * z).sqrt();
        let phi = x.atan2(-z);
        (y, phi, r)
    }

    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        (r * phi.sin(), s, -r * phi.cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revolution_cylinder() {
        // Constant radius = cylinder
        let s = SurfaceOfRevolution::new(vec![0.0, 100.0], vec![5.0, 5.0]);
        assert!((s.local_r(50.0, 0.0) - 5.0).abs() < 1e-12);
        assert!((s.local_r(0.0, 1.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn revolution_taper() {
        // Linear taper: r goes from 5 to 10
        let s = SurfaceOfRevolution::new(vec![0.0, 100.0], vec![5.0, 10.0]);
        assert!((s.local_r(50.0, 0.0) - 7.5).abs() < 1e-10);
    }

    #[test]
    fn revolution_round_trip() {
        let s = SurfaceOfRevolution::new(vec![0.0, 100.0], vec![5.0, 10.0]);
        let r = s.local_r(30.0, 0.5);
        let (x, y, z) = s.param_to_xyz(30.0, 0.5, r);
        let (s2, phi2, r2) = s.xyz_to_param(x, y, z);
        assert!((s2 - 30.0).abs() < 1e-10);
        assert!((phi2 - 0.5).abs() < 1e-10);
        assert!((r2 - r).abs() < 1e-10);
    }
}
