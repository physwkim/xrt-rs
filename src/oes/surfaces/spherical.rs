//! Spherical mirror surface.
//!
//! z = R - sqrt(R² - x² - y²)
//!
//! Ported from oes.py spherical surface logic.

use crate::oes::surface::Surface;

/// A spherical mirror surface.
#[derive(Debug, Clone, Copy)]
pub struct SphericalSurface {
    /// Radius of curvature R [mm]
    pub r: f64,
}

impl SphericalSurface {
    pub fn new(r: f64) -> Self {
        Self { r }
    }
}

impl Surface for SphericalSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        let r2 = self.r * self.r;
        let rho2 = x * x + y * y;
        if rho2 < r2 {
            self.r - (r2 - rho2).sqrt()
        } else {
            self.r // clamp at rim
        }
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let r2 = self.r * self.r;
        let rho2 = x * x + y * y;
        if rho2 < r2 {
            let denom = (r2 - rho2).sqrt();
            let dz_dx = x / denom;
            let dz_dy = y / denom;
            let nx = -dz_dx;
            let ny = -dz_dy;
            let nz = 1.0;
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            [nx / norm, ny / norm, nz / norm]
        } else {
            [0.0, 0.0, 1.0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_at_origin() {
        let s = SphericalSurface::new(1000.0);
        assert_eq!(s.local_z(0.0, 0.0), 0.0);
        let n = s.local_n(0.0, 0.0);
        assert!((n[0]).abs() < 1e-15);
        assert!((n[1]).abs() < 1e-15);
        assert!((n[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn spherical_z_formula() {
        let r = 1000.0;
        let s = SphericalSurface::new(r);
        let x = 10.0;
        let expected = r - (r * r - x * x).sqrt();
        let got = s.local_z(x, 0.0);
        assert!(
            (got - expected).abs() < 1e-10,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn spherical_symmetry() {
        let s = SphericalSurface::new(500.0);
        // z(x, 0) == z(-x, 0)
        assert!((s.local_z(5.0, 0.0) - s.local_z(-5.0, 0.0)).abs() < 1e-15);
        // z(x, y) == z(x, -y)
        assert!((s.local_z(3.0, 4.0) - s.local_z(3.0, -4.0)).abs() < 1e-15);
    }
}
