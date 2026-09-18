//! Toroidal mirror surface.
//!
//! z = y²/(2R) + r(1 - sqrt(1 - (x/r)²))
//!
//! Where R is the major (meridional) radius and r is the minor (sagittal) radius.
//! Ported from oes.py:1092-1181.

use crate::oes::surface::Surface;

/// A toroidal mirror surface.
#[derive(Debug, Clone, Copy)]
pub struct ToroidSurface {
    /// Meridional radius R [mm] (large radius, along y)
    pub r_major: f64,
    /// Sagittal radius r [mm] (small radius, along x)
    pub r_minor: f64,
}

impl ToroidSurface {
    pub fn new(r_major: f64, r_minor: f64) -> Self {
        Self { r_major, r_minor }
    }
}

impl Surface for ToroidSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        let z_meridional = y * y / (2.0 * self.r_major);
        let arg = x / self.r_minor;
        let arg2 = arg * arg;
        let z_sagittal = if arg2 < 1.0 {
            self.r_minor * (1.0 - (1.0 - arg2).sqrt())
        } else {
            self.r_minor // clamp at rim
        };
        z_meridional + z_sagittal
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        // Partial derivatives:
        // dz/dx = x / (r * sqrt(1 - (x/r)²))  (sagittal)
        // dz/dy = y / R                         (meridional)
        // normal = normalize(-dz/dx, -dz/dy, 1)
        let arg = x / self.r_minor;
        let arg2 = arg * arg;
        let dz_dx = if arg2 < 1.0 {
            x / (self.r_minor * (1.0 - arg2).sqrt())
        } else {
            x.signum() * 1e6 // near-vertical at rim
        };
        let dz_dy = y / self.r_major;

        let nx = -dz_dx;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toroid_at_origin() {
        let s = ToroidSurface::new(5e6, 50.0);
        assert_eq!(s.local_z(0.0, 0.0), 0.0);
        let n = s.local_n(0.0, 0.0);
        assert!((n[0]).abs() < 1e-15);
        assert!((n[1]).abs() < 1e-15);
        assert!((n[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn toroid_sagittal_z() {
        let r = 50.0;
        let s = ToroidSurface::new(1e9, r);
        let x = 10.0;
        // z ≈ r(1 - sqrt(1 - (x/r)²))
        let expected = r * (1.0 - (1.0 - (x / r).powi(2)).sqrt());
        let got = s.local_z(x, 0.0);
        assert!(
            (got - expected).abs() < 1e-10,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn toroid_meridional_z() {
        let big_r = 5e6;
        let s = ToroidSurface::new(big_r, 1e9);
        let y = 100.0;
        let expected = y * y / (2.0 * big_r);
        let got = s.local_z(0.0, y);
        assert!(
            (got - expected).abs() < 1e-10,
            "got {got}, expected {expected}"
        );
    }
}
