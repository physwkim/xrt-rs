//! Cylindrical mirror surface (shadow3 FCYL=1).
//!
//! A cylinder with radius R, axis along y.
//! z = R - sqrt(R² - x²) (sagittal only, no meridional curvature).

use crate::oes::surface::Surface;

/// Cylindrical mirror (curvature in x only).
#[derive(Debug, Clone, Copy)]
pub struct CylindricalSurface {
    /// Cylinder radius [mm]
    pub r: f64,
}

impl CylindricalSurface {
    pub fn new(r: f64) -> Self {
        Self { r }
    }
}

impl Surface for CylindricalSurface {
    fn local_z(&self, x: f64, _y: f64) -> f64 {
        let arg = x / self.r;
        let arg2 = arg * arg;
        if arg2 < 1.0 {
            self.r * (1.0 - (1.0 - arg2).sqrt())
        } else {
            self.r
        }
    }

    fn local_n(&self, x: f64, _y: f64) -> [f64; 3] {
        let arg = x / self.r;
        let arg2 = arg * arg;
        if arg2 < 1.0 {
            let dz_dx = x / (self.r * (1.0 - arg2).sqrt());
            let nx = -dz_dx;
            let nz = 1.0;
            let norm = (nx * nx + nz * nz).sqrt();
            [nx / norm, 0.0, nz / norm]
        } else {
            [0.0, 0.0, 1.0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cylindrical_at_origin() {
        let s = CylindricalSurface::new(100.0);
        assert!(s.local_z(0.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn cylindrical_y_independent() {
        let s = CylindricalSurface::new(100.0);
        assert_eq!(s.local_z(5.0, 0.0), s.local_z(5.0, 100.0));
    }

    #[test]
    fn cylindrical_normal_y_zero() {
        let s = CylindricalSurface::new(100.0);
        let [_, ny, _] = s.local_n(5.0, 50.0);
        assert!(ny.abs() < 1e-15, "cylindrical ny should be 0");
    }
}
