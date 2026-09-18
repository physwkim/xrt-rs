//! Bent flat mirror surface.
//!
//! z = (y² - y_min²) / (2R) — cylindrical bending in meridional plane.

use crate::oes::surface::Surface;

/// A flat mirror bent to radius R in the meridional plane.
#[derive(Debug, Clone, Copy)]
pub struct BentFlatSurface {
    /// Bending radius [mm] (meridional)
    pub r: f64,
    /// Fixed end y-position [mm] (where z = 0)
    pub y_min: f64,
}

impl BentFlatSurface {
    pub fn new(r: f64, y_min: f64) -> Self {
        Self { r, y_min }
    }
}

impl Surface for BentFlatSurface {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        (y * y - self.y_min * self.y_min) / (2.0 * self.r)
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        let dz_dy = y / self.r;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (ny * ny + nz * nz).sqrt();
        [0.0, ny / norm, nz / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bent_flat_at_y_min() {
        let s = BentFlatSurface::new(1000.0, -50.0);
        let z = s.local_z(0.0, -50.0);
        assert!(z.abs() < 1e-12, "z at y_min should be 0: {z}");
    }

    #[test]
    fn bent_flat_symmetry() {
        let s = BentFlatSurface::new(1000.0, 0.0);
        let z1 = s.local_z(0.0, 10.0);
        let z2 = s.local_z(0.0, -10.0);
        assert!((z1 - z2).abs() < 1e-12, "should be symmetric about y=0");
    }

    #[test]
    fn bent_flat_curvature() {
        let r = 5000.0;
        let s = BentFlatSurface::new(r, 0.0);
        let y = 100.0;
        let expected = y * y / (2.0 * r);
        assert!((s.local_z(0.0, y) - expected).abs() < 1e-12);
    }

    #[test]
    fn bent_flat_normal_at_origin() {
        let s = BentFlatSurface::new(1000.0, 0.0);
        let [nx, ny, nz] = s.local_n(0.0, 0.0);
        assert!(nx.abs() < 1e-15);
        assert!(ny.abs() < 1e-15);
        assert!((nz - 1.0).abs() < 1e-15);
    }
}
