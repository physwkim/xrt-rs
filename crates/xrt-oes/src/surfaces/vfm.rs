//! Vertically focusing mirror surface.
//!
//! Toroid with meridional bending and sagittal cylinder,
//! optionally with fixed-end mounting (y_min offset).

use crate::surface::Surface;

/// A vertically focusing mirror (cylindrical sagittal + meridional bending).
#[derive(Debug, Clone, Copy)]
pub struct VfmSurface {
    /// Meridional bending radius R [mm]
    pub r_major: f64,
    /// Sagittal radius r [mm]
    pub r_minor: f64,
    /// Fixed end y-position [mm]
    pub y_min: f64,
}

impl VfmSurface {
    pub fn new(r_major: f64, r_minor: f64, y_min: f64) -> Self {
        Self { r_major, r_minor, y_min }
    }
}

impl Surface for VfmSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        // Meridional: (y² - y_min²) / (2R)
        let z_meridional = (y * y - self.y_min * self.y_min) / (2.0 * self.r_major);
        // Sagittal: r - sqrt(r² - x²)
        let arg = x / self.r_minor;
        let arg2 = arg * arg;
        let z_sagittal = if arg2 < 1.0 {
            self.r_minor * (1.0 - (1.0 - arg2).sqrt())
        } else {
            self.r_minor
        };
        z_meridional + z_sagittal
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let arg = x / self.r_minor;
        let arg2 = arg * arg;
        let dz_dx = if arg2 < 1.0 {
            x / (self.r_minor * (1.0 - arg2).sqrt())
        } else {
            x.signum() * 1e6
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
    fn vfm_at_y_min() {
        let s = VfmSurface::new(5e6, 50.0, -100.0);
        let z = s.local_z(0.0, -100.0);
        assert!(z.abs() < 1e-10, "z at y_min, x=0 should be ~0: {z}");
    }

    #[test]
    fn vfm_reduces_to_toroid() {
        // With y_min = 0, should match toroid behavior at origin
        let vfm = VfmSurface::new(5e6, 50.0, 0.0);
        let z = vfm.local_z(1.0, 10.0);
        // Compare with toroid formula
        let z_mer = 10.0 * 10.0 / (2.0 * 5e6);
        let arg: f64 = 1.0 / 50.0;
        let z_sag = 50.0 * (1.0 - (1.0 - arg * arg).sqrt());
        assert!((z - z_mer - z_sag).abs() < 1e-12);
    }
}
