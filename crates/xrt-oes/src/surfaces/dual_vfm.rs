//! Dual vertically focusing mirror surface.
//!
//! Two side-by-side VFM strips with independently adjustable sagittal radii,
//! sharing the same meridional bending.

use crate::surface::Surface;

/// Dual VFM: two sagittal strips side-by-side.
#[derive(Debug, Clone, Copy)]
pub struct DualVfmSurface {
    /// Meridional bending radius R [mm]
    pub r_major: f64,
    /// Sagittal radius for the negative-x strip [mm]
    pub r_minor_1: f64,
    /// Sagittal radius for the positive-x strip [mm]
    pub r_minor_2: f64,
    /// x-boundary between the two strips [mm]
    pub x_boundary: f64,
    /// Fixed end y-position [mm]
    pub y_min: f64,
}

impl DualVfmSurface {
    pub fn new(r_major: f64, r_minor_1: f64, r_minor_2: f64, x_boundary: f64, y_min: f64) -> Self {
        Self {
            r_major,
            r_minor_1,
            r_minor_2,
            x_boundary,
            y_min,
        }
    }

    fn r_minor_at(&self, x: f64) -> f64 {
        if x < self.x_boundary {
            self.r_minor_1
        } else {
            self.r_minor_2
        }
    }
}

impl Surface for DualVfmSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        let z_mer = (y * y - self.y_min * self.y_min) / (2.0 * self.r_major);
        let r = self.r_minor_at(x);
        let arg = x / r;
        let arg2 = arg * arg;
        let z_sag = if arg2 < 1.0 {
            r * (1.0 - (1.0 - arg2).sqrt())
        } else {
            r
        };
        z_mer + z_sag
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let r = self.r_minor_at(x);
        let arg = x / r;
        let arg2 = arg * arg;
        let dz_dx = if arg2 < 1.0 {
            x / (r * (1.0 - arg2).sqrt())
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
    fn dual_vfm_at_origin() {
        let s = DualVfmSurface::new(5e6, 50.0, 60.0, 0.0, 0.0);
        assert!(s.local_z(0.0, 0.0).abs() < 1e-10);
    }

    #[test]
    fn dual_vfm_different_radii() {
        let s = DualVfmSurface::new(5e6, 50.0, 100.0, 0.0, 0.0);
        let z_neg = s.local_z(-5.0, 0.0);
        let z_pos = s.local_z(5.0, 0.0);
        // Different sagittal radii → different z
        assert!(
            (z_neg - z_pos).abs() > 1e-6,
            "different radii should give different z: {z_neg} vs {z_pos}"
        );
    }
}
