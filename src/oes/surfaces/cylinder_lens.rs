//! Parabolic cylinder lens surfaces.
//!
//! One-dimensional parabolic lens (focuses in one plane only).

use crate::oes::surface::Surface;

/// Parabolic cylinder flat lens (focuses in x only).
#[derive(Debug, Clone, Copy)]
pub struct ParabolicCylinderLensSurface {
    /// Focal length [mm]
    pub focus: f64,
    /// Maximum sag [mm]
    pub z_max: Option<f64>,
}

impl ParabolicCylinderLensSurface {
    pub fn new(focus: f64, z_max: Option<f64>) -> Self {
        Self { focus, z_max }
    }
}

impl Surface for ParabolicCylinderLensSurface {
    fn local_z(&self, x: f64, _y: f64) -> f64 {
        let z = x * x / (4.0 * self.focus);
        match self.z_max {
            Some(zmax) => z.min(zmax),
            None => z,
        }
    }

    fn local_n(&self, x: f64, _y: f64) -> [f64; 3] {
        if let Some(zmax) = self.z_max
            && x * x / (4.0 * self.focus) >= zmax
        {
            return [0.0, 0.0, 1.0];
        }
        let dz_dx = x / (2.0 * self.focus);
        let nx = -dz_dx;
        let nz = 1.0;
        let norm = (nx * nx + nz * nz).sqrt();
        [nx / norm, 0.0, nz / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cylinder_lens_at_origin() {
        let s = ParabolicCylinderLensSurface::new(100.0, None);
        assert_eq!(s.local_z(0.0, 0.0), 0.0);
        // y should not affect z
        assert_eq!(s.local_z(0.0, 100.0), 0.0);
    }

    #[test]
    fn cylinder_lens_focuses_x() {
        let s = ParabolicCylinderLensSurface::new(100.0, None);
        let z = s.local_z(10.0, 0.0);
        assert!(z > 0.0);
        // Same z for any y
        assert_eq!(s.local_z(10.0, 50.0), z);
    }
}
