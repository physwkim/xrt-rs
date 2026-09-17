//! Paraboloid lens surface.
//!
//! z = (x² + y²) / (4f), clamped at z_max.
//!
//! Ported from oes.py lens surface logic.

use crate::surface::Surface;

/// A paraboloid lens surface (plano-convex/concave).
#[derive(Debug, Clone, Copy)]
pub struct ParaboloidLensSurface {
    /// Focal length f [mm]
    pub focus: f64,
    /// Maximum sag (clipping height) [mm]. None = no clip.
    pub z_max: Option<f64>,
}

impl ParaboloidLensSurface {
    pub fn new(focus: f64, z_max: Option<f64>) -> Self {
        Self { focus, z_max }
    }
}

impl Surface for ParaboloidLensSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        let rho2 = x * x + y * y;
        let z = rho2 / (4.0 * self.focus);
        match self.z_max {
            Some(zmax) => z.min(zmax),
            None => z,
        }
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let rho2 = x * x + y * y;
        let z = rho2 / (4.0 * self.focus);

        // If clipped, return flat normal
        if let Some(zmax) = self.z_max
            && z >= zmax
        {
            return [0.0, 0.0, 1.0];
        }

        // dz/dx = x / (2f), dz/dy = y / (2f)
        let inv_2f = 1.0 / (2.0 * self.focus);
        let dz_dx = x * inv_2f;
        let dz_dy = y * inv_2f;

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
    fn lens_at_origin() {
        let s = ParaboloidLensSurface::new(100.0, None);
        assert_eq!(s.local_z(0.0, 0.0), 0.0);
        let n = s.local_n(0.0, 0.0);
        assert!((n[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn lens_z_formula() {
        let f = 100.0;
        let s = ParaboloidLensSurface::new(f, None);
        let x = 10.0;
        let expected = x * x / (4.0 * f);
        let got = s.local_z(x, 0.0);
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn lens_z_clipping() {
        let s = ParaboloidLensSurface::new(10.0, Some(0.5));
        // Without clipping, z(10,0) = 100/40 = 2.5
        // With z_max=0.5, should clamp
        let z = s.local_z(10.0, 0.0);
        assert!((z - 0.5).abs() < 1e-15, "z = {z}, expected 0.5");
    }

    #[test]
    fn lens_normal_at_clip() {
        let s = ParaboloidLensSurface::new(10.0, Some(0.5));
        let n = s.local_n(10.0, 0.0);
        // Clipped → flat normal
        assert!((n[2] - 1.0).abs() < 1e-15);
    }
}
