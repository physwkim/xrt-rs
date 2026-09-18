//! Conical mirror surface.
//!
//! Used for KB (Kirkpatrick-Baez) focusing mirrors.

use crate::oes::surface::Surface;

/// A conical mirror surface.
#[derive(Debug, Clone, Copy)]
pub struct ConicalSurface {
    /// Half-opening angle [rad]
    pub half_angle: f64,
    /// Distance to cone vertex [mm]
    pub l0: f64,
}

impl ConicalSurface {
    pub fn new(half_angle: f64, l0: f64) -> Self {
        Self { half_angle, l0 }
    }
}

impl Surface for ConicalSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        // Cone: z = -tan(half_angle) * sqrt(x² + (y - l0)²) + apex_z
        // Simplified for small angles: z ≈ -(x² + (y-l0)²) * tan(half_angle) / (2*(y-l0))
        let dy = y - self.l0;
        let r = (x * x + dy * dy).sqrt();
        if r < 1e-30 {
            return 0.0;
        }
        -r * self.half_angle.tan() + self.l0.abs() * self.half_angle.tan()
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let dy = y - self.l0;
        let r = (x * x + dy * dy).sqrt();
        if r < 1e-30 {
            return [0.0, 0.0, 1.0];
        }
        let tan_a = self.half_angle.tan();
        let nx = -(-x / r * tan_a);
        let ny = -(-dy / r * tan_a);
        let nz = 1.0;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conical_at_vertex() {
        let s = ConicalSurface::new(0.01, 1000.0);
        let n = s.local_n(0.0, 1000.0);
        assert!(
            (n[2] - 1.0).abs() < 0.01,
            "normal at vertex should be ~vertical"
        );
    }

    #[test]
    fn conical_symmetric_in_x() {
        let s = ConicalSurface::new(0.01, 500.0);
        let z1 = s.local_z(5.0, 100.0);
        let z2 = s.local_z(-5.0, 100.0);
        assert!((z1 - z2).abs() < 1e-12, "should be symmetric in x");
    }
}
