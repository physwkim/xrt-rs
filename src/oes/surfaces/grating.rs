//! Diffraction grating surfaces.
//!
//! - BlazedGrating: sawtooth (triangular) groove profile
//! - LaminarGrating: rectangular groove profile
//!
//! Both implement the `Surface` trait and provide `local_g` for the
//! grating vector used in diffraction calculations.
//!
//! Ported from oes.py grating surface logic.

use crate::oes::surface::Surface;

/// A blazed (sawtooth / triangular profile) diffraction grating.
///
/// The grating grooves run along the x-direction.
/// Groove density ρ is in lines/mm.
#[derive(Debug, Clone, Copy)]
pub struct BlazedGrating {
    /// Groove density [lines/mm]
    pub rho: f64,
    /// Blaze angle [rad]
    pub blaze_angle: f64,
    /// Anti-blaze angle [rad] (typically π/2 or larger than blaze)
    pub anti_blaze_angle: f64,
}

impl BlazedGrating {
    pub fn new(rho: f64, blaze_angle: f64, anti_blaze_angle: f64) -> Self {
        Self {
            rho,
            blaze_angle,
            anti_blaze_angle,
        }
    }
}

impl Surface for BlazedGrating {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        // Sawtooth profile along y
        let d = 1.0 / self.rho; // groove period [mm]
        let y_mod = ((y % d) + d) % d; // positive modulo
        let blaze_width = d * self.anti_blaze_angle.tan()
            / (self.blaze_angle.tan() + self.anti_blaze_angle.tan());

        if y_mod < blaze_width {
            y_mod * self.blaze_angle.tan()
        } else {
            (d - y_mod) * self.anti_blaze_angle.tan()
        }
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        let d = 1.0 / self.rho;
        let y_mod = ((y % d) + d) % d;
        let blaze_width = d * self.anti_blaze_angle.tan()
            / (self.blaze_angle.tan() + self.anti_blaze_angle.tan());

        let dz_dy = if y_mod < blaze_width {
            self.blaze_angle.tan()
        } else {
            -self.anti_blaze_angle.tan()
        };

        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (ny * ny + nz * nz).sqrt();
        [0.0, ny / norm, nz / norm]
    }

    fn local_g(&self, _x: f64, _y: f64) -> Option<[f64; 3]> {
        // Grating vector along -y (grooves along x)
        Some([0.0, -self.rho, 0.0])
    }
}

/// A laminar (rectangular profile) diffraction grating.
///
/// Grooves are rectangular with a specified depth and duty cycle.
#[derive(Debug, Clone, Copy)]
pub struct LaminarGrating {
    /// Groove density [lines/mm]
    pub rho: f64,
    /// Groove depth [mm]
    pub depth: f64,
    /// Duty cycle: ratio of groove width to period (0..1)
    pub duty_cycle: f64,
}

impl LaminarGrating {
    pub fn new(rho: f64, depth: f64, duty_cycle: f64) -> Self {
        Self {
            rho,
            depth,
            duty_cycle: duty_cycle.clamp(0.0, 1.0),
        }
    }
}

impl Surface for LaminarGrating {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        // Rectangular profile along y
        let d = 1.0 / self.rho;
        let y_mod = ((y % d) + d) % d;
        if y_mod < d * self.duty_cycle {
            self.depth // inside groove
        } else {
            0.0 // land (top surface)
        }
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        // Flat-top profile: normal is always [0, 0, 1]
        // (groove walls are vertical, contribute no scattered light)
        let d = 1.0 / self.rho;
        let y_mod = ((y % d) + d) % d;
        let groove_width = d * self.duty_cycle;

        // At groove edges, normal tilts, but for practical raytracing
        // the flat-top approximation is standard
        let _ = (y_mod, groove_width);
        [0.0, 0.0, 1.0]
    }

    fn local_g(&self, _x: f64, _y: f64) -> Option<[f64; 3]> {
        Some([0.0, -self.rho, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blazed_grating_g_vector() {
        let g = BlazedGrating::new(1200.0, 0.01, 0.5);
        let gv = g.local_g(0.0, 0.0).unwrap();
        assert!((gv[0]).abs() < 1e-15);
        assert!((gv[1] + 1200.0).abs() < 1e-10);
        assert!((gv[2]).abs() < 1e-15);
    }

    #[test]
    fn blazed_grating_z_at_zero() {
        let g = BlazedGrating::new(600.0, 0.02, 1.0);
        let z = g.local_z(0.0, 0.0);
        assert!(z.abs() < 1e-15 || z >= 0.0, "z = {z}");
    }

    #[test]
    fn laminar_grating_z_depth() {
        let g = LaminarGrating::new(1200.0, 0.005, 0.5);
        let d = 1.0 / 1200.0;
        // In the groove region
        let z_groove = g.local_z(0.0, d * 0.25);
        assert!((z_groove - 0.005).abs() < 1e-10, "z = {z_groove}");
        // On the land
        let z_land = g.local_z(0.0, d * 0.75);
        assert!(z_land.abs() < 1e-10, "z = {z_land}");
    }

    #[test]
    fn laminar_grating_g_vector() {
        let g = LaminarGrating::new(600.0, 0.01, 0.5);
        let gv = g.local_g(0.0, 0.0).unwrap();
        assert!((gv[1] + 600.0).abs() < 1e-10);
    }

    #[test]
    fn blazed_normal_orientation() {
        let g = BlazedGrating::new(1200.0, 0.02, 0.5);
        let n = g.local_n(0.0, 0.0);
        // Normal should point mostly upward
        assert!(n[2] > 0.9);
    }

    #[test]
    fn laminar_normal_flat() {
        let g = LaminarGrating::new(1200.0, 0.005, 0.5);
        let n = g.local_n(0.0, 0.0);
        assert!((n[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn laminar_grating_local_g() {
        let lg = LaminarGrating::new(600.0, 0.005, 0.5);
        let g = lg.local_g(0.0, 0.0);
        assert!(g.is_some(), "laminar should have grating vector");
        let [gx, gy, gz] = g.unwrap();
        assert!((gx - 0.0).abs() < 1e-15, "gx should be 0");
        assert!((gy - (-600.0)).abs() < 1e-12, "gy should be -rho = -600");
        assert!((gz - 0.0).abs() < 1e-15, "gz should be 0");
    }

    #[test]
    fn blazed_grating_local_g() {
        let bg = BlazedGrating::new(600.0, 0.02, 0.5);
        let g = bg.local_g(0.0, 0.0);
        assert!(g.is_some());
        let [_gx, gy, _gz] = g.unwrap();
        assert!((gy - (-600.0)).abs() < 1e-12, "gy = -rho = -600");
    }
}
