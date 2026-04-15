//! Variable line-spacing (VLS) grating surfaces.
//!
//! Groove density varies with position: rho(y) = rho0 × (1 + c1*y + c2*y² + c3*y³)

use crate::surface::Surface;

/// A VLS laminar grating with position-dependent groove density.
#[derive(Debug, Clone)]
pub struct VlsGrating {
    /// Base groove density [lines/mm]
    pub rho0: f64,
    /// VLS polynomial coefficients [c1, c2, c3, ...] (dimensionless)
    pub coeffs: Vec<f64>,
    /// Groove depth [mm]
    pub depth: f64,
    /// Duty cycle (groove-width / period)
    pub duty: f64,
}

impl VlsGrating {
    pub fn new(rho0: f64, coeffs: Vec<f64>, depth: f64, duty: f64) -> Self {
        Self {
            rho0,
            coeffs,
            depth,
            duty,
        }
    }

    /// Compute local groove density at position y.
    pub fn rho_at(&self, y: f64) -> f64 {
        let mut factor = 1.0;
        let mut y_power = y;
        for &c in &self.coeffs {
            factor += c * y_power;
            y_power *= y;
        }
        self.rho0 * factor
    }
}

impl Surface for VlsGrating {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        let rho = self.rho_at(y);
        let d = 1.0 / rho;
        let y_mod = ((y % d) + d) % d;
        if y_mod < d * self.duty {
            self.depth
        } else {
            0.0
        }
    }

    fn local_n(&self, _x: f64, _y: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0] // laminar grating: flat normal
    }

    fn local_g(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        let rho = self.rho_at(y);
        Some([0.0, -rho, 0.0])
    }
}

/// A VLS blazed grating.
#[derive(Debug, Clone)]
pub struct VlsBlazedGrating {
    /// Base groove density [lines/mm]
    pub rho0: f64,
    /// VLS polynomial coefficients
    pub coeffs: Vec<f64>,
    /// Blaze angle [rad]
    pub blaze: f64,
    /// Anti-blaze angle [rad]
    pub anti_blaze: f64,
}

impl VlsBlazedGrating {
    pub fn new(rho0: f64, coeffs: Vec<f64>, blaze: f64, anti_blaze: f64) -> Self {
        Self {
            rho0,
            coeffs,
            blaze,
            anti_blaze,
        }
    }

    pub fn rho_at(&self, y: f64) -> f64 {
        let mut factor = 1.0;
        let mut y_power = y;
        for &c in &self.coeffs {
            factor += c * y_power;
            y_power *= y;
        }
        self.rho0 * factor
    }
}

impl Surface for VlsBlazedGrating {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        let rho = self.rho_at(y);
        let d = 1.0 / rho;
        let y_mod = ((y % d) + d) % d;
        let blaze_width = d * self.anti_blaze.tan() / (self.blaze.tan() + self.anti_blaze.tan());
        if y_mod < blaze_width {
            y_mod * self.blaze.tan()
        } else {
            (d - y_mod) * self.anti_blaze.tan()
        }
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        let rho = self.rho_at(y);
        let d = 1.0 / rho;
        let y_mod = ((y % d) + d) % d;
        let blaze_width = d * self.anti_blaze.tan() / (self.blaze.tan() + self.anti_blaze.tan());
        let dz_dy = if y_mod < blaze_width {
            self.blaze.tan()
        } else {
            -self.anti_blaze.tan()
        };
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (ny * ny + nz * nz).sqrt();
        [0.0, ny / norm, nz / norm]
    }

    fn local_g(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        let rho = self.rho_at(y);
        Some([0.0, -rho, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vls_uniform_matches_constant() {
        // With empty coeffs, rho should be constant
        let g = VlsGrating::new(600.0, vec![], 0.005, 0.5);
        assert!((g.rho_at(0.0) - 600.0).abs() < 1e-12);
        assert!((g.rho_at(100.0) - 600.0).abs() < 1e-12);
    }

    #[test]
    fn vls_linear_variation() {
        let g = VlsGrating::new(600.0, vec![0.001], 0.005, 0.5);
        // At y=100: rho = 600 * (1 + 0.001*100) = 600 * 1.1 = 660
        assert!((g.rho_at(100.0) - 660.0).abs() < 1e-10);
    }

    #[test]
    fn vls_grating_vector() {
        let g = VlsGrating::new(600.0, vec![0.001], 0.005, 0.5);
        let gv = g.local_g(0.0, 100.0).unwrap();
        assert!((gv[1] - (-660.0)).abs() < 1e-10);
    }
}
