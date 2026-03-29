//! Takagi-Taupin crystal definition.
//!
//! Defines the crystal parameters needed for solving the TT equations,
//! including susceptibilities, asymmetry angle, and compliance tensor.

use num_complex::Complex64;

/// Crystal parameters for Takagi-Taupin calculations.
#[derive(Debug, Clone)]
pub struct TtCrystal {
    /// Miller indices (h, k, l)
    pub hkl: [i32; 3],
    /// d-spacing [Å]
    pub d_spacing: f64,
    /// Crystal thickness [Å]
    pub thickness: f64,
    /// Zeroth-order susceptibility χ₀
    pub chi_0: Complex64,
    /// Diffraction susceptibility χ_h
    pub chi_h: Complex64,
    /// Conjugate susceptibility χ_h̄
    pub chi_hbar: Complex64,
    /// Asymmetry angle α [rad] (deviation from symmetric Bragg)
    pub asymmetry: f64,
    /// Compliance tensor component s₁₃ [Pa⁻¹] for deformation calculations
    pub compliance_s13: f64,
}

impl TtCrystal {
    pub fn new(
        hkl: [i32; 3],
        d_spacing: f64,
        thickness: f64,
        chi_0: Complex64,
        chi_h: Complex64,
        chi_hbar: Complex64,
    ) -> Self {
        Self {
            hkl,
            d_spacing,
            thickness,
            chi_0,
            chi_h,
            chi_hbar,
            asymmetry: 0.0,
            compliance_s13: 0.0,
        }
    }

    /// Set asymmetry angle [rad].
    pub fn with_asymmetry(mut self, alpha: f64) -> Self {
        self.asymmetry = alpha;
        self
    }

    /// Set compliance tensor.
    pub fn with_compliance(mut self, s13: f64) -> Self {
        self.compliance_s13 = s13;
        self
    }

    /// Asymmetry parameter b = -cos(θ_B + α) / cos(θ_B - α).
    pub fn asymmetry_b(&self, theta_b: f64) -> f64 {
        -(theta_b + self.asymmetry).cos() / (theta_b - self.asymmetry).cos()
    }

    /// Bragg angle θ_B [rad] for given wavelength [Å].
    pub fn bragg_angle(&self, wavelength: f64) -> Option<f64> {
        let arg = wavelength / (2.0 * self.d_spacing);
        if arg.abs() <= 1.0 {
            Some(arg.asin())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si111() -> TtCrystal {
        TtCrystal::new(
            [1, 1, 1],
            3.1356,     // Si(111) d-spacing [Å]
            1_000_000.0, // 100 μm = 1e6 Å
            Complex64::new(-1.5e-5, 1e-7),
            Complex64::new(-8e-6, 5e-8),
            Complex64::new(-8e-6, 5e-8),
        )
    }

    #[test]
    fn bragg_angle_si111_10kev() {
        let c = si111();
        let lambda = 12398.419 / 10000.0; // 1.2398 Å at 10 keV
        let theta = c.bragg_angle(lambda).unwrap();
        // θ_B ≈ 11.4° for Si(111) at 10 keV
        let deg = theta.to_degrees();
        assert!(
            (deg - 11.4).abs() < 0.5,
            "θ_B = {deg}°, expected ~11.4°"
        );
    }

    #[test]
    fn symmetric_b_equals_minus_one() {
        let c = si111(); // asymmetry = 0
        let theta = 0.2; // ~11.5°
        let b = c.asymmetry_b(theta);
        assert!(
            (b + 1.0).abs() < 1e-10,
            "b = {b}, expected -1 for symmetric"
        );
    }

    #[test]
    fn asymmetric_b() {
        let c = si111().with_asymmetry(0.05); // 50 mrad asymmetry
        let theta = 0.2;
        let b = c.asymmetry_b(theta);
        // With positive asymmetry, b differs from -1
        assert!(
            (b + 1.0).abs() > 0.01,
            "b = {b}, expected to differ from -1 with asymmetry"
        );
        assert!(b < 0.0, "b = {b}, expected negative");
    }

    #[test]
    fn bragg_angle_out_of_range() {
        let c = si111();
        // Very long wavelength → no diffraction
        let result = c.bragg_angle(100.0);
        assert!(result.is_none());
    }
}
