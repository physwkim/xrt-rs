//! Crystal deformation models for Takagi-Taupin calculations.
//!
//! Provides a `Deformation` trait and concrete implementations for
//! isotropic and anisotropic plate bending.

/// Trait for crystal deformation models.
///
/// Returns the strain (du_h/dz) at a given depth z in the crystal.
pub trait Deformation: Send + Sync {
    /// Strain derivative du_h/dz at depth z [Å] from the surface.
    fn strain(&self, z: f64) -> f64;
}

/// No deformation (perfect crystal).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoDeformation;

impl Deformation for NoDeformation {
    #[inline]
    fn strain(&self, _z: f64) -> f64 {
        0.0
    }
}

/// Isotropic plate bending deformation.
///
/// For a crystal plate bent to radius R, the strain varies linearly with depth:
///   ε(z) = (z - t/2) / R × (1 + ν) / (1 - ν)
/// where t is the thickness, ν is Poisson's ratio.
///
/// The strain term for TT is: du_h/dz = d_spacing × dε/dz = d / R × factor
#[derive(Debug, Clone, Copy)]
pub struct IsotropicPlate {
    /// Bending radius [Å]
    pub radius: f64,
    /// Crystal thickness [Å]
    pub thickness: f64,
    /// Poisson's ratio ν
    pub poisson: f64,
    /// d-spacing [Å]
    pub d_spacing: f64,
}

impl IsotropicPlate {
    pub fn new(radius_mm: f64, thickness_um: f64, poisson: f64, d_spacing: f64) -> Self {
        Self {
            radius: radius_mm * 1e7,        // mm → Å
            thickness: thickness_um * 1e4,    // μm → Å
            poisson,
            d_spacing,
        }
    }
}

impl Deformation for IsotropicPlate {
    fn strain(&self, _z: f64) -> f64 {
        // Linear strain gradient: dε/dz = 1/R × (1 + ν) / (1 - ν)
        let factor = (1.0 + self.poisson) / (1.0 - self.poisson);
        self.d_spacing / self.radius * factor
    }
}

/// Anisotropic plate bending deformation.
///
/// Uses the full compliance tensor (s₁₃) instead of Poisson's ratio:
///   ε(z) = -(z - t/2) × s₁₃ / (s₁₁ × R)
/// The strain derivative is constant: dε/dz = -s₁₃/(s₁₁ × R)
#[derive(Debug, Clone, Copy)]
pub struct AnisotropicPlate {
    /// Bending radius [Å]
    pub radius: f64,
    /// Crystal thickness [Å]
    pub thickness: f64,
    /// Compliance ratio s₁₃/s₁₁
    pub compliance_ratio: f64,
    /// d-spacing [Å]
    pub d_spacing: f64,
}

impl AnisotropicPlate {
    pub fn new(
        radius_mm: f64,
        thickness_um: f64,
        compliance_ratio: f64,
        d_spacing: f64,
    ) -> Self {
        Self {
            radius: radius_mm * 1e7,
            thickness: thickness_um * 1e4,
            compliance_ratio,
            d_spacing,
        }
    }
}

impl Deformation for AnisotropicPlate {
    fn strain(&self, _z: f64) -> f64 {
        // dε/dz = -s₁₃/(s₁₁ × R), scaled by d_spacing
        -self.compliance_ratio / self.radius * self.d_spacing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_deformation_zero() {
        let d = NoDeformation;
        assert_eq!(d.strain(0.0), 0.0);
        assert_eq!(d.strain(1e6), 0.0);
    }

    #[test]
    fn isotropic_plate_strain() {
        let d = IsotropicPlate::new(
            1000.0, // R = 1 m
            100.0,  // t = 100 μm
            0.28,   // Si Poisson
            3.1356, // Si(111)
        );
        let s = d.strain(5e5); // mid-crystal
        // Should be non-zero for bent crystal
        assert!(s != 0.0, "strain = {s}");
        assert!(s.abs() < 1.0, "strain should be small: {s}");
    }

    #[test]
    fn anisotropic_plate_strain() {
        let d = AnisotropicPlate::new(
            1000.0,  // R = 1 m
            100.0,   // t = 100 μm
            -0.073,  // s₁₃/s₁₁ for Si
            3.1356,
        );
        let s = d.strain(0.0);
        assert!(s != 0.0, "strain = {s}");
    }

    #[test]
    fn larger_radius_less_strain() {
        let d1 = IsotropicPlate::new(1000.0, 100.0, 0.28, 3.1356);
        let d2 = IsotropicPlate::new(10000.0, 100.0, 0.28, 3.1356);
        assert!(
            d1.strain(5e5).abs() > d2.strain(5e5).abs(),
            "smaller radius = more strain"
        );
    }
}
