//! Surface roughness model (shadow3 compatibility).
//!
//! Applies roughness-induced reflectivity reduction using the
//! Debye-Waller factor: R_rough = R_smooth × exp(-Q²σ²)
//! where Q = 4π sin(θ)/λ is the momentum transfer and σ is the RMS roughness.

/// Roughness parameters for a surface.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceRoughness {
    /// RMS roughness σ [mm]
    pub sigma: f64,
}

impl SurfaceRoughness {
    pub fn new(sigma_mm: f64) -> Self {
        Self { sigma: sigma_mm }
    }

    /// From RMS roughness in nm.
    pub fn from_nm(sigma_nm: f64) -> Self {
        Self {
            sigma: sigma_nm * 1e-6,
        }
    }

    /// From RMS roughness in Angstroms.
    pub fn from_angstrom(sigma_a: f64) -> Self {
        Self {
            sigma: sigma_a * 1e-7,
        }
    }

    /// Compute the Debye-Waller roughness factor.
    ///
    /// Returns a multiplicative factor for reflectivity [0, 1].
    /// `sin_theta` is the grazing angle sine, `energy` is photon energy in eV.
    pub fn debye_waller_factor(&self, sin_theta: f64, energy: f64) -> f64 {
        use crate::core::consts::CH;
        let wavelength_mm = CH / energy * 1e-7; // Å → mm
        let q = 4.0 * std::f64::consts::PI * sin_theta / wavelength_mm;
        (-q * q * self.sigma * self.sigma).exp()
    }

    /// Compute roughness factor for an array of angles.
    pub fn debye_waller_array(&self, sin_thetas: &[f64], energy: f64) -> Vec<f64> {
        sin_thetas
            .iter()
            .map(|&st| self.debye_waller_factor(st, energy))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_roughness_factor_is_one() {
        let r = SurfaceRoughness::new(0.0);
        assert!((r.debye_waller_factor(0.01, 10000.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn roughness_reduces_reflectivity() {
        let r = SurfaceRoughness::from_nm(3.0);
        let factor = r.debye_waller_factor(0.01, 10000.0);
        assert!(
            factor > 0.0 && factor < 1.0,
            "roughness factor should be in (0,1): {factor}"
        );
    }

    #[test]
    fn roughness_increases_with_angle() {
        let r = SurfaceRoughness::from_nm(3.0);
        let f1 = r.debye_waller_factor(0.01, 10000.0);
        let f2 = r.debye_waller_factor(0.05, 10000.0);
        assert!(
            f2 < f1,
            "higher angle should reduce factor more: {f1} vs {f2}"
        );
    }

    #[test]
    fn roughness_increases_with_energy() {
        let r = SurfaceRoughness::from_nm(3.0);
        let f1 = r.debye_waller_factor(0.01, 5000.0);
        let f2 = r.debye_waller_factor(0.01, 20000.0);
        assert!(
            f2 < f1,
            "higher energy should reduce factor more: {f1} vs {f2}"
        );
    }
}
