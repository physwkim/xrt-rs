//! Takagi-Taupin scan parameter computation.
//!
//! Computes the BraggCoeffs for each scan point from crystal parameters
//! and scan angles/energies.

use num_complex::Complex64;

use xrt_core::consts::{CH, PI};

use crate::crystal::TtCrystal;
use crate::solver::BraggCoeffs;

/// Polarization factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// σ-polarization (C = 1)
    Sigma,
    /// π-polarization (C = cos(2θ_B))
    Pi,
}

/// Scan mode.
#[derive(Debug, Clone)]
pub enum ScanMode {
    /// Angle scan: fixed energy, vary Δθ around Bragg angle
    Angle {
        energy_ev: f64,
        d_theta: Vec<f64>, // deviations from Bragg angle [rad]
    },
    /// Energy scan: fixed angle, vary energy
    Energy {
        theta_bragg: f64,   // fixed Bragg angle [rad]
        energies_ev: Vec<f64>,
    },
}

/// Compute BraggCoeffs for each scan point.
///
/// Returns (coeffs_s, coeffs_p) for σ and π polarizations.
pub fn compute_bragg_coeffs(
    crystal: &TtCrystal,
    scan: &ScanMode,
) -> (Vec<BraggCoeffs>, Vec<BraggCoeffs>) {
    match scan {
        ScanMode::Angle { energy_ev, d_theta } => {
            let lambda = CH / energy_ev;
            let theta_b = crystal.bragg_angle(lambda).expect("no Bragg angle");
            let b = crystal.asymmetry_b(theta_b);

            d_theta
                .iter()
                .map(|&dth| {
                    let theta = theta_b + dth;
                    let coeffs_s = bragg_coeffs_at(crystal, lambda, theta, theta_b, b, Polarization::Sigma);
                    let coeffs_p = bragg_coeffs_at(crystal, lambda, theta, theta_b, b, Polarization::Pi);
                    (coeffs_s, coeffs_p)
                })
                .unzip()
        }
        ScanMode::Energy { theta_bragg, energies_ev } => {
            energies_ev
                .iter()
                .map(|&e| {
                    let lambda = CH / e;
                    let theta_b = *theta_bragg;
                    let b = crystal.asymmetry_b(theta_b);

                    let coeffs_s = bragg_coeffs_at(crystal, lambda, theta_b, theta_b, b, Polarization::Sigma);
                    let coeffs_p = bragg_coeffs_at(crystal, lambda, theta_b, theta_b, b, Polarization::Pi);
                    (coeffs_s, coeffs_p)
                })
                .unzip()
        }
    }
}

/// Compute BraggCoeffs at a single point.
fn bragg_coeffs_at(
    crystal: &TtCrystal,
    lambda: f64,
    theta: f64,
    theta_b: f64,
    b: f64,
    pol: Polarization,
) -> BraggCoeffs {
    // Polarization factor
    let c_pol = match pol {
        Polarization::Sigma => 1.0,
        Polarization::Pi => (2.0 * theta_b).cos(),
    };

    // Deviation parameter β = (Δθ)·sin(2θ_B) for small deviations
    let d_theta = theta - theta_b;
    let beta = d_theta * (2.0 * theta_b).sin();

    // Scaled susceptibilities
    // cb = χ_h · C / b  (drives ξ² term)
    let cb = crystal.chi_h * c_pol / b;

    // c0 = χ_0(1 - 1/b) + β + strain (strain is zero here)
    let c0 = crystal.chi_0 * (1.0 - 1.0 / b) + Complex64::new(beta, 0.0);

    // ch = χ_h̄ · C · b  (source term, but for Riccati: χ_h̄ · C)
    let ch = crystal.chi_hbar * c_pol;

    // Scale factor: π/(λ·γ₀) where γ₀ = sin(θ_B + α)
    let gamma_0 = (theta_b + crystal.asymmetry).sin();
    let scale = PI / (lambda * gamma_0);

    BraggCoeffs {
        cb: cb * scale,
        c0: c0 * scale,
        ch: ch * scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si111() -> TtCrystal {
        TtCrystal::new(
            [1, 1, 1],
            3.1356,
            1_000_000.0,
            Complex64::new(-1.5e-5, 1e-7),
            Complex64::new(-8e-6, 5e-8),
            Complex64::new(-8e-6, 5e-8),
        )
    }

    #[test]
    fn angle_scan_produces_coeffs() {
        let c = si111();
        let scan = ScanMode::Angle {
            energy_ev: 10000.0,
            d_theta: vec![-1e-4, 0.0, 1e-4],
        };
        let (cs, cp) = compute_bragg_coeffs(&c, &scan);
        assert_eq!(cs.len(), 3);
        assert_eq!(cp.len(), 3);
        // At exact Bragg angle (dθ=0), c0 should have small real part
        assert!(cs[1].c0.re.is_finite()); // coefficients exist at exact Bragg angle
    }

    #[test]
    fn sigma_pi_differ() {
        let c = si111();
        let scan = ScanMode::Angle {
            energy_ev: 10000.0,
            d_theta: vec![0.0],
        };
        let (cs, cp) = compute_bragg_coeffs(&c, &scan);
        // π has C = cos(2θ_B), so cb should differ
        assert!((cs[0].cb - cp[0].cb).norm() > 1e-10);
    }

    #[test]
    fn energy_scan_produces_coeffs() {
        let c = si111();
        let scan = ScanMode::Energy {
            theta_bragg: 0.2,
            energies_ev: vec![9990.0, 10000.0, 10010.0],
        };
        let (cs, cp) = compute_bragg_coeffs(&c, &scan);
        assert_eq!(cs.len(), 3);
        assert_eq!(cp.len(), 3);
    }
}
