//! Atomic form factor f0 evaluation using Gaussian sum parameterization.
//!
//! Ported from `xrt/backends/raycing/materials.py:237-241`.
//!
//! The parameterization is:
//!   f0(q/(4π)) = c + Σ(a_i * exp(-b_i * (q/(4π))²)) for i=1..5
//!
//! Coefficients layout: [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]

use ndarray::Array1;

/// Coefficients for the f0 Gaussian sum: [a1..a5, c, b1..b5]
pub type F0Coeffs = [f64; 11];

/// Evaluate f0 for a single q/(4π) value.
#[inline]
pub fn f0_scalar(coeffs: &F0Coeffs, q_over_4pi: f64) -> f64 {
    let q2 = q_over_4pi * q_over_4pi;
    let c = coeffs[5];
    let mut sum = c;
    for i in 0..5 {
        sum += coeffs[i] * (-coeffs[6 + i] * q2).exp();
    }
    sum
}

/// Evaluate f0 for an array of q/(4π) values.
pub fn f0_array(coeffs: &F0Coeffs, q_over_4pi: &Array1<f64>) -> Array1<f64> {
    q_over_4pi.mapv(|q| f0_scalar(coeffs, q))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Si (Z=14) coefficients from f0_xop.dat
    const SI_COEFFS: F0Coeffs = [
        5.275_329_53,
        3.191_038_26,
        1.511_468_12,
        1.356_499_03,
        2.519_459_79,
        0.145_667_43,
        2.631_386_59,
        33.730_728_22,
        0.081_119_04,
        86.288_643_53,
        1.170_862_42,
    ];

    #[test]
    fn si_f0_at_zero_equals_z() {
        // At q=0, f0 should equal the atomic number Z=14
        let f0 = f0_scalar(&SI_COEFFS, 0.0);
        assert!((f0 - 14.0).abs() < 0.01, "f0(0) = {f0}, expected ~14");
    }

    #[test]
    fn f0_decreases_with_q() {
        // f0 should decrease monotonically with increasing q
        let f0_0 = f0_scalar(&SI_COEFFS, 0.0);
        let f0_1 = f0_scalar(&SI_COEFFS, 0.5);
        let f0_2 = f0_scalar(&SI_COEFFS, 1.0);
        assert!(f0_0 > f0_1);
        assert!(f0_1 > f0_2);
    }

    #[test]
    fn f0_array_matches_scalar() {
        let qs = Array1::from_vec(vec![0.0, 0.25, 0.5, 1.0]);
        let arr = f0_array(&SI_COEFFS, &qs);
        for (q, &val) in qs.iter().zip(arr.iter()) {
            let scalar = f0_scalar(&SI_COEFFS, *q);
            assert!((val - scalar).abs() < 1e-15);
        }
    }
}
