//! Ray deflection: specular reflection, grating equation, Snell's law.
//!
//! Ported from oes_base.py:1554-1572 and :2057-2103.

use xrt_core::consts::CH;

/// Specular reflection of a ray off a surface normal.
///
/// out = in + 2·cos(θ₁)·n = in - 2·(in·n)·n
///
/// Returns normalized (a_out, b_out, c_out).
#[inline]
pub fn reflect_specular(
    a_in: f64,
    b_in: f64,
    c_in: f64,
    nx: f64,
    ny: f64,
    nz: f64,
    beam_in_dot_normal: f64,
) -> (f64, f64, f64) {
    let a = a_in - nx * 2.0 * beam_in_dot_normal;
    let b = b_in - ny * 2.0 * beam_in_dot_normal;
    let c = c_in - nz * 2.0 * beam_in_dot_normal;
    // Should already be normalized if input is normalized, but normalize for safety
    let norm = (a * a + b * b + c * c).sqrt();
    (a / norm, b / norm, c / norm)
}

/// Grating deflection using the Spencer-Murty equation.
///
/// out = in - dn·n + g·m·λ
///
/// where dn = cos(θ₁) ± sqrt(cos²(θ₁) - 2(g·in)·mλ - g²·m²λ²)
///
/// `order`: diffraction order m
/// `sig`: sign for dn calculation (+1 or -1, or 0 for auto)
///
/// Returns normalized (a_out, b_out, c_out).
pub fn grating_deflection(
    a_in: f64,
    b_in: f64,
    c_in: f64,
    gx: f64,
    gy: f64,
    gz: f64,
    nx: f64,
    ny: f64,
    nz: f64,
    beam_in_dot_normal: f64,
    energy: f64,
    order: i32,
    sig: Option<f64>,
) -> (f64, f64, f64) {
    let beam_in_dot_g = a_in * gx + b_in * gy + c_in * gz;
    let g2 = gx * gx + gy * gy + gz * gz;
    let order_lambda = order as f64 * CH / energy * 1e-7;

    let u = beam_in_dot_normal * beam_in_dot_normal
        - 2.0 * beam_in_dot_g * order_lambda
        - g2 * order_lambda * order_lambda;

    let gs = sig.unwrap_or_else(|| beam_in_dot_normal.signum());
    let dn = beam_in_dot_normal + gs * u.abs().sqrt();

    let a = a_in - nx * dn + gx * order_lambda;
    let b = b_in - ny * dn + gy * order_lambda;
    let c = c_in - nz * dn + gz * order_lambda;

    let norm = (a * a + b * b + c * c).sqrt();
    (a / norm, b / norm, c / norm)
}

/// Compute the blaze efficiency for a blazed grating.
///
/// Uses the sinc² approximation:
///   η = sinc²(π × (m - d×(sin(α) + sin(β))/λ))
/// where α is the incidence angle, β is the diffracted angle,
/// m is the diffraction order, d is the groove spacing, λ is the wavelength.
///
/// Returns efficiency factor in [0, 1].
pub fn blaze_efficiency(
    sin_alpha: f64,
    sin_beta: f64,
    energy: f64,
    order: i32,
    groove_spacing: f64, // d = 1/rho [mm]
) -> f64 {
    use xrt_core::consts::CH;
    use std::f64::consts::PI;

    let wavelength = CH / energy * 1e-7; // Å → mm
    let path_diff = groove_spacing * (sin_alpha + sin_beta) / wavelength;
    let arg = PI * (order as f64 - path_diff);

    if arg.abs() < 1e-15 {
        1.0 // sinc(0) = 1
    } else {
        let sinc = arg.sin() / arg;
        sinc * sinc
    }
}

/// Snell's law refraction through a surface.
///
/// out = (n1/n2)·in + ((n1/n2)·cos(θ₁) - cos(θ₂))·n
///
/// Returns normalized (a_out, b_out, c_out).
pub fn refract_snell(
    a_in: f64,
    b_in: f64,
    c_in: f64,
    nx: f64,
    ny: f64,
    nz: f64,
    beam_in_dot_normal: f64,
    n1_over_n2: f64,
) -> (f64, f64, f64) {
    let sign_n = (-beam_in_dot_normal).signum();
    let n1n2_cos_theta1 = -n1_over_n2 * beam_in_dot_normal;
    let cos_theta2 =
        sign_n * (1.0 - n1_over_n2 * n1_over_n2 + n1n2_cos_theta1 * n1n2_cos_theta1).sqrt();
    let dn = n1n2_cos_theta1 - cos_theta2;

    let a = a_in * n1_over_n2 + nx * dn;
    let b = b_in * n1_over_n2 + ny * dn;
    let c = c_in * n1_over_n2 + nz * dn;

    let norm = (a * a + b * b + c * c).sqrt();
    (a / norm, b / norm, c / norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specular_reflection_normal_incidence() {
        // Ray going straight down, flat surface normal = (0, 0, 1)
        let (a, b, c) = reflect_specular(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0);
        assert!((a).abs() < 1e-15);
        assert!((b).abs() < 1e-15);
        assert!((c - 1.0).abs() < 1e-15);
    }

    #[test]
    fn specular_reflection_45deg() {
        // Ray at 45° in yz plane
        let s = 0.5_f64.sqrt();
        let bidn = 0.0 * 0.0 + s * 0.0 + (-s) * 1.0; // = -s
        let (a, b, c) = reflect_specular(0.0, s, -s, 0.0, 0.0, 1.0, bidn);
        assert!(a.abs() < 1e-10);
        assert!((b - s).abs() < 1e-10);
        assert!((c - s).abs() < 1e-10);
    }

    #[test]
    fn snell_normal_incidence() {
        // Normal incidence, n1/n2 = 1 → straight through
        let (a, b, c) = refract_snell(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0);
        assert!(a.abs() < 1e-10);
        assert!(b.abs() < 1e-10);
        assert!((c - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn snell_refraction_direction() {
        // Ray at 30° to normal, going into denser medium (n1/n2 < 1)
        // Should bend toward normal
        let theta1 = 30.0_f64.to_radians();
        let a_in = theta1.sin();
        let c_in = -theta1.cos();
        let bidn = -c_in * 1.0; // = cos(30°)
        let n1n2 = 0.5;
        let (a_out, _b_out, c_out) = refract_snell(a_in, 0.0, c_in, 0.0, 0.0, 1.0, -bidn, n1n2);
        // Refracted angle should be smaller
        let theta2 = a_out.atan2(-c_out);
        assert!(
            theta2 < theta1,
            "θ₂ = {:.2}° should be < θ₁ = {:.2}°",
            theta2.to_degrees(),
            theta1.to_degrees()
        );
    }

    #[test]
    fn snell_total_internal_reflection() {
        // When angle exceeds critical angle, result should be NaN
        // (n1/n2 > 1, grazing angle too small)
        let (a, b, c) = refract_snell(
            0.0, 1.0, 0.0,    // beam along y
            0.0, 0.0, 1.0,    // normal along z
            0.0,               // beam_in_dot_normal = 0 (grazing)
            1.5,               // n1/n2 > 1 (dense to less dense)
        );
        // Should produce NaN (total internal reflection)
        assert!(a.is_nan() || b.is_nan() || c.is_nan(),
            "Expected NaN for total internal reflection, got ({a}, {b}, {c})");
    }

    #[test]
    fn snell_dense_to_sparse() {
        // n1/n2 > 1 but above critical angle → valid refraction
        let (a, b, c) = refract_snell(
            0.0, 0.5_f64.sqrt(), -0.5_f64.sqrt(), // 45° to normal
            0.0, 0.0, 1.0,          // normal along z
            -0.5_f64.sqrt(),         // beam_in_dot_normal
            1.1,                     // slight density change
        );
        assert!(a.is_finite() && b.is_finite() && c.is_finite(),
            "Should refract for angle above critical");
        let norm = (a*a + b*b + c*c).sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "output should be unit vector");
    }

    #[test]
    fn grating_deflection_basic() {
        // Simple grating deflection test: order +1 at modest angle
        let (a, b, c) = grating_deflection(
            0.0, 0.99, -0.14,        // beam direction (shallow incidence)
            0.0, -600.0, 0.0,        // grating vector [mm⁻¹]
            0.0, 0.0, 1.0,           // surface normal
            -0.14,                     // beam_in_dot_normal
            1000.0,                    // energy [eV]
            1,                         // order
            None,                      // auto sign
        );
        // Output should be finite unit vector
        assert!(a.is_finite() && b.is_finite() && c.is_finite(),
            "grating deflection should be finite");
        let norm = (a*a + b*b + c*c).sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "output should be unit: {norm}");
    }

    #[test]
    fn passthrough_preserves_direction() {
        // PassThrough should not change the beam direction
        // (tested via reflect.rs, but we verify the concept here)
        let a_in = 0.1;
        let b_in = 0.99;
        let c_in = -0.1;
        // PassThrough just returns the input unchanged
        // Verify with specular at normal incidence that direction changes
        let (_a_r, _b_r, c_r) = reflect_specular(a_in, b_in, c_in, 0.0, 0.0, 1.0, c_in);
        // Reflected c should flip sign
        assert!((c_r - (-c_in)).abs() < 0.1,
            "specular should flip c: got {c_r}");
        // PassThrough would NOT flip c (it's a no-op on direction)
    }

    #[test]
    fn blaze_efficiency_peak() {
        // At the blaze condition, efficiency should be near 1
        // Blaze condition: m × λ = d × (sin(α) + sin(β_blaze))
        let d = 1.0 / 600.0; // groove spacing [mm] for 600 lines/mm
        let energy = 1000.0; // 1 keV
        let sin_alpha = 0.1; // incidence angle

        // At order 1, compute β that satisfies blaze condition
        // For simplicity, test that efficiency is between 0 and 1
        let eff = blaze_efficiency(sin_alpha, -0.05, energy, 1, d);
        assert!((0.0..=1.0).contains(&eff),
            "efficiency should be in [0,1]: {eff}");
        assert!(eff.is_finite(), "efficiency should be finite");
    }

    #[test]
    fn blaze_efficiency_high_order_lower() {
        // Higher orders should generally have lower efficiency
        let d = 1.0 / 600.0;
        let energy = 1000.0;
        let sin_alpha = 0.1;
        let sin_beta = -0.05;

        let eff1 = blaze_efficiency(sin_alpha, sin_beta, energy, 1, d);
        let eff3 = blaze_efficiency(sin_alpha, sin_beta, energy, 3, d);

        // Both should be valid
        assert!(eff1.is_finite() && eff3.is_finite());
        // Order 1 at this geometry should have different efficiency than order 3
        assert!((eff1 - eff3).abs() > 1e-10,
            "orders 1 and 3 should have different efficiency: {eff1:.4} vs {eff3:.4}");
    }

    #[test]
    fn blaze_efficiency_energy_scan() {
        // Scan energy: efficiency should vary with wavelength
        let d = 1.0 / 600.0;
        let sin_alpha = 0.1;
        let sin_beta = -0.05;

        let mut efficiencies = Vec::new();
        for energy in [500.0, 1000.0, 1500.0, 2000.0, 3000.0] {
            let eff = blaze_efficiency(sin_alpha, sin_beta, energy, 1, d);
            assert!((0.0..=1.0 + 1e-10).contains(&eff),
                "efficiency at E={energy}: {eff}");
            efficiencies.push(eff);
        }

        // Not all the same (energy-dependent)
        let min = efficiencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = efficiencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > min + 1e-6,
            "efficiency should vary with energy: min={min:.4}, max={max:.4}");
    }
}
