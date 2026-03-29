//! Property-based tests for ray deflection physics.

use proptest::prelude::*;
use xrt_oes::deflection::{grating_deflection, reflect_specular, refract_snell};

proptest! {
    /// Specular reflection preserves direction vector normalization.
    #[test]
    fn specular_preserves_norm(
        // Generate a normalized direction (a, b, c)
        theta in 0.001..1.5_f64,
        phi in 0.0..6.28_f64,
    ) {
        let a_in = theta.sin() * phi.cos();
        let b_in = theta.cos();
        let c_in = theta.sin() * phi.sin();
        // Flat surface normal
        let (nx, ny, nz) = (0.0, 0.0, 1.0);
        let bidn = a_in * nx + b_in * ny + c_in * nz;

        let (a, b, c) = reflect_specular(a_in, b_in, c_in, nx, ny, nz, bidn);
        let norm = (a * a + b * b + c * c).sqrt();
        prop_assert!((norm - 1.0).abs() < 1e-10, "norm = {norm}");
    }

    /// Specular reflection: angle of incidence = angle of reflection.
    #[test]
    fn specular_angle_equality(
        grazing in 0.001..0.5_f64, // grazing angle [rad]
    ) {
        // Ray in y-z plane at grazing angle
        let a_in = 0.0;
        let b_in = grazing.cos();
        let c_in = -grazing.sin();
        let (nx, ny, nz) = (0.0, 0.0, 1.0);
        let bidn = c_in; // = -sin(grazing)

        let (a, b, c) = reflect_specular(a_in, b_in, c_in, nx, ny, nz, bidn);

        // Reflected c should be +sin(grazing)
        prop_assert!((c - grazing.sin()).abs() < 1e-10,
            "c_out = {c}, expected {}", grazing.sin());
        // b should be preserved
        prop_assert!((b - b_in).abs() < 1e-10, "b changed: {} → {}", b_in, b);
    }

    /// Refraction preserves normalization (below critical angle).
    #[test]
    fn refract_preserves_norm(
        theta in 0.01..0.5_f64,  // small angles to avoid total internal reflection
        n_ratio in 0.3..1.0_f64, // into denser medium (no TIR)
    ) {
        let a_in = theta.sin();
        let c_in = -theta.cos();
        let bidn = a_in * 0.0 + 0.0 * 0.0 + c_in * 1.0;

        let (a, b, c) = refract_snell(a_in, 0.0, c_in, 0.0, 0.0, 1.0, bidn, n_ratio);
        let norm = (a * a + b * b + c * c).sqrt();
        prop_assert!((norm - 1.0).abs() < 1e-8, "norm = {norm}");
    }

    /// Refraction with n1/n2 = 1 is identity.
    #[test]
    fn refract_unity_is_identity(
        theta in 0.01..1.4_f64,
    ) {
        let a_in = theta.sin();
        let c_in = -theta.cos();
        let bidn = c_in;

        let (a, _b, c) = refract_snell(a_in, 0.0, c_in, 0.0, 0.0, 1.0, bidn, 1.0);
        prop_assert!((a - a_in).abs() < 1e-10, "a: {} → {}", a_in, a);
        prop_assert!((c - c_in).abs() < 1e-10, "c: {} → {}", c_in, c);
    }

    /// Specular reflection preserves unit vector for arbitrary input directions.
    #[test]
    fn specular_reflection_preserves_unit_vector(
        a_in in -1.0f64..1.0,
        b_in in 0.1f64..1.0,
        c_in in -1.0f64..1.0,
    ) {
        // Normalize input
        let norm_in = (a_in*a_in + b_in*b_in + c_in*c_in).sqrt();
        if norm_in < 1e-10 { return Ok(()); }
        let (ai, bi, ci) = (a_in/norm_in, b_in/norm_in, c_in/norm_in);

        // Normal along z
        let bidn = ci; // beam_in_dot_normal for nz=1
        let (a_out, b_out, c_out) = reflect_specular(ai, bi, ci, 0.0, 0.0, 1.0, bidn);

        // Output should be unit vector
        let norm_out = (a_out*a_out + b_out*b_out + c_out*c_out).sqrt();
        prop_assert!((norm_out - 1.0).abs() < 1e-10,
            "output norm = {}, expected 1.0", norm_out);
    }

    /// Snell refraction preserves unit vector for non-TIR conditions.
    #[test]
    fn snell_refraction_preserves_unit_vector(
        angle in 0.1f64..1.4, // avoid grazing and TIR
        n_ratio in 0.8f64..1.2,
    ) {
        let a_in = 0.0;
        let b_in = angle.cos();
        let c_in = -angle.sin();
        let bidn = c_in; // dot with (0,0,1) normal

        let (a_out, b_out, c_out) = refract_snell(a_in, b_in, c_in, 0.0, 0.0, 1.0, bidn, n_ratio);

        if a_out.is_finite() {
            let norm = (a_out*a_out + b_out*b_out + c_out*c_out).sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-10,
                "refracted norm = {}, expected 1.0", norm);
        }
        // NaN is ok for TIR conditions
    }

    #[test]
    fn grating_deflection_preserves_unit_vector(
        angle in 0.05f64..0.5,
        energy in 500.0f64..30000.0,
        order in -2i32..3,
    ) {
        if order == 0 { return Ok(()); }
        let b_in = angle.cos();
        let c_in = -angle.sin();
        let bidn = c_in;
        let (a, b, c) = grating_deflection(
            0.0, b_in, c_in,
            0.0, -600.0, 0.0,
            0.0, 0.0, 1.0,
            bidn, energy, order, None,
        );
        if a.is_finite() {
            let norm = (a*a + b*b + c*c).sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-10,
                "grating output norm = {}, expected 1.0", norm);
        }
    }
}
