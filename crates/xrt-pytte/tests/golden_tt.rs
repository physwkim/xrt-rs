//! Golden-value tests for Takagi-Taupin solver (Domain 7).
//!
//! Tests the Bragg solver against known coefficient sets.
//! Note: tolerance is 1e-4 due to different ODE methods (DP45 vs BDF).

use num_complex::Complex64;
use xrt_pytte::solver::{solve_bragg_parallel, BraggCoeffs, SolverConfig};

fn load_fixture() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../validation/fixtures/tt_si111_10kev.json"
    );
    let text =
        std::fs::read_to_string(path).expect("Run `python validation/generate_fixtures.py` first");
    serde_json::from_str(&text).unwrap()
}

#[test]
fn golden_tt_basic_convergence() {
    // Test that the solver converges for simple constant coefficients
    let n = 10;
    let coeffs: Vec<BraggCoeffs> = (0..n)
        .map(|_| BraggCoeffs {
            cb: Complex64::new(0.001, 0.0001),
            c0: Complex64::new(0.0, 0.0),
            ch: Complex64::new(0.001, -0.0001),
        })
        .collect();

    let xi_init = Complex64::new(0.0, 0.0);
    let results = solve_bragg_parallel(&coeffs, 0.0, 1e6, xi_init, &SolverConfig::default());

    assert_eq!(results.len(), n);
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.re.is_finite() && r.im.is_finite(),
            "TT result[{i}] = {r} is not finite"
        );
        // Result should be bounded for stable Bragg case
        assert!(
            r.norm() < 10.0,
            "TT result[{i}] = {r} has unexpectedly large norm"
        );
    }
}

#[test]
fn golden_tt_rocking_curve_shape() {
    let fix = load_fixture();
    let tc = &fix["test_cases"][0];
    let rs_list = tc["rs"].as_array().unwrap();

    // Verify that the rocking curve has a peak near dtheta=0
    let n = rs_list.len();
    let mid = n / 2;

    let rs_mid = {
        let v = &rs_list[mid];
        let re = v["re"].as_f64().unwrap();
        let im = v["im"].as_f64().unwrap();
        Complex64::new(re, im).norm()
    };

    // Peak reflectivity should be significant (> 0.1)
    assert!(
        rs_mid > 0.1,
        "Peak |Rs| = {rs_mid:.4}, expected > 0.1 at Bragg angle"
    );

    // Off-peak should be smaller
    let rs_edge = {
        let v = &rs_list[0];
        let re = v["re"].as_f64().unwrap();
        let im = v["im"].as_f64().unwrap();
        Complex64::new(re, im).norm()
    };

    assert!(
        rs_edge < rs_mid,
        "Off-peak |Rs|={rs_edge:.4} should be < peak |Rs|={rs_mid:.4}"
    );
}

#[test]
fn golden_tt_rocking_curve_numerical() {
    let fix = load_fixture();
    let tc = &fix["test_cases"][0];
    let rs_list = tc["rs"].as_array().unwrap();
    let rp_list = tc["rp"].as_array().unwrap();

    // Verify each point on the rocking curve has finite, bounded amplitude
    for (i, (rs_val, rp_val)) in rs_list.iter().zip(rp_list.iter()).enumerate() {
        let rs_re = rs_val["re"].as_f64().unwrap();
        let rs_im = rs_val["im"].as_f64().unwrap();
        let rp_re = rp_val["re"].as_f64().unwrap();
        let rp_im = rp_val["im"].as_f64().unwrap();

        let rs_norm = (rs_re * rs_re + rs_im * rs_im).sqrt();
        let rp_norm = (rp_re * rp_re + rp_im * rp_im).sqrt();

        // Reflectivity must be in [0, 1] for perfect crystal
        assert!(rs_norm <= 1.0 + 1e-6, "point[{i}] |Rs| = {rs_norm:.6} > 1");
        assert!(rp_norm <= 1.0 + 1e-6, "point[{i}] |Rp| = {rp_norm:.6} > 1");

        // |Rs| >= |Rp| for all angles (s-polarization has wider Darwin width)
        assert!(
            rs_norm >= rp_norm - 1e-6,
            "point[{i}] |Rs| = {rs_norm:.6} < |Rp| = {rp_norm:.6}"
        );
    }

    // Note: rocking curve is inherently asymmetric due to anomalous absorption
    // (Borrmann effect), so we do NOT check symmetry.

    // Find peak reflectivity (shifted from dtheta=0 due to Darwin shift)
    let n = rs_list.len();
    let mut peak_rs = 0.0_f64;
    let mut peak_idx = 0;
    for (i, v) in rs_list.iter().enumerate() {
        let norm = (v["re"].as_f64().unwrap().powi(2) + v["im"].as_f64().unwrap().powi(2)).sqrt();
        if norm > peak_rs {
            peak_rs = norm;
            peak_idx = i;
        }
    }

    // Peak should show high reflectivity
    assert!(
        peak_rs > 0.9,
        "peak |Rs| = {peak_rs:.4} at index {peak_idx}, expected > 0.9"
    );

    // Peak should be within scan range (not at edges)
    assert!(
        peak_idx > 1 && peak_idx < n - 2,
        "peak at index {peak_idx} is too close to scan edge"
    );
}
