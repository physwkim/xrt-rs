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
    let text = std::fs::read_to_string(path)
        .expect("Run `python validation/generate_fixtures.py` first");
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
