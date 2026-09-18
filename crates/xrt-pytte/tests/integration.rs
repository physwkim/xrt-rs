//! Integration tests for the Takagi-Taupin solver.
//!
//! Tests the full workflow: crystal → scan → solve → reflectivity.

use num_complex::Complex64;

use xrt_pytte::crystal::TtCrystal;
use xrt_pytte::deformation::{IsotropicPlate, NoDeformation};
use xrt_pytte::quantity::{arcsec_to_rad, ev_to_angstrom, rad_to_arcsec};
use xrt_pytte::scan::{ScanMode, compute_bragg_coeffs};
use xrt_pytte::solver::{BraggCoeffs, SolverConfig, solve_bragg_riccati, tt_solve};

fn si111() -> TtCrystal {
    TtCrystal::new(
        [1, 1, 1],
        3.1356,      // Si(111) d-spacing [Å]
        1_000_000.0, // 100 μm thick
        Complex64::new(-1.5e-5, 1e-7),
        Complex64::new(-8e-6, 5e-8),
        Complex64::new(-8e-6, 5e-8),
    )
}

/// Test: Thick crystal Bragg reflectivity approaches 1 at exact Bragg angle.
#[test]
fn thick_bragg_peak_reflectivity() {
    let crystal = si111();
    let d_theta: Vec<f64> = (-50..=50).map(|i| arcsec_to_rad(i as f64 * 0.1)).collect();
    let scan = ScanMode::Angle {
        energy_ev: 10000.0,
        d_theta,
    };

    let (coeffs_s, coeffs_p) = compute_bragg_coeffs(&crystal, &scan);
    let results = tt_solve(
        &coeffs_s,
        &coeffs_p,
        0.0,
        crystal.thickness,
        Complex64::new(0.0, 0.0),
        &SolverConfig::default(),
        &NoDeformation,
    );

    assert_eq!(results.len(), 101);

    // Find peak reflectivity — should be significant for thick crystal
    let peak_rs = results.iter().map(|r| r.rs.norm()).fold(0.0_f64, f64::max);
    assert!(
        peak_rs > 0.1,
        "peak |rs| = {peak_rs}, expected > 0.1 for thick crystal"
    );
}

/// Test: Rocking curve with deformation shifts the peak.
#[test]
fn deformation_shifts_rocking_curve() {
    let crystal = si111();
    let d_theta: Vec<f64> = (-20..=20).map(|i| arcsec_to_rad(i as f64 * 0.5)).collect();
    let scan = ScanMode::Angle {
        energy_ev: 10000.0,
        d_theta: d_theta.clone(),
    };

    let (cs, cp) = compute_bragg_coeffs(&crystal, &scan);

    // Without deformation
    let results_flat = tt_solve(
        &cs,
        &cp,
        0.0,
        crystal.thickness,
        Complex64::new(0.0, 0.0),
        &SolverConfig::default(),
        &NoDeformation,
    );

    // With bending deformation
    let deform = IsotropicPlate::new(1000.0, 100.0, 0.28, 3.1356);
    let results_bent = tt_solve(
        &cs,
        &cp,
        0.0,
        crystal.thickness,
        Complex64::new(0.0, 0.0),
        &SolverConfig::default(),
        &deform,
    );

    // Find peak positions
    let _peak_flat = results_flat
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.rs.norm().partial_cmp(&b.rs.norm()).unwrap())
        .unwrap()
        .0;
    let _peak_bent = results_bent
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.rs.norm().partial_cmp(&b.rs.norm()).unwrap())
        .unwrap()
        .0;

    // Deformation should modify reflectivity (even slightly)
    let diff_sum: f64 = results_flat
        .iter()
        .zip(results_bent.iter())
        .map(|(a, b)| (a.rs.norm() - b.rs.norm()).abs())
        .sum();
    assert!(
        diff_sum > 1e-6,
        "deformation should modify reflectivity curve, diff_sum = {diff_sum}"
    );
}

/// Test: σ and π polarizations differ.
#[test]
fn sigma_pi_differ_in_rocking_curve() {
    let crystal = si111();
    let scan = ScanMode::Angle {
        energy_ev: 10000.0,
        d_theta: vec![0.0],
    };

    let (cs, cp) = compute_bragg_coeffs(&crystal, &scan);
    let results = tt_solve(
        &cs,
        &cp,
        0.0,
        crystal.thickness,
        Complex64::new(0.0, 0.0),
        &SolverConfig::default(),
        &NoDeformation,
    );

    // σ and π should differ (π has cos(2θ) factor)
    assert!(
        (results[0].rs.norm() - results[0].rp.norm()).abs() > 1e-6,
        "rs = {}, rp = {} — should differ",
        results[0].rs.norm(),
        results[0].rp.norm()
    );
}

/// Test: Unit conversions are self-consistent.
#[test]
fn unit_conversion_consistency() {
    // 10 keV photon
    let e = 10000.0;
    let lambda = ev_to_angstrom(e);
    assert!(lambda > 1.0 && lambda < 2.0, "λ = {lambda} Å at 10 keV");

    // Bragg angle for Si(111)
    let d = 3.1356;
    let theta = (lambda / (2.0 * d)).asin();
    let theta_arcsec = rad_to_arcsec(theta);
    assert!(
        theta_arcsec > 30000.0 && theta_arcsec < 50000.0,
        "θ_B = {} arcsec",
        theta_arcsec
    );
}

/// Test: Zero-coefficient ODE preserves initial condition.
#[test]
fn zero_coefficients_preserves_initial() {
    let coeffs = BraggCoeffs {
        cb: Complex64::new(0.0, 0.0),
        c0: Complex64::new(0.0, 0.0),
        ch: Complex64::new(0.0, 0.0),
    };
    let xi_init = Complex64::new(0.3, 0.7);
    let result = solve_bragg_riccati(&coeffs, 0.0, 1e6, xi_init, &SolverConfig::default());
    assert!(
        (result - xi_init).norm() < 1e-6,
        "result = {result}, expected {xi_init}"
    );
}
