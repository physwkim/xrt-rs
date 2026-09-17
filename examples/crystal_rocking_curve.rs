//! Example: Si(111) crystal rocking curve calculation.
//!
//! Demonstrates: TtCrystal → scan → solve → plot reflectivity vs angle
//!
//! Run: cargo run --example crystal_rocking_curve

use num_complex::Complex64;

use xrt_pytte::crystal::TtCrystal;
use xrt_pytte::deformation::{IsotropicPlate, NoDeformation};
use xrt_pytte::quantity::{ev_to_angstrom, rad_to_arcsec};
use xrt_pytte::scan::{ScanMode, compute_bragg_coeffs};
use xrt_pytte::solver::{SolverConfig, tt_solve};

fn main() {
    println!("=== XRT-RS: Si(111) Rocking Curve ===\n");

    // 1. Define Si(111) crystal
    let crystal = TtCrystal::new(
        [1, 1, 1],
        3.1356,                            // d-spacing [Å]
        1_000_000.0,                       // 100 μm thick
        Complex64::new(-1.53e-5, 1.88e-7), // χ₀
        Complex64::new(-8.51e-6, 1.34e-7), // χ_h
        Complex64::new(-8.51e-6, 1.34e-7), // χ_h̄
    );

    let energy = 10_000.0; // 10 keV
    let lambda = ev_to_angstrom(energy);
    let theta_b = crystal.bragg_angle(lambda).expect("Bragg angle exists");

    println!("Crystal: Si(111)");
    println!("  d = {:.4} Å", crystal.d_spacing);
    println!("  Energy = {} eV (λ = {:.4} Å)", energy, lambda);
    println!(
        "  θ_B = {:.4}° ({:.1} arcsec)\n",
        theta_b.to_degrees(),
        rad_to_arcsec(theta_b)
    );

    // 2. Scan: ±50 arcsec around Bragg angle in 0.5 arcsec steps
    let n_points = 201;
    let d_theta: Vec<f64> = (0..n_points)
        .map(|i| {
            let arcsec = (i as f64 - 100.0) * 0.5;
            arcsec / 206_264.806
        })
        .collect();

    let scan = ScanMode::Angle {
        energy_ev: energy,
        d_theta: d_theta.clone(),
    };

    // 3. Solve — perfect crystal
    let (cs, cp) = compute_bragg_coeffs(&crystal, &scan);
    let results_perfect = tt_solve(
        &cs,
        &cp,
        0.0,
        crystal.thickness,
        Complex64::new(0.0, 0.0),
        &SolverConfig::default(),
        &NoDeformation,
    );

    // 4. Solve — bent crystal (R = 1 m)
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

    // 5. Print rocking curve
    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "Δθ [arcsec]", "|Rs|² perf", "|Rp|² perf", "|Rs|² bent", "|Rp|² bent"
    );
    println!("{}", "-".repeat(60));

    let mut peak_rs = 0.0_f64;
    let mut peak_angle = 0.0_f64;

    for i in (0..n_points).step_by(10) {
        let arcsec = (i as f64 - 100.0) * 0.5;
        let rs2_p = results_perfect[i].rs.norm_sqr();
        let rp2_p = results_perfect[i].rp.norm_sqr();
        let rs2_b = results_bent[i].rs.norm_sqr();
        let rp2_b = results_bent[i].rp.norm_sqr();

        if rs2_p > peak_rs {
            peak_rs = rs2_p;
            peak_angle = arcsec;
        }

        println!(
            "{:>10.1} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            arcsec, rs2_p, rp2_p, rs2_b, rp2_b
        );
    }

    // 6. Summary
    let fwhm_points: Vec<usize> = results_perfect
        .iter()
        .enumerate()
        .filter(|(_, r)| r.rs.norm_sqr() > peak_rs * 0.5)
        .map(|(i, _)| i)
        .collect();

    let fwhm_arcsec = if fwhm_points.len() >= 2 {
        let first = (fwhm_points[0] as f64 - 100.0) * 0.5;
        let last = (*fwhm_points.last().unwrap() as f64 - 100.0) * 0.5;
        last - first
    } else {
        0.0
    };

    println!("\n--- Summary ---");
    println!(
        "  Peak |Rs|² = {:.6} at Δθ = {:.1} arcsec",
        peak_rs, peak_angle
    );
    println!("  FWHM ≈ {:.1} arcsec", fwhm_arcsec);

    let integrated: f64 = results_perfect.iter().map(|r| r.rs.norm_sqr()).sum::<f64>() * 0.5; // × step size in arcsec
    println!("  Integrated |Rs|² = {:.2} arcsec", integrated);
}
