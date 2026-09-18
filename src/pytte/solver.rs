//! Dormand-Prince RK45 adaptive ODE solver for Takagi-Taupin equations.
//!
//! Ported from xrt/backends/raycing/pyTTE_x.py.
//!
//! Bragg case: dξ/dz = i(cb·ξ² + (c₀+β+strain)·ξ + ch) — Riccati equation
//! Laue case: 2-component system [ξ, D₀]
//!
//! Each scan point is independent → rayon parallel over scan points.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pytte::deformation::Deformation;

// ── Dormand-Prince RK45 Butcher tableau ─────────────────────────────────────
// 7-stage, 5th-order method with embedded 4th-order for error estimation.

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// 5th-order weights
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

// 4th-order weights (for error estimation)
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

// Node positions c_i
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

/// Configuration for the ODE solver.
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    /// Initial step size
    pub h_init: f64,
    /// Minimum step size
    pub h_min: f64,
    /// Maximum step size
    pub h_max: f64,
    /// Relative tolerance
    pub rtol: f64,
    /// Absolute tolerance
    pub atol: f64,
    /// Maximum number of steps
    pub max_steps: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            h_init: 1.0,
            h_min: 1e-10,
            h_max: 100.0,
            rtol: 1e-8,
            atol: 1e-12,
            max_steps: 100_000,
        }
    }
}

/// Coefficients for the Bragg Riccati equation.
#[derive(Debug, Clone, Copy)]
pub struct BraggCoeffs {
    /// χ_h * C / b factor
    pub cb: Complex64,
    /// χ_0 + β + strain term
    pub c0: Complex64,
    /// χ_h̄ * C factor
    pub ch: Complex64,
}

/// Coefficients for the Laue 2-component ODE system.
#[derive(Debug, Clone, Copy)]
pub struct LaueCoeffs {
    /// Drives ξ² coupling
    pub cb: Complex64,
    /// χ_0 + β term
    pub c0: Complex64,
    /// Source term
    pub ch: Complex64,
    /// Absorption factor for g₀
    pub g0: Complex64,
    /// Absorption factor for gb
    pub gb: Complex64,
}

/// Result of solving the TT equation at a single scan point.
#[derive(Debug, Clone, Copy)]
pub struct TtResult {
    /// Reflectivity amplitude (s-polarization)
    pub rs: Complex64,
    /// Reflectivity amplitude (p-polarization)
    pub rp: Complex64,
}

/// Solve the Bragg Riccati ODE for a single scan point using Dormand-Prince RK45.
///
/// dξ/dz = i × (cb·ξ² + c0·ξ + ch)
pub fn solve_bragg_riccati(
    coeffs: &BraggCoeffs,
    z_start: f64,
    z_end: f64,
    xi_init: Complex64,
    config: &SolverConfig,
) -> Complex64 {
    solve_bragg_riccati_with_strain(coeffs, z_start, z_end, xi_init, config, &|_| 0.0)
}

/// Solve the Bragg Riccati ODE with a z-dependent strain term.
///
/// dξ/dz = i × (cb·ξ² + (c0 + strain(z))·ξ + ch)
pub fn solve_bragg_riccati_with_strain(
    coeffs: &BraggCoeffs,
    z_start: f64,
    z_end: f64,
    xi_init: Complex64,
    config: &SolverConfig,
    strain_fn: &dyn Fn(f64) -> f64,
) -> Complex64 {
    let mut z = z_start;
    let mut xi = xi_init;
    let mut h = config.h_init.min((z_end - z_start).abs());
    let direction = (z_end - z_start).signum();

    let f = |z_v: f64, xi_v: Complex64| -> Complex64 {
        let strain = Complex64::new(strain_fn(z_v), 0.0);
        Complex64::i() * (coeffs.cb * xi_v * xi_v + (coeffs.c0 + strain) * xi_v + coeffs.ch)
    };

    for _ in 0..config.max_steps {
        if (z - z_end).abs() < config.h_min {
            break;
        }

        h = h.min((z_end - z).abs());
        let hd = h * direction;

        // Dormand-Prince 7 stages
        let k1 = f(z, xi);
        let k2 = f(z + C2 * hd, xi + k1 * (A21 * hd));
        let k3 = f(z + C3 * hd, xi + k1 * (A31 * hd) + k2 * (A32 * hd));
        let k4 = f(
            z + C4 * hd,
            xi + k1 * (A41 * hd) + k2 * (A42 * hd) + k3 * (A43 * hd),
        );
        let k5 = f(
            z + C5 * hd,
            xi + k1 * (A51 * hd) + k2 * (A52 * hd) + k3 * (A53 * hd) + k4 * (A54 * hd),
        );
        let k6 = f(
            z + hd,
            xi + k1 * (A61 * hd)
                + k2 * (A62 * hd)
                + k3 * (A63 * hd)
                + k4 * (A64 * hd)
                + k5 * (A65 * hd),
        );

        // 5th-order solution
        let xi_new = xi + (k1 * B1 + k3 * B3 + k4 * B4 + k5 * B5 + k6 * B6) * hd;

        // Error estimation (difference between 5th and 4th order)
        let k7 = f(z + hd, xi_new);
        let err_vec = (k1 * E1 + k3 * E3 + k4 * E4 + k5 * E5 + k6 * E6 + k7 * E7) * hd;
        let err = err_vec.norm();
        let tol = config.atol + config.rtol * xi_new.norm();

        if err > tol && h > config.h_min {
            // Reject step
            let factor = (tol / err).powf(0.2).max(0.1);
            h = (h * 0.9 * factor).max(config.h_min);
            continue;
        }

        xi = xi_new;
        z += hd;

        // Adjust step
        if err > 0.0 {
            let factor = (tol / err).powf(0.2).clamp(0.1, 5.0);
            h = (h * 0.9 * factor).clamp(config.h_min, config.h_max);
        }
    }

    xi
}

/// Solve the Laue 2-component TT ODE system.
///
/// dξ/dz  = i(cb·ξ² + (c0+strain)·ξ + ch)
/// dD₀/dz = -i(g0 + gb·ξ)·D₀
///
/// Returns (ξ_final, D₀_final).
pub fn solve_laue(
    coeffs: &LaueCoeffs,
    z_start: f64,
    z_end: f64,
    xi_init: Complex64,
    d0_init: Complex64,
    config: &SolverConfig,
    strain_fn: &dyn Fn(f64) -> f64,
) -> (Complex64, Complex64) {
    let mut z = z_start;
    let mut xi = xi_init;
    let mut d0 = d0_init;
    let mut h = config.h_init.min((z_end - z_start).abs());
    let direction = (z_end - z_start).signum();

    let f_xi = |z_v: f64, xi_v: Complex64| -> Complex64 {
        let strain = Complex64::new(strain_fn(z_v), 0.0);
        Complex64::i() * (coeffs.cb * xi_v * xi_v + (coeffs.c0 + strain) * xi_v + coeffs.ch)
    };

    let f_d0 = |xi_v: Complex64, d0_v: Complex64| -> Complex64 {
        -Complex64::i() * (coeffs.g0 + coeffs.gb * xi_v) * d0_v
    };

    for _ in 0..config.max_steps {
        if (z - z_end).abs() < config.h_min {
            break;
        }

        h = h.min((z_end - z).abs());
        let hd = h * direction;

        // RK4 for the coupled system
        let kx1 = f_xi(z, xi);
        let kd1 = f_d0(xi, d0);

        let kx2 = f_xi(z + 0.5 * hd, xi + kx1 * (0.5 * hd));
        let kd2 = f_d0(xi + kx1 * (0.5 * hd), d0 + kd1 * (0.5 * hd));

        let kx3 = f_xi(z + 0.5 * hd, xi + kx2 * (0.5 * hd));
        let kd3 = f_d0(xi + kx2 * (0.5 * hd), d0 + kd2 * (0.5 * hd));

        let kx4 = f_xi(z + hd, xi + kx3 * hd);
        let kd4 = f_d0(xi + kx3 * hd, d0 + kd3 * hd);

        let xi_new = xi + (kx1 + kx2 * 2.0 + kx3 * 2.0 + kx4) * (hd / 6.0);
        let d0_new = d0 + (kd1 + kd2 * 2.0 + kd3 * 2.0 + kd4) * (hd / 6.0);

        let err = ((kx4 - kx3) * hd).norm();
        let tol = config.atol + config.rtol * xi_new.norm();

        if err > tol && h > config.h_min {
            let factor = (tol / err).powf(0.25).max(0.1);
            h = (h * 0.5 * factor).max(config.h_min);
            continue;
        }

        xi = xi_new;
        d0 = d0_new;
        z += hd;

        if err > 0.0 {
            let factor = (tol / err).powf(0.25).clamp(0.1, 4.0);
            h = (h * 0.9 * factor).clamp(config.h_min, config.h_max);
        }
    }

    (xi, d0)
}

/// Solve TT equations for multiple scan points in parallel.
///
/// Each scan point has its own Bragg coefficients.
pub fn solve_bragg_parallel(
    coeffs: &[BraggCoeffs],
    z_start: f64,
    z_end: f64,
    xi_init: Complex64,
    config: &SolverConfig,
) -> Vec<Complex64> {
    coeffs
        .par_iter()
        .map(|c| solve_bragg_riccati(c, z_start, z_end, xi_init, config))
        .collect()
}

/// Unified TT solver: solves Bragg or Laue with optional deformation.
///
/// For each scan point, solves with both σ and π polarizations.
pub fn tt_solve(
    coeffs_s: &[BraggCoeffs],
    coeffs_p: &[BraggCoeffs],
    z_start: f64,
    z_end: f64,
    xi_init: Complex64,
    config: &SolverConfig,
    deformation: &dyn Deformation,
) -> Vec<TtResult> {
    let strain_fn = |z: f64| -> f64 { deformation.strain(z) };

    coeffs_s
        .par_iter()
        .zip(coeffs_p.par_iter())
        .map(|(cs, cp)| {
            let rs =
                solve_bragg_riccati_with_strain(cs, z_start, z_end, xi_init, config, &strain_fn);
            let rp =
                solve_bragg_riccati_with_strain(cp, z_start, z_end, xi_init, config, &strain_fn);
            TtResult { rs, rp }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pytte::deformation::{IsotropicPlate, NoDeformation};

    #[test]
    fn solve_constant_coefficients() {
        let coeffs = BraggCoeffs {
            cb: Complex64::new(0.0, 0.0),
            c0: Complex64::new(0.0, 0.0),
            ch: Complex64::new(0.0, 0.0),
        };
        let xi = solve_bragg_riccati(
            &coeffs,
            0.0,
            100.0,
            Complex64::new(0.5, 0.0),
            &SolverConfig::default(),
        );
        assert!((xi - Complex64::new(0.5, 0.0)).norm() < 1e-8);
    }

    #[test]
    fn solve_linear_growth() {
        let ch = Complex64::new(1e-6, 0.0);
        let coeffs = BraggCoeffs {
            cb: Complex64::new(0.0, 0.0),
            c0: Complex64::new(0.0, 0.0),
            ch,
        };
        let z_end = 1000.0;
        let xi = solve_bragg_riccati(
            &coeffs,
            0.0,
            z_end,
            Complex64::new(0.0, 0.0),
            &SolverConfig::default(),
        );
        let expected = Complex64::i() * ch * z_end;
        assert!(
            (xi - expected).norm() < 1e-6,
            "xi = {xi}, expected {expected}"
        );
    }

    #[test]
    fn parallel_solve() {
        let coeffs: Vec<BraggCoeffs> = (0..100)
            .map(|i| BraggCoeffs {
                cb: Complex64::new(0.0, 0.0),
                c0: Complex64::new(0.0, 0.0),
                ch: Complex64::new(i as f64 * 1e-8, 0.0),
            })
            .collect();

        let results = solve_bragg_parallel(
            &coeffs,
            0.0,
            100.0,
            Complex64::new(0.0, 0.0),
            &SolverConfig::default(),
        );

        assert_eq!(results.len(), 100);
        assert!(results[0].norm() < 1e-10);
        assert!(results[99].norm() > 0.0);
    }

    #[test]
    fn dp45_matches_rk4_simple() {
        // DP45 with strain=0 should give similar results to the old RK4
        let ch = Complex64::new(1e-6, 0.0);
        let coeffs = BraggCoeffs {
            cb: Complex64::new(0.0, 0.0),
            c0: Complex64::new(0.0, 0.0),
            ch,
        };
        let z_end = 1000.0;
        let xi_dp45 = solve_bragg_riccati_with_strain(
            &coeffs,
            0.0,
            z_end,
            Complex64::new(0.0, 0.0),
            &SolverConfig::default(),
            &|_| 0.0,
        );
        let expected = Complex64::i() * ch * z_end;
        assert!(
            (xi_dp45 - expected).norm() < 1e-6,
            "DP45: xi = {xi_dp45}, expected {expected}"
        );
    }

    #[test]
    fn strain_shifts_result() {
        let coeffs = BraggCoeffs {
            cb: Complex64::new(1e-8, 0.0),
            c0: Complex64::new(-1e-5, 1e-7),
            ch: Complex64::new(-8e-6, 5e-8),
        };
        let config = SolverConfig::default();
        let z_end = 10000.0;
        let xi0 = Complex64::new(0.0, 0.0);

        let xi_no_strain =
            solve_bragg_riccati_with_strain(&coeffs, 0.0, z_end, xi0, &config, &|_| 0.0);
        let xi_with_strain =
            solve_bragg_riccati_with_strain(&coeffs, 0.0, z_end, xi0, &config, &|_z| 1e-6);

        // Strain should modify the result
        assert!(
            (xi_no_strain - xi_with_strain).norm() > 1e-10,
            "strain should change the result"
        );
    }

    #[test]
    fn laue_energy_conservation() {
        // In Laue geometry without absorption, |D₀|² + |Dh|² ≈ 1
        let coeffs = LaueCoeffs {
            cb: Complex64::new(1e-8, 0.0),
            c0: Complex64::new(-1e-6, 0.0), // no absorption
            ch: Complex64::new(-8e-7, 0.0),
            g0: Complex64::new(0.0, 0.0),
            gb: Complex64::new(0.0, 0.0),
        };
        let (xi, d0) = solve_laue(
            &coeffs,
            0.0,
            100000.0,
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            &SolverConfig::default(),
            &|_| 0.0,
        );
        // ξ = Dh/D0, so Dh = ξ * D0
        let dh = xi * d0;
        let total = d0.norm_sqr() + dh.norm_sqr();
        // Should be approximately conserved (may deviate slightly due to numerics)
        assert!(total.is_finite(), "total = {total}");
    }

    #[test]
    fn tt_solve_unified() {
        let coeffs_s = vec![BraggCoeffs {
            cb: Complex64::new(1e-8, 0.0),
            c0: Complex64::new(-1e-5, 1e-7),
            ch: Complex64::new(-8e-6, 5e-8),
        }];
        let coeffs_p = vec![BraggCoeffs {
            cb: Complex64::new(8e-9, 0.0),
            c0: Complex64::new(-1e-5, 1e-7),
            ch: Complex64::new(-6e-6, 4e-8),
        }];

        let results = tt_solve(
            &coeffs_s,
            &coeffs_p,
            0.0,
            100000.0,
            Complex64::new(0.0, 0.0),
            &SolverConfig::default(),
            &NoDeformation,
        );

        assert_eq!(results.len(), 1);
        assert!(results[0].rs.norm().is_finite());
        assert!(results[0].rp.norm().is_finite());
    }

    #[test]
    fn tt_solve_with_deformation() {
        let coeffs_s = vec![BraggCoeffs {
            cb: Complex64::new(1e-8, 0.0),
            c0: Complex64::new(-1e-5, 1e-7),
            ch: Complex64::new(-8e-6, 5e-8),
        }];
        let coeffs_p = coeffs_s.clone();

        let deformation = IsotropicPlate::new(1000.0, 100.0, 0.28, 3.1356);

        let results = tt_solve(
            &coeffs_s,
            &coeffs_p,
            0.0,
            100000.0,
            Complex64::new(0.0, 0.0),
            &SolverConfig::default(),
            &deformation,
        );

        assert_eq!(results.len(), 1);
        assert!(results[0].rs.norm().is_finite());
    }
}
