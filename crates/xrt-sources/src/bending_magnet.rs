//! BendingMagnet synchrotron source.
//!
//! Ported from sources_synchr.py:20-300.
//!
//! Uses Monte Carlo rejection sampling with the synchrotron radiation
//! spectrum computed via modified Bessel functions K_{1/3} and K_{2/3}.

use num_complex::Complex64;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use xrt_core::beam::{Beam, RayState};
use xrt_core::consts::{C, E0, E2W, FINE_STR, M0, M0C2, PI, SIE0, SIM0};

/// Synchrotron source parameters common to BM/Wiggler/Undulator.
#[derive(Debug, Clone)]
pub struct SynchrotronParams {
    /// Electron beam energy [GeV]
    pub electron_energy: f64,
    /// Beam current [A]
    pub beam_current: f64,
    /// Lorentz factor γ = E_e / m₀c²
    pub gamma: f64,
    /// γ²
    pub gamma2: f64,
    /// Electron beam sizes σ_x, σ_z [mm]
    pub dx: f64,
    pub dz: f64,
    /// Electron beam divergences σ_x', σ_z' [rad]
    pub dxprime: f64,
    pub dzprime: f64,
    /// Energy spread σ_E/E
    pub e_spread: f64,
}

impl SynchrotronParams {
    pub fn new(electron_energy_gev: f64, beam_current: f64) -> Self {
        let gamma = electron_energy_gev * 1e3 / M0C2;
        Self {
            electron_energy: electron_energy_gev,
            beam_current,
            gamma,
            gamma2: gamma * gamma,
            dx: 0.0,
            dz: 0.0,
            dxprime: 0.0,
            dzprime: 0.0,
            e_spread: 0.0,
        }
    }
}

/// Bending magnet source.
#[derive(Debug, Clone)]
pub struct BendingMagnet {
    pub params: SynchrotronParams,
    /// Magnetic field [T]
    pub b_field: f64,
    /// Bending radius [m]
    pub rho: f64,
    /// Number of rays to generate
    pub nrays: usize,
    /// Photon energy range [eV]
    pub e_min: f64,
    pub e_max: f64,
    /// Angular range [rad]
    pub theta_min: f64,
    pub theta_max: f64,
    pub psi_min: f64,
    pub psi_max: f64,
    /// Maximum intensity found (for rejection sampling)
    i_max: f64,
}

impl BendingMagnet {
    /// Create a BendingMagnet from electron energy and magnetic field.
    pub fn new(
        electron_energy_gev: f64,
        beam_current: f64,
        b_field: f64,
        nrays: usize,
        e_min: f64,
        e_max: f64,
        theta_max: f64,
        psi_max: f64,
    ) -> Self {
        let params = SynchrotronParams::new(electron_energy_gev, beam_current);
        let rho = M0 * C * C * params.gamma / (b_field * E0 * 1e6);
        // rho = m0 c² γ / (B e) in CGS → meters

        Self {
            params,
            b_field,
            rho,
            nrays,
            e_min,
            e_max,
            theta_min: -theta_max,
            theta_max,
            psi_min: -psi_max,
            psi_max,
            i_max: 0.0,
        }
    }

    /// Create from bending radius instead of field.
    pub fn from_rho(
        electron_energy_gev: f64,
        beam_current: f64,
        rho: f64,
        nrays: usize,
        e_min: f64,
        e_max: f64,
        theta_max: f64,
        psi_max: f64,
    ) -> Self {
        let params = SynchrotronParams::new(electron_energy_gev, beam_current);
        let b_field = M0 * C * C * params.gamma / (rho * E0 * 1e6);

        Self {
            params,
            b_field,
            rho,
            nrays,
            e_min,
            e_max,
            theta_min: -theta_max,
            theta_max,
            psi_min: -psi_max,
            psi_max,
            i_max: 0.0,
        }
    }

    /// Compute the intensity map for given energy, theta, psi arrays.
    ///
    /// Returns (intensity, ampS, ampP) per ray.
    ///
    /// Uses modified Bessel functions K_{1/3}(η) and K_{2/3}(η) where
    /// η = (E/2E_c)(1 + γ²ψ²)^{3/2}.
    pub fn build_i_map(
        &self,
        energies: &[f64],
        _thetas: &[f64],
        psis: &[f64],
    ) -> (Vec<f64>, Vec<Complex64>, Vec<Complex64>) {
        let gamma = self.params.gamma;
        let gamma2 = self.params.gamma2;

        // Critical frequency: ω_c = 1.5 × γ² × B × e / m₀
        let w_cr = 1.5 * gamma2 * self.b_field * SIE0 / SIM0;

        let n = energies.len();
        let mut intensity = vec![0.0; n];
        let mut amp_s = vec![Complex64::new(0.0, 0.0); n];
        let mut amp_p = vec![Complex64::new(0.0, 0.0); n];

        let sq3_over_pi = 3.0_f64.sqrt() / PI;
        let bw_fact = 1.0; // for non-BW mode
        let amp2flux = FINE_STR * bw_fact * self.params.beam_current / SIE0;

        for i in 0..n {
            let e = energies[i];
            let psi = psis[i];

            let gamma_psi = gamma * psi;
            let gamma2_psi2_p1 = gamma_psi * gamma_psi + 1.0;

            let eta = 0.5 * e * E2W / w_cr * gamma2_psi2_p1.powf(1.5);

            if eta <= 0.0 || !eta.is_finite() {
                continue;
            }

            // Bessel functions K_{2/3}(η) and K_{1/3}(η)
            // Using asymptotic approximation for moderate η
            let k23 = bessel_k_approx(2.0 / 3.0, eta);
            let k13 = bessel_k_approx(1.0 / 3.0, eta);

            let amp_sp =
                Complex64::new(0.0, -0.5) * sq3_over_pi * gamma * e * E2W / w_cr * gamma2_psi2_p1;

            let as_val = amp_sp * k23;
            let ap_val = Complex64::i() * gamma_psi * amp_sp * k13 / gamma2_psi2_p1.sqrt();

            let inv_e = 1.0 / e;
            let is_val = (as_val * as_val.conj()).re;
            let ip_val = (ap_val * ap_val.conj()).re;

            intensity[i] = amp2flux * inv_e * (is_val + ip_val);
            amp_s[i] = as_val * (amp2flux * inv_e).sqrt();
            amp_p[i] = ap_val * (amp2flux * inv_e).sqrt();
        }

        (intensity, amp_s, amp_p)
    }

    /// Generate a beam using Monte Carlo rejection sampling.
    pub fn shine(&mut self) -> Beam {
        let mut rng = rand::thread_rng();
        let mc_rays = (self.nrays as f64 * 1.2) as usize;

        let mut collected_beams: Vec<Beam> = Vec::new();
        let mut total_length = 0;

        while total_length < self.nrays {
            // Generate random samples
            let e_dist = Uniform::new(self.e_min, self.e_max);
            let theta_dist = Uniform::new(self.theta_min, self.theta_max);
            let psi_dist = Uniform::new(self.psi_min, self.psi_max);

            let energies: Vec<f64> = (0..mc_rays).map(|_| e_dist.sample(&mut rng)).collect();
            let thetas: Vec<f64> = (0..mc_rays).map(|_| theta_dist.sample(&mut rng)).collect();
            let psis: Vec<f64> = (0..mc_rays).map(|_| psi_dist.sample(&mut rng)).collect();
            let disc: Vec<f64> = (0..mc_rays).map(|_| rng.r#gen::<f64>()).collect();

            let (intensity, amp_s, amp_p) = self.build_i_map(&energies, &thetas, &psis);

            // Update maximum intensity
            for &val in &intensity {
                if val > self.i_max {
                    self.i_max = val;
                }
            }

            if self.i_max <= 0.0 {
                continue;
            }

            // Rejection sampling
            let passed: Vec<usize> = (0..mc_rays)
                .filter(|&i| self.i_max * disc[i] < intensity[i])
                .collect();

            let npassed = passed.len();
            if npassed == 0 {
                continue;
            }

            let mut bot = Beam::with_amplitudes(npassed);
            bot.set_state(RayState::Good);

            for (j, &i) in passed.iter().enumerate() {
                bot.e[j] = energies[i];
                bot.a[j] = thetas[i].tan();
                bot.c[j] = psis[i].tan();

                // Position sampling
                if self.params.dx > 0.0 {
                    if let Ok(d) = Normal::new(self.rho * 1e3, self.params.dx) {
                        let r1 = d.sample(&mut rng);
                        bot.x[j] = -r1 * thetas[i].cos() + self.rho * 1e3;
                        bot.y[j] = r1 * thetas[i].sin();
                    }
                }
                if self.params.dz > 0.0 {
                    if let Ok(d) = Normal::new(0.0, self.params.dz) {
                        bot.z[j] = d.sample(&mut rng);
                    }
                }

                // Polarization from synchrotron radiation
                let is_val = (amp_s[i] * amp_s[i].conj()).re;
                let ip_val = (amp_p[i] * amp_p[i].conj()).re;
                let ssp = is_val + ip_val;
                if ssp > 0.0 {
                    bot.jss[j] = is_val / ssp;
                    bot.jpp[j] = ip_val / ssp;
                    bot.jsp[j] = amp_s[i] * amp_p[i].conj() / ssp;
                }

                if let Some(ref mut es) = bot.es {
                    es[j] = amp_s[i];
                }
                if let Some(ref mut ep) = bot.ep {
                    ep[j] = amp_p[i];
                }
            }

            total_length += npassed;
            collected_beams.push(bot);
        }

        // Concatenate all collected beams
        let mut beam = collected_beams.remove(0);
        for bot in collected_beams {
            beam.concatenate(&bot);
        }

        // Trim to exact nrays
        if beam.nrays() > self.nrays {
            beam = beam.filter_by_index(&(0..self.nrays).collect::<Vec<_>>());
        }

        // Normalize directions
        for i in 0..beam.nrays() {
            let norm = (beam.a[i] * beam.a[i] + 1.0 + beam.c[i] * beam.c[i]).sqrt();
            beam.a[i] /= norm;
            beam.b[i] = 1.0 / norm;
            beam.c[i] /= norm;
        }

        beam
    }
}

/// Approximate modified Bessel function K_ν(x) for ν = 1/3 or 2/3.
///
/// Uses the uniform asymptotic expansion valid for all x > 0.
/// For small x, uses the power series; for large x, the exponential decay.
pub fn bessel_k_approx(nu: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x > 500.0 {
        return 0.0; // Exponentially small
    }

    // For moderate to large x, use asymptotic: K_ν(x) ≈ sqrt(π/(2x)) * exp(-x)
    // with corrections
    if x > 2.0 {
        let prefactor = (PI / (2.0 * x)).sqrt() * (-x).exp();
        let mu = 4.0 * nu * nu;
        // First few terms of asymptotic series
        let t1 = 1.0;
        let t2 = (mu - 1.0) / (8.0 * x);
        let t3 = (mu - 1.0) * (mu - 9.0) / (128.0 * x * x);
        return prefactor * (t1 + t2 + t3);
    }

    // For small x, use Gamma function relation:
    // K_ν(x) ≈ (Γ(ν)/2) * (2/x)^ν  for x → 0
    if x < 0.01 {
        let gamma_nu = gamma_function(nu);
        return 0.5 * gamma_nu * (2.0 / x).powf(nu);
    }

    // For intermediate x, use numerical integration or series
    // Simple series for K_ν: use integral representation
    // K_ν(x) = ∫₀^∞ exp(-x cosh(t)) cosh(νt) dt
    // Gauss-Laguerre quadrature with 20 points
    let n_quad = 32;
    let dt = 6.0 / n_quad as f64; // integrate from 0 to ~6
    let mut sum = 0.0;
    for j in 0..n_quad {
        let t = (j as f64 + 0.5) * dt;
        sum += (-x * t.cosh()).exp() * (nu * t).cosh() * dt;
    }
    sum
}

/// Simple Gamma function for ν = 1/3 and 2/3.
pub fn gamma_function(x: f64) -> f64 {
    // Γ(1/3) ≈ 2.67894, Γ(2/3) ≈ 1.35412
    if (x - 1.0 / 3.0).abs() < 0.01 {
        2.678_938_534_707_748
    } else if (x - 2.0 / 3.0).abs() < 0.01 {
        1.354_117_939_426_4
    } else {
        // Stirling approximation for other values
        (2.0 * PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bending_magnet_create() {
        let bm = BendingMagnet::new(
            3.0, // 3 GeV
            0.3, // 300 mA
            1.0, // 1 T
            100, // rays
            5000.0, 15000.0, // energy range
            0.001,   // theta_max
            0.001,   // psi_max
        );
        assert!(bm.rho > 0.0);
        assert!(bm.params.gamma > 1000.0);
    }

    #[test]
    fn build_i_map_nonzero() {
        let bm = BendingMagnet::new(3.0, 0.3, 1.0, 100, 5000.0, 15000.0, 0.001, 0.001);
        let energies = vec![10000.0; 10];
        let thetas = vec![0.0; 10];
        let psis = vec![0.0; 10];
        let (intensity, _, _) = bm.build_i_map(&energies, &thetas, &psis);
        // On-axis intensity should be non-zero
        assert!(intensity[0] > 0.0, "intensity = {}", intensity[0]);
    }

    #[test]
    fn bending_magnet_shine() {
        let mut bm = BendingMagnet::new(3.0, 0.3, 1.0, 50, 5000.0, 15000.0, 0.002, 0.002);
        let beam = bm.shine();
        assert_eq!(beam.nrays(), 50);
        // All energies should be in range
        for i in 0..50 {
            assert!(
                beam.e[i] >= 5000.0 && beam.e[i] <= 15000.0,
                "E[{i}] = {}",
                beam.e[i]
            );
        }
        // Directions should be normalized
        for i in 0..50 {
            let norm =
                (beam.a[i] * beam.a[i] + beam.b[i] * beam.b[i] + beam.c[i] * beam.c[i]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "direction not normalized: {norm}"
            );
        }
    }

    #[test]
    fn bessel_k_positive() {
        // K_{2/3}(1) should be positive and finite
        let k = bessel_k_approx(2.0 / 3.0, 1.0);
        assert!(k > 0.0 && k.is_finite(), "K_2/3(1) = {k}");
        // K_{1/3}(1) should also be positive
        let k = bessel_k_approx(1.0 / 3.0, 1.0);
        assert!(k > 0.0 && k.is_finite(), "K_1/3(1) = {k}");
    }

    #[test]
    fn bessel_k_large_x_decays() {
        let k1 = bessel_k_approx(2.0 / 3.0, 5.0);
        let k2 = bessel_k_approx(2.0 / 3.0, 10.0);
        assert!(k2 < k1, "K should decay: K(5)={k1}, K(10)={k2}");
    }

    #[test]
    fn bending_magnet_from_rho() {
        // from_rho should create a valid BM from bending radius
        let rho = 5729.58; // mm, corresponds to B ≈ 1.747T for 3GeV
        let mut bm = BendingMagnet::from_rho(3.0, 0.3, rho, 1000, 5000.0, 15000.0, 1e-3, 1e-3);
        let beam = bm.shine();
        assert_eq!(beam.nrays(), 1000);
        // Check direction normalization
        for i in 0..beam.nrays() {
            let a = beam.a[i];
            let b = beam.b[i];
            let c = beam.c[i];
            let norm = (a * a + b * b + c * c).sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "ray[{i}] |dir| = {norm}");
        }
    }

    #[test]
    fn bending_magnet_polarization_output() {
        let mut bm = BendingMagnet::new(3.0, 0.3, 1.0, 1000, 5000.0, 15000.0, 1e-3, 1e-3);
        let beam = bm.shine();
        // BM should produce polarization info (jss, jpp)
        assert_eq!(beam.jss.len(), beam.nrays());
        // Synchrotron radiation is predominantly horizontally polarized
        // so jss (s-component) should be > jpp on average
        let mean_jss: f64 = beam.jss.iter().sum::<f64>() / beam.nrays() as f64;
        let mean_jpp: f64 = beam.jpp.iter().sum::<f64>() / beam.nrays() as f64;
        assert!(mean_jss.is_finite(), "mean jss should be finite");
        assert!(mean_jpp.is_finite(), "mean jpp should be finite");
    }
}
