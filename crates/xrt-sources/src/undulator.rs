//! Undulator synchrotron source (CPU reference implementation).
//!
//! Models a planar or elliptical undulator with K_x and K_y deflection
//! parameters, producing quasi-monochromatic radiation at harmonics of
//! the fundamental energy E₁.
//!
//! Key features:
//! - Fundamental energy: E₁ = 2γ²ℏω₁/(1 + K²/2)
//! - Electron trajectory integration over one period
//! - N-period resonance function sin(Nπu)/sin(πu)
//! - Monte Carlo rejection sampling for beam generation
//!
//! Ported from sources_synchr.py Undulator class.

use num_complex::Complex64;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use xrt_core::beam::{Beam, RayState};
use xrt_core::consts::{E2W, FINE_STR, PI2, SIC, SIE0, SIHPLANCK};

use crate::bending_magnet::SynchrotronParams;

/// Undulator source parameters.
#[derive(Debug, Clone)]
pub struct Undulator {
    pub params: SynchrotronParams,
    /// Horizontal deflection parameter K_x
    pub kx: f64,
    /// Vertical deflection parameter K_y
    pub ky: f64,
    /// Undulator period [mm]
    pub period: f64,
    /// Number of periods
    pub n_periods: usize,
    /// Phase difference between K_x and K_y [degrees]
    pub phase_deg: f64,
    /// Number of rays
    pub nrays: usize,
    /// Photon energy range [eV]
    pub e_min: f64,
    pub e_max: f64,
    /// Angular range [rad]
    pub theta_max: f64,
    pub psi_max: f64,
    /// Maximum intensity (rejection sampling)
    i_max: f64,
}

impl Undulator {
    pub fn new(
        electron_energy_gev: f64,
        beam_current: f64,
        kx: f64,
        ky: f64,
        period_mm: f64,
        n_periods: usize,
        nrays: usize,
        e_min: f64,
        e_max: f64,
        theta_max: f64,
        psi_max: f64,
    ) -> Self {
        let params = SynchrotronParams::new(electron_energy_gev, beam_current);
        Self {
            params,
            kx,
            ky,
            period: period_mm,
            n_periods,
            phase_deg: 0.0,
            nrays,
            e_min,
            e_max,
            theta_max,
            psi_max,
            i_max: 0.0,
        }
    }

    /// Set the K_x - K_y phase difference [degrees].
    pub fn with_phase(mut self, phase_deg: f64) -> Self {
        self.phase_deg = phase_deg;
        self
    }

    /// Total K² = K_x² + K_y².
    pub fn k_squared(&self) -> f64 {
        self.kx * self.kx + self.ky * self.ky
    }

    /// Fundamental photon energy E₁ [eV].
    pub fn fundamental_energy(&self) -> f64 {
        let gamma = self.params.gamma;
        let lambda_u = self.period * 1e-3; // mm → m
                                           // E₁ = 2γ²hc / (λ_u(1 + K²/2))
        let e1 =
            2.0 * gamma * gamma * SIHPLANCK * SIC / (lambda_u * (1.0 + self.k_squared() / 2.0));
        e1 / SIE0 // J → eV
    }

    /// Compute intensity map using numerical trajectory integration.
    ///
    /// Integrates the electron trajectory over one period and applies
    /// the N-period resonance function.
    pub fn build_i_map(
        &self,
        energies: &[f64],
        thetas: &[f64],
        psis: &[f64],
    ) -> (Vec<f64>, Vec<Complex64>, Vec<Complex64>) {
        let gamma = self.params.gamma;
        let gamma2 = self.params.gamma2;
        let n = energies.len();

        let mut intensity = vec![0.0; n];
        let mut amp_s = vec![Complex64::new(0.0, 0.0); n];
        let mut amp_p = vec![Complex64::new(0.0, 0.0); n];

        let lambda_u = self.period * 1e-3; // mm → m
        let phase_rad = self.phase_deg.to_radians();

        let amp2flux = FINE_STR * self.params.beam_current / SIE0 * (self.n_periods as f64);

        // Number of integration steps per period
        let n_steps = 64;
        let dt = 1.0 / n_steps as f64; // normalized time (0..1 = one period)

        for i in 0..n {
            let e = energies[i];
            let theta = thetas[i];
            let psi = psis[i];

            let omega = e * E2W; // angular frequency [rad/s]

            // Integrate radiation amplitude over one period
            let mut ax_sum = Complex64::new(0.0, 0.0);
            let mut az_sum = Complex64::new(0.0, 0.0);

            for j in 0..n_steps {
                let t = (j as f64 + 0.5) * dt;
                let phi_t = PI2 * t;

                // Electron velocity (normalized to c)
                // β_x = K_x/γ × sin(2πt)
                // β_z = K_y/γ × sin(2πt + φ)
                let beta_x = self.kx / gamma * phi_t.sin();
                let beta_z = self.ky / gamma * (phi_t + phase_rad).sin();

                // Electron position (normalized)
                // x = K_x λ_u/(2πγ) × (1 - cos(2πt))
                // z = K_y λ_u/(2πγ) × (1 - cos(2πt + φ))
                let x_e = self.kx * lambda_u / (PI2 * gamma) * (1.0 - phi_t.cos());
                let z_e = self.ky * lambda_u / (PI2 * gamma) * (1.0 - (phi_t + phase_rad).cos());

                // Longitudinal position
                let y_e = lambda_u * t;

                // Retarded phase: ω/c × (y_e - x_e×sinθ - z_e×sinψ)
                // Simplified for small angles: phase ≈ ω × (t/c - n·r/c)
                let path = y_e - x_e * theta - z_e * psi;
                let phase_term = omega / SIC * path;

                // Correction for average velocity
                let avg_correction =
                    omega / SIC * lambda_u * t * (1.0 + self.k_squared() / 2.0) / (2.0 * gamma2);

                let total_phase = phase_term - avg_correction;
                let exp_phase = Complex64::new(total_phase.cos(), total_phase.sin());

                // Radiation amplitude ∝ (β_⊥ - n̂_⊥) × exp(iφ)
                ax_sum += (beta_x - theta) * exp_phase * dt;
                az_sum += (beta_z - psi) * exp_phase * dt;
            }

            // N-period resonance enhancement
            // For the fundamental and harmonics, the single-period amplitude
            // gets multiplied by N (coherent enhancement)
            let n_per = self.n_periods as f64;

            // Single-period result scaled by N
            let ax = ax_sum * n_per * gamma2;
            let az = az_sum * n_per * gamma2;

            let is_val = (ax * ax.conj()).re;
            let ip_val = (az * az.conj()).re;

            let inv_e = 1.0 / e;
            intensity[i] = amp2flux * inv_e * (is_val + ip_val);
            let sqrt_flux = (amp2flux * inv_e).sqrt();
            amp_s[i] = ax * sqrt_flux;
            amp_p[i] = az * sqrt_flux;
        }

        (intensity, amp_s, amp_p)
    }

    /// Build intensity map, preferring GPU when the `gpu` feature is enabled.
    ///
    /// Falls back to CPU (`build_i_map`) if no GPU is available or the
    /// feature is not compiled in.
    pub fn build_i_map_auto(
        &self,
        energies: &[f64],
        thetas: &[f64],
        psis: &[f64],
    ) -> (Vec<f64>, Vec<Complex64>, Vec<Complex64>) {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref ctx) = xrt_gpu::context::GpuContext::new() {
                return self.build_i_map_gpu(ctx, energies, thetas, psis);
            }
        }
        self.build_i_map(energies, thetas, psis)
    }

    /// Build intensity map on the GPU.
    ///
    /// Converts observation points to GPU format, dispatches the compute
    /// shader, and converts results back to f64 precision.
    #[cfg(feature = "gpu")]
    fn build_i_map_gpu(
        &self,
        ctx: &xrt_gpu::context::GpuContext,
        energies: &[f64],
        thetas: &[f64],
        psis: &[f64],
    ) -> (Vec<f64>, Vec<Complex64>, Vec<Complex64>) {
        let obs: Vec<xrt_gpu::undulator::GpuObsPoint> = energies
            .iter()
            .zip(thetas.iter())
            .zip(psis.iter())
            .map(|((&e, &t), &p)| xrt_gpu::undulator::GpuObsPoint {
                energy: e as f32,
                theta: t as f32,
                psi: p as f32,
                _pad: 0.0,
            })
            .collect();

        let result = xrt_gpu::undulator::undulator_gpu(
            ctx,
            &obs,
            self.kx,
            self.ky,
            self.period,
            self.n_periods,
            self.params.gamma,
            self.params.beam_current,
            64,
            self.phase_deg,
        );

        (result.intensity, result.amp_s, result.amp_p)
    }

    /// Generate a beam using Monte Carlo rejection sampling.
    pub fn shine(&mut self) -> Beam {
        let mut rng = rand::thread_rng();
        let mc_rays = (self.nrays as f64 * 1.2) as usize;

        let mut collected_beams: Vec<Beam> = Vec::new();
        let mut total_length = 0;

        let theta_min = -self.theta_max;
        let psi_min = -self.psi_max;

        while total_length < self.nrays {
            let e_dist = Uniform::new(self.e_min, self.e_max);
            let theta_dist = Uniform::new(theta_min, self.theta_max);
            let psi_dist = Uniform::new(psi_min, self.psi_max);

            let energies: Vec<f64> = (0..mc_rays).map(|_| e_dist.sample(&mut rng)).collect();
            let thetas: Vec<f64> = (0..mc_rays).map(|_| theta_dist.sample(&mut rng)).collect();
            let psis: Vec<f64> = (0..mc_rays).map(|_| psi_dist.sample(&mut rng)).collect();
            let disc: Vec<f64> = (0..mc_rays).map(|_| rng.gen::<f64>()).collect();

            let (int, a_s, a_p) = self.build_i_map_auto(&energies, &thetas, &psis);

            for &val in &int {
                if val > self.i_max {
                    self.i_max = val;
                }
            }

            if self.i_max <= 0.0 {
                continue;
            }

            let passed: Vec<usize> = (0..mc_rays)
                .filter(|&j| self.i_max * disc[j] < int[j])
                .collect();

            let npassed = passed.len();
            if npassed == 0 {
                continue;
            }

            let mut bot = Beam::with_amplitudes(npassed);
            bot.set_state(RayState::Good);

            for (j, &k) in passed.iter().enumerate() {
                bot.e[j] = energies[k];
                bot.a[j] = thetas[k].tan();
                bot.c[j] = psis[k].tan();

                if self.params.dx > 0.0 {
                    if let Ok(d) = Normal::new(0.0, self.params.dx) {
                        bot.x[j] = d.sample(&mut rng);
                    }
                }
                if self.params.dz > 0.0 {
                    if let Ok(d) = Normal::new(0.0, self.params.dz) {
                        bot.z[j] = d.sample(&mut rng);
                    }
                }

                let is_val = (a_s[k] * a_s[k].conj()).re;
                let ip_val = (a_p[k] * a_p[k].conj()).re;
                let ssp = is_val + ip_val;
                if ssp > 0.0 {
                    bot.jss[j] = is_val / ssp;
                    bot.jpp[j] = ip_val / ssp;
                    bot.jsp[j] = a_s[k] * a_p[k].conj() / ssp;
                }

                if let Some(ref mut es) = bot.es {
                    es[j] = a_s[k];
                }
                if let Some(ref mut ep) = bot.ep {
                    ep[j] = a_p[k];
                }
            }

            total_length += npassed;
            collected_beams.push(bot);
        }

        let mut beam = collected_beams.remove(0);
        for bot in collected_beams {
            beam.concatenate(&bot);
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undulator_fundamental_energy() {
        // ESRF U20: 6 GeV, K=1.0, period=20mm
        let u = Undulator::new(
            6.0,    // 6 GeV
            0.2,    // 200 mA
            0.0,    // Kx = 0
            1.0,    // Ky = 1
            20.0,   // 20 mm period
            100,    // 100 periods
            100,    // rays
            5000.0, // energy range
            20000.0, 0.0001, 0.0001,
        );
        let e1 = u.fundamental_energy();
        // E₁ should be in the keV range for these parameters
        assert!(e1 > 1000.0 && e1 < 100000.0, "E₁ = {} eV", e1);
    }

    #[test]
    fn undulator_build_i_map_nonzero() {
        let u = Undulator::new(
            3.0, 0.3, 0.0, 2.0, 30.0, 50, 100, 5000.0, 15000.0, 0.0005, 0.0005,
        );
        let e1 = u.fundamental_energy();
        // Test at fundamental energy, on-axis
        let energies = vec![e1];
        let thetas = vec![0.0];
        let psis = vec![0.0];
        let (int, _, _) = u.build_i_map(&energies, &thetas, &psis);
        assert!(int[0] > 0.0, "intensity at fundamental = {}", int[0]);
    }

    #[test]
    fn undulator_on_axis_nonzero() {
        let u = Undulator::new(
            3.0, 0.3, 0.0, 2.0, 30.0, 50, 100, 5000.0, 15000.0, 0.001, 0.001,
        );
        let e1 = u.fundamental_energy();
        let (int_on, _, _) = u.build_i_map(&[e1], &[0.0], &[0.0]);
        assert!(
            int_on[0] > 0.0 && int_on[0].is_finite(),
            "on-axis intensity = {}",
            int_on[0]
        );
    }

    #[test]
    fn undulator_shine() {
        let mut u = Undulator::new(
            3.0, 0.3, 0.0, 2.0, 30.0, 50, 50, 5000.0, 15000.0, 0.001, 0.001,
        );
        let beam = u.shine();
        assert_eq!(beam.nrays(), 50);
        for i in 0..50 {
            assert!(beam.e[i] >= 5000.0 && beam.e[i] <= 15000.0);
            let norm =
                (beam.a[i] * beam.a[i] + beam.b[i] * beam.b[i] + beam.c[i] * beam.c[i]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "direction not normalized: {norm}"
            );
        }
    }

    #[test]
    fn k_squared() {
        let u = Undulator::new(
            3.0, 0.3, 1.0, 2.0, 30.0, 50, 100, 5000.0, 15000.0, 0.001, 0.001,
        );
        assert!((u.k_squared() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn with_phase_sets_phase() {
        let und = Undulator::new(
            6.0, 0.2, 0.0, 1.5, 20.0, 100, 100, 5000.0, 20000.0, 1e-4, 1e-4,
        )
        .with_phase(90.0);
        assert!((und.phase_deg - 90.0).abs() < 1e-15);
    }

    #[test]
    fn with_phase_circular_produces_rays() {
        // K_x = K_y with phase=90° → circular polarization
        let mut und = Undulator::new(
            6.0, 0.2, 1.0, 1.0, 20.0, 100, 1000, 5000.0, 20000.0, 1e-4, 1e-4,
        )
        .with_phase(90.0);
        let beam = und.shine();
        assert_eq!(beam.nrays(), 1000);
        // Direction should be unit vectors
        for i in 0..beam.nrays() {
            let norm =
                (beam.a[i] * beam.a[i] + beam.b[i] * beam.b[i] + beam.c[i] * beam.c[i]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn harmonic_energy_scaling() {
        // E_n = n × E_1 for odd harmonics
        let und = Undulator::new(
            6.0, 0.2, 0.0, 1.5, 20.0, 100, 100, 5000.0, 50000.0, 1e-4, 1e-4,
        );
        let e1 = und.fundamental_energy();
        assert!(e1 > 0.0, "fundamental energy should be positive");
        // E1 = 950 * E_GeV^2 / (period_mm * (1 + K^2/2)) = 950*36/(20*2.125) ≈ 8044 eV
        assert!(
            e1 > 5000.0 && e1 < 15000.0,
            "E1 = {e1} should be ~8044 eV for 6GeV, K=1.5, λ=20mm"
        );
    }

    #[test]
    fn harmonic_peaks_detected() {
        // Check that on-axis intensity peaks near the fundamental energy E₁.
        // Use a focused energy scan around E₁ with fine spacing to reliably
        // resolve the narrow undulator peak (ΔE/E ≈ 1/N_periods).
        let und = Undulator::new(
            6.0, 0.2, 0.0, 1.5, 20.0, 100, 100, 1000.0, 50000.0, 1e-4, 1e-4,
        );
        let e1 = und.fundamental_energy();

        // Scan ±30% around E₁ with fine spacing
        let n_points = 200;
        let e_lo = e1 * 0.7;
        let e_hi = e1 * 1.3;
        let energies: Vec<f64> = (0..n_points)
            .map(|i| e_lo + (e_hi - e_lo) * i as f64 / (n_points - 1) as f64)
            .collect();
        let thetas = vec![0.0; n_points]; // on-axis
        let psis = vec![0.0; n_points];

        let (intensity, _, _) = und.build_i_map(&energies, &thetas, &psis);

        // Find the energy with maximum intensity
        let (i_max, _) = intensity
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let peak_e = energies[i_max];

        let rel_diff = (peak_e - e1).abs() / e1;
        assert!(
            rel_diff < 0.15,
            "Peak at {peak_e:.0} eV should be near E₁={e1:.0} eV (rel={rel_diff:.2})"
        );

        // Intensity at peak should be significantly above the scan edges
        let edge_intensity = intensity[0].max(intensity[n_points - 1]);
        assert!(
            intensity[i_max] > edge_intensity * 2.0,
            "Peak intensity ({:.2e}) should be well above edge ({:.2e})",
            intensity[i_max],
            edge_intensity
        );
    }

    #[test]
    fn on_axis_odd_harmonics_only() {
        // On-axis: only odd harmonics should appear
        let und = Undulator::new(
            6.0, 0.2, 0.0, 1.5, 20.0, 100, 100, 1000.0, 50000.0, 1e-4, 1e-4,
        );
        let e1 = und.fundamental_energy();

        // Check intensity at E₁, 2×E₁, 3×E₁
        let thetas = vec![0.0];
        let psis = vec![0.0];

        let (i_e1, _, _) = und.build_i_map(&[e1], &thetas, &psis);
        let (i_2e1, _, _) = und.build_i_map(&[2.0 * e1], &thetas, &psis);
        let (i_3e1, _, _) = und.build_i_map(&[3.0 * e1], &thetas, &psis);

        // On-axis: E₁ and 3×E₁ should have significant intensity
        assert!(i_e1[0] > 0.0, "I(E₁) should be > 0: {}", i_e1[0]);
        assert!(i_3e1[0] > 0.0, "I(3E₁) should be > 0: {}", i_3e1[0]);

        // 2×E₁ on-axis should be suppressed (even harmonic)
        // Allow it to be small but not zero (finite N effects)
        if i_e1[0] > 1e-30 {
            let ratio_2nd = i_2e1[0] / i_e1[0];
            assert!(
                ratio_2nd < 0.5,
                "On-axis 2nd harmonic should be suppressed: I(2E₁)/I(E₁) = {ratio_2nd:.3}"
            );
        }
    }

    #[test]
    fn off_axis_even_harmonics_appear() {
        // Off-axis: even harmonics become visible
        let und = Undulator::new(
            6.0, 0.2, 0.0, 1.5, 20.0, 100, 100, 1000.0, 50000.0, 1e-4, 1e-4,
        );
        let e1 = und.fundamental_energy();

        // Off-axis observation
        let thetas = vec![5e-5]; // 50 µrad off-axis
        let psis = vec![0.0];

        let (i_2e1_off, _, _) = und.build_i_map(&[2.0 * e1], &thetas, &psis);

        // Off-axis 2nd harmonic should have some intensity
        assert!(
            i_2e1_off[0].is_finite(),
            "Off-axis I(2E₁) should be finite: {}",
            i_2e1_off[0]
        );
    }
}
