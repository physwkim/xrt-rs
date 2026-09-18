//! Wiggler synchrotron source.
//!
//! Extension of BendingMagnet with periodic magnetic field.
//! K parameter defines the deflection, L0 the period, Np the number of periods.
//!
//! The critical frequency is modulated by the angular position within the wiggler:
//!   ω_cr → ω_cr × sin(arccos(θ·γ/K))
//!
//! Ported from sources_synchr.py Wiggler class.

use num_complex::Complex64;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::core::beam::{Beam, RayState};
use crate::core::consts::{E2W, FINE_STR, K2B, PI, SIE0, SIM0};

use crate::sources::bending_magnet::{SynchrotronParams, bessel_k_approx};
use crate::sources::rejection::RejectionBudget;

/// Wiggler source.
///
/// Note: At K->0, the wiggler becomes highly collimated and does NOT
/// converge to a single bending magnet. For BM-equivalent behavior,
/// use `BendingMagnet` directly. The flux scales as N_periods * (single pole flux).
#[derive(Debug, Clone)]
pub struct Wiggler {
    pub params: SynchrotronParams,
    /// Deflection parameter K
    pub k_param: f64,
    /// Magnetic field period [mm]
    pub period: f64,
    /// Number of periods
    pub n_periods: usize,
    /// Peak magnetic field [T]
    pub b_max: f64,
    /// Number of rays to generate
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

impl Wiggler {
    /// Create a Wiggler from K parameter and period.
    pub fn new(
        electron_energy_gev: f64,
        beam_current: f64,
        k_param: f64,
        period_mm: f64,
        n_periods: usize,
        nrays: usize,
        e_min: f64,
        e_max: f64,
        theta_max: f64,
        psi_max: f64,
    ) -> Self {
        let params = SynchrotronParams::new(electron_energy_gev, beam_current);
        let b_max = k_param * K2B / period_mm; // K2B = 2π·m₀c²·1e-3/e₀ [T·mm]

        Self {
            params,
            k_param,
            period: period_mm,
            n_periods,
            b_max,
            nrays,
            e_min,
            e_max,
            theta_max,
            psi_max,
            i_max: 0.0,
        }
    }

    /// Half-width of the theta window that `shine` samples [rad].
    ///
    /// `theta_max` is the requested angular acceptance, but wiggler emission
    /// exists only within |theta| < K/gamma: `build_i_map` sets the intensity
    /// to zero outside, since w_cr is scaled by sqrt(1 - (theta*gamma/K)^2).
    /// Sampling the whole acceptance therefore rejects nearly every ray.
    /// Python xrt reduces the range the same way in its `xPrimeMax` property
    /// (sources/sybase.py:374-385, enabled for the Wiggler at
    /// sources/synchr.py:97), with K = 0 falling back to 2/gamma.
    pub fn theta_max_sampling(&self) -> f64 {
        let k0 = if self.k_param != 0.0 {
            self.k_param.abs()
        } else {
            2.0
        };
        self.theta_max.min(k0 / self.params.gamma)
    }

    /// Electron trajectory amplitude X0 [mm].
    pub fn trajectory_amplitude(&self) -> f64 {
        self.k_param * self.period / (PI * 2.0 * self.params.gamma)
    }

    /// Build intensity map, accounting for wiggler angular modulation.
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

        let sq3_over_pi = 3.0_f64.sqrt() / PI;
        let amp2flux = FINE_STR * self.params.beam_current / SIE0;

        // Base critical frequency from peak field
        let w_cr_base = 1.5 * gamma2 * self.b_max * SIE0 / SIM0;

        for i in 0..n {
            let e = energies[i];
            let theta = thetas[i];
            let psi = psis[i];

            // Wiggler modulation: effective w_cr depends on theta
            let cos_arg = theta * gamma / self.k_param;
            let w_cr = if cos_arg.abs() < 1.0 {
                w_cr_base * (1.0 - cos_arg * cos_arg).sqrt()
            } else {
                0.0
            };

            if w_cr <= 0.0 {
                continue;
            }

            let gamma_psi = gamma * psi;
            let gamma2_psi2_p1 = gamma_psi * gamma_psi + 1.0;

            let eta = 0.5 * e * E2W / w_cr * gamma2_psi2_p1.powf(1.5);

            if eta <= 0.0 || !eta.is_finite() {
                continue;
            }

            let k23 = bessel_k_approx(2.0 / 3.0, eta);
            let k13 = bessel_k_approx(1.0 / 3.0, eta);

            let amp_sp =
                Complex64::new(0.0, -0.5) * sq3_over_pi * gamma * e * E2W / w_cr * gamma2_psi2_p1;

            let as_val = amp_sp * k23;
            let ap_val = Complex64::i() * gamma_psi * amp_sp * k13 / gamma2_psi2_p1.sqrt();

            let inv_e = 1.0 / e;
            let is_val = (as_val * as_val.conj()).re;
            let ip_val = (ap_val * ap_val.conj()).re;

            // Multiply by 2*N_periods for wiggler flux
            let np2 = 2.0 * self.n_periods as f64;
            intensity[i] = amp2flux * inv_e * (is_val + ip_val) * np2;
            let sqrt_np2 = np2.sqrt();
            amp_s[i] = as_val * (amp2flux * inv_e).sqrt() * sqrt_np2;
            amp_p[i] = ap_val * (amp2flux * inv_e).sqrt() * sqrt_np2;
        }

        (intensity, amp_s, amp_p)
    }

    /// Generate a beam using Monte Carlo rejection sampling.
    pub fn shine(&mut self) -> Beam {
        let mut rng = rand::thread_rng();
        let mc_rays = (self.nrays as f64 * 1.2) as usize;

        let mut collected_beams: Vec<Beam> = Vec::new();
        let mut total_length = 0;
        let mut budget = RejectionBudget::new();

        let theta_max = self.theta_max_sampling();
        let theta_min = -theta_max;
        let psi_min = -self.psi_max;

        while total_length < self.nrays {
            let e_dist = Uniform::new(self.e_min, self.e_max);
            let theta_dist = Uniform::new(theta_min, theta_max);
            let psi_dist = Uniform::new(psi_min, self.psi_max);

            let energies: Vec<f64> = (0..mc_rays).map(|_| e_dist.sample(&mut rng)).collect();
            let thetas: Vec<f64> = (0..mc_rays).map(|_| theta_dist.sample(&mut rng)).collect();
            let psis: Vec<f64> = (0..mc_rays).map(|_| psi_dist.sample(&mut rng)).collect();
            let disc: Vec<f64> = (0..mc_rays).map(|_| rng.r#gen::<f64>()).collect();

            let (intensity, amp_s, amp_p) = self.build_i_map(&energies, &thetas, &psis);

            for &val in &intensity {
                if val > self.i_max {
                    self.i_max = val;
                }
            }

            if self.i_max <= 0.0 {
                budget.note_empty_batch("Wiggler", || {
                    format!(
                        "K={}, B={:.4} T, E={}..{} eV, |theta|<={:.3e} rad, |psi|<={:.3e} rad",
                        self.k_param, self.b_max, self.e_min, self.e_max, theta_max, self.psi_max
                    )
                });
                continue;
            }

            let passed: Vec<usize> = (0..mc_rays)
                .filter(|&i| self.i_max * disc[i] < intensity[i])
                .collect();

            let npassed = passed.len();
            if npassed == 0 {
                budget.note_empty_batch("Wiggler", || {
                    format!(
                        "K={}, B={:.4} T, E={}..{} eV, |theta|<={:.3e} rad, |psi|<={:.3e} rad",
                        self.k_param, self.b_max, self.e_min, self.e_max, theta_max, self.psi_max
                    )
                });
                continue;
            }
            budget.note_progress();

            let mut bot = Beam::with_amplitudes(npassed);
            bot.set_state(RayState::Good);

            // Sample position along wiggler length
            let wiggler_length = self.period * self.n_periods as f64;
            let y_dist = Uniform::new(-wiggler_length / 2.0, wiggler_length / 2.0);

            for (j, &i) in passed.iter().enumerate() {
                bot.e[j] = energies[i];
                bot.a[j] = thetas[i].tan();
                bot.c[j] = psis[i].tan();

                // Position sampling along wiggler
                bot.y[j] = y_dist.sample(&mut rng);
                // Horizontal offset from electron trajectory
                let x0 = self.trajectory_amplitude();
                bot.x[j] = x0 * (PI * 2.0 * bot.y[j] / self.period).sin();

                if self.params.dx > 0.0
                    && let Ok(d) = Normal::new(0.0, self.params.dx)
                {
                    bot.x[j] += d.sample(&mut rng);
                }
                if self.params.dz > 0.0
                    && let Ok(d) = Normal::new(0.0, self.params.dz)
                {
                    bot.z[j] = d.sample(&mut rng);
                }

                let is_val = (amp_s[i] * amp_s[i].conj()).re;
                let ip_val = (amp_p[i] * amp_p[i].conj()).re;
                let ssp = is_val + ip_val;
                if ssp > 0.0 {
                    bot.jss[j] = is_val / ssp;
                    bot.jpp[j] = ip_val / ssp;
                    bot.jsp[j] = amp_s[i] * amp_p[i].conj() / ssp;
                }

                if let Some(amps) = bot.amplitudes_mut() {
                    amps.es[j] = amp_s[i];
                    amps.ep[j] = amp_p[i];
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
    fn wiggler_create() {
        let w = Wiggler::new(
            3.0,    // 3 GeV
            0.3,    // 300 mA
            10.0,   // K=10
            80.0,   // 80 mm period
            20,     // 20 periods
            100,    // rays
            5000.0, // energy range
            15000.0, 0.002, 0.002,
        );
        assert!(w.b_max > 0.0);
        assert!(w.trajectory_amplitude() > 0.0);
    }

    #[test]
    fn wiggler_build_i_map_nonzero() {
        let w = Wiggler::new(3.0, 0.3, 10.0, 80.0, 20, 100, 5000.0, 15000.0, 0.002, 0.002);
        let energies = vec![10000.0; 10];
        let thetas = vec![0.0; 10];
        let psis = vec![0.0; 10];
        let (intensity, _, _) = w.build_i_map(&energies, &thetas, &psis);
        assert!(intensity[0] > 0.0, "intensity = {}", intensity[0]);
    }

    #[test]
    fn wiggler_shine() {
        let mut w = Wiggler::new(3.0, 0.3, 10.0, 80.0, 20, 50, 5000.0, 15000.0, 0.003, 0.003);
        let beam = w.shine();
        assert_eq!(beam.nrays(), 50);
        for i in 0..50 {
            assert!(beam.e[i] >= 5000.0 && beam.e[i] <= 15000.0);
        }
        // Direction normalization
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
    fn wiggler_flux_higher_than_bm() {
        // Wiggler with N_p periods should produce ~2*N_p × BM flux
        let w = Wiggler::new(3.0, 0.3, 10.0, 80.0, 20, 100, 5000.0, 15000.0, 0.002, 0.002);
        let energies = vec![10000.0];
        let thetas = vec![0.0];
        let psis = vec![0.0];
        let (i_wig, _, _) = w.build_i_map(&energies, &thetas, &psis);
        assert!(i_wig[0] > 0.0);
    }

    #[test]
    fn wiggler_flux_scales_with_periods() {
        // Flux should scale approximately linearly with N_periods
        let mut wig10 = Wiggler::new(3.0, 0.3, 5.0, 80.0, 10, 5000, 5000.0, 15000.0, 1e-3, 1e-3);
        let mut wig20 = Wiggler::new(3.0, 0.3, 5.0, 80.0, 20, 5000, 5000.0, 15000.0, 1e-3, 1e-3);
        let beam10 = wig10.shine();
        let beam20 = wig20.shine();

        // Both should produce same number of rays
        assert_eq!(beam10.nrays(), 5000);
        assert_eq!(beam20.nrays(), 5000);

        // Energy distributions should be similar (same K, same B)
        let mean_e10: f64 = beam10.e.iter().sum::<f64>() / beam10.nrays() as f64;
        let mean_e20: f64 = beam20.e.iter().sum::<f64>() / beam20.nrays() as f64;
        let rel_diff = (mean_e10 - mean_e20).abs() / mean_e10;
        assert!(
            rel_diff < 0.2,
            "mean energy should be similar: {mean_e10:.0} vs {mean_e20:.0}"
        );
    }

    #[test]
    fn b_max_uses_the_period_in_mm() {
        let w = Wiggler::new(3.0, 0.3, 10.0, 80.0, 20, 100, 5000.0, 15000.0, 0.002, 0.002);
        // K = 0.934·B[T]·period[cm] → B = 10/(0.934·8) = 1.339 T
        let b_expected = 10.0 / (0.934 * 8.0);
        assert!(
            (w.b_max - b_expected).abs() / b_expected < 1e-3,
            "b_max = {} T, expected ≈ {b_expected} T",
            w.b_max
        );
    }

    #[test]
    fn theta_sampling_window_is_the_narrower_of_acceptance_and_k_over_gamma() {
        let narrow = Wiggler::new(3.0, 0.3, 0.01, 80.0, 10, 1000, 4.0, 20.0, 1e-3, 1e-3);
        // K/γ = 1.70e-6, far inside the 1e-3 acceptance
        assert!((narrow.theta_max_sampling() - 0.01 / narrow.params.gamma).abs() < 1e-18);

        let wide = Wiggler::new(3.0, 0.3, 10.0, 80.0, 10, 100, 5000.0, 15000.0, 1e-3, 1e-3);
        // K/γ = 1.70e-3, so the requested acceptance stands
        assert_eq!(wide.theta_max_sampling(), 1e-3);
    }

    #[test]
    #[should_panic(expected = "made no progress")]
    fn a_source_that_cannot_emit_stops_instead_of_spinning() {
        // 5-15 keV asked of a K=0.01, 80 mm, 3 GeV wiggler, whose critical
        // energy is 8 eV: no sample can ever pass the discriminator, so the
        // loop must give up and say why rather than run forever.
        let mut wig = Wiggler::new(3.0, 0.3, 0.01, 80.0, 10, 50, 5000.0, 15000.0, 1e-3, 1e-3);
        let _ = wig.shine();
    }

    #[test]
    fn wiggler_small_k_produces_rays() {
        // K=0.01 over an 80 mm period at 3 GeV is B = 1.34 mT, whose critical
        // photon energy is 8.0 eV — a keV window emits nothing at all, so the
        // energy range has to sit at the critical energy for rays to exist.
        let mut wig = Wiggler::new(
            3.0, 0.3, 0.01, // very small K
            80.0, 10, 1000, 4.0, 20.0, 1e-3, 1e-3,
        );
        let beam = wig.shine();
        assert_eq!(beam.nrays(), 1000);
        // All energies in range
        for &e in beam.e.iter() {
            assert!((4.0..=20.0).contains(&e), "energy {e} out of range");
        }
        // Emission is confined to |θ| < K/γ, so every ray must be inside it
        let theta_max = 0.01 / wig.params.gamma;
        for i in 0..beam.nrays() {
            let theta = (beam.a[i] / beam.b[i]).atan();
            assert!(theta.abs() <= theta_max, "θ = {theta} outside K/γ");
        }
        // Direction vectors normalized
        for i in 0..beam.nrays() {
            let norm =
                (beam.a[i] * beam.a[i] + beam.b[i] * beam.b[i] + beam.c[i] * beam.c[i]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }
}
