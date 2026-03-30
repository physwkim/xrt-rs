//! Crystal base: dynamical diffraction theory.
//!
//! Ported from `xrt/backends/raycing/materials.py:1151-1708`.
//!
//! Implements Bragg and Laue diffraction using the formalism of
//! Belyakov & Dmitrienko (1989).

use ndarray::Array1;
use num_complex::Complex64;

use xrt_core::consts::{CH, PI, PI2, R0};
use xrt_core::error::XrtError;

use crate::material::Material;

/// Crystal diffraction geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrystalGeometry {
    /// Bragg reflected (geometry=2)
    BraggReflected,
    /// Bragg transmitted (geometry=3)
    BraggTransmitted,
    /// Laue reflected (geometry=0)
    LaueReflected,
    /// Laue transmitted (geometry=1)
    LaueTransmitted,
}

impl CrystalGeometry {
    pub fn from_str_xrt(s: &str) -> Result<Self, XrtError> {
        let s_lower = s.to_lowercase();
        let is_bragg = s_lower.starts_with("bragg");
        let is_transmitted = s_lower.contains("transmitted");
        match (is_bragg, is_transmitted) {
            (true, false) => Ok(CrystalGeometry::BraggReflected),
            (true, true) => Ok(CrystalGeometry::BraggTransmitted),
            (false, true) => Ok(CrystalGeometry::LaueTransmitted),
            (false, false) => Ok(CrystalGeometry::LaueReflected),
        }
    }

    pub fn is_bragg(&self) -> bool {
        matches!(
            self,
            CrystalGeometry::BraggReflected | CrystalGeometry::BraggTransmitted
        )
    }

    pub fn is_transmitted(&self) -> bool {
        matches!(
            self,
            CrystalGeometry::BraggTransmitted | CrystalGeometry::LaueTransmitted
        )
    }
}

/// Trait for computing structure factors.
pub trait StructureFactor {
    /// Compute structure factors F0, Fhkl, Fhkl_bar.
    ///
    /// - `e`: photon energies [eV]
    /// - `sin_theta_over_lambda`: sin(θ)/λ for f0 evaluation
    /// - `need_fhkl`: if false, Fhkl can be zero (optimization)
    fn get_structure_factor(
        &self,
        e: &Array1<f64>,
        sin_theta_over_lambda: &Array1<f64>,
        need_fhkl: bool,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>, Array1<Complex64>), XrtError>;
}

/// Result of get_f_chi calculation.
#[derive(Debug, Clone)]
pub struct FChiResult {
    pub f0: Array1<Complex64>,
    pub fhkl: Array1<Complex64>,
    pub fhkl_bar: Array1<Complex64>,
    pub chi0: Array1<Complex64>,
    pub chih: Array1<Complex64>,
    pub chih_bar: Array1<Complex64>,
}

/// Crystal base parameters (shared by all crystal variants).
#[derive(Debug, Clone)]
pub struct CrystalBase {
    /// The underlying material (elements, density, etc.)
    pub material: Material,
    /// Miller indices
    pub hkl: [i32; 3],
    /// sqrt(h² + k² + l²)
    pub sqrt_hkl2: f64,
    /// Interplanar spacing [Å]
    pub d: f64,
    /// Unit cell volume [ų]
    pub v: f64,
    /// χ to F conversion factor = -R0 / (π V)
    pub chi_to_f: f64,
    /// |chiToF| * d²
    pub chi_to_f_d2: f64,
    /// Diffraction geometry
    pub geom: CrystalGeometry,
    /// Debye-Waller factor
    pub fact_dw: f64,
    /// Crystal thickness [mm] (None = semi-infinite)
    pub thickness: Option<f64>,
    /// Mosaicity [rad]
    pub mosaicity: f64,
}

impl CrystalBase {
    /// Create a new CrystalBase.
    pub fn new(
        material: Material,
        hkl: [i32; 3],
        d: f64,
        v: Option<f64>,
        geom: CrystalGeometry,
        fact_dw: f64,
        thickness: Option<f64>,
        mosaicity: f64,
    ) -> Self {
        let sqrt_hkl2 = ((hkl[0] * hkl[0] + hkl[1] * hkl[1] + hkl[2] * hkl[2]) as f64).sqrt();
        let v = v.unwrap_or_else(|| (d * sqrt_hkl2).powi(3));
        let chi_to_f = -R0 / PI / v;
        let chi_to_f_d2 = chi_to_f.abs() * d * d;

        Self {
            material,
            hkl,
            sqrt_hkl2,
            d,
            v,
            chi_to_f,
            chi_to_f_d2,
            geom,
            fact_dw,
            thickness,
            mosaicity,
        }
    }

    /// Bragg angle θ_B = arcsin(λ / (2d)) for given energies.
    pub fn get_bragg_angle(&self, e: &Array1<f64>) -> Array1<f64> {
        e.mapv(|energy| {
            let sin_theta = CH / (2.0 * self.d * energy);
            let sin_theta = sin_theta.clamp(-1.0 + 1e-16, 1.0 - 1e-16);
            sin_theta.asin()
        })
    }

    /// Compute χ₀, χh, χh̄ from structure factors.
    pub fn get_f_chi(
        &self,
        e: &Array1<f64>,
        sin_theta_over_lambda: &Array1<f64>,
        sf: &dyn StructureFactor,
    ) -> Result<FChiResult, XrtError> {
        let (f0, fhkl, fhkl_bar) =
            sf.get_structure_factor(e, sin_theta_over_lambda, true)?;

        let result = ndarray::Zip::from(e)
            .and(&f0)
            .and(&fhkl)
            .and(&fhkl_bar)
            .map_collect(|&energy, &f0_val, &fhkl_val, &fhkl_bar_val| {
                let wavelength = CH / energy;
                let lambda2 = wavelength * wavelength;
                let chi_to_f_lambda2 = self.chi_to_f * lambda2;
                // Note: conjugate needed for Belyakov & Dmitrienko formulas
                let chi0 = f0_val.conj() * chi_to_f_lambda2;
                let chih = fhkl_val.conj() * chi_to_f_lambda2;
                let chih_bar = fhkl_bar_val.conj() * chi_to_f_lambda2;
                (chi0, chih, chih_bar)
            });

        let len = e.len();
        let mut chi0 = Array1::<Complex64>::zeros(len);
        let mut chih = Array1::<Complex64>::zeros(len);
        let mut chih_bar = Array1::<Complex64>::zeros(len);
        for (i, &(c0, ch_, ch_bar)) in result.iter().enumerate() {
            chi0[i] = c0;
            chih[i] = ch_;
            chih_bar[i] = ch_bar;
        }

        Ok(FChiResult {
            f0,
            fhkl,
            fhkl_bar,
            chi0,
            chih,
            chih_bar,
        })
    }

    /// Darwin width: 2δ = |C| × |√(χh × χh̄)| / (sin(2θ) × √|b|)
    ///
    /// Uses the absolute value of the complex square root (not just the
    /// real part), consistent with XRT's `get_Darwin_width()`.
    pub fn get_darwin_width(
        &self,
        e: &Array1<f64>,
        b: f64,
        polarization: Polarization,
        sf: &dyn StructureFactor,
    ) -> Result<Array1<f64>, XrtError> {
        let theta0 = self.get_bragg_angle(e);
        let sin2theta = theta0.mapv(|t| (2.0 * t).sin());
        let wavelength = e.mapv(|energy| CH / energy);
        let sin_theta_over_l = ndarray::Zip::from(&theta0)
            .and(&wavelength)
            .map_collect(|&t, &l| t.sin() / l);

        let fchi = self.get_f_chi(e, &sin_theta_over_l, sf)?;

        let pol_factor = match polarization {
            Polarization::S => Array1::ones(e.len()),
            Polarization::P => theta0.mapv(|t| (2.0 * t).cos().abs()),
        };

        let b_sqrt = b.abs().sqrt();
        let width = ndarray::Zip::from(&pol_factor)
            .and(&fchi.chih)
            .and(&fchi.chih_bar)
            .and(&sin2theta)
            .map_collect(|&pf, &chih, &chih_bar, &s2t| {
                2.0 * pf * (chih * chih_bar).sqrt().norm() / (s2t * b_sqrt)
            });

        // Apply mosaicity broadening: total width = sqrt(dynamical² + mosaicity²)
        let width = if self.mosaicity > 0.0 {
            width.mapv(|w| (w * w + self.mosaicity * self.mosaicity).sqrt())
        } else {
            width
        };
        Ok(width)
    }

    /// Calculate crystal amplitude reflectivity/transmittivity.
    ///
    /// Based on Belyakov & Dmitrienko (1989).
    /// If mosaicity > 0, applies Gaussian convolution over the rocking curve.
    pub fn get_amplitude(
        &self,
        e: &Array1<f64>,
        beam_in_dot_normal: &Array1<f64>,
        beam_out_dot_normal: Option<&Array1<f64>>,
        beam_in_dot_h_normal: Option<&Array1<f64>>,
        sf: &dyn StructureFactor,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>), XrtError> {
        if self.mosaicity <= 0.0 {
            return self.compute_amplitude_core(
                e,
                beam_in_dot_normal,
                beam_out_dot_normal,
                beam_in_dot_h_normal,
                sf,
            );
        }

        // Gaussian convolution with 7 points spanning -3σ to +3σ
        let sigma = self.mosaicity;
        let n_conv: i32 = 7;
        let half = (n_conv - 1) / 2; // = 3
        let len = e.len();
        let mut total_s = Array1::<Complex64>::zeros(len);
        let mut total_p = Array1::<Complex64>::zeros(len);
        let mut w_sum = 0.0_f64;

        for k in 0..n_conv {
            let offset = (k - half) as f64 * sigma;
            let weight = (-0.5 * ((k - half) as f64).powi(2)).exp();
            w_sum += weight;

            let bidn_shifted = beam_in_dot_normal.mapv(|bidn| {
                let theta = (-bidn).acos();
                -(theta + offset).cos()
            });

            let (s, p) = self.compute_amplitude_core(
                e,
                &bidn_shifted,
                beam_out_dot_normal,
                beam_in_dot_h_normal,
                sf,
            )?;
            total_s = total_s + s.mapv(|v| v * weight);
            total_p = total_p + p.mapv(|v| v * weight);
        }

        Ok((total_s / w_sum, total_p / w_sum))
    }

    /// Core amplitude computation without mosaicity convolution.
    ///
    /// This is the Belyakov & Dmitrienko dynamical diffraction calculation,
    /// factored out so `get_amplitude` can call it at multiple shifted angles
    /// for mosaic crystal convolution.
    fn compute_amplitude_core(
        &self,
        e: &Array1<f64>,
        beam_in_dot_normal: &Array1<f64>,
        beam_out_dot_normal: Option<&Array1<f64>>,
        beam_in_dot_h_normal: Option<&Array1<f64>>,
        sf: &dyn StructureFactor,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>), XrtError> {
        let len = e.len();
        let k = e.mapv(|energy| PI2 / (CH / energy));

        // k0s = -beamInDotNormal * k
        let k0s = ndarray::Zip::from(beam_in_dot_normal)
            .and(&k)
            .map_collect(|&bidn, &kv| -bidn * kv);

        let default_beam_out = beam_in_dot_normal.mapv(|v| -v);
        let beam_out_dn = beam_out_dot_normal.unwrap_or(&default_beam_out);

        // kHs = -beamOutDotNormal * k
        let k_hs = ndarray::Zip::from(beam_out_dn)
            .and(&k)
            .map_collect(|&bodn, &kv| -bodn * kv);

        let beam_in_hn = beam_in_dot_h_normal.unwrap_or(beam_in_dot_normal);

        let hh = PI2 / self.d;
        let h2 = hh * hh;

        // b = k0s / kHs (asymmetry ratio)
        let b_arr = ndarray::Zip::from(&k0s)
            .and(&k_hs)
            .map_collect(|&k0, &kh| {
                if kh.abs() < 1e-300 {
                    -1.0
                } else {
                    k0 / kh
                }
            });

        // k0H = |beamInDotHNormal| * HH * k
        let k0h = ndarray::Zip::from(beam_in_hn)
            .and(&k)
            .map_collect(|&bihn, &kv| bihn.abs() * hh * kv);

        let k02 = k.mapv(|kv| kv * kv);

        // Get structure factors at θ = sin⁻¹(λ/(2d))
        let stol = Array1::from_elem(len, 0.5 / self.d);
        let fchi = self.get_f_chi(e, &stol, sf)?;

        let theta_b = self.get_bragg_angle(e);

        // α = (H²/2 - k0H) / k0² + χ₀/2 × (1/b - 1)
        let alpha = ndarray::Zip::from(&k0h)
            .and(&k02)
            .and(&fchi.chi0)
            .and(&b_arr)
            .map_collect(|&k0h_v, &k02_v, &chi0, &b_v| {
                let b_c = Complex64::new(b_v, 0.0);
                Complex64::new((h2 / 2.0 - k0h_v) / k02_v, 0.0)
                    + chi0 / 2.0 * (Complex64::new(1.0, 0.0) / b_c - Complex64::new(1.0, 0.0))
            });

        // For each polarization
        let curve_s = self.for_one_polarization(
            &alpha,
            &fchi.chih,
            &fchi.chih_bar,
            &fchi.chi0,
            &b_arr,
            &k02,
            &k_hs,
            &k0s,
            &theta_b,
            1.0, // C_s = 1
        );

        let curve_p = ndarray::Zip::from(&theta_b)
            .map_collect(|&tb| (2.0 * tb).cos());
        let curve_p = self.for_one_polarization_array(
            &alpha,
            &fchi.chih,
            &fchi.chih_bar,
            &fchi.chi0,
            &b_arr,
            &k02,
            &k_hs,
            &k0s,
            &curve_p,
        );

        Ok((curve_s, curve_p))
    }

    /// Compute amplitude for a single polarization factor (scalar C).
    fn for_one_polarization(
        &self,
        alpha: &Array1<Complex64>,
        chih: &Array1<Complex64>,
        chih_bar: &Array1<Complex64>,
        chi0: &Array1<Complex64>,
        b_arr: &Array1<f64>,
        k02: &Array1<f64>,
        k_hs: &Array1<f64>,
        k0s: &Array1<f64>,
        _theta_b: &Array1<f64>,
        pol_factor: f64,
    ) -> Array1<Complex64> {
        let pf_arr = Array1::from_elem(alpha.len(), pol_factor);
        self.for_one_polarization_array(alpha, chih, chih_bar, chi0, b_arr, k02, k_hs, k0s, &pf_arr)
    }

    /// Compute amplitude for per-element polarization factors.
    fn for_one_polarization_array(
        &self,
        alpha: &Array1<Complex64>,
        chih: &Array1<Complex64>,
        chih_bar: &Array1<Complex64>,
        chi0: &Array1<Complex64>,
        b_arr: &Array1<f64>,
        k02: &Array1<f64>,
        k_hs: &Array1<f64>,
        k0s: &Array1<f64>,
        pol_factor: &Array1<f64>,
    ) -> Array1<Complex64> {
        let len = alpha.len();
        let mut result = Array1::<Complex64>::zeros(len);

        for i in 0..len {
            let a = alpha[i];
            let ch = chih[i];
            let ch_bar = chih_bar[i];
            let c0 = chi0[i];
            let b = b_arr[i];
            let pf = pol_factor[i];
            let b_c = Complex64::new(b, 0.0);

            let delta = (a * a + pf * pf * ch * ch_bar / b_c).sqrt();

            if self.thickness.is_none() {
                // Thick Bragg crystal
                let ra = ch * pf / (a + delta);
                let ad = a - delta;
                let rb = if ad.norm() > 1e-100 {
                    ch * pf / ad
                } else {
                    ra
                };

                let mut r = if ra.is_nan() { rb } else { ra };
                if rb.norm() < ra.norm() && !rb.is_nan() {
                    r = rb;
                }
                result[i] = r / b.abs().sqrt();
            } else {
                let t = self.thickness.unwrap_or(0.0) * 1e7; // mm → Å (1mm = 1e7 Å)
                let l = delta * k02[i] / (2.0 * k_hs[i]); // per unit length
                let l_t = l * t;

                let exp_factor =
                    (Complex64::i() * k02[i] * t * (c0 - a * b_c) / (2.0 * k0s[i])).exp();

                if self.geom.is_bragg() {
                    if self.geom.is_transmitted() {
                        let cos_l = l_t.cos();
                        let sin_l = l_t.sin();
                        result[i] = exp_factor
                            / (cos_l - Complex64::i() * a * sin_l / delta);
                    } else {
                        let cot_l = l_t.cos() / l_t.sin();
                        result[i] = ch * pf
                            / (a + Complex64::i() * delta * cot_l)
                            / b.abs().sqrt();
                    }
                } else {
                    // Laue
                    let cos_l = l_t.cos();
                    let sin_l = l_t.sin();
                    if self.geom.is_transmitted() {
                        result[i] = (cos_l + Complex64::i() * a * sin_l / delta) * exp_factor;
                    } else {
                        result[i] = ch * pf * sin_l / delta * exp_factor / b.abs().sqrt();
                    }
                }
            }
        }

        result
    }
}

/// Polarization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarization {
    S,
    P,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal_variants::CrystalSi;
    use crate::data::ScatteringTable;
    use ndarray::array;

    #[test]
    fn si111_bragg_angle_10kev() {
        let si = CrystalSi::new(
            [1, 1, 1],
            297.15,
            CrystalGeometry::BraggReflected,
            1.0,
            None,
            0.0,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let e = array![10000.0];
        let theta = si.base.get_bragg_angle(&e);
        let theta_deg = theta[0].to_degrees();
        // Si(111) at 10 keV: θ_B ≈ 11.4°
        assert!(
            (theta_deg - 11.4).abs() < 0.2,
            "θ_B = {theta_deg:.2}°, expected ~11.4°"
        );
    }

    #[test]
    fn si111_darwin_width() {
        let si = CrystalSi::new(
            [1, 1, 1],
            297.15,
            CrystalGeometry::BraggReflected,
            1.0,
            None,
            0.0,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let e = array![10000.0];
        let dw = si
            .base
            .get_darwin_width(&e, 1.0, Polarization::S, &si)
            .unwrap();
        let dw_urad = dw[0] * 1e6;
        // Si(111) Darwin width ~20-30 μrad at 10 keV
        assert!(
            dw_urad > 10.0 && dw_urad < 50.0,
            "Darwin width = {dw_urad:.1} μrad, expected ~25 μrad"
        );
    }

    #[test]
    fn si111_thick_bragg_peak_reflectivity() {
        let si = CrystalSi::new(
            [1, 1, 1],
            297.15,
            CrystalGeometry::BraggReflected,
            1.0,
            None, // thick crystal
            0.0,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let e = array![10000.0];
        let theta_b = si.base.get_bragg_angle(&e);
        // At exact Bragg angle, beam_in_dot_normal = -sin(θ_B)
        let bidn = theta_b.mapv(|t| -t.sin());

        let (curve_s, _curve_p) = si.base.get_amplitude(&e, &bidn, None, None, &si).unwrap();
        let rs = curve_s[0].norm();
        // Reflectivity should be significant at Bragg angle
        assert!(
            rs > 0.1,
            "|Rs| = {rs:.4}, expected significant reflectivity at Bragg peak"
        );
    }
}
