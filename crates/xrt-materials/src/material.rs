//! Material: refractive index, absorption, Fresnel reflection/transmission.
//!
//! Ported from `xrt/backends/raycing/materials.py:298-679`.

use ndarray::Array1;
use num_complex::Complex64;

use xrt_core::consts::{AVOGADRO, CH, CHBAR, PI2, R0};
use xrt_core::error::XrtError;

use crate::data::ScatteringTable;
use crate::elements::Element;

/// Kind of optical material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialKind {
    Mirror,
    ThinMirror,
    Plate,
    Lens,
    Grating,
    Fzp,
    Auto,
}

impl MaterialKind {
    pub fn from_str_xrt(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "mirror" => MaterialKind::Mirror,
            "thin mirror" => MaterialKind::ThinMirror,
            "plate" => MaterialKind::Plate,
            "lens" => MaterialKind::Lens,
            "grating" => MaterialKind::Grating,
            "fzp" => MaterialKind::Fzp,
            _ => MaterialKind::Auto,
        }
    }
}

/// Result of amplitude calculation.
#[derive(Debug, Clone)]
pub struct AmplitudeResult {
    /// s-polarization reflectivity/transmittivity amplitude
    pub rs: Array1<Complex64>,
    /// p-polarization reflectivity/transmittivity amplitude
    pub rp: Array1<Complex64>,
    /// Linear absorption coefficient [cm⁻¹]
    pub abs_coeff: Array1<f64>,
    /// Phase (n.real * E / CHBAR * 1e8)
    pub phase: Array1<f64>,
}

/// A material defined by its chemical formula and density.
#[derive(Debug, Clone)]
pub struct Material {
    pub elements: Vec<Element>,
    pub quantities: Vec<f64>,
    pub rho: f64,
    pub kind: MaterialKind,
    pub thickness: Option<f64>, // mm, for thin mirror
    pub mass: f64,              // total molar mass
    pub name: String,
}

impl Material {
    /// Create a new Material from element symbols and quantities.
    ///
    /// # Arguments
    /// * `elem_names` - Element symbols (e.g. ["Si", "O"])
    /// * `quantities` - Stoichiometric coefficients (e.g. [1.0, 2.0] for SiO₂)
    /// * `rho` - Density in g/cm³
    /// * `kind` - Material kind
    /// * `thickness` - Thickness in mm (required for ThinMirror)
    /// * `table` - Scattering factor table
    pub fn new(
        elem_names: &[&str],
        quantities: Option<&[f64]>,
        rho: f64,
        kind: MaterialKind,
        thickness: Option<f64>,
        table: ScatteringTable,
    ) -> Result<Self, XrtError> {
        let quantities: Vec<f64> = quantities
            .map(|q| q.to_vec())
            .unwrap_or_else(|| vec![1.0; elem_names.len()]);

        let mut elements = Vec::with_capacity(elem_names.len());
        let mut mass = 0.0;
        let mut name = String::new();

        for (&elem, &xi) in elem_names.iter().zip(quantities.iter()) {
            let element = Element::new(elem, table)?;
            mass += xi * element.mass;
            name.push_str(elem);
            if xi != 1.0 {
                name.push_str(&format!("{xi}"));
            }
            elements.push(element);
        }

        Ok(Self {
            elements,
            quantities,
            rho,
            kind,
            thickness,
            mass,
            name,
        })
    }

    /// Calculate complex refractive index at given energies.
    ///
    /// n = 1 - (r₀λ²Nₐρ)/(2πM) × Σ(xᵢ(Zᵢ + f1ᵢ + i·f2ᵢ))
    pub fn get_refractive_index(&self, e: &Array1<f64>) -> Result<Array1<Complex64>, XrtError> {
        let mut xf = Array1::<Complex64>::zeros(e.len());

        for (elem, &xi) in self.elements.iter().zip(self.quantities.iter()) {
            let f1f2 = elem.get_f1f2(e)?;
            let z = elem.z as f64;
            ndarray::Zip::from(&mut xf)
                .and(&f1f2)
                .for_each(|xf_val, &f| {
                    *xf_val += (Complex64::new(z, 0.0) + f) * xi;
                });
        }

        // n = 1 - 1e-24 * AVOGADRO * R0 / PI2 * (CH/E)^2 * rho * xf / mass
        let factor = 1e-24 * AVOGADRO * R0 / PI2 * self.rho / self.mass;
        let result = ndarray::Zip::from(e)
            .and(&xf)
            .map_collect(|&energy, &xf_val| {
                let lambda = CH / energy;
                let lambda2 = lambda * lambda;
                Complex64::new(1.0, 0.0) - factor * lambda2 * xf_val
            });

        Ok(result)
    }

    /// Calculate linear absorption coefficient μ [cm⁻¹].
    ///
    /// μ = 2 × |Im(n)| × E/cℏ × 1e8
    pub fn get_absorption_coefficient(&self, e: &Array1<f64>) -> Result<Array1<f64>, XrtError> {
        let n = self.get_refractive_index(e)?;
        Ok(ndarray::Zip::from(&n)
            .and(e)
            .map_collect(|&n_val, &energy| n_val.im.abs() * energy / CHBAR * 2e8))
    }

    /// Calculate Fresnel amplitudes for reflection or transmission.
    ///
    /// Returns (rs, rp, absorption_coeff, phase).
    pub fn get_amplitude(
        &self,
        e: &Array1<f64>,
        beam_in_dot_normal: &Array1<f64>,
        from_vacuum: bool,
    ) -> Result<AmplitudeResult, XrtError> {
        let kind = if self.kind == MaterialKind::Auto {
            MaterialKind::Mirror
        } else {
            self.kind
        };

        if kind == MaterialKind::Fzp {
            let ones = Array1::from_elem(e.len(), Complex64::new(1.0, 0.0));
            let zeros = Array1::zeros(e.len());
            return Ok(AmplitudeResult {
                rs: ones.clone(),
                rp: ones,
                abs_coeff: zeros.clone(),
                phase: zeros,
            });
        }

        let n = self.get_refractive_index(e)?;

        let len = e.len();
        let mut rs = Array1::<Complex64>::zeros(len);
        let mut rp = Array1::<Complex64>::zeros(len);
        let mut abs_coeff = Array1::<f64>::zeros(len);
        let mut phase = Array1::<f64>::zeros(len);

        for i in 0..len {
            let (n1, n2) = if from_vacuum {
                (Complex64::new(1.0, 0.0), n[i])
            } else {
                (n[i], Complex64::new(1.0, 0.0))
            };

            let cos_alpha = beam_in_dot_normal[i].abs();
            let sin_alpha2 = (1.0 - beam_in_dot_normal[i] * beam_in_dot_normal[i]).max(0.0);

            let n1_cos_alpha = n1 * cos_alpha;
            let cos_beta = (Complex64::new(1.0, 0.0) - (n1 / n2).powi(2) * sin_alpha2).sqrt();
            let n2_cos_beta = n2 * cos_beta;

            match kind {
                MaterialKind::Mirror | MaterialKind::ThinMirror | MaterialKind::Grating => {
                    let rs_val = (n1_cos_alpha - n2_cos_beta) / (n1_cos_alpha + n2_cos_beta);
                    let rp_val =
                        (n2 * cos_alpha - n1 * cos_beta) / (n2 * cos_alpha + n1 * cos_beta);

                    if kind == MaterialKind::ThinMirror {
                        let t = self.thickness.unwrap_or(0.0);
                        let p2 =
                            (Complex64::new(0.0, 2.0) * e[i] / CHBAR * n2_cos_beta * t * 1e7).exp();
                        rs[i] = rs_val * (Complex64::new(1.0, 0.0) - p2)
                            / (Complex64::new(1.0, 0.0) - rs_val * rs_val * p2);
                        rp[i] = rp_val * (Complex64::new(1.0, 0.0) - p2)
                            / (Complex64::new(1.0, 0.0) - rp_val * rp_val * p2);
                    } else {
                        rs[i] = rs_val;
                        rp[i] = rp_val;
                    }
                }
                MaterialKind::Plate | MaterialKind::Lens => {
                    let tf = ((n2_cos_beta * n1.conj()).re / cos_alpha).sqrt() / n1.norm();
                    rs[i] =
                        Complex64::new(2.0, 0.0) * n1_cos_alpha / (n1_cos_alpha + n2_cos_beta) * tf;
                    rp[i] = Complex64::new(2.0, 0.0) * n1_cos_alpha
                        / (n2 * cos_alpha + n1 * cos_beta)
                        * tf;
                }
                _ => {}
            }

            abs_coeff[i] = n[i].im.abs() * e[i] / CHBAR * 2e8;
            phase[i] = n[i].re * e[i] / CHBAR * 1e8;
        }

        Ok(AmplitudeResult {
            rs,
            rp,
            abs_coeff,
            phase,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn si_material() -> Material {
        Material::new(
            &["Si"],
            None,
            2.33,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap()
    }

    #[test]
    fn si_refractive_index_10kev() {
        let si = si_material();
        let e = array![10000.0];
        let n = si.get_refractive_index(&e).unwrap();
        // δ = 1 - Re(n) should be ~4.9e-6 for Si at 10 keV
        let delta = 1.0 - n[0].re;
        assert!(
            delta > 4e-6 && delta < 6e-6,
            "delta = {delta:.6e}, expected ~4.9e-6"
        );
        // β = -Im(n) should be small positive
        assert!(n[0].im.abs() < 1e-6);
    }

    #[test]
    fn si_refractive_index_8kev() {
        // δ ≈ 7.6e-6 at 8 keV (Cu Kα) — commonly cited value
        let si = si_material();
        let e = array![8000.0];
        let n = si.get_refractive_index(&e).unwrap();
        let delta = 1.0 - n[0].re;
        assert!(
            (delta - 7.6e-6).abs() < 1e-6,
            "delta = {delta:.6e}, expected ~7.6e-6 at 8 keV"
        );
    }

    #[test]
    fn si_absorption_coefficient() {
        let si = si_material();
        let e = array![10000.0];
        let mu = si.get_absorption_coefficient(&e).unwrap();
        // μ for Si at 10 keV should be ~150 cm⁻¹
        assert!(mu[0] > 50.0 && mu[0] < 500.0, "mu = {} cm⁻¹", mu[0]);
    }

    #[test]
    fn total_reflection_below_critical_angle() {
        let si = si_material();
        let e = array![10000.0];
        // Very small grazing angle → total reflection
        // beam_in_dot_normal = sin(grazing_angle)
        // Critical angle for Si at 10 keV: ~3.9 mrad
        let small_angle = array![0.001]; // ~1 mrad << critical angle
        let result = si.get_amplitude(&e, &small_angle, true).unwrap();
        let rs_abs = result.rs[0].norm();
        assert!(
            rs_abs > 0.99,
            "|rs| = {rs_abs}, expected ~1 for total reflection"
        );
    }

    #[test]
    fn reflection_above_critical_angle() {
        let si = si_material();
        let e = array![10000.0];
        // Large angle → no total reflection
        let large_angle = array![0.5]; // ~30°
        let result = si.get_amplitude(&e, &large_angle, true).unwrap();
        let rs_abs = result.rs[0].norm();
        assert!(
            rs_abs < 0.1,
            "|rs| = {rs_abs}, expected <<1 above critical angle"
        );
    }

    #[test]
    fn sio2_material() {
        let sio2 = Material::new(
            &["Si", "O"],
            Some(&[1.0, 2.0]),
            2.2,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();
        assert_eq!(sio2.elements.len(), 2);
        assert!((sio2.mass - (28.0855 + 2.0 * 15.9994)).abs() < 0.01);
    }

    #[test]
    fn three_element_compound() {
        // LaAlO3: La + Al + 3×O, density 6.52 g/cm³
        let mat = Material::new(
            &["La", "Al", "O"],
            Some(&[1.0, 1.0, 3.0]),
            6.52,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let e = Array1::from_vec(vec![10000.0]);
        let n = mat.get_refractive_index(&e).unwrap();
        // Real part should be close to 1 (1 - delta)
        assert!(
            n[0].re < 1.0 && n[0].re > 0.999,
            "LaAlO3 n.re at 10keV should be ~1: {}",
            n[0].re
        );
        // Imaginary part should be small (absorption, sign depends on convention)
        assert!(
            n[0].im.abs() < 1e-3,
            "LaAlO3 n.im at 10keV should be small: {}",
            n[0].im
        );
    }

    #[test]
    fn negative_density_handled() {
        // Negative density might produce physically wrong but finite results
        let mat = Material::new(
            &["Si"],
            None,
            -2.33,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        );
        // Either returns error or produces finite result
        if let Ok(m) = mat {
            let e = Array1::from_vec(vec![10000.0]);
            let n = m.get_refractive_index(&e).unwrap();
            assert!(
                n[0].re.is_finite(),
                "negative rho should still give finite n"
            );
        }
    }

    #[test]
    fn zero_density_handled() {
        let mat = Material::new(
            &["Si"],
            None,
            0.0,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        );
        if let Ok(m) = mat {
            let e = Array1::from_vec(vec![10000.0]);
            let n = m.get_refractive_index(&e).unwrap();
            // Zero density -> n should be exactly 1 (vacuum)
            assert!(
                (n[0].re - 1.0).abs() < 1e-10,
                "zero density n.re should be 1.0: {}",
                n[0].re
            );
        }
    }

    #[test]
    fn low_energy_scattering() {
        // Test at very low energy (50 eV) — near limits of tabulated data
        let mat = Material::new(
            &["Si"],
            None,
            2.33,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();
        let e = Array1::from_vec(vec![50.0]);
        let n = mat.get_refractive_index(&e);
        if let Ok(n_val) = n {
            assert!(n_val[0].re.is_finite(), "n.re at 50eV should be finite");
            assert!(n_val[0].im.is_finite(), "n.im at 50eV should be finite");
        }
    }

    #[test]
    fn fresnel_reflectivity_bounded() {
        // For any material and angle, |Rs| ≤ 1 and |Rp| ≤ 1
        let mat = Material::new(
            &["Si"],
            None,
            2.33,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let e = Array1::from_vec(vec![10000.0]);
        for sin_theta in [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9] {
            let bidn = Array1::from_vec(vec![sin_theta]);
            let result = mat.get_amplitude(&e, &bidn, true).unwrap();
            assert!(
                result.rs[0].norm() <= 1.0 + 1e-10,
                "Si |Rs| at sin_θ={sin_theta}: {:.6} > 1",
                result.rs[0].norm()
            );
            assert!(
                result.rp[0].norm() <= 1.0 + 1e-10,
                "Si |Rp| at sin_θ={sin_theta}: {:.6} > 1",
                result.rp[0].norm()
            );
        }
    }
}
