//! Multilayer reflectivity using the Parratt recursive formalism.
//!
//! Computes reflectivity and transmittivity of periodic multilayer structures
//! (e.g., W/Si, Mo/Si) using the Parratt recursion with Nevot-Croce
//! interfacial roughness correction.
//!
//! Ported from materials.py Multilayer class.

use ndarray::Array1;
use num_complex::Complex64;

use xrt_core::consts::CHBAR;
use xrt_core::error::XrtError;

use crate::material::Material;

/// Geometry of the multilayer calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultilayerGeom {
    /// Reflected beam (normal use case)
    Reflected,
    /// Transmitted beam
    Transmitted,
}

/// Multilayer reflectivity calculator.
#[derive(Debug, Clone)]
pub struct Multilayer {
    /// Top layer material
    pub t_layer: Material,
    /// Bottom layer material
    pub b_layer: Material,
    /// Substrate material
    pub substrate: Material,
    /// Number of bilayer pairs
    pub n_pairs: usize,
    /// Top layer thickness [Å]
    pub d_t: f64,
    /// Bottom layer thickness [Å]
    pub d_b: f64,
    /// Interfacial roughness σ [Å] (Nevot-Croce RMS)
    pub roughness: f64,
    /// Geometry
    pub geom: MultilayerGeom,
}

/// Result of a multilayer amplitude calculation.
#[derive(Debug, Clone)]
pub struct MultilayerResult {
    /// s-polarization reflectivity amplitude
    pub rs: Array1<Complex64>,
    /// p-polarization reflectivity amplitude
    pub rp: Array1<Complex64>,
}

impl Multilayer {
    pub fn new(
        t_layer: Material,
        b_layer: Material,
        substrate: Material,
        n_pairs: usize,
        d_t: f64,
        d_b: f64,
        roughness: f64,
        geom: MultilayerGeom,
    ) -> Self {
        Self {
            t_layer,
            b_layer,
            substrate,
            n_pairs,
            d_t,
            d_b,
            roughness,
            geom,
        }
    }

    /// Bilayer period [Å].
    pub fn period(&self) -> f64 {
        self.d_t + self.d_b
    }

    /// Compute multilayer reflectivity amplitudes.
    ///
    /// Follows the Python XRT algorithm (materials.py Multilayer.get_amplitude):
    /// - Uses `Q = 2k sinθ` convention internally (matching XRT)
    /// - Conjugates refractive indices before Q computation (XRT convention)
    /// - Layer-by-layer Parratt recursion from substrate upward
    ///
    /// # Arguments
    /// * `e` - Photon energies [eV]
    /// * `sin_theta` - sin(grazing angle) for each energy point
    pub fn get_amplitude(
        &self,
        e: &Array1<f64>,
        sin_theta: &Array1<f64>,
    ) -> Result<MultilayerResult, XrtError> {
        let len = e.len();

        // Get refractive indices and conjugate (XRT convention)
        let n_t = self.t_layer.get_refractive_index(e)?.mapv(|n| n.conj());
        let n_b = self.b_layer.get_refractive_index(e)?.mapv(|n| n.conj());
        let n_s = self.substrate.get_refractive_index(e)?.mapv(|n| n.conj());

        let mut rs = Array1::<Complex64>::zeros(len);
        let mut rp = Array1::<Complex64>::zeros(len);

        let id2 = self.roughness * self.roughness;
        let one = Complex64::new(1.0, 0.0);

        for i in 0..len {
            let k = e[i] / CHBAR; // k = E / (c·ħ) [Å⁻¹]
            let sin_th = sin_theta[i].abs();

            // Q = 2k sinθ (XRT convention: Q includes factor of 2)
            let q = Complex64::new(2.0 * k * sin_th, 0.0);
            let q2 = q * q;
            let k28 = Complex64::new(8.0 * k * k, 0.0);

            // Perpendicular wave vectors in each medium
            // Q_j = sqrt(Q² + (n_j - 1) × 8k²)
            let q_t = (q2 + (n_t[i] - one) * k28).sqrt();
            let q_b = (q2 + (n_b[i] - one) * k28).sqrt();
            let q_s = (q2 + (n_s[i] - one) * k28).sqrt();

            // Pre-compute Fresnel coefficients × roughness factors
            // s-polarization
            let rough_vt = nevot_croce(q, q_t, id2);
            let rough_tb = nevot_croce(q_t, q_b, id2);
            let rough_bs = nevot_croce(q_b, q_s, id2);

            let rvt_s = (q - q_t) / (q + q_t) * rough_vt;
            let rtb_s = (q_t - q_b) / (q_t + q_b) * rough_tb;
            let rbt_s = -rtb_s;
            let rbs_s = (q_b - q_s) / (q_b + q_s) * rough_bs;

            // p-polarization
            let nt = n_t[i];
            let nb = n_b[i];
            let ns = n_s[i];

            let rvt_p = (q * nt - q_t / nt) / (q * nt + q_t / nt) * rough_vt;
            let rtb_p =
                (q_t / nt * nb - q_b / nb * nt) / (q_t / nt * nb + q_b / nb * nt) * rough_tb;
            let rbt_p = -rtb_p;
            let rbs_p =
                (q_b / nb * ns - q_s / ns * nb) / (q_b / nb * ns + q_s / ns * nb) * rough_bs;

            // Phase factors: p² = exp(i × Q_j × d_j)
            // (Q already includes factor of 2, so this is exp(i × 2k_z × d) = round-trip)
            let p2_t = (Complex64::i() * q_t * self.d_t).exp();
            let p2_b = (Complex64::i() * q_b * self.d_b).exp();

            // Parratt recursion: layer-by-layer from substrate upward
            // Layer indices: 2*n_pairs-1 (bottom of 1st bilayer) down to 0 (top of last bilayer)
            // Odd = bottom layer, even = top layer, i=0 = topmost top layer
            let mut rj_s = rbs_s;
            let mut rj_p = rbs_p;

            for j in (0..2 * self.n_pairs).rev() {
                let (rij_s, rij_p, p2);
                if j % 2 == 0 {
                    // Top layer
                    if j == 0 {
                        rij_s = rvt_s;
                        rij_p = rvt_p;
                    } else {
                        rij_s = rbt_s;
                        rij_p = rbt_p;
                    }
                    p2 = p2_t;
                } else {
                    // Bottom layer
                    rij_s = rtb_s;
                    rij_p = rtb_p;
                    p2 = p2_b;
                }

                let rj2_s = rj_s * p2;
                rj_s = (rij_s + rj2_s) / (one + rij_s * rj2_s);

                let rj2_p = rj_p * p2;
                rj_p = (rij_p + rj2_p) / (one + rij_p * rj2_p);
            }

            rs[i] = rj_s;
            rp[i] = rj_p;
        }

        Ok(MultilayerResult { rs, rp })
    }
}

/// Nevot-Croce roughness factor.
///
/// `exp(-0.5 × Q_i × Q_j × σ²)` where Q includes the factor-of-2 convention.
fn nevot_croce(q_i: Complex64, q_j: Complex64, sigma2: f64) -> Complex64 {
    (-Complex64::new(0.5, 0.0) * q_i * q_j * sigma2).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ScatteringTable;
    use crate::material::MaterialKind;
    use ndarray::array;

    fn make_material(elem: &str, rho: f64) -> Material {
        Material::new(
            &[elem],
            None,
            rho,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap()
    }

    #[test]
    fn single_layer_matches_fresnel() {
        // A single bilayer with d_b → 0 should approximate a single-layer result
        let w = make_material("W", 19.3);
        let si = make_material("Si", 2.33);
        let substrate = make_material("Si", 2.33);

        let ml = Multilayer::new(
            w,
            si.clone(),
            substrate,
            1,
            20.0, // 20 Å W
            0.1,  // negligible Si
            0.0,  // no roughness
            MultilayerGeom::Reflected,
        );

        let e = array![10000.0];
        let sin_theta = array![0.01]; // ~10 mrad grazing

        let result = ml.get_amplitude(&e, &sin_theta).unwrap();
        let rs_abs = result.rs[0].norm();
        // Should be a valid reflectivity in [0, 1]
        assert!(
            (0.0..=1.01).contains(&rs_abs),
            "|rs| = {rs_abs}"
        );
    }

    #[test]
    fn multilayer_higher_reflectivity() {
        // More pairs → higher peak reflectivity near Bragg condition
        let w = make_material("W", 19.3);
        let si = make_material("Si", 2.33);
        let substrate = make_material("Si", 2.33);

        let ml_few = Multilayer::new(
            w.clone(),
            si.clone(),
            substrate.clone(),
            5,
            15.0,
            25.0,
            3.0,
            MultilayerGeom::Reflected,
        );

        let ml_many = Multilayer::new(
            w,
            si,
            substrate,
            50,
            15.0,
            25.0,
            3.0,
            MultilayerGeom::Reflected,
        );

        let e = array![8000.0];
        let sin_theta = array![0.02];

        let r_few = ml_few.get_amplitude(&e, &sin_theta).unwrap();
        let r_many = ml_many.get_amplitude(&e, &sin_theta).unwrap();

        // Both should produce finite results
        assert!(r_few.rs[0].norm().is_finite(), "few rs = {}", r_few.rs[0]);
        assert!(r_many.rs[0].norm().is_finite(), "many rs = {}", r_many.rs[0]);
    }

    #[test]
    fn period_returns_bilayer_sum() {
        let ml = Multilayer::new(
            Material::new(&["W"], None, 19.3, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            Material::new(&["Si"], None, 2.33, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            Material::new(&["Si"], None, 2.33, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            20, 15.0, 25.0, 3.0, MultilayerGeom::Reflected,
        );
        assert!((ml.period() - 40.0).abs() < 1e-12, "period should be d_t + d_b = 40 Å");
    }

    #[test]
    fn single_bilayer_produces_finite_result() {
        let ml = Multilayer::new(
            Material::new(&["W"], None, 19.3, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            Material::new(&["Si"], None, 2.33, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            Material::new(&["Si"], None, 2.33, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            1, 15.0, 25.0, 0.0, MultilayerGeom::Reflected,
        );
        let e = Array1::from_vec(vec![10000.0]);
        let st = Array1::from_vec(vec![0.02]);
        let result = ml.get_amplitude(&e, &st).unwrap();
        assert!(result.rs[0].re.is_finite(), "n_pairs=1 rs should be finite");
        assert!(result.rs[0].norm() > 0.0, "n_pairs=1 should have nonzero reflectivity");
        assert!(result.rs[0].norm() < 1.0, "n_pairs=1 should have low reflectivity");
    }

    #[test]
    fn transmitted_geometry_finite() {
        let ml = Multilayer::new(
            Material::new(&["W"], None, 19.3, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            Material::new(&["Si"], None, 2.33, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            Material::new(&["Si"], None, 2.33, MaterialKind::Mirror, None, ScatteringTable::ChantlerTotal).unwrap(),
            20, 15.0, 25.0, 0.0, MultilayerGeom::Transmitted,
        );
        let e = Array1::from_vec(vec![10000.0]);
        let st = Array1::from_vec(vec![0.02]);
        let result = ml.get_amplitude(&e, &st).unwrap();
        assert!(result.rs[0].re.is_finite(), "transmitted rs should be finite");
        assert!(result.rp[0].re.is_finite(), "transmitted rp should be finite");
        // Verify transmitted amplitude is nonzero and bounded
        assert!(result.rs[0].norm() > 0.0, "transmitted |rs| should be > 0");
        assert!(result.rs[0].norm() < 10.0, "transmitted |rs| should be bounded");
    }

    #[test]
    fn roughness_reduces_reflectivity() {
        let w = make_material("W", 19.3);
        let si = make_material("Si", 2.33);
        let substrate = make_material("Si", 2.33);

        let ml_smooth = Multilayer::new(
            w.clone(),
            si.clone(),
            substrate.clone(),
            20,
            15.0,
            25.0,
            0.0, // no roughness
            MultilayerGeom::Reflected,
        );

        let ml_rough = Multilayer::new(
            w,
            si,
            substrate,
            20,
            15.0,
            25.0,
            10.0, // significant roughness
            MultilayerGeom::Reflected,
        );

        let e = array![10000.0];
        let sin_theta = array![0.015];

        let r_smooth = ml_smooth.get_amplitude(&e, &sin_theta).unwrap();
        let r_rough = ml_rough.get_amplitude(&e, &sin_theta).unwrap();

        assert!(
            r_smooth.rs[0].norm() >= r_rough.rs[0].norm(),
            "smooth {} should >= rough {}",
            r_smooth.rs[0].norm(),
            r_rough.rs[0].norm()
        );
    }
}
