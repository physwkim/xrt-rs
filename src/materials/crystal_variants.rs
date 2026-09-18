//! Crystal variants: Fcc, Diamond, Si, FromCell.
//!
//! Ported from `xrt/backends/raycing/materials.py:2292-2606`.

use std::f64::consts::PI;

use ndarray::Array1;
use num_complex::Complex64;

use crate::core::consts::AVOGADRO;
use crate::core::error::XrtError;
use crate::math::f0::f0_scalar;

use crate::materials::crystal::{CrystalBase, CrystalGeometry, StructureFactor};
use crate::materials::data::ScatteringTable;
use crate::materials::elements::Element;
use crate::materials::material::{Material, MaterialKind};

// ── CrystalFcc ──────────────────────────────────────────────────────────────

/// FCC crystal: F_hkl = 4f if all h,k,l even or all odd, else 0.
#[derive(Debug, Clone)]
pub struct CrystalFcc {
    pub base: CrystalBase,
}

impl CrystalFcc {
    pub fn new(
        element: &str,
        hkl: [i32; 3],
        d: f64,
        v: Option<f64>,
        rho: f64,
        geom: CrystalGeometry,
        fact_dw: f64,
        thickness: Option<f64>,
        mosaicity: f64,
        table: ScatteringTable,
    ) -> Result<Self, XrtError> {
        let material = Material::new(&[element], None, rho, MaterialKind::Mirror, None, table)?;
        let base = CrystalBase::new(material, hkl, d, v, geom, fact_dw, thickness, mosaicity);
        Ok(Self { base })
    }

    fn hkl_all_even_or_odd(&self) -> bool {
        let residue: i32 = self.base.hkl.iter().map(|h| h.rem_euclid(2)).sum();
        residue == 0 || residue == 3
    }
}

impl StructureFactor for CrystalFcc {
    fn get_structure_factor(
        &self,
        e: &Array1<f64>,
        sin_theta_over_lambda: &Array1<f64>,
        need_fhkl: bool,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>, Array1<Complex64>), XrtError> {
        let elem = &self.base.material.elements[0];
        let anomalous = elem.get_f1f2(e)?;
        let z = elem.z as f64;
        let dw = self.base.fact_dw;

        let f0_arr = anomalous.mapv(|ap| Complex64::new(z, 0.0) + ap);
        let f0_struct = f0_arr.mapv(|f| f * 4.0 * dw);

        if self.hkl_all_even_or_odd() && need_fhkl {
            // Fast path: when all stol values are the same (common for DCM/crystal),
            // compute f0 once and broadcast
            let n = e.len();
            let stol0 = sin_theta_over_lambda[0];
            let uniform_stol = n <= 1 || sin_theta_over_lambda.iter().all(|&s| s == stol0);

            let fhkl = if uniform_stol && n > 1 {
                let f0_val = f0_scalar(&elem.f0_coeffs, stol0);
                let f0_c = Complex64::new(f0_val, 0.0);
                anomalous.mapv(|ap| (f0_c + ap) * 4.0 * dw)
            } else {
                ndarray::Zip::from(sin_theta_over_lambda)
                    .and(&anomalous)
                    .map_collect(|&stol, &ap| {
                        let f0_val = f0_scalar(&elem.f0_coeffs, stol);
                        (Complex64::new(f0_val, 0.0) + ap) * 4.0 * dw
                    })
            };
            Ok((f0_struct, fhkl.clone(), fhkl))
        } else {
            let zeros = Array1::from_elem(e.len(), Complex64::new(0.0, 0.0));
            Ok((f0_struct, zeros.clone(), zeros))
        }
    }
}

// ── CrystalDiamond ──────────────────────────────────────────────────────────

/// Diamond-structure crystal: F_diamond = F_fcc × (1 + exp(iπ(h+k+l)/2))
#[derive(Debug, Clone)]
pub struct CrystalDiamond {
    pub base: CrystalBase,
    /// Lattice parameter [Å]
    pub lattice_a: f64,
    /// FCC sub-structure for structure factor
    fcc: CrystalFcc,
}

impl CrystalDiamond {
    pub fn new(
        element: &str,
        hkl: [i32; 3],
        lattice_a: f64,
        rho: f64,
        geom: CrystalGeometry,
        fact_dw: f64,
        thickness: Option<f64>,
        mosaicity: f64,
        table: ScatteringTable,
    ) -> Result<Self, XrtError> {
        let sqrt_hkl2 = ((hkl[0] * hkl[0] + hkl[1] * hkl[1] + hkl[2] * hkl[2]) as f64).sqrt();
        let d = lattice_a / sqrt_hkl2;
        let v = Some(lattice_a.powi(3));

        let material = Material::new(&[element], None, rho, MaterialKind::Mirror, None, table)?;
        let base = CrystalBase::new(material, hkl, d, v, geom, fact_dw, thickness, mosaicity);

        let fcc = CrystalFcc { base: base.clone() };

        Ok(Self {
            base,
            lattice_a,
            fcc,
        })
    }

    fn diamond_to_fcc_factor(&self) -> Complex64 {
        let hkl_sum: i32 = self.base.hkl.iter().sum();
        Complex64::new(1.0, 0.0) + (Complex64::i() * PI * hkl_sum as f64 / 2.0).exp()
    }
}

impl StructureFactor for CrystalDiamond {
    fn get_structure_factor(
        &self,
        e: &Array1<f64>,
        sin_theta_over_lambda: &Array1<f64>,
        need_fhkl: bool,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>, Array1<Complex64>), XrtError> {
        let (f0, fhkl, fhkl_bar) =
            self.fcc
                .get_structure_factor(e, sin_theta_over_lambda, need_fhkl)?;

        let factor = self.diamond_to_fcc_factor();
        let factor_conj = factor.conj();

        Ok((
            f0.mapv(|f| f * 2.0), // F0 doubles for diamond
            fhkl.mapv(|f| f * factor),
            fhkl_bar.mapv(|f| f * factor_conj),
        ))
    }
}

// ── CrystalSi ───────────────────────────────────────────────────────────────

/// Silicon crystal with temperature-dependent lattice constant.
///
/// Uses the Swenson (1983) parameterization for thermal expansion.
#[derive(Debug, Clone)]
pub struct CrystalSi {
    pub base: CrystalBase,
    /// Lattice parameter at reference temperature [Å]
    pub a0: f64,
    /// dl/l at reference temperature (19.9°C)
    pub dl_l0: f64,
    /// Temperature [K]
    pub t_k: f64,
    /// Diamond sub-structure for structure factor
    diamond: CrystalDiamond,
}

impl CrystalSi {
    /// Create a new Si crystal.
    ///
    /// * `hkl`: Miller indices
    /// * `t_k`: Temperature in Kelvin (default 297.15 = 24°C)
    /// * `geom`: Diffraction geometry
    /// * `fact_dw`: Debye-Waller factor (default 1.0)
    /// * `thickness`: Crystal thickness in mm (None = semi-infinite)
    /// * `mosaicity`: Mosaicity in radians
    /// * `table`: Scattering table
    pub fn new(
        hkl: [i32; 3],
        t_k: f64,
        geom: CrystalGeometry,
        fact_dw: f64,
        thickness: Option<f64>,
        mosaicity: f64,
        table: ScatteringTable,
    ) -> Result<Self, XrtError> {
        let a0 = 5.430_710;
        let dl_l0 = dl_l_swenson(273.15 + 19.9);
        let dl_l_t = dl_l_swenson(t_k);
        let a = a0 * (dl_l_t - dl_l0 + 1.0);

        let rho = 2.33; // Si density g/cm³
        let diamond = CrystalDiamond::new(
            "Si", hkl, a, rho, geom, fact_dw, thickness, mosaicity, table,
        )?;

        Ok(Self {
            base: diamond.base.clone(),
            a0,
            dl_l0,
            t_k,
            diamond,
        })
    }

    /// Get current lattice parameter [Å] at the crystal temperature.
    pub fn get_a(&self) -> f64 {
        self.a0 * (dl_l_swenson(self.t_k) - self.dl_l0 + 1.0)
    }
}

impl StructureFactor for CrystalSi {
    fn get_structure_factor(
        &self,
        e: &Array1<f64>,
        sin_theta_over_lambda: &Array1<f64>,
        need_fhkl: bool,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>, Array1<Complex64>), XrtError> {
        self.diamond
            .get_structure_factor(e, sin_theta_over_lambda, need_fhkl)
    }
}

/// Swenson (1983) thermal expansion parameterization for Si.
///
/// Returns dl/l (fractional length change) at temperature t [K].
/// Reference temperature is 19.9°C = 293.05 K.
fn dl_l_swenson(t: f64) -> f64 {
    if (0.0..30.0).contains(&t) {
        -2.154_537e-4
    } else if (30.0..130.0).contains(&t) {
        -2.303_956e-14 * t.powi(4) + 7.834_799e-11 * t.powi(3) - 1.724_143e-8 * t.powi(2)
            + 8.396_104e-7 * t
            - 2.276_144e-4
    } else if (130.0..293.0).contains(&t) {
        -1.223_001e-11 * t.powi(3) + 1.532_991e-8 * t.powi(2) - 3.263_667e-6 * t - 5.217_231e-5
    } else if (293.0..=1000.0).contains(&t) {
        -1.161_022e-12 * t.powi(3) + 3.311_476e-9 * t.powi(2) + 1.124_129e-6 * t - 5.844_535e-4
    } else {
        1.0e100
    }
}

// ── CrystalFromCell ─────────────────────────────────────────────────────────

/// General crystal defined by unit cell parameters and atomic positions.
#[derive(Debug, Clone)]
pub struct CrystalFromCell {
    pub base: CrystalBase,
    /// Cell parameters
    pub cell_a: f64,
    pub cell_b: f64,
    pub cell_c: f64,
    pub cell_alpha: f64, // radians
    pub cell_beta: f64,
    pub cell_gamma: f64,
    /// Atoms (element objects)
    pub elements: Vec<Element>,
    /// Fractional atomic coordinates
    pub atoms_xyz: Vec<[f64; 3]>,
    /// Atomic fractions (occupancy)
    pub atoms_fraction: Vec<f64>,
}

impl CrystalFromCell {
    /// Create a crystal from unit cell parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hkl: [i32; 3],
        a: f64,
        b: Option<f64>,
        c: Option<f64>,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        atoms: &[&str], // element names or Z as strings
        atoms_xyz: &[[f64; 3]],
        atoms_fraction: Option<&[f64]>,
        geom: CrystalGeometry,
        fact_dw: f64,
        thickness: Option<f64>,
        mosaicity: f64,
        table: ScatteringTable,
    ) -> Result<Self, XrtError> {
        let b_val = b.unwrap_or(a);
        let c_val = c.unwrap_or(a);
        let alpha = alpha_deg.to_radians();
        let beta = beta_deg.to_radians();
        let gamma = gamma_deg.to_radians();

        let (ca, cb, cg) = (alpha.cos(), beta.cos(), gamma.cos());
        let (sa, sb, sg) = (alpha.sin(), beta.sin(), gamma.sin());

        let v = a * b_val * c_val * (1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg).sqrt();

        let h = hkl[0] as f64;
        let k = hkl[1] as f64;
        let l = hkl[2] as f64;

        let d = v / (a * b_val * c_val)
            * ((h * sa / a).powi(2)
                + (k * sb / b_val).powi(2)
                + (l * sg / c_val).powi(2)
                + 2.0 * h * k * (ca * cb - cg) / (a * b_val)
                + 2.0 * h * l * (ca * cg - cb) / (a * c_val)
                + 2.0 * k * l * (cb * cg - ca) / (b_val * c_val))
                .powf(-0.5);

        // Build elements
        let mut elements = Vec::with_capacity(atoms.len());
        let mut unique: std::collections::HashMap<usize, Element> =
            std::collections::HashMap::new();
        let mut mass = 0.0;

        let fractions: Vec<f64> = atoms_fraction
            .map(|f| f.to_vec())
            .unwrap_or_else(|| vec![1.0; atoms.len()]);

        for (&atom_name, &frac) in atoms.iter().zip(fractions.iter()) {
            let elem = if let Ok(z) = atom_name.parse::<usize>() {
                if let Some(cached) = unique.get(&z) {
                    cached.clone()
                } else {
                    let e = Element::from_z(z, table)?;
                    unique.insert(z, e.clone());
                    e
                }
            } else {
                let z = crate::materials::data::element_z(atom_name)
                    .ok_or_else(|| XrtError::ElementNotFound(atom_name.to_string()))?;
                if let Some(cached) = unique.get(&z) {
                    cached.clone()
                } else {
                    let e = Element::new(atom_name, table)?;
                    unique.insert(z, e.clone());
                    e
                }
            };
            mass += frac * elem.mass;
            elements.push(elem);
        }

        let rho = mass / AVOGADRO / v * 1e24;

        // Build a minimal Material for the base
        let material = Material {
            elements: elements.clone(),
            quantities: fractions.clone(),
            rho,
            kind: MaterialKind::Mirror,
            thickness: None,
            mass,
            name: String::new(),
        };

        let base = CrystalBase::new(
            material,
            hkl,
            d,
            Some(v),
            geom,
            fact_dw,
            thickness,
            mosaicity,
        );

        Ok(Self {
            base,
            cell_a: a,
            cell_b: b_val,
            cell_c: c_val,
            cell_alpha: alpha,
            cell_beta: beta,
            cell_gamma: gamma,
            elements,
            atoms_xyz: atoms_xyz.to_vec(),
            atoms_fraction: fractions,
        })
    }
}

impl StructureFactor for CrystalFromCell {
    fn get_structure_factor(
        &self,
        e: &Array1<f64>,
        sin_theta_over_lambda: &Array1<f64>,
        need_fhkl: bool,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>, Array1<Complex64>), XrtError> {
        let len = e.len();
        let mut f0_total = Array1::<Complex64>::zeros(len);
        let mut fhkl = Array1::<Complex64>::zeros(len);
        let mut fhkl_bar = Array1::<Complex64>::zeros(len);

        let dw = self.base.fact_dw;
        let hkl_f = [
            self.base.hkl[0] as f64,
            self.base.hkl[1] as f64,
            self.base.hkl[2] as f64,
        ];

        // Cache per unique Z
        let mut cache: std::collections::HashMap<usize, (Array1<f64>, Array1<Complex64>)> =
            std::collections::HashMap::new();

        for (idx, (elem, xyz)) in self.elements.iter().zip(self.atoms_xyz.iter()).enumerate() {
            let af = self.atoms_fraction[idx];

            let (f0_vals, anomalous) = if let Some(cached) = cache.get(&elem.z) {
                cached.clone()
            } else {
                let f0_vals = if need_fhkl {
                    sin_theta_over_lambda.mapv(|stol| f0_scalar(&elem.f0_coeffs, stol))
                } else {
                    Array1::zeros(len)
                };
                let ap = elem.get_f1f2(e)?;
                cache.insert(elem.z, (f0_vals.clone(), ap.clone()));
                (f0_vals, ap)
            };

            let z = elem.z as f64;

            // F0 += af * (Z + anomalous) * factDW
            ndarray::Zip::from(&mut f0_total)
                .and(&anomalous)
                .for_each(|f0, &ap| {
                    *f0 += (Complex64::new(z, 0.0) + ap) * af * dw;
                });

            // exp(2πi × hkl · xyz)
            let hkl_dot_xyz = hkl_f[0] * xyz[0] + hkl_f[1] * xyz[1] + hkl_f[2] * xyz[2];
            let exp_ihr = (Complex64::i() * 2.0 * PI * hkl_dot_xyz).exp();
            let exp_ihr_inv = Complex64::new(1.0, 0.0) / exp_ihr;

            // Fhkl += af * (f0 + anomalous) * factDW * exp(2πi hkl·xyz)
            ndarray::Zip::from(&mut fhkl)
                .and(&f0_vals)
                .and(&anomalous)
                .for_each(|fh, &f0v, &ap| {
                    *fh += (Complex64::new(f0v, 0.0) + ap) * af * dw * exp_ihr;
                });

            ndarray::Zip::from(&mut fhkl_bar)
                .and(&f0_vals)
                .and(&anomalous)
                .for_each(|fhb, &f0v, &ap| {
                    *fhb += (Complex64::new(f0v, 0.0) + ap) * af * dw * exp_ihr_inv;
                });
        }

        Ok((f0_total, fhkl, fhkl_bar))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn si_lattice_constant_room_temp() {
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
        let a = si.get_a();
        // Si lattice constant at room temp ≈ 5.4310 Å
        assert!((a - 5.431).abs() < 0.002, "a = {a:.6} Å, expected ~5.431 Å");
    }

    #[test]
    fn si111_d_spacing() {
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
        // d_111 = a / sqrt(3) ≈ 3.1356 Å
        assert!((si.base.d - 3.1356).abs() < 0.01, "d = {:.4} Å", si.base.d);
    }

    #[test]
    fn diamond_factor_111() {
        // For (111): h+k+l = 3, factor = 1 + exp(i*3π/2) = 1 - i
        let diamond = CrystalDiamond::new(
            "Si",
            [1, 1, 1],
            5.431,
            2.33,
            CrystalGeometry::BraggReflected,
            1.0,
            None,
            0.0,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();
        let factor = diamond.diamond_to_fcc_factor();
        assert!((factor.re - 1.0).abs() < 1e-10);
        assert!((factor.im - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn diamond_factor_220() {
        // For (220): h+k+l = 4, factor = 1 + exp(i*2π) = 2
        let diamond = CrystalDiamond::new(
            "Si",
            [2, 2, 0],
            5.431,
            2.33,
            CrystalGeometry::BraggReflected,
            1.0,
            None,
            0.0,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();
        let factor = diamond.diamond_to_fcc_factor();
        assert!((factor.re - 2.0).abs() < 1e-10);
        assert!(factor.im.abs() < 1e-10);
    }

    #[test]
    fn from_cell_si_structure_factor() {
        // Build Si as CrystalFromCell and compare with CrystalSi
        let si_from_cell = CrystalFromCell::new(
            [1, 1, 1],
            5.431,
            None,
            None,
            90.0,
            90.0,
            90.0,
            &["Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si"],
            &[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ],
            None,
            CrystalGeometry::BraggReflected,
            1.0,
            None,
            0.0,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let e = array![10000.0];
        let stol = array![0.5 / si_from_cell.base.d];
        let (f0, fhkl, _) = si_from_cell.get_structure_factor(&e, &stol, true).unwrap();

        // F0 should be ~8 × (14 + f1 + if2) since 8 atoms
        assert!(f0[0].norm() > 100.0, "F0 = {}", f0[0]);
        // Fhkl should be non-zero for (111) in diamond structure
        assert!(fhkl[0].norm() > 1.0, "Fhkl = {}", fhkl[0]);
    }

    #[test]
    fn swenson_thermal_expansion() {
        // At 0 K, dl/l should be ~ -2.15e-4
        assert!((dl_l_swenson(10.0) - (-2.154_537e-4)).abs() < 1e-8);
        // At room temp 293 K
        let dll = dl_l_swenson(293.0);
        assert!(dll > -1e-3 && dll < 1e-3, "dl/l(293K) = {dll}");
    }

    #[test]
    fn crystal_from_cell_simple_cubic() {
        // Simple cubic crystal: single atom at origin
        let cr = CrystalFromCell::new(
            [1, 0, 0], // hkl
            5.43,      // a [Å]
            None,
            None, // b, c (defaults to a)
            90.0,
            90.0,
            90.0, // alpha, beta, gamma
            &["Si"],
            &[[0.0, 0.0, 0.0]], // atom at origin
            None,               // occupancy = 1.0
            CrystalGeometry::BraggReflected,
            1.0,  // DW
            None, // thickness
            0.0,  // mosaicity
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let e = array![10000.0];
        let theta = cr.base.get_bragg_angle(&e);
        assert!(
            theta[0] > 0.0 && theta[0] < std::f64::consts::FRAC_PI_2,
            "Bragg angle should be in (0, π/2): {}",
            theta[0]
        );
    }
}
