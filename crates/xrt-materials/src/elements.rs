//! Element: atomic scattering factors f0, f1, f2.
//!
//! Ported from `xrt/backends/raycing/materials.py:164-295`.

use ndarray::Array1;
use num_complex::Complex64;

use xrt_core::error::XrtError;
use xrt_math::f0::{f0_array, f0_scalar, F0Coeffs};
use xrt_math::interp::interp_linear;

use crate::data::{
    element_z, read_atomic_mass, read_f0_coeffs, read_f1f2_table, ScatteringTable, ELEMENTS_LIST,
};

/// A chemical element with tabulated scattering factors.
#[derive(Debug, Clone)]
pub struct Element {
    /// Atomic number
    pub z: usize,
    /// Element symbol
    pub name: String,
    /// f0 Gaussian sum coefficients [a1..a5, c, b1..b5]
    pub f0_coeffs: F0Coeffs,
    /// Tabulated energy values [eV]
    pub e_table: Vec<f64>,
    /// Tabulated f1 scattering factor (anomalous, relative to Z)
    pub f1_table: Vec<f64>,
    /// Tabulated f2 scattering factor
    pub f2_table: Vec<f64>,
    /// Which table was used
    pub table: ScatteringTable,
    /// Atomic mass [g/mol]
    pub mass: f64,
}

impl Element {
    /// Create a new Element from name or atomic number.
    ///
    /// `elem` can be a chemical symbol (e.g. "Si") or an atomic number as string (e.g. "14").
    pub fn new(elem: &str, table: ScatteringTable) -> Result<Self, XrtError> {
        let (z, name) = if let Ok(z_num) = elem.parse::<usize>() {
            let name = ELEMENTS_LIST
                .get(z_num)
                .ok_or_else(|| XrtError::ElementNotFound(format!("Z={z_num}")))?;
            (z_num, name.to_string())
        } else {
            let z = element_z(elem)
                .ok_or_else(|| XrtError::ElementNotFound(elem.to_string()))?;
            (z, elem.to_string())
        };

        let f0_coeffs = read_f0_coeffs(z)?;
        let (e_table, f1_table, f2_table) = read_f1f2_table(&name, table, None)?;
        let mass = read_atomic_mass(z)?;

        Ok(Self {
            z,
            name,
            f0_coeffs,
            e_table,
            f1_table,
            f2_table,
            table,
            mass,
        })
    }

    /// Create from atomic number directly.
    pub fn from_z(z: usize, table: ScatteringTable) -> Result<Self, XrtError> {
        let name = ELEMENTS_LIST
            .get(z)
            .ok_or_else(|| XrtError::ElementNotFound(format!("Z={z}")))?;
        Self::new(name, table)
    }

    /// Calculate f0 for a single q/(4π) value.
    #[inline]
    pub fn get_f0_scalar(&self, q_over_4pi: f64) -> f64 {
        f0_scalar(&self.f0_coeffs, q_over_4pi)
    }

    /// Calculate f0 for an array of q/(4π) = sin(θ)/λ values.
    pub fn get_f0(&self, q_over_4pi: &Array1<f64>) -> Array1<f64> {
        f0_array(&self.f0_coeffs, q_over_4pi)
    }

    /// Calculate (interpolate) f1 + i·f2 for given energy array E [eV].
    ///
    /// Returns Complex64 array where real = f1, imag = f2.
    /// Errors if any E is outside the table range.
    pub fn get_f1f2(&self, e: &Array1<f64>) -> Result<Array1<Complex64>, XrtError> {
        // Check range
        let e_min = self.e_table[0];
        let e_max = self.e_table[self.e_table.len() - 1];
        for &energy in e.iter() {
            if energy < e_min || energy > e_max {
                return Err(XrtError::EnergyOutOfRange {
                    energy,
                    min: e_min,
                    max: e_max,
                });
            }
        }

        let f1 = interp_linear(e, &self.e_table, &self.f1_table);
        let f2 = interp_linear(e, &self.e_table, &self.f2_table);

        Ok(ndarray::Zip::from(&f1)
            .and(&f2)
            .map_collect(|&r, &i| Complex64::new(r, i)))
    }

    /// Calculate f1 + i·f2 for a single energy value.
    pub fn get_f1f2_scalar(&self, energy: f64) -> Result<Complex64, XrtError> {
        let e_min = self.e_table[0];
        let e_max = self.e_table[self.e_table.len() - 1];
        if energy < e_min || energy > e_max {
            return Err(XrtError::EnergyOutOfRange {
                energy,
                min: e_min,
                max: e_max,
            });
        }

        let f1 = xrt_math::interp::interp_linear_scalar(energy, &self.e_table, &self.f1_table);
        let f2 = xrt_math::interp::interp_linear_scalar(energy, &self.e_table, &self.f2_table);
        Ok(Complex64::new(f1, f2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn create_si_element() {
        let si = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
        assert_eq!(si.z, 14);
        assert_eq!(si.name, "Si");
        assert!((si.mass - 28.0855).abs() < 0.001);
    }

    #[test]
    fn create_from_z() {
        let si = Element::from_z(14, ScatteringTable::ChantlerTotal).unwrap();
        assert_eq!(si.name, "Si");
    }

    #[test]
    fn f0_at_zero_equals_z() {
        let si = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
        let q = array![0.0];
        let f0 = si.get_f0(&q);
        assert!(
            (f0[0] - 14.0).abs() < 0.05,
            "f0(0) = {}, expected ~14",
            f0[0]
        );
    }

    #[test]
    fn f1f2_at_10kev() {
        let si = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
        let e = array![10000.0];
        let f1f2 = si.get_f1f2(&e).unwrap();
        // f1 should be small negative (few tenths), f2 should be small positive
        assert!(f1f2[0].re.abs() < 2.0, "f1 = {}", f1f2[0].re);
        assert!(f1f2[0].im >= 0.0, "f2 = {} (should be >= 0)", f1f2[0].im);
    }

    #[test]
    fn energy_out_of_range() {
        let si = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
        let e = array![0.001]; // Way below table range
        assert!(si.get_f1f2(&e).is_err());
    }

    #[test]
    fn element_not_found() {
        assert!(Element::new("Xx", ScatteringTable::Chantler).is_err());
    }

    #[test]
    fn invalid_element_returns_error() {
        let result = Element::new("Xx", ScatteringTable::ChantlerTotal);
        assert!(result.is_err(), "Invalid element 'Xx' should return Err");
    }

    #[test]
    fn valid_element_by_symbol() {
        let si = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
        assert_eq!(si.z, 14);
        let au = Element::new("Au", ScatteringTable::ChantlerTotal).unwrap();
        assert_eq!(au.z, 79);
    }
}
