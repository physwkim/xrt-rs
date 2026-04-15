//! Data file parsers for atomic scattering data.
//!
//! - `f0_xop.dat`: f0 Gaussian sum coefficients (include_str! embedded)
//! - `AtomicData.dat`: atomic masses and other properties (include_str! embedded)
//! - `Chantler.npz`, `Henke.npz`, `BrCo.npz`: f1/f2 scattering factors (runtime NPZ loading)

use std::collections::HashMap;
use std::io::{Cursor, Read as IoRead};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use xrt_core::error::XrtError;
use xrt_math::f0::F0Coeffs;

// Embed the text data files at compile time
const F0_XOP_DATA: &str = include_str!("../data/f0_xop.dat");
const ATOMIC_DATA: &str = include_str!("../data/AtomicData.dat");

/// The periodic table, index 0 = "none" (placeholder).
pub const ELEMENTS_LIST: &[&str] = &[
    "none", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
];

/// Scattering factor table selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatteringTable {
    /// Chantler (photoelectric only), 11 eV < E < 405 keV
    Chantler,
    /// Chantler with total absorption cross-sections
    ChantlerTotal,
    /// Henke, 10 eV < E < 30 keV
    Henke,
    /// Brennan & Cowan, 30 eV < E < 509 keV
    BrCo,
}

impl ScatteringTable {
    /// Returns the NPZ filename (without path) for this table.
    pub fn filename(&self) -> &'static str {
        match self {
            ScatteringTable::Chantler | ScatteringTable::ChantlerTotal => "Chantler.npz",
            ScatteringTable::Henke => "Henke.npz",
            ScatteringTable::BrCo => "BrCo.npz",
        }
    }

    /// Returns the f2 key suffix for this table.
    pub fn f2_key_suffix(&self) -> &'static str {
        match self {
            ScatteringTable::ChantlerTotal => "_f2tot",
            _ => "_f2",
        }
    }

    /// Parse from string (matching Python XRT conventions).
    pub fn from_str_xrt(s: &str) -> Self {
        let s_lower = s.to_lowercase();
        if s_lower.contains("henke") {
            ScatteringTable::Henke
        } else if s_lower.contains("brco") || s_lower.contains("brennan") {
            ScatteringTable::BrCo
        } else if s_lower.contains("total") {
            ScatteringTable::ChantlerTotal
        } else {
            ScatteringTable::Chantler
        }
    }
}

/// Look up element Z (atomic number) from name.
pub fn element_z(name: &str) -> Option<usize> {
    ELEMENTS_LIST.iter().position(|&e| e == name)
}

/// Look up element name from Z.
pub fn element_name(z: usize) -> Option<&'static str> {
    ELEMENTS_LIST.get(z).copied()
}

// ── Cached f0 coefficients ──────────────────────────────────────────────────

static F0_CACHE: OnceLock<HashMap<usize, F0Coeffs>> = OnceLock::new();

fn parse_f0_data() -> HashMap<usize, F0Coeffs> {
    let mut map = HashMap::new();
    let mut current_z: Option<usize> = None;
    let mut looking_for_up = false;

    for line in F0_XOP_DATA.lines() {
        if line.starts_with("#S") {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 2 {
                if let Ok(z) = fields[1].parse::<usize>() {
                    current_z = Some(z);
                    looking_for_up = true;
                }
            }
        } else if looking_for_up && line.starts_with("#UP") {
            // The data line follows #UP
            looking_for_up = false;
        } else if let Some(z) = current_z {
            if !looking_for_up && !line.starts_with('#') {
                // This is the coefficient data line
                let vals: Vec<f64> = line
                    .split_whitespace()
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if vals.len() == 11 {
                    let mut coeffs: F0Coeffs = [0.0; 11];
                    coeffs.copy_from_slice(&vals);
                    map.insert(z, coeffs);
                    current_z = None;
                }
            }
        }
    }
    map
}

/// Read f0 coefficients for element with atomic number Z.
pub fn read_f0_coeffs(z: usize) -> Result<F0Coeffs, XrtError> {
    let cache = F0_CACHE.get_or_init(parse_f0_data);
    cache
        .get(&z)
        .copied()
        .ok_or_else(|| XrtError::ElementNotFound(format!("f0 coefficients not found for Z={}", z)))
}

// ── Cached atomic masses ────────────────────────────────────────────────────

static MASS_CACHE: OnceLock<HashMap<usize, f64>> = OnceLock::new();

fn parse_atomic_data() -> HashMap<usize, f64> {
    let mut map = HashMap::new();
    for line in ATOMIC_DATA.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("0 ") {
            // Skip header line (starts with "0  Atomic...")
            if line.starts_with("0 ") && line.contains("AtomicRadius") {
                continue;
            }
            if line.starts_with("0 ") {
                // Z=0 doesn't exist, skip
                continue;
            }
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 4 {
            if let (Ok(z), Ok(mass)) = (fields[0].parse::<usize>(), fields[3].parse::<f64>()) {
                map.insert(z, mass);
            }
        }
    }
    map
}

/// Read atomic mass for element with atomic number Z.
pub fn read_atomic_mass(z: usize) -> Result<f64, XrtError> {
    let cache = MASS_CACHE.get_or_init(parse_atomic_data);
    cache
        .get(&z)
        .copied()
        .ok_or_else(|| XrtError::ElementNotFound(format!("atomic mass not found for Z={}", z)))
}

// ── NPZ f1/f2 loading ──────────────────────────────────────────────────────

/// Default data directory (relative to the crate root at compile time).
fn default_data_dir() -> PathBuf {
    // Try the crate's data directory
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_dir.join("data")
}

/// Read f1/f2 scattering factor tables from an NPZ file.
///
/// Returns (E, f1, f2) as Vec<f64>.
pub fn read_f1f2_table(
    element_name: &str,
    table: ScatteringTable,
    data_dir: Option<&Path>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), XrtError> {
    let dir = data_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_data_dir);
    let npz_path = dir.join(table.filename());

    let file = std::fs::File::open(&npz_path)
        .map_err(|e| XrtError::DataParse(format!("cannot open {}: {}", npz_path.display(), e)))?;

    let mut archive = zip::ZipArchive::new(file).map_err(|e| {
        XrtError::DataParse(format!("cannot read NPZ {}: {}", npz_path.display(), e))
    })?;

    let e_key = format!("{element_name}_E.npy");
    let f1_key = format!("{element_name}_f1.npy");
    let f2_suffix = table.f2_key_suffix();
    let f2_key = format!("{element_name}{f2_suffix}.npy");

    let e_arr = read_npy_f32_from_zip(&mut archive, &e_key, &npz_path)?;
    let f1_arr = read_npy_f32_from_zip(&mut archive, &f1_key, &npz_path)?;
    let f2_arr = read_npy_f32_from_zip(&mut archive, &f2_key, &npz_path)?;

    Ok((e_arr, f1_arr, f2_arr))
}

/// Read a single f32 NPY array from within a ZIP archive, converting to f64.
fn read_npy_f32_from_zip(
    archive: &mut zip::ZipArchive<std::fs::File>,
    name: &str,
    npz_path: &Path,
) -> Result<Vec<f64>, XrtError> {
    let mut entry = archive.by_name(name).map_err(|_| {
        XrtError::DataParse(format!(
            "key '{}' not found in {}",
            name,
            npz_path.display()
        ))
    })?;

    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).map_err(|e| {
        XrtError::DataParse(format!(
            "failed to read '{}' from {}: {}",
            name,
            npz_path.display(),
            e
        ))
    })?;

    let reader = npyz::NpyFile::new(Cursor::new(&buf))
        .map_err(|e| XrtError::DataParse(format!("failed to parse NPY '{}': {}", name, e)))?;

    let data: Vec<f32> = reader
        .into_vec()
        .map_err(|e| XrtError::DataParse(format!("failed to read NPY data '{}': {}", name, e)))?;

    Ok(data.into_iter().map(|v| v as f64).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elements_list_correct_length() {
        // 0=none + 1..92 elements = 93 entries
        assert_eq!(ELEMENTS_LIST.len(), 93);
    }

    #[test]
    fn element_z_lookup() {
        assert_eq!(element_z("H"), Some(1));
        assert_eq!(element_z("Si"), Some(14));
        assert_eq!(element_z("U"), Some(92));
        assert_eq!(element_z("Xx"), None);
    }

    #[test]
    fn read_f0_coeffs_si() {
        let coeffs = read_f0_coeffs(14).unwrap();
        // a1..a5,c should sum to ~14 (Z of Si) within parameterization error
        let sum: f64 = coeffs[..6].iter().sum();
        assert!(
            (sum - 14.0).abs() < 0.05,
            "sum of coefficients = {sum}, expected ~14"
        );
    }

    #[test]
    fn read_f0_coeffs_hydrogen() {
        let coeffs = read_f0_coeffs(1).unwrap();
        let sum: f64 = coeffs[..6].iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "sum of coefficients = {sum}, expected ~1"
        );
    }

    #[test]
    fn read_atomic_mass_si() {
        let mass = read_atomic_mass(14).unwrap();
        assert!(
            (mass - 28.0855).abs() < 0.001,
            "Si mass = {mass}, expected 28.0855"
        );
    }

    #[test]
    fn read_f1f2_chantler_si() {
        let (e, f1, f2) = read_f1f2_table("Si", ScatteringTable::ChantlerTotal, None).unwrap();
        assert!(!e.is_empty());
        assert_eq!(e.len(), f1.len());
        assert_eq!(e.len(), f2.len());
        // Energy range should span from ~5 eV to ~400 keV
        assert!(e[0] < 10.0);
        assert!(e[e.len() - 1] > 100_000.0);
    }

    #[test]
    fn scattering_table_from_str() {
        assert_eq!(
            ScatteringTable::from_str_xrt("Chantler total"),
            ScatteringTable::ChantlerTotal
        );
        assert_eq!(
            ScatteringTable::from_str_xrt("Henke"),
            ScatteringTable::Henke
        );
        assert_eq!(ScatteringTable::from_str_xrt("BrCo"), ScatteringTable::BrCo);
    }
}
