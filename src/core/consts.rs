//! Physical and mathematical constants for X-ray tracing.
//!
//! Ported from `xrt/backends/raycing/physconsts.py`.
//! All values are in CGS unless prefixed with `SI`.

// ── Mathematical constants ──────────────────────────────────────────────────

pub const PI: f64 = std::f64::consts::PI;
pub const PI2: f64 = 2.0 * PI;
pub const SQRT2PI: f64 = 2.506_628_274_631_000_5; // (2π)^0.5
pub const SQ3: f64 = 1.732_050_807_568_877_2;
pub const SQ2: f64 = std::f64::consts::SQRT_2;
pub const SQPI: f64 = 1.772_453_850_905_516; // π^0.5

// ── Fundamental physical constants (CGS) ────────────────────────────────────

/// Elementary charge [SI: C]
pub const SIE0: f64 = 1.602_176_565e-19;

/// Speed of light [cm/s]
pub const C: f64 = 2.997_924_58e10;

/// Elementary charge [esu]
pub const E0: f64 = SIE0 * C / 10.0; // = 4.803204...e-10

/// Electron rest mass [g]
pub const M0: f64 = 9.109_383_701_528e-28;

/// Electron rest mass [kg]
pub const SIM0: f64 = 9.109_383_701_528e-31;

/// Electron rest mass energy [MeV]
pub const M0C2: f64 = 0.510_998_928;

/// Planck constant [erg·s]
pub const HPLANCK: f64 = 6.626_069_573e-27;

/// Energy conversion: eV → erg
pub const EV2ERG: f64 = 1.602_176_565e-12;

/// K to B conversion factor (= 2π m₀ c² × 0.001 / e₀)
pub const K2B: f64 = 10.710_201_593_926_415;

/// e / (m₀ · c[mm])
pub const EMC: f64 = 0.586_679_180_241_648_7;

/// Planck constant [SI: J·s]
pub const SIHPLANCK: f64 = 6.626_069_573e-34;

/// Speed of light [SI: m/s]
pub const SIC: f64 = C * 1e-2;

/// Fine-structure constant
pub const FINE_STR: f64 = 1.0 / 137.035_999_76;

/// Energy to angular frequency: ω = E2W × E[eV]
pub const E2W: f64 = 1_519_267_514_747_457.9;

pub const E2WC: f64 = 5_067.730_939_206_809;

/// Classical electron radius [Å]
pub const R0: f64 = 2.817_940_285e-5;

/// Avogadro's number [atoms/mol]
pub const AVOGADRO: f64 = 6.022_141_99e23;

/// c·h [eV·cm]
pub const CH_EV_CM: f64 = 1.239_841_929_761_767_8e-4; // HPLANCK * C / EV2ERG

/// c·h [eV·Å]
pub const CH: f64 = 12_398.419_297_617_678;

/// c·ħ [eV·Å]  (= c·h / 2π)
pub const CHBAR: f64 = 1_973.269_717_741_798_6;

/// Default photon energy [eV]
pub const DEFAULT_ENERGY: f64 = 9000.0;
