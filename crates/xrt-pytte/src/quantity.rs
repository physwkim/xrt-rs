//! Unit conversion utilities for Takagi-Taupin calculations.
//!
//! Provides conversions between common X-ray crystallography units.

use xrt_core::consts::CH;

/// Convert photon energy [eV] to wavelength [Å].
#[inline]
pub fn ev_to_angstrom(energy_ev: f64) -> f64 {
    CH / energy_ev
}

/// Convert wavelength [Å] to photon energy [eV].
#[inline]
pub fn angstrom_to_ev(wavelength_a: f64) -> f64 {
    CH / wavelength_a
}

/// Convert radians to arcseconds.
#[inline]
pub fn rad_to_arcsec(rad: f64) -> f64 {
    rad * 206_264.806_247_096_36
}

/// Convert arcseconds to radians.
#[inline]
pub fn arcsec_to_rad(arcsec: f64) -> f64 {
    arcsec / 206_264.806_247_096_36
}

/// Convert millimeters to Ångströms.
#[inline]
pub fn mm_to_angstrom(mm: f64) -> f64 {
    mm * 1e7
}

/// Convert Ångströms to millimeters.
#[inline]
pub fn angstrom_to_mm(angstrom: f64) -> f64 {
    angstrom * 1e-7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ev_angstrom_roundtrip() {
        let e = 10000.0;
        let lambda = ev_to_angstrom(e);
        let e2 = angstrom_to_ev(lambda);
        assert!((e2 - e).abs() < 1e-6);
    }

    #[test]
    fn cu_ka_wavelength() {
        // Cu Kα: ~8048 eV → ~1.5406 Å
        let lambda = ev_to_angstrom(8048.0);
        assert!(
            (lambda - 1.5406).abs() < 0.01,
            "λ = {lambda}, expected ~1.5406"
        );
    }

    #[test]
    fn rad_arcsec_roundtrip() {
        let angle = 0.001;
        let arcsec = rad_to_arcsec(angle);
        let angle2 = arcsec_to_rad(arcsec);
        assert!((angle2 - angle).abs() < 1e-15);
    }

    #[test]
    fn one_arcsec_in_rad() {
        let one_arcsec = arcsec_to_rad(1.0);
        assert!(
            (one_arcsec - 4.8481e-6).abs() < 1e-9,
            "1 arcsec = {} rad",
            one_arcsec
        );
    }

    #[test]
    fn mm_angstrom_roundtrip() {
        let mm = 0.001;
        let a = mm_to_angstrom(mm);
        let mm2 = angstrom_to_mm(a);
        assert!((mm2 - mm).abs() < 1e-15);
    }
}
