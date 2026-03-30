//! Holographic grating surface.
//!
//! Groove pattern formed by interference of two recording beams.
//! The grating vector varies with position based on the geometry
//! of the two recording sources.
//!
//! Compatible with shadow3's HOLO recording geometry.

use crate::surface::Surface;

/// A holographic grating recorded by two point sources.
///
/// The groove density at each point (x, y) is determined by the
/// path difference from the two recording sources to that point.
#[derive(Debug, Clone, Copy)]
pub struct HolographicGrating {
    /// Recording wavelength [mm]
    pub recording_wavelength: f64,
    /// Recording source 1 position [x, y, z] in mm
    pub source1: [f64; 3],
    /// Recording source 2 position [x, y, z] in mm
    pub source2: [f64; 3],
}

impl HolographicGrating {
    pub fn new(recording_wavelength: f64, source1: [f64; 3], source2: [f64; 3]) -> Self {
        Self { recording_wavelength, source1, source2 }
    }

    /// Compute the local grating density at (x, y, z=0).
    fn local_density(&self, x: f64, y: f64) -> [f64; 3] {
        let z = 0.0;
        // Distance from point to each source
        let dx1 = x - self.source1[0];
        let dy1 = y - self.source1[1];
        let dz1 = z - self.source1[2];
        let r1 = (dx1 * dx1 + dy1 * dy1 + dz1 * dz1).sqrt();

        let dx2 = x - self.source2[0];
        let dy2 = y - self.source2[1];
        let dz2 = z - self.source2[2];
        let r2 = (dx2 * dx2 + dy2 * dy2 + dz2 * dz2).sqrt();

        if r1 < 1e-30 || r2 < 1e-30 {
            return [0.0, 0.0, 0.0];
        }

        // Grating vector = gradient of (r1 - r2) / recording_wavelength
        let inv_lam = 1.0 / self.recording_wavelength;
        let gx = inv_lam * (dx1 / r1 - dx2 / r2);
        let gy = inv_lam * (dy1 / r1 - dy2 / r2);
        let gz = inv_lam * (dz1 / r1 - dz2 / r2);

        [gx, gy, gz]
    }
}

impl Surface for HolographicGrating {
    fn local_z(&self, _x: f64, _y: f64) -> f64 {
        0.0 // flat substrate
    }

    fn local_n(&self, _x: f64, _y: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }

    fn local_g(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        Some(self.local_density(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holographic_flat_substrate() {
        let g = HolographicGrating::new(
            0.0005, // 500 nm recording wavelength
            [0.0, -1000.0, 100.0],
            [0.0, -1000.0, -100.0],
        );
        assert_eq!(g.local_z(0.0, 0.0), 0.0);
    }

    #[test]
    fn holographic_has_grating_vector() {
        let g = HolographicGrating::new(
            0.0005,
            [0.0, -1000.0, 100.0],
            [0.0, -1000.0, -100.0],
        );
        let gv = g.local_g(0.0, 0.0).unwrap();
        // Should have nonzero gz component (from z-separated sources)
        assert!(gv[2].abs() > 0.0, "should have grating vector");
    }

    #[test]
    fn holographic_varies_with_position() {
        let g = HolographicGrating::new(
            0.0005,
            [100.0, -1000.0, 100.0],
            [-100.0, -1000.0, -100.0],
        );
        let g1 = g.local_g(0.0, 0.0).unwrap();
        let g2 = g.local_g(10.0, 50.0).unwrap();
        let diff = ((g1[0]-g2[0]).powi(2) + (g1[1]-g2[1]).powi(2) + (g1[2]-g2[2]).powi(2)).sqrt();
        assert!(diff > 1e-6, "grating vector should vary with position");
    }
}
