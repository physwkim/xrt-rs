//! Plate (transmission slab) surface.
//!
//! A flat plate of given thickness. Used as a transmission optical element
//! (e.g., Laue crystal plate, filter, window).
//! The plate itself is flat; transmission is handled by DeflectionMode::Refract
//! or by the crystal amplitude calculation.

use crate::surface::Surface;

/// A flat plate with finite thickness.
///
/// The surface shape is flat (z = 0). The thickness is stored as metadata
/// for use by the OE wrapper (e.g., for absorption calculation or
/// Laue crystal diffraction through the plate volume).
#[derive(Debug, Clone, Copy)]
pub struct PlateSurface {
    /// Plate thickness [mm]
    pub thickness: f64,
}

impl PlateSurface {
    pub fn new(thickness: f64) -> Self {
        Self { thickness }
    }
}

impl Surface for PlateSurface {
    fn local_z(&self, _x: f64, _y: f64) -> f64 {
        0.0
    }

    fn local_n(&self, _x: f64, _y: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plate_is_flat() {
        let p = PlateSurface::new(1.0);
        assert_eq!(p.local_z(10.0, 50.0), 0.0);
        let n = p.local_n(10.0, 50.0);
        assert_eq!(n, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn plate_stores_thickness() {
        let p = PlateSurface::new(0.5);
        assert_eq!(p.thickness, 0.5);
    }
}
