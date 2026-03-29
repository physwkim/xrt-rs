//! Flat mirror surface: z = 0, n = [0, 0, 1].

use crate::surface::Surface;

/// A perfectly flat mirror surface.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatSurface;

impl Surface for FlatSurface {
    #[inline]
    fn local_z(&self, _x: f64, _y: f64) -> f64 {
        0.0
    }

    #[inline]
    fn local_n(&self, _x: f64, _y: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_surface_z() {
        let s = FlatSurface;
        assert_eq!(s.local_z(1.0, 2.0), 0.0);
        assert_eq!(s.local_z(-100.0, 500.0), 0.0);
    }

    #[test]
    fn flat_surface_normal() {
        let s = FlatSurface;
        assert_eq!(s.local_n(0.0, 0.0), [0.0, 0.0, 1.0]);
        assert_eq!(s.local_n(10.0, -5.0), [0.0, 0.0, 1.0]);
    }
}
