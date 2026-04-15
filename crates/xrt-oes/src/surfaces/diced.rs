//! Diced (segmented) surface wrapper.
//!
//! Wraps any [`Surface`] and divides it into rectangular facets with gaps.
//! Rays hitting gaps are considered lost (z returns `NaN`).

use crate::surface::Surface;

/// A diced (segmented) surface.
///
/// Divides the underlying surface into rectangular facets of size
/// `dx_facet` x `dy_facet`, separated by gaps of `dx_gap` and `dy_gap`.
/// At gap positions, [`Surface::local_z`] returns [`f64::NAN`] to signal
/// ray loss.
#[derive(Debug, Clone, Copy)]
pub struct DicedSurface<S: Surface> {
    /// Underlying surface
    pub base: S,
    /// Facet width in x [mm]
    pub dx_facet: f64,
    /// Facet height in y [mm]
    pub dy_facet: f64,
    /// Gap width in x [mm]
    pub dx_gap: f64,
    /// Gap height in y [mm]
    pub dy_gap: f64,
}

impl<S: Surface> DicedSurface<S> {
    /// Create a new diced surface.
    ///
    /// # Arguments
    /// * `base` - The underlying surface to segment
    /// * `dx_facet` - Facet width in x [mm]
    /// * `dy_facet` - Facet height in y [mm]
    /// * `dx_gap` - Gap width in x [mm]
    /// * `dy_gap` - Gap height in y [mm]
    pub fn new(base: S, dx_facet: f64, dy_facet: f64, dx_gap: f64, dy_gap: f64) -> Self {
        Self {
            base,
            dx_facet,
            dy_facet,
            dx_gap,
            dy_gap,
        }
    }

    /// Check if (x, y) falls in a gap between facets.
    fn is_in_gap(&self, x: f64, y: f64) -> bool {
        let pitch_x = self.dx_facet + self.dx_gap;
        let pitch_y = self.dy_facet + self.dy_gap;
        // Euclidean remainder: always non-negative
        let x_mod = ((x % pitch_x) + pitch_x) % pitch_x;
        let y_mod = ((y % pitch_y) + pitch_y) % pitch_y;
        x_mod >= self.dx_facet || y_mod >= self.dy_facet
    }

    /// Return the center of the facet containing (x, y).
    fn facet_center(&self, x: f64, y: f64) -> (f64, f64) {
        let pitch_x = self.dx_facet + self.dx_gap;
        let pitch_y = self.dy_facet + self.dy_gap;
        let facet_cx = (x / pitch_x).round() * pitch_x;
        let facet_cy = (y / pitch_y).round() * pitch_y;
        (facet_cx, facet_cy)
    }
}

impl<S: Surface> Surface for DicedSurface<S> {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        if self.is_in_gap(x, y) {
            f64::NAN
        } else {
            let (cx, cy) = self.facet_center(x, y);
            self.base.local_z(cx, cy)
        }
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        if self.is_in_gap(x, y) {
            [0.0, 0.0, 1.0] // arbitrary; ray will be lost
        } else {
            let (cx, cy) = self.facet_center(x, y);
            self.base.local_n(cx, cy)
        }
    }

    fn local_n_bragg(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        if self.is_in_gap(x, y) {
            None
        } else {
            let (cx, cy) = self.facet_center(x, y);
            self.base.local_n_bragg(cx, cy)
        }
    }

    fn local_g(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        self.base.local_g(x, y)
    }
}

/// Diced Johann toroid crystal.
pub type DicedJohannToroid = DicedSurface<super::johann_toroid::JohannToroidSurface>;

/// Diced Johansson toroid crystal.
pub type DicedJohanssonToroid = DicedSurface<super::johann_toroid::JohanssonToroidSurface>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::flat::FlatSurface;

    #[test]
    fn diced_flat_gap_detection() {
        let d = DicedSurface::new(FlatSurface, 10.0, 10.0, 1.0, 1.0);
        // Center of facet: should NOT be in gap
        assert!(!d.is_in_gap(5.0, 5.0));
        // In gap (x > facet width within pitch):
        assert!(d.is_in_gap(10.5, 5.0));
    }

    #[test]
    fn diced_flat_z_in_facet() {
        let d = DicedSurface::new(FlatSurface, 10.0, 10.0, 1.0, 1.0);
        let z = d.local_z(5.0, 5.0);
        assert!(z.is_finite(), "z in facet should be finite");
        assert!(z.abs() < 1e-10, "flat diced z should be ~0");
    }

    #[test]
    fn diced_flat_z_in_gap() {
        let d = DicedSurface::new(FlatSurface, 10.0, 10.0, 1.0, 1.0);
        let z = d.local_z(10.5, 5.0);
        assert!(z.is_nan(), "z in gap should be NaN");
    }
}
