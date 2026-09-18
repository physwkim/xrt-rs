//! Aperture / rays_good: determine which rays hit the optical element.
//!
//! Ported from oes_base.py:1000-1062.

use crate::core::beam::RayState;

/// Aperture shape.
#[derive(Debug, Clone)]
pub enum ApertureShape {
    /// Rectangular aperture
    Rectangular,
    /// Round (elliptical) aperture
    Round,
}

/// Aperture limits for an optical element.
#[derive(Debug, Clone)]
pub struct Aperture {
    pub shape: ApertureShape,
    /// Physical limits in x: [min, max]
    pub phys_x: [f64; 2],
    /// Physical limits in y: [min, max]
    pub phys_y: [f64; 2],
    /// Optional optical (good) limits in x: [min, max]
    pub opt_x: Option<[f64; 2]>,
    /// Optional optical (good) limits in y: [min, max]
    pub opt_y: Option<[f64; 2]>,
}

impl Default for Aperture {
    fn default() -> Self {
        let half = 1e6; // maxHalfSizeOfOE
        Self {
            shape: ApertureShape::Rectangular,
            phys_x: [-half, half],
            phys_y: [-half, half],
            opt_x: None,
            opt_y: None,
        }
    }
}

impl Aperture {
    /// Determine ray state for a single ray at local coordinates (x, y).
    ///
    /// Returns:
    /// - `Good` (1): ray is within the optical area
    /// - `Over` (3): ray hit the physical surface but outside optical area
    /// - `Out` (2): ray missed the element entirely
    pub fn classify_ray(&self, x: f64, y: f64) -> RayState {
        match self.shape {
            ApertureShape::Rectangular => self.classify_rect(x, y),
            ApertureShape::Round => self.classify_round(x, y),
        }
    }

    fn classify_rect(&self, x: f64, y: f64) -> RayState {
        // Check physical limits
        if x < self.phys_x[0] || x > self.phys_x[1] {
            return RayState::Out;
        }
        if y < self.phys_y[0] || y > self.phys_y[1] {
            return RayState::Out;
        }

        // Check optical limits (if defined)
        if let Some(opt_x) = self.opt_x
            && (x < opt_x[0] || x > opt_x[1])
        {
            return RayState::Over;
        }
        if let Some(opt_y) = self.opt_y
            && (y < opt_y[0] || y > opt_y[1])
        {
            return RayState::Over;
        }

        RayState::Good
    }

    fn classify_round(&self, x: f64, y: f64) -> RayState {
        let cx = (self.phys_x[0] + self.phys_x[1]) * 0.5;
        let rx = (self.phys_x[1] - self.phys_x[0]) * 0.5;
        let cy = (self.phys_y[0] + self.phys_y[1]) * 0.5;
        let ry = (self.phys_y[1] - self.phys_y[0]) * 0.5;

        if rx.is_infinite() || ry.is_infinite() {
            return RayState::Good;
        }

        let r2 = ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2);
        if r2 > 1.0 {
            RayState::Out
        } else {
            RayState::Good
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_inside() {
        let ap = Aperture {
            shape: ApertureShape::Rectangular,
            phys_x: [-10.0, 10.0],
            phys_y: [-50.0, 50.0],
            opt_x: None,
            opt_y: None,
        };
        assert_eq!(ap.classify_ray(0.0, 0.0), RayState::Good);
        assert_eq!(ap.classify_ray(9.0, 49.0), RayState::Good);
    }

    #[test]
    fn rectangular_outside() {
        let ap = Aperture {
            shape: ApertureShape::Rectangular,
            phys_x: [-10.0, 10.0],
            phys_y: [-50.0, 50.0],
            opt_x: None,
            opt_y: None,
        };
        assert_eq!(ap.classify_ray(11.0, 0.0), RayState::Out);
        assert_eq!(ap.classify_ray(0.0, 51.0), RayState::Out);
    }

    #[test]
    fn rectangular_optical_limit() {
        let ap = Aperture {
            shape: ApertureShape::Rectangular,
            phys_x: [-10.0, 10.0],
            phys_y: [-50.0, 50.0],
            opt_x: Some([-5.0, 5.0]),
            opt_y: None,
        };
        assert_eq!(ap.classify_ray(0.0, 0.0), RayState::Good);
        assert_eq!(ap.classify_ray(7.0, 0.0), RayState::Over);
    }

    #[test]
    fn round_inside() {
        let ap = Aperture {
            shape: ApertureShape::Round,
            phys_x: [-10.0, 10.0],
            phys_y: [-10.0, 10.0],
            opt_x: None,
            opt_y: None,
        };
        assert_eq!(ap.classify_ray(0.0, 0.0), RayState::Good);
        assert_eq!(ap.classify_ray(5.0, 5.0), RayState::Good);
    }

    #[test]
    fn round_outside() {
        let ap = Aperture {
            shape: ApertureShape::Round,
            phys_x: [-10.0, 10.0],
            phys_y: [-10.0, 10.0],
            opt_x: None,
            opt_y: None,
        };
        assert_eq!(ap.classify_ray(9.0, 9.0), RayState::Out);
    }
}
