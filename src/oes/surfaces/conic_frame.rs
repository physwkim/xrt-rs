//! Placement of a parametric conic's own axis in mirror-local coordinates.
//!
//! xrt's `EllipticalMirrorParam`, `ParabolicalMirrorParam` and
//! `HyperbolicMirrorParam` evaluate r(s, φ) about the conic axis, whose origin
//! (y0, z0) and tilt γ in the mirror frame follow from p, q and the grazing
//! angle (backends/raycing/oes/parametric.py).

use crate::core::transforms::rotate_x;

/// r returned on the half of an open mirror that does not reflect.
pub const OPEN_SIDE_R: f64 = 1e20;

/// How the meridional profile r0(s) is swept about the conic axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Figure {
    /// Surface of revolution about the axis.
    #[default]
    Revolution,
    /// The profile extruded along local x: curved in the meridional plane only.
    Cylinder,
}

impl Figure {
    /// r at azimuth φ of the point whose profile radius is `r0`.
    pub fn r(self, r0: f64, phi: f64) -> f64 {
        match self {
            Figure::Revolution => r0,
            Figure::Cylinder => r0 / phi.cos().abs(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConicFrame {
    /// Conic origin (ellipse/hyperbola center, parabola focus), local y [mm]
    pub y0: f64,
    /// Conic origin, local z [mm]
    pub z0: f64,
    /// cos γ, γ being the angle between the conic axis and the mirror surface
    pub cos_gamma: f64,
    /// sin γ
    pub sin_gamma: f64,
}

impl ConicFrame {
    pub fn new(y0: f64, z0: f64, gamma: f64) -> Self {
        let (sin_gamma, cos_gamma) = gamma.sin_cos();
        Self {
            y0,
            z0,
            cos_gamma,
            sin_gamma,
        }
    }

    /// Conic axis along local y with its origin at y = `y0`.
    pub fn on_axis(y0: f64) -> Self {
        Self::new(y0, 0.0, 0.0)
    }

    /// Mirror-local (x, y, z) to conic (s, φ, r).
    pub fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let (s, z_axis) = rotate_x(y - self.y0, z - self.z0, self.cos_gamma, self.sin_gamma);
        (s, x.atan2(z_axis), x.hypot(z_axis))
    }

    /// Conic (s, φ, r) to mirror-local (x, y, z).
    pub fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        let (sin_phi, cos_phi) = phi.sin_cos();
        let (y, z) = rotate_x(s, r * cos_phi, self.cos_gamma, -self.sin_gamma);
        (r * sin_phi, y + self.y0, z + self.z0)
    }

    /// A vector in conic axes, expressed in mirror-local axes.
    pub fn vector_to_local(&self, a: f64, b: f64, c: f64) -> [f64; 3] {
        let (b, c) = rotate_x(b, c, self.cos_gamma, -self.sin_gamma);
        [a, b, c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_xyz_roundtrip_tilted() {
        let f = ConicFrame::new(-4850.0, 15.4, 0.0028);
        let (x, y, z) = (0.7, 12.0, -0.3);
        let (s, phi, r) = f.xyz_to_param(x, y, z);
        let (x2, y2, z2) = f.param_to_xyz(s, phi, r);
        assert!((x2 - x).abs() < 1e-9 && (y2 - y).abs() < 1e-9 && (z2 - z).abs() < 1e-9);
    }
}
