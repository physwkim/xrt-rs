//! Parabolical parametric mirror: paraboloid of revolution about its axis.
//!
//! r(s) = 2·sqrt(P·s + P²), s measured along the axis from the focus, φ = 0
//! pointing up. The mirror is open: only the lower half (|φ| > π/2) reflects.
//! A paraboloid has one finite focus, so it is built from either `p`
//! (collimator) or `q` (focuser), never both.
//!
//! Ported from xrt `ParabolicalMirrorParam` (backends/raycing/oes/parametric.py).

use std::f64::consts::FRAC_PI_2;

use crate::oes::surface::ParametricSurface;
use crate::oes::surfaces::conic_frame::{ConicFrame, Figure, OPEN_SIDE_R};

#[derive(Debug, Clone, Copy)]
pub struct ParabolicalSurface {
    /// Parabola parameter P [mm]: positive for a collimator, negative for a focuser
    pub param: f64,
    /// Focus and axis tilt in mirror-local coordinates
    pub frame: ConicFrame,
    /// Revolution about the axis, or a cylinder (upstream `isCylindrical`)
    pub figure: Figure,
}

impl ParabolicalSurface {
    /// Collimator: source focus at distance `p` upstream, grazing angle `theta` [rad].
    pub fn from_p(p: f64, theta: f64) -> Self {
        let theta = theta.abs();
        let (sin_t, cos_t) = theta.sin_cos();
        Self {
            param: p * sin_t * sin_t,
            frame: ConicFrame::new(-p * cos_t, p * sin_t, -theta),
            figure: Figure::Revolution,
        }
    }

    /// Focuser: image focus at distance `q` downstream, grazing angle `theta` [rad].
    pub fn from_q(q: f64, theta: f64) -> Self {
        let theta = theta.abs();
        let (sin_t, cos_t) = theta.sin_cos();
        Self {
            param: -q * sin_t * sin_t,
            frame: ConicFrame::new(q * cos_t, q * sin_t, theta),
            figure: Figure::Revolution,
        }
    }

    /// Paraboloid with its axis along local y and the focus at y = `y0`.
    pub fn new(param: f64, y0: f64) -> Self {
        Self {
            param,
            frame: ConicFrame::on_axis(y0),
            figure: Figure::Revolution,
        }
    }

    /// The same profile as a cylinder, curved in the meridional plane only.
    pub fn cylindrical(self) -> Self {
        Self {
            figure: Figure::Cylinder,
            ..self
        }
    }
}

impl ParametricSurface for ParabolicalSurface {
    fn local_r(&self, s: f64, phi: f64) -> f64 {
        if phi.abs() <= FRAC_PI_2 {
            return OPEN_SIDE_R;
        }
        let r0 = 2.0 * (self.param * s + self.param * self.param).max(0.0).sqrt();
        self.figure.r(r0, phi)
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let nr = self.param / (self.param * s + self.param * self.param).sqrt();
        let norm = nr.hypot(1.0);
        let (sin_phi, cos_phi) = phi.sin_cos();
        let (a, c) = match self.figure {
            Figure::Revolution => (-sin_phi, -cos_phi),
            Figure::Cylinder => (0.0, 1.0),
        };
        self.frame.vector_to_local(a / norm, nr / norm, c / norm)
    }

    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        self.frame.xyz_to_param(x, y, z)
    }

    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        self.frame.param_to_xyz(s, phi, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn parabolical_r_at_focus_plane() {
        let s = ParabolicalSurface::new(10.0, 0.0);
        // s = 0: r = 2·sqrt(P²) = 2P
        assert!((s.local_r(0.0, PI) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn parabolical_r_positive_s() {
        let p = 5.0;
        let s = ParabolicalSurface::new(p, 0.0);
        let sv = 10.0;
        let expected = 2.0 * (p * sv + p * p).sqrt();
        let got = s.local_r(sv, PI);
        assert!(
            (got - expected).abs() < 1e-10,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn upper_half_is_open() {
        let s = ParabolicalSurface::new(10.0, 0.0);
        assert_eq!(s.local_r(0.0, 0.0), OPEN_SIDE_R);
    }

    #[test]
    fn pole_lies_on_surface_with_upward_normal() {
        for s in [
            ParabolicalSurface::from_p(10_000.0, 0.003),
            ParabolicalSurface::from_q(300.0, 0.003),
        ] {
            let (sv, phi, r) = s.xyz_to_param(0.0, 0.0, 0.0);
            assert!(phi.abs() > FRAC_PI_2, "pole on the open side, phi = {phi}");
            assert!((s.local_r(sv, phi) - r).abs() < 1e-9 * r);
            let n = s.local_n(sv, phi);
            assert!(n[0].abs() < 1e-12 && n[1].abs() < 1e-9 && (n[2] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn param_xyz_roundtrip() {
        let s = ParabolicalSurface::from_q(300.0, 0.003);
        let (x, y, z) = (3.0, 5.0, 7.0);
        let (sv, phi, r) = s.xyz_to_param(x, y, z);
        let (x2, y2, z2) = s.param_to_xyz(sv, phi, r);
        assert!((x2 - x).abs() < 1e-9);
        assert!((y2 - y).abs() < 1e-9);
        assert!((z2 - z).abs() < 1e-9);
    }
}
