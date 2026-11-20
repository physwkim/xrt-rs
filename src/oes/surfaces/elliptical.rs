//! Elliptical parametric mirror: ellipsoid of revolution about the major axis.
//!
//! r(s) = B·sqrt(|1 − s²/A²|), s measured along the major axis from the
//! ellipse center, φ = 0 pointing up. The mirror is open: only the lower half
//! (|φ| > π/2) reflects.
//!
//! Ported from xrt `EllipticalMirrorParam` (backends/raycing/oes/parametric.py).

use std::f64::consts::FRAC_PI_2;

use crate::oes::surface::ParametricSurface;
use crate::oes::surfaces::conic_frame::{ConicFrame, OPEN_SIDE_R};

#[derive(Debug, Clone, Copy)]
pub struct EllipticalSurface {
    /// Semi-major axis A [mm]
    pub a: f64,
    /// Semi-minor axis B [mm]
    pub b: f64,
    /// Ellipse center and major-axis tilt in mirror-local coordinates
    pub frame: ConicFrame,
}

impl EllipticalSurface {
    /// Ellipse through the local origin with foci at distance `p` upstream
    /// and `q` downstream, met at grazing angle `theta` [rad].
    pub fn from_pq(p: f64, q: f64, theta: f64) -> Self {
        let (sin_t, cos_t) = theta.abs().sin_cos();
        let gamma = ((p - q) * sin_t).atan2((p + q) * cos_t);
        Self {
            a: (p + q) * 0.5,
            b: (p * q).sqrt() * sin_t,
            frame: ConicFrame::new((q - p) * cos_t * 0.5, (p + q) * sin_t * 0.5, gamma),
        }
    }

    /// Ellipse with its major axis along local y, centered at y = `y0`.
    pub fn new(a: f64, b: f64, y0: f64) -> Self {
        Self {
            a,
            b,
            frame: ConicFrame::on_axis(y0),
        }
    }
}

impl ParametricSurface for EllipticalSurface {
    fn local_r(&self, s: f64, phi: f64) -> f64 {
        if phi.abs() <= FRAC_PI_2 {
            return OPEN_SIDE_R;
        }
        self.b * (1.0 - s * s / (self.a * self.a)).abs().sqrt()
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let a2s2 = self.a * self.a - s * s;
        // Beyond the ellipse ends: a near-flat normal, as upstream (ray is lost).
        let a2s2 = if a2s2 <= 0.0 { 1e22 } else { a2s2 };
        let nr = -self.b / self.a * s / a2s2.sqrt();
        let norm = nr.hypot(1.0);
        let (sin_phi, cos_phi) = phi.sin_cos();
        self.frame
            .vector_to_local(-sin_phi / norm, nr / norm, -cos_phi / norm)
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
    fn elliptical_from_pq() {
        let (p, q, theta) = (10_000.0, 5_000.0, 0.003);
        let e = EllipticalSurface::from_pq(p, q, theta);
        assert!((e.a - 7500.0).abs() < 1e-9);
        assert!((e.b - (p * q).sqrt() * theta.sin()).abs() < 1e-12);
    }

    #[test]
    fn pole_lies_on_surface_with_upward_normal() {
        let e = EllipticalSurface::from_pq(10_000.0, 300.0, 0.003);
        let (s, phi, r) = e.xyz_to_param(0.0, 0.0, 0.0);
        assert!(phi.abs() > FRAC_PI_2, "pole on the open side, phi = {phi}");
        assert!((e.local_r(s, phi) - r).abs() < 1e-9 * r);
        let n = e.local_n(s, phi);
        assert!(n[0].abs() < 1e-12 && n[1].abs() < 1e-9 && (n[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn elliptical_r_at_center() {
        let e = EllipticalSurface::new(100.0, 50.0, 0.0);
        assert!((e.local_r(0.0, PI) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn elliptical_r_at_end() {
        let e = EllipticalSurface::new(100.0, 50.0, 0.0);
        assert!(e.local_r(100.0, PI).abs() < 1e-10);
    }

    #[test]
    fn upper_half_is_open() {
        let e = EllipticalSurface::new(100.0, 50.0, 0.0);
        assert_eq!(e.local_r(0.0, 0.0), OPEN_SIDE_R);
    }

    #[test]
    fn param_xyz_roundtrip() {
        let e = EllipticalSurface::from_pq(10_000.0, 300.0, 0.003);
        let (x, y, z) = (3.0, 5.0, 7.0);
        let (s, phi, r) = e.xyz_to_param(x, y, z);
        let (x2, y2, z2) = e.param_to_xyz(s, phi, r);
        assert!((x2 - x).abs() < 1e-9);
        assert!((y2 - y).abs() < 1e-9);
        assert!((z2 - z).abs() < 1e-9);
    }
}
