//! Hyperbolic parametric mirror: hyperboloid of revolution about the real axis.
//!
//! r(s) = B·sqrt(|s²/A² − 1|), s measured along the real axis from the
//! hyperbola center, φ = 0 pointing up. The mirror is open: only the upper half
//! (|φ| < π/2) reflects.
//!
//! Ported from xrt `HyperbolicMirrorParam` (backends/raycing/oes/parametric.py).

use std::f64::consts::FRAC_PI_2;

use crate::oes::surface::ParametricSurface;
use crate::oes::surfaces::conic_frame::{ConicFrame, OPEN_SIDE_R};

#[derive(Debug, Clone, Copy)]
pub struct HyperbolicSurface {
    /// Real semi-axis A [mm]
    pub a: f64,
    /// Imaginary semi-axis B [mm]
    pub b: f64,
    /// Hyperbola center and real-axis tilt in mirror-local coordinates
    pub frame: ConicFrame,
}

impl HyperbolicSurface {
    /// Hyperbola through the local origin with foci at distance `p` upstream
    /// and `q` (virtual) downstream, met at grazing angle `theta` [rad].
    pub fn from_pq(p: f64, q: f64, theta: f64) -> Self {
        let (sin_t, cos_t) = theta.abs().sin_cos();
        let gamma = ((p + q) * sin_t).atan2((p - q) * cos_t);
        Self {
            a: (p - q).abs() * 0.5,
            b: (p * q).sqrt() * sin_t,
            frame: ConicFrame::new(-(p + q) * cos_t * 0.5, (p - q) * sin_t * 0.5, gamma),
        }
    }

    /// Hyperbola with its real axis along local y, centered at y = `y0`.
    pub fn new(a: f64, b: f64, y0: f64) -> Self {
        Self {
            a,
            b,
            frame: ConicFrame::on_axis(y0),
        }
    }
}

impl ParametricSurface for HyperbolicSurface {
    fn local_r(&self, s: f64, phi: f64) -> f64 {
        if phi.abs() >= FRAC_PI_2 {
            return OPEN_SIDE_R;
        }
        self.b * (s * s / (self.a * self.a) - 1.0).abs().sqrt()
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let s2a2 = s * s - self.a * self.a;
        // Between the branches: a near-flat normal, as upstream (ray is lost).
        let s2a2 = if s2a2 <= 0.0 { 1e22 } else { s2a2 };
        let nr = -self.b / self.a * s / s2a2.sqrt();
        let norm = nr.hypot(1.0);
        let (sin_phi, cos_phi) = phi.sin_cos();
        self.frame
            .vector_to_local(sin_phi / norm, nr / norm, cos_phi / norm)
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

    #[test]
    fn hyperbolic_r_at_vertex() {
        let s = HyperbolicSurface::new(100.0, 50.0, 0.0);
        assert!(s.local_r(100.0, 0.0).abs() < 1e-10);
    }

    #[test]
    fn hyperbolic_r_beyond_vertex() {
        let (a, b) = (100.0, 50.0);
        let s = HyperbolicSurface::new(a, b, 0.0);
        let sv = 200.0;
        let expected = b * ((sv / a).powi(2) - 1.0).sqrt();
        let got = s.local_r(sv, 0.0);
        assert!(
            (got - expected).abs() < 1e-10,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn lower_half_is_open() {
        let s = HyperbolicSurface::new(100.0, 50.0, 0.0);
        assert_eq!(s.local_r(200.0, std::f64::consts::PI), OPEN_SIDE_R);
    }

    #[test]
    fn pole_lies_on_surface_with_upward_normal() {
        let s = HyperbolicSurface::from_pq(10_000.0, 300.0, 0.003);
        let (sv, phi, r) = s.xyz_to_param(0.0, 0.0, 0.0);
        assert!(phi.abs() < FRAC_PI_2, "pole on the open side, phi = {phi}");
        assert!((s.local_r(sv, phi) - r).abs() < 1e-9 * r);
        let n = s.local_n(sv, phi);
        assert!(n[0].abs() < 1e-12 && n[1].abs() < 1e-9 && (n[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn param_xyz_roundtrip() {
        let s = HyperbolicSurface::from_pq(10_000.0, 300.0, 0.003);
        let (x, y, z) = (3.0, 200.0, 7.0);
        let (sv, phi, r) = s.xyz_to_param(x, y, z);
        let (x2, y2, z2) = s.param_to_xyz(sv, phi, r);
        assert!((x2 - x).abs() < 1e-9);
        assert!((y2 - y).abs() < 1e-9);
        assert!((z2 - z).abs() < 1e-9);
    }
}
