//! Paraboloidal parametric surface (paraboloid of revolution).
//!
//! Parametric form: r(s) = 2√(p·s + p²) where p is the parameter.
//! s is the arc parameter along the meridional direction.
//! φ is the azimuthal angle.
//! Supports γ rotation (pitch angle offset).
//!
//! Ported from oes.py:1740-1900.

use crate::surface::ParametricSurface;

/// Paraboloid of revolution defined by p (focal parameter) and q (image distance).
#[derive(Debug, Clone, Copy)]
pub struct ParabolicalSurface {
    /// Focal parameter p [mm]
    pub p_param: f64,
    /// Center offset along the meridional axis
    pub y0: f64,
}

impl ParabolicalSurface {
    /// Create from p (source distance) and q (image distance) at grazing angle θ.
    ///
    /// For a paraboloid: focal parameter p = q·sin²(θ) / 2
    pub fn from_pq(p_dist: f64, q_dist: f64, theta: f64) -> Self {
        // For collimating: p_param = p·sin²(θ)/2
        // For focusing: p_param = q·sin²(θ)/2
        let p_param = q_dist * theta.sin().powi(2) / 2.0;
        let y0 = (p_dist - q_dist) / 2.0;
        Self { p_param, y0 }
    }

    /// Create directly from focal parameter.
    pub fn new(p_param: f64, y0: f64) -> Self {
        Self { p_param, y0 }
    }
}

impl ParametricSurface for ParabolicalSurface {
    fn local_r(&self, s: f64, _phi: f64) -> f64 {
        let s_shifted = s + self.y0;
        let arg = self.p_param * s_shifted + self.p_param * self.p_param;
        if arg > 0.0 {
            2.0 * arg.sqrt()
        } else {
            0.0
        }
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let s_shifted = s + self.y0;
        let arg = self.p_param * s_shifted + self.p_param * self.p_param;

        if arg <= 0.0 {
            return [0.0, 0.0, 1.0];
        }

        let r = 2.0 * arg.sqrt();
        // dr/ds = p / sqrt(p·s + p²) = p / (r/2) = 2p/r
        let dr_ds = 2.0 * self.p_param / r;

        let (sin_phi, cos_phi) = phi.sin_cos();

        // Tangent vectors
        let ts = [dr_ds * sin_phi, 1.0, dr_ds * cos_phi];
        let tp = [r * cos_phi, 0.0, -r * sin_phi];

        // Normal = cross(tangent_s, tangent_phi)
        let nx = ts[1] * tp[2] - ts[2] * tp[1];
        let ny = ts[2] * tp[0] - ts[0] * tp[2];
        let nz = ts[0] * tp[1] - ts[1] * tp[0];

        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        if norm > 1e-300 {
            let sign = if nx * sin_phi + nz * cos_phi > 0.0 {
                1.0
            } else {
                -1.0
            };
            [sign * nx / norm, sign * ny / norm, sign * nz / norm]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r = (x * x + z * z).sqrt();
        let phi = x.atan2(z);
        let s = y;
        (s, phi, r)
    }

    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        let x = r * phi.sin();
        let y = s;
        let z = r * phi.cos();
        (x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parabolical_r_at_center() {
        let s = ParabolicalSurface::new(10.0, 0.0);
        // At s=0: r = 2*sqrt(0 + p²) = 2*p = 20
        let r = s.local_r(0.0, 0.0);
        assert!((r - 20.0).abs() < 1e-10, "r = {r}, expected 20.0");
    }

    #[test]
    fn parabolical_r_positive_s() {
        let p = 5.0;
        let s = ParabolicalSurface::new(p, 0.0);
        let sv = 10.0;
        let expected = 2.0 * (p * sv + p * p).sqrt();
        let got = s.local_r(sv, 0.0);
        assert!(
            (got - expected).abs() < 1e-10,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn parabolical_normal_at_origin() {
        let s = ParabolicalSurface::new(10.0, 0.0);
        let n = s.local_n(0.0, 0.0);
        // At phi=0, normal should point outward in +z direction
        assert!(n[2].abs() > 0.0);
    }

    #[test]
    fn param_xyz_roundtrip() {
        let s = ParabolicalSurface::new(10.0, 0.0);
        let (x, y, z) = (3.0, 5.0, 7.0);
        let (sv, phi, r) = s.xyz_to_param(x, y, z);
        let (x2, y2, z2) = s.param_to_xyz(sv, phi, r);
        assert!((x2 - x).abs() < 1e-10);
        assert!((y2 - y).abs() < 1e-10);
        assert!((z2 - z).abs() < 1e-10);
    }
}
