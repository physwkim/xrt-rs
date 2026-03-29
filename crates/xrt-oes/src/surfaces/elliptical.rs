//! Elliptical parametric surface (ellipsoid of revolution).
//!
//! Parametric form: r(s) = b × sqrt(1 - s²/a²)
//! where a is the semi-major axis, b is the semi-minor axis.
//! s is the arc parameter along the meridional direction.
//! φ is the azimuthal angle.
//!
//! Ported from oes.py:1500-1737.

use crate::surface::ParametricSurface;

/// Ellipsoid of revolution defined by p (source distance) and q (image distance).
#[derive(Debug, Clone, Copy)]
pub struct EllipticalSurface {
    /// Semi-major axis a
    pub a: f64,
    /// Semi-minor axis b
    pub b: f64,
    /// Center offset along the meridional axis
    pub y0: f64,
}

impl EllipticalSurface {
    /// Create from p (source-to-mirror) and q (mirror-to-image) distances
    /// and grazing angle theta [rad].
    pub fn from_pq(p: f64, q: f64, _theta: f64) -> Self {
        let a = (p + q) / 2.0;
        let c = ((p - q) / 2.0).abs(); // focal distance
        let b = (a * a - c * c).sqrt();
        // y0: the position of the mirror center on the ellipse
        let y0 = (p - q) / 2.0;
        Self { a, b, y0 }
    }

    /// Create directly from semi-axes.
    pub fn new(a: f64, b: f64, y0: f64) -> Self {
        Self { a, b, y0 }
    }
}

impl ParametricSurface for EllipticalSurface {
    fn local_r(&self, s: f64, _phi: f64) -> f64 {
        let s_shifted = s + self.y0;
        let arg = s_shifted / self.a;
        let arg2 = arg * arg;
        if arg2 < 1.0 {
            self.b * (1.0 - arg2).sqrt()
        } else {
            0.0 // beyond the ellipsoid
        }
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let s_shifted = s + self.y0;
        let arg = s_shifted / self.a;
        let arg2 = arg * arg;

        if arg2 >= 1.0 {
            return [0.0, 0.0, 1.0];
        }

        let r = self.b * (1.0 - arg2).sqrt();
        // dr/ds = -b * s / (a² * sqrt(1 - s²/a²))
        let dr_ds = -self.b * s_shifted / (self.a * self.a * (1.0 - arg2).sqrt());

        // Normal in parametric coordinates, then convert
        let (sin_phi, cos_phi) = phi.sin_cos();

        // Surface point: (r*sin(φ), s, r*cos(φ))
        // Tangent along s: (dr_ds*sin(φ), 1, dr_ds*cos(φ))
        // Tangent along φ: (r*cos(φ), 0, -r*sin(φ))
        // Normal = cross(tangent_s, tangent_phi)
        let ts = [dr_ds * sin_phi, 1.0, dr_ds * cos_phi];
        let tp = [r * cos_phi, 0.0, -r * sin_phi];

        let nx = ts[1] * tp[2] - ts[2] * tp[1];
        let ny = ts[2] * tp[0] - ts[0] * tp[2];
        let nz = ts[0] * tp[1] - ts[1] * tp[0];

        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        if norm > 1e-300 {
            // Ensure normal points outward (positive radial direction)
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
    fn elliptical_from_pq() {
        // p = 10m, q = 5m, theta = 3 mrad
        let e = EllipticalSurface::from_pq(10_000.0, 5_000.0, 0.003);
        assert!((e.a - 7500.0).abs() < 0.1);
        // c = 2500, b = sqrt(7500² - 2500²) = sqrt(50e6) ≈ 7071
        assert!((e.b - 7071.07).abs() < 1.0);
    }

    #[test]
    fn elliptical_r_at_center() {
        let e = EllipticalSurface::new(100.0, 50.0, 0.0);
        // At s=0 (center), r = b = 50
        let r = e.local_r(0.0, 0.0);
        assert!((r - 50.0).abs() < 1e-10);
    }

    #[test]
    fn elliptical_r_at_end() {
        let e = EllipticalSurface::new(100.0, 50.0, 0.0);
        // At s=a (end), r = 0
        let r = e.local_r(100.0, 0.0);
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn param_xyz_roundtrip() {
        let e = EllipticalSurface::new(100.0, 50.0, 0.0);
        let (x, y, z) = (3.0, 5.0, 7.0);
        let (s, phi, r) = e.xyz_to_param(x, y, z);
        let (x2, y2, z2) = e.param_to_xyz(s, phi, r);
        assert!((x2 - x).abs() < 1e-10);
        assert!((y2 - y).abs() < 1e-10);
        assert!((z2 - z).abs() < 1e-10);
    }
}
