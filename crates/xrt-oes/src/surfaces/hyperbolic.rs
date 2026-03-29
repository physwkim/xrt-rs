//! Hyperbolic parametric surface (hyperboloid of revolution).
//!
//! Parametric form: r(s) = b × sqrt(s²/a² - 1)
//! where a is the real semi-axis and b is the imaginary semi-axis.
//!
//! Ported from oes.py hyperboloid surface logic.

use crate::surface::ParametricSurface;

/// Hyperboloid of revolution.
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicSurface {
    /// Real semi-axis a
    pub a: f64,
    /// Imaginary semi-axis b
    pub b: f64,
    /// Center offset along the meridional axis
    pub y0: f64,
    /// Sign convention: +1 for upper branch, -1 for lower
    pub sign: f64,
}

impl HyperbolicSurface {
    /// Create from p and q distances.
    ///
    /// For a hyperboloid: a = |p - q| / 2, c = (p + q) / 2, b = sqrt(c² - a²)
    pub fn from_pq(p: f64, q: f64, _theta: f64) -> Self {
        let a = ((p - q) / 2.0).abs();
        let c = (p + q) / 2.0;
        let b = (c * c - a * a).sqrt();
        let y0 = (p - q) / 2.0;
        Self {
            a,
            b,
            y0,
            sign: 1.0,
        }
    }

    /// Create directly from semi-axes.
    pub fn new(a: f64, b: f64, y0: f64) -> Self {
        Self {
            a,
            b,
            y0,
            sign: 1.0,
        }
    }
}

impl ParametricSurface for HyperbolicSurface {
    fn local_r(&self, s: f64, _phi: f64) -> f64 {
        let s_shifted = s + self.y0;
        let arg = s_shifted / self.a;
        let arg2 = arg * arg;
        if arg2 > 1.0 {
            self.sign * self.b * (arg2 - 1.0).sqrt()
        } else {
            0.0 // inside the asymptotic cone
        }
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let s_shifted = s + self.y0;
        let arg = s_shifted / self.a;
        let arg2 = arg * arg;

        if arg2 <= 1.0 {
            return [0.0, 0.0, 1.0];
        }

        let r = self.sign * self.b * (arg2 - 1.0).sqrt();
        // dr/ds = sign * b * s / (a² * sqrt(s²/a² - 1))
        let dr_ds =
            self.sign * self.b * s_shifted / (self.a * self.a * (arg2 - 1.0).sqrt());

        let (sin_phi, cos_phi) = phi.sin_cos();

        let ts = [dr_ds * sin_phi, 1.0, dr_ds * cos_phi];
        let tp = [r * cos_phi, 0.0, -r * sin_phi];

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
    fn hyperbolic_r_at_vertex() {
        let s = HyperbolicSurface::new(100.0, 50.0, 0.0);
        // At s = a (vertex), r = b * sqrt(1-1) = 0
        let r = s.local_r(100.0, 0.0);
        assert!(r.abs() < 1e-10, "r at vertex = {r}");
    }

    #[test]
    fn hyperbolic_r_beyond_vertex() {
        let a = 100.0;
        let b = 50.0;
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
    fn hyperbolic_inside_cone() {
        let s = HyperbolicSurface::new(100.0, 50.0, 0.0);
        // s < a → r = 0
        let r = s.local_r(50.0, 0.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn param_xyz_roundtrip() {
        let s = HyperbolicSurface::new(100.0, 50.0, 0.0);
        let (x, y, z) = (3.0, 200.0, 7.0);
        let (sv, phi, r) = s.xyz_to_param(x, y, z);
        let (x2, y2, z2) = s.param_to_xyz(sv, phi, r);
        assert!((x2 - x).abs() < 1e-10);
        assert!((y2 - y).abs() < 1e-10);
        assert!((z2 - z).abs() < 1e-10);
    }
}
