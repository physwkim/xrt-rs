//! Capillary mirror surfaces (surfaces of revolution).
//!
//! Ellipsoid, hyperboloid, and paraboloid capillary optics
//! for focusing X-rays via total external reflection.

use crate::surface::ParametricSurface;

/// Paraboloid capillary mirror.
#[derive(Debug, Clone, Copy)]
pub struct ParaboloidCapillary {
    /// Focus parameter [mm]
    pub focus: f64,
    /// Distance from focus to entrance [mm]
    pub s0: f64,
}

impl ParaboloidCapillary {
    pub fn new(q: f64, r0: f64) -> Self {
        let focus = -0.5 * (q - (q * q + r0 * r0).sqrt());
        let s0 = focus + q;
        Self { focus, s0 }
    }
}

impl ParametricSurface for ParaboloidCapillary {
    fn local_r(&self, s: f64, _phi: f64) -> f64 {
        let ds = self.s0 - s;
        if ds > 0.0 {
            2.0 * (ds * self.focus).sqrt()
        } else {
            0.0
        }
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let ds = self.s0 - s;
        if ds <= 0.0 {
            return [0.0, 1.0, 0.0];
        }
        let dr_ds = -(self.focus / ds).sqrt();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let nx = -sin_phi;
        let ny = dr_ds;
        let nz = -cos_phi;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }

    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r = (x * x + z * z).sqrt();
        let phi = x.atan2(-z);
        let s = y;
        (s, phi, r)
    }

    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        let x = r * phi.sin();
        let y = s;
        let z = -r * phi.cos();
        (x, y, z)
    }
}

/// Ellipsoid capillary mirror.
#[derive(Debug, Clone, Copy)]
pub struct EllipsoidCapillary {
    pub a: f64,   // semi-major axis [mm]
    pub b: f64,   // semi-minor axis [mm]
    pub ctd: f64, // center-to-device distance [mm]
}

impl EllipsoidCapillary {
    pub fn new(a: f64, b: f64, working_distance: f64, length: f64) -> Self {
        let c = (a * a - b * b).sqrt();
        let ctd = c - working_distance - 0.5 * length;
        Self { a, b, ctd }
    }
}

impl ParametricSurface for EllipsoidCapillary {
    fn local_r(&self, s: f64, _phi: f64) -> f64 {
        let ss = self.ctd + s;
        let arg = 1.0 - ss * ss / (self.a * self.a);
        if arg > 0.0 {
            self.b * arg.sqrt()
        } else {
            0.0
        }
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let ss = self.ctd + s;
        let arg = 1.0 - ss * ss / (self.a * self.a);
        if arg <= 0.0 {
            return [0.0, 1.0, 0.0];
        }
        let dr_ds = -self.b * ss / (self.a * self.a * arg.sqrt());
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let nx = -sin_phi;
        let ny = dr_ds;
        let nz = -cos_phi;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }

    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r = (x * x + z * z).sqrt();
        let phi = x.atan2(-z);
        (y, phi, r)
    }

    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        (r * phi.sin(), s, -r * phi.cos())
    }
}

/// Hyperboloid capillary mirror.
#[derive(Debug, Clone, Copy)]
pub struct HyperboloidCapillary {
    pub a: f64,
    pub b: f64,
    pub ctd: f64,
}

impl HyperboloidCapillary {
    pub fn new(a: f64, b: f64, working_distance: f64, length: f64) -> Self {
        let c = (a * a + b * b).sqrt();
        let ctd = c + working_distance + 0.5 * length;
        Self { a, b, ctd }
    }
}

impl ParametricSurface for HyperboloidCapillary {
    fn local_r(&self, s: f64, _phi: f64) -> f64 {
        let ss = self.ctd + s;
        let arg = ss * ss / (self.a * self.a) - 1.0;
        if arg > 0.0 {
            self.b * arg.sqrt()
        } else {
            0.0
        }
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        let ss = self.ctd + s;
        let arg = ss * ss / (self.a * self.a) - 1.0;
        if arg <= 0.0 {
            return [0.0, 1.0, 0.0];
        }
        let dr_ds = self.b * ss / (self.a * self.a * arg.sqrt());
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let nx = -sin_phi;
        let ny = dr_ds;
        let nz = -cos_phi;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }

    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r = (x * x + z * z).sqrt();
        let phi = x.atan2(-z);
        (y, phi, r)
    }

    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        (r * phi.sin(), s, -r * phi.cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paraboloid_capillary_radius_positive() {
        let cap = ParaboloidCapillary::new(100.0, 0.5);
        let r = cap.local_r(0.0, 0.0);
        assert!(r > 0.0, "radius at entrance should be positive: {r}");
    }

    #[test]
    fn ellipsoid_capillary_radius() {
        let cap = EllipsoidCapillary::new(500.0, 5.0, 50.0, 100.0);
        let r = cap.local_r(0.0, 0.0);
        assert!(r > 0.0 && r < 10.0, "radius should be reasonable: {r}");
    }

    #[test]
    fn hyperboloid_capillary_radius() {
        let cap = HyperboloidCapillary::new(500.0, 5.0, 50.0, 100.0);
        let r = cap.local_r(0.0, 0.0);
        assert!(r > 0.0, "radius should be positive: {r}");
    }

    #[test]
    fn round_trip_paraboloid() {
        let cap = ParaboloidCapillary::new(100.0, 0.5);
        let r = cap.local_r(10.0, 0.3);
        let (x, y, z) = cap.param_to_xyz(10.0, 0.3, r);
        let (s2, phi2, r2) = cap.xyz_to_param(x, y, z);
        assert!((s2 - 10.0).abs() < 1e-10);
        assert!((phi2 - 0.3).abs() < 1e-10);
        assert!((r2 - r).abs() < 1e-10);
    }
}
