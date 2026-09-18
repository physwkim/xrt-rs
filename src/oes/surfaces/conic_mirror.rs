//! Cartesian conic section mirrors (elliptical, hyperbolic, parabolic).
//!
//! These use the Cartesian z=f(x,y) representation, unlike the parametric
//! versions in elliptical.rs/hyperbolic.rs which use r=f(s,φ).

use crate::oes::surface::Surface;

/// Elliptical mirror in Cartesian coordinates.
///
/// z = p*y²/(4*q²) where p,q are source/image distances at grazing angle.
/// Uses the cylindrical approximation for meridional focusing.
#[derive(Debug, Clone, Copy)]
pub struct EllipticalMirrorSurface {
    /// Source distance p [mm]
    pub p: f64,
    /// Image distance q [mm]
    pub q: f64,
    /// Grazing angle θ [rad]
    pub theta: f64,
}

impl EllipticalMirrorSurface {
    pub fn new(p: f64, q: f64, theta: f64) -> Self {
        Self { p, q, theta }
    }

    fn curvature_meridional(&self) -> f64 {
        // R_meridional = 2pq sin(θ) / (p+q)
        2.0 * self.p * self.q * self.theta.sin() / (self.p + self.q)
    }
}

impl Surface for EllipticalMirrorSurface {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        let r = self.curvature_meridional();
        y * y / (2.0 * r)
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        let r = self.curvature_meridional();
        let dz_dy = y / r;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (ny * ny + nz * nz).sqrt();
        [0.0, ny / norm, nz / norm]
    }
}

/// Hyperbolic mirror in Cartesian coordinates.
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicMirrorSurface {
    pub p: f64,
    pub q: f64,
    pub theta: f64,
}

impl HyperbolicMirrorSurface {
    pub fn new(p: f64, q: f64, theta: f64) -> Self {
        Self { p, q, theta }
    }

    fn curvature_meridional(&self) -> f64 {
        // For hyperbolic (virtual focus): R = 2|p*q| sin(θ) / |p-q|
        let denom = (self.p - self.q).abs();
        if denom < 1e-30 {
            return 1e30;
        }
        2.0 * (self.p * self.q).abs() * self.theta.sin() / denom
    }
}

impl Surface for HyperbolicMirrorSurface {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        let r = self.curvature_meridional();
        y * y / (2.0 * r)
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        let r = self.curvature_meridional();
        let dz_dy = y / r;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (ny * ny + nz * nz).sqrt();
        [0.0, ny / norm, nz / norm]
    }
}

/// Parabolic mirror in Cartesian coordinates.
#[derive(Debug, Clone, Copy)]
pub struct ParabolicMirrorSurface {
    pub p: f64,
    pub theta: f64,
}

impl ParabolicMirrorSurface {
    pub fn new(p: f64, theta: f64) -> Self {
        Self { p, theta }
    }
}

impl Surface for ParabolicMirrorSurface {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        let r = 2.0 * self.p * self.theta.sin();
        y * y / (2.0 * r)
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        let r = 2.0 * self.p * self.theta.sin();
        let dz_dy = y / r;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (ny * ny + nz * nz).sqrt();
        [0.0, ny / norm, nz / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elliptical_mirror_at_origin() {
        let m = EllipticalMirrorSurface::new(5000.0, 1000.0, 0.003);
        assert!(m.local_z(0.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn hyperbolic_mirror_at_origin() {
        let m = HyperbolicMirrorSurface::new(5000.0, 1000.0, 0.003);
        assert!(m.local_z(0.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn parabolic_mirror_at_origin() {
        let m = ParabolicMirrorSurface::new(5000.0, 0.003);
        assert!(m.local_z(0.0, 0.0).abs() < 1e-15);
    }
}
