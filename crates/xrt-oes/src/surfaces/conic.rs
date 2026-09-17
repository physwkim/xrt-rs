//! General conic coefficient surface (shadow3 compatibility).
//!
//! Defined by 10 coefficients in the implicit equation:
//!   CCC[0]x² + CCC[1]y² + CCC[2]z² + CCC[3]xy + CCC[4]yz + CCC[5]xz
//!   + CCC[6]x + CCC[7]y + CCC[8]z + CCC[9] = 0
//!
//! This is the most general second-order surface and can represent
//! spheres, ellipsoids, hyperboloids, paraboloids, cones, and cylinders.

use crate::surface::Surface;

/// A surface defined by 10 conic coefficients (shadow3 CCC array).
#[derive(Debug, Clone, Copy)]
pub struct ConicCoefficientSurface {
    /// The 10 conic coefficients [c0..c9]
    pub ccc: [f64; 10],
}

impl ConicCoefficientSurface {
    pub fn new(ccc: [f64; 10]) -> Self {
        Self { ccc }
    }

    /// Create a sphere with radius R.
    pub fn sphere(r: f64) -> Self {
        let mut ccc = [0.0; 10];
        ccc[0] = 1.0; // x²
        ccc[1] = 1.0; // y²
        ccc[2] = 1.0; // z²
        ccc[8] = -2.0 * r; // -2Rz
        Self { ccc }
    }

    /// Create a cylinder with radius R along y-axis.
    pub fn cylinder_y(r: f64) -> Self {
        let mut ccc = [0.0; 10];
        ccc[0] = 1.0; // x²
        ccc[2] = 1.0; // z²
        ccc[8] = -2.0 * r; // -2Rz
        Self { ccc }
    }

    /// Create a plane (z = 0).
    pub fn plane() -> Self {
        let mut ccc = [0.0; 10];
        ccc[8] = 1.0; // z = 0
        Self { ccc }
    }

    /// Create an ellipsoid with semi-axes a, b, c.
    pub fn ellipsoid(a: f64, b: f64, c: f64) -> Self {
        let mut ccc = [0.0; 10];
        ccc[0] = 1.0 / (a * a);
        ccc[1] = 1.0 / (b * b);
        ccc[2] = 1.0 / (c * c);
        ccc[9] = -1.0;
        Self { ccc }
    }
}

impl Surface for ConicCoefficientSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        let c = &self.ccc;
        // Quadratic in z: c[2]*z² + (c[4]*y + c[5]*x + c[8])*z + rest = 0
        let a_coeff = c[2];
        let b_coeff = c[4] * y + c[5] * x + c[8];
        let c_coeff = c[0] * x * x + c[1] * y * y + c[3] * x * y + c[6] * x + c[7] * y + c[9];

        if a_coeff.abs() < 1e-30 {
            // Linear in z
            if b_coeff.abs() < 1e-30 {
                return 0.0;
            }
            return -c_coeff / b_coeff;
        }

        let discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
        if discriminant < 0.0 {
            return 0.0; // no real solution
        }

        let sqrt_d = discriminant.sqrt();
        let z1 = (-b_coeff + sqrt_d) / (2.0 * a_coeff);
        let z2 = (-b_coeff - sqrt_d) / (2.0 * a_coeff);

        // Return the solution closer to z = 0
        if z1.abs() < z2.abs() { z1 } else { z2 }
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let z = self.local_z(x, y);
        let c = &self.ccc;
        // Gradient of F(x,y,z) = Σ CCC terms
        let dfdx = 2.0 * c[0] * x + c[3] * y + c[5] * z + c[6];
        let dfdy = 2.0 * c[1] * y + c[3] * x + c[4] * z + c[7];
        let dfdz = 2.0 * c[2] * z + c[4] * y + c[5] * x + c[8];

        if dfdz.abs() < 1e-30 {
            return [0.0, 0.0, 1.0];
        }

        // Normal = ∇F / |∇F|, but we want outward pointing (dz positive)
        let sign = if dfdz > 0.0 { 1.0 } else { -1.0 };
        let nx = sign * dfdx;
        let ny = sign * dfdy;
        let nz = sign * dfdz;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conic_plane() {
        let s = ConicCoefficientSurface::plane();
        assert_eq!(s.local_z(5.0, 10.0), 0.0);
        assert_eq!(s.local_n(5.0, 10.0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn conic_sphere_at_origin() {
        let s = ConicCoefficientSurface::sphere(1000.0);
        assert!(s.local_z(0.0, 0.0).abs() < 1e-10);
    }

    #[test]
    fn conic_sphere_matches_spherical() {
        let r = 1000.0;
        let conic = ConicCoefficientSurface::sphere(r);
        // At x=10, y=0: z ≈ x²/(2R) = 100/2000 = 0.05
        let z = conic.local_z(10.0, 0.0);
        let expected = r - (r * r - 100.0).sqrt();
        assert!(
            (z - expected).abs() < 1e-6,
            "conic sphere z={z:.6e} vs expected={expected:.6e}"
        );
    }

    #[test]
    fn conic_cylinder() {
        let s = ConicCoefficientSurface::cylinder_y(100.0);
        // z should not depend on y
        let z1 = s.local_z(5.0, 0.0);
        let z2 = s.local_z(5.0, 100.0);
        assert!((z1 - z2).abs() < 1e-12, "cylinder should be y-independent");
    }
}
