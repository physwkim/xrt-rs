//! Bent Laue crystal surfaces.
//!
//! Cylindrically or spherically bent crystals in Laue (transmission) geometry.
//! The Bragg plane normal is perpendicular to the surface normal (rotated 90°).

use crate::surface::{CrossSection, Surface};

/// Cylindrically bent Laue crystal.
#[derive(Debug, Clone, Copy)]
pub struct BentLaueCylinderSurface {
    /// Bending radius R [mm]
    pub r: f64,
    /// Cross-section type
    pub cross_section: CrossSection,
    /// Asymmetry angle [rad] (0 = symmetric Laue)
    pub alpha: f64,
}

impl BentLaueCylinderSurface {
    pub fn new(r: f64, cross_section: CrossSection, alpha: f64) -> Self {
        Self { r, cross_section, alpha }
    }
}

impl Surface for BentLaueCylinderSurface {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        match self.cross_section {
            CrossSection::Circular => {
                let r2 = self.r * self.r;
                if y * y < r2 {
                    self.r - (r2 - y * y).sqrt()
                } else {
                    self.r
                }
            }
            CrossSection::Parabolic => y * y / (2.0 * self.r),
        }
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        match self.cross_section {
            CrossSection::Circular => {
                let r2 = self.r * self.r;
                if y * y < r2 {
                    let sq = (r2 - y * y).sqrt();
                    let ny = -y / self.r;
                    let nz = sq / self.r;
                    [0.0, ny, nz]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
            CrossSection::Parabolic => {
                let dz_dy = y / self.r;
                let ny = -dz_dy;
                let nz = 1.0;
                let norm = (ny * ny + nz * nz).sqrt();
                [0.0, ny / norm, nz / norm]
            }
        }
    }

    fn local_n_bragg(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        // Laue: Bragg normal is ~90° rotated from surface normal
        let [_, ny, nz] = self.local_n(_x, y);
        // Base Laue rotation: swap ny<->nz with sign
        let (mut ny_b, mut nz_b) = (nz, -ny);
        // Apply asymmetry angle
        if self.alpha.abs() > 1e-15 {
            let (sin_a, cos_a) = self.alpha.sin_cos();
            let ny_r = ny_b * cos_a - nz_b * sin_a;
            let nz_r = ny_b * sin_a + nz_b * cos_a;
            ny_b = ny_r;
            nz_b = nz_r;
        }
        let norm = (ny_b * ny_b + nz_b * nz_b).sqrt();
        Some([0.0, ny_b / norm, nz_b / norm])
    }
}

/// Spherically bent Laue crystal (2D bending).
#[derive(Debug, Clone, Copy)]
pub struct BentLaue2DSurface {
    /// Meridional bending radius [mm]
    pub rm: f64,
    /// Sagittal bending radius [mm]
    pub rs: f64,
}

impl BentLaue2DSurface {
    pub fn new(rm: f64, rs: f64) -> Self {
        Self { rm, rs }
    }
}

impl Surface for BentLaue2DSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        0.5 * x * x / self.rs + 0.5 * y * y / self.rm
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let dz_dx = x / self.rs;
        let dz_dy = y / self.rm;
        let nx = -dz_dx;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }

    fn local_n_bragg(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        // 2D Laue: Bragg normal swapped from surface normal
        let [nx, ny, nz] = self.local_n(x, y);
        let norm = (nz * nz + ny * ny).sqrt();
        Some([nx, nz / norm, -ny / norm])
    }
}

/// Spherically bent Laue crystal.
#[derive(Debug, Clone, Copy)]
pub struct BentLaueSphereSurface {
    /// Bending radius [mm]
    pub r: f64,
    /// Cross-section type
    pub cross_section: CrossSection,
}

impl BentLaueSphereSurface {
    pub fn new(r: f64, cross_section: CrossSection) -> Self {
        Self { r, cross_section }
    }
}

impl Surface for BentLaueSphereSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        match self.cross_section {
            CrossSection::Circular => {
                let rho2 = x * x + y * y;
                let r2 = self.r * self.r;
                if rho2 < r2 {
                    self.r - (r2 - rho2).sqrt()
                } else {
                    self.r
                }
            }
            CrossSection::Parabolic => {
                (x * x + y * y) / (2.0 * self.r)
            }
        }
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        match self.cross_section {
            CrossSection::Circular => {
                let rho2 = x * x + y * y;
                let r2 = self.r * self.r;
                if rho2 < r2 {
                    let sq = (r2 - rho2).sqrt();
                    let nx = -x / self.r;
                    let ny = -y / self.r;
                    let nz = sq / self.r;
                    [nx, ny, nz]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
            CrossSection::Parabolic => {
                let dz_dx = x / self.r;
                let dz_dy = y / self.r;
                let nx = -dz_dx;
                let ny = -dz_dy;
                let nz = 1.0;
                let norm = (nx * nx + ny * ny + nz * nz).sqrt();
                [nx / norm, ny / norm, nz / norm]
            }
        }
    }

    fn local_n_bragg(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        let [_, ny, nz] = self.local_n(0.0, y); // meridional component only
        Some([0.0, nz, -ny]) // 90° rotation
    }
}

/// Ground-bent Laue cylinder (Johansson-type for Laue).
#[derive(Debug, Clone, Copy)]
pub struct GroundBentLaueCylinderSurface {
    pub r: f64,
    pub cross_section: CrossSection,
    pub alpha: f64,
}

impl GroundBentLaueCylinderSurface {
    pub fn new(r: f64, cross_section: CrossSection, alpha: f64) -> Self {
        Self { r, cross_section, alpha }
    }
}

impl Surface for GroundBentLaueCylinderSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        BentLaueCylinderSurface::new(self.r, self.cross_section, 0.0).local_z(x, y)
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        BentLaueCylinderSurface::new(self.r, self.cross_section, 0.0).local_n(x, y)
    }

    fn local_n_bragg(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        // Ground-bent Laue: Johansson formula then 90° rotation
        let r2 = self.r * self.r;
        let sq = if y * y < r2 { (r2 - y * y).sqrt() } else { 0.0 };
        let ny_j = -y;
        let nz_j = sq + self.r;
        // 90° rotation for Laue
        let (mut ny_b, mut nz_b) = (nz_j, -ny_j);
        // Apply asymmetry
        if self.alpha.abs() > 1e-15 {
            let (sin_a, cos_a) = self.alpha.sin_cos();
            let ny_r = ny_b * cos_a - nz_b * sin_a;
            let nz_r = ny_b * sin_a + nz_b * cos_a;
            ny_b = ny_r;
            nz_b = nz_r;
        }
        let norm = (ny_b * ny_b + nz_b * nz_b).sqrt();
        Some([0.0, ny_b / norm, nz_b / norm])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bent_laue_cylinder_at_origin() {
        let s = BentLaueCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        assert!(s.local_z(0.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn bent_laue_has_bragg_normal() {
        let s = BentLaueCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        let bragg = s.local_n_bragg(0.0, 10.0);
        assert!(bragg.is_some());
        // Bragg should be ~perpendicular to surface normal
        let [_, ny_s, nz_s] = s.local_n(0.0, 10.0);
        let [_, ny_b, nz_b] = bragg.unwrap();
        let dot = ny_s * ny_b + nz_s * nz_b;
        assert!(dot.abs() < 0.1, "Bragg should be ~perpendicular to surface: dot={dot}");
    }

    #[test]
    fn bent_laue_2d_symmetric() {
        let s = BentLaue2DSurface::new(5000.0, 500.0);
        let z1 = s.local_z(1.0, 10.0);
        let z2 = s.local_z(-1.0, 10.0);
        assert!((z1 - z2).abs() < 1e-12);
    }

    #[test]
    fn bent_laue_sphere_at_origin() {
        let s = BentLaueSphereSurface::new(1000.0, CrossSection::Circular);
        assert!(s.local_z(0.0, 0.0).abs() < 1e-15);
        let n = s.local_n(0.0, 0.0);
        assert!((n[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ground_bent_laue_has_bragg() {
        let s = GroundBentLaueCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        assert!(s.local_n_bragg(0.0, 10.0).is_some());
    }
}
