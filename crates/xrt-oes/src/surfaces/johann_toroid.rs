//! Johann and Johansson toroidally bent crystal surfaces.
//!
//! Adds sagittal curvature to the cylindrical Johann/Johansson geometry.

use crate::surface::{CrossSection, Surface};

/// Johann toroidally bent crystal.
#[derive(Debug, Clone, Copy)]
pub struct JohannToroidSurface {
    /// Meridional bending radius [mm]
    pub rm: f64,
    /// Sagittal bending radius [mm]
    pub rs: f64,
    /// Cross-section type
    pub cross_section: CrossSection,
    /// Asymmetry angle [rad]
    pub alpha: f64,
}

impl JohannToroidSurface {
    pub fn new(rm: f64, rs: f64, cross_section: CrossSection, alpha: f64) -> Self {
        Self { rm, rs, cross_section, alpha }
    }
}

impl Surface for JohannToroidSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        // Meridional bending
        let z_mer = match self.cross_section {
            CrossSection::Circular => {
                let r2 = self.rm * self.rm;
                if y * y < r2 { self.rm - (r2 - y * y).sqrt() } else { self.rm }
            }
            CrossSection::Parabolic => y * y / (2.0 * self.rm),
        };
        // Sagittal curvature: r(y) = Rs - z_mer
        let r_local = self.rs - z_mer;
        let z_sag = if r_local > 0.0 && x * x < r_local * r_local {
            r_local * (1.0 - (1.0 - x * x / (r_local * r_local)).sqrt())
        } else {
            0.0
        };
        z_mer + z_sag
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        // Numerical gradient for the toroid
        let eps = 1e-7;
        let dz_dx = (self.local_z(x + eps, y) - self.local_z(x - eps, y)) / (2.0 * eps);
        let dz_dy = (self.local_z(x, y + eps) - self.local_z(x, y - eps)) / (2.0 * eps);
        let nx = -dz_dx;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }

    fn local_n_bragg(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        if self.alpha.abs() < 1e-15 {
            return None; // symmetric
        }
        let [nx, ny, nz] = self.local_n(x, y);
        let (sin_a, cos_a) = self.alpha.sin_cos();
        let ny_b = ny * cos_a - nz * sin_a;
        let nz_b = ny * sin_a + nz * cos_a;
        let norm = (nx * nx + ny_b * ny_b + nz_b * nz_b).sqrt();
        Some([nx / norm, ny_b / norm, nz_b / norm])
    }
}

/// Johansson toroidally bent crystal.
#[derive(Debug, Clone, Copy)]
pub struct JohanssonToroidSurface {
    pub rm: f64,
    pub rs: f64,
    pub cross_section: CrossSection,
    pub alpha: f64,
}

impl JohanssonToroidSurface {
    pub fn new(rm: f64, rs: f64, cross_section: CrossSection, alpha: f64) -> Self {
        Self { rm, rs, cross_section, alpha }
    }
}

impl Surface for JohanssonToroidSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        JohannToroidSurface::new(self.rm, self.rs, self.cross_section, self.alpha)
            .local_z(x, y)
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        JohannToroidSurface::new(self.rm, self.rs, self.cross_section, self.alpha)
            .local_n(x, y)
    }

    fn local_n_bragg(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        // Johansson: ground-bent formula for Bragg normal
        let r2 = self.rm * self.rm;
        let sq = if y * y < r2 { (r2 - y * y).sqrt() } else { 0.0 };
        let ny_b = -y;
        let nz_b = sq + self.rm;
        let norm = (ny_b * ny_b + nz_b * nz_b).sqrt();
        let (mut ny_b, mut nz_b) = (ny_b / norm, nz_b / norm);
        if self.alpha.abs() > 1e-15 {
            let (sin_a, cos_a) = self.alpha.sin_cos();
            let ny_r = ny_b * cos_a - nz_b * sin_a;
            let nz_r = ny_b * sin_a + nz_b * cos_a;
            ny_b = ny_r;
            nz_b = nz_r;
        }
        Some([0.0, ny_b, nz_b])
    }
}

/// General Bragg toroid with independent surface and Bragg radii.
#[derive(Debug, Clone, Copy)]
pub struct GeneralBraggToroidSurface {
    /// Surface meridional radius [mm]
    pub rm: f64,
    /// Surface sagittal radius [mm]
    pub rs: f64,
    /// Bragg plane meridional radius [mm]
    pub rm_bragg: f64,
    /// Bragg plane sagittal radius [mm]
    pub rs_bragg: f64,
    pub cross_section: CrossSection,
}

impl GeneralBraggToroidSurface {
    pub fn new(rm: f64, rs: f64, rm_bragg: f64, rs_bragg: f64, cross_section: CrossSection) -> Self {
        Self { rm, rs, rm_bragg, rs_bragg, cross_section }
    }
}

impl Surface for GeneralBraggToroidSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        JohannToroidSurface::new(self.rm, self.rs, self.cross_section, 0.0)
            .local_z(x, y)
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        JohannToroidSurface::new(self.rm, self.rs, self.cross_section, 0.0)
            .local_n(x, y)
    }

    fn local_n_bragg(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        // Use Bragg radii for the Bragg plane normal
        let bragg_surf = JohannToroidSurface::new(
            self.rm_bragg, self.rs_bragg, self.cross_section, 0.0
        );
        Some(bragg_surf.local_n(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn johann_toroid_at_origin() {
        let s = JohannToroidSurface::new(5000.0, 500.0, CrossSection::Circular, 0.0);
        assert!(s.local_z(0.0, 0.0).abs() < 1e-10);
    }

    #[test]
    fn johansson_toroid_has_bragg() {
        let s = JohanssonToroidSurface::new(5000.0, 500.0, CrossSection::Circular, 0.0);
        assert!(s.local_n_bragg(0.0, 10.0).is_some());
    }

    #[test]
    fn general_bragg_toroid_different_radii() {
        let s = GeneralBraggToroidSurface::new(
            5000.0, 500.0,   // surface
            4000.0, 400.0,   // Bragg
            CrossSection::Parabolic,
        );
        let n_surf = s.local_n(0.0, 10.0);
        let n_bragg = s.local_n_bragg(0.0, 10.0).unwrap();
        // Different radii -> different normals
        let diff = (n_surf[1] - n_bragg[1]).abs();
        assert!(diff > 1e-6, "surface and Bragg normals should differ");
    }
}
