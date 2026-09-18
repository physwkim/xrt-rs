//! Johann and Johansson cylindrically bent crystal surfaces.
//!
//! Johann: crystal bent to radius R, diffracting planes at radius R.
//! Johansson: crystal surface ground to R, planes at 2R (exact focusing).

use crate::oes::surface::{CrossSection, Surface};

/// Johann cylindrically bent crystal.
#[derive(Debug, Clone, Copy)]
pub struct JohannCylinderSurface {
    /// Bending radius R [mm]
    pub rm: f64,
    /// Cross-section type
    pub cross_section: CrossSection,
    /// Asymmetry angle [rad] (0 = symmetric)
    pub alpha: f64,
}

impl JohannCylinderSurface {
    pub fn new(rm: f64, cross_section: CrossSection, alpha: f64) -> Self {
        Self {
            rm,
            cross_section,
            alpha,
        }
    }
}

impl Surface for JohannCylinderSurface {
    fn local_z(&self, _x: f64, y: f64) -> f64 {
        match self.cross_section {
            CrossSection::Circular => {
                let r2 = self.rm * self.rm;
                if y * y < r2 {
                    self.rm - (r2 - y * y).sqrt()
                } else {
                    self.rm
                }
            }
            CrossSection::Parabolic => y * y / (2.0 * self.rm),
        }
    }

    fn local_n(&self, _x: f64, y: f64) -> [f64; 3] {
        match self.cross_section {
            CrossSection::Circular => {
                let r2 = self.rm * self.rm;
                if y * y < r2 {
                    let sq = (r2 - y * y).sqrt();
                    let ny = -y / self.rm;
                    let nz = sq / self.rm;
                    let norm = (ny * ny + nz * nz).sqrt();
                    [0.0, ny / norm, nz / norm]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
            CrossSection::Parabolic => {
                let dz_dy = y / self.rm;
                let ny = -dz_dy;
                let nz = 1.0;
                let norm = (ny * ny + nz * nz).sqrt();
                [0.0, ny / norm, nz / norm]
            }
        }
    }

    fn local_n_bragg(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        // For Johann geometry, Bragg planes follow the bending
        // but the asymmetry angle rotates the plane normal
        if self.alpha.abs() < 1e-15 {
            return None; // symmetric: bragg == surface
        }
        // Rotate surface normal by -alpha around x-axis
        let [_, ny, nz] = self.local_n(_x, y);
        let (sin_a, cos_a) = self.alpha.sin_cos();
        let ny_b = ny * cos_a - nz * sin_a;
        let nz_b = ny * sin_a + nz * cos_a;
        let norm = (ny_b * ny_b + nz_b * nz_b).sqrt();
        Some([0.0, ny_b / norm, nz_b / norm])
    }
}

/// Johansson (ground-bent) cylindrically bent crystal.
///
/// Surface is ground to radius Rm (like Johann), but Bragg planes
/// have effective radius 2Rm, providing exact Rowland circle focusing.
#[derive(Debug, Clone, Copy)]
pub struct JohanssonCylinderSurface {
    /// Bending radius R [mm]
    pub rm: f64,
    /// Cross-section type
    pub cross_section: CrossSection,
    /// Asymmetry angle [rad]
    pub alpha: f64,
}

impl JohanssonCylinderSurface {
    pub fn new(rm: f64, cross_section: CrossSection, alpha: f64) -> Self {
        Self {
            rm,
            cross_section,
            alpha,
        }
    }
}

impl Surface for JohanssonCylinderSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        // Same surface shape as Johann
        JohannCylinderSurface::new(self.rm, self.cross_section, self.alpha).local_z(x, y)
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        // Same surface normal as Johann
        JohannCylinderSurface::new(self.rm, self.cross_section, self.alpha).local_n(x, y)
    }

    fn local_n_bragg(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        // Johansson: Bragg normal uses the exact ground-bent formula
        // bragg_n = normalize(0, -y, sqrt(Rm²-y²) + Rm)
        let r2 = self.rm * self.rm;
        let sq = if y * y < r2 { (r2 - y * y).sqrt() } else { 0.0 };
        let ny_b = -y;
        let nz_b = sq + self.rm;
        let norm = (ny_b * ny_b + nz_b * nz_b).sqrt();
        let (mut ny_b, mut nz_b) = (ny_b / norm, nz_b / norm);

        // Apply asymmetry rotation if nonzero
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn johann_at_origin() {
        let s = JohannCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        assert!(s.local_z(0.0, 0.0).abs() < 1e-15);
        let n = s.local_n(0.0, 0.0);
        assert!((n[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn johann_parabolic_approx() {
        let r = 5000.0;
        let circ = JohannCylinderSurface::new(r, CrossSection::Circular, 0.0);
        let para = JohannCylinderSurface::new(r, CrossSection::Parabolic, 0.0);
        let y = 10.0; // small compared to R
        let diff = (circ.local_z(0.0, y) - para.local_z(0.0, y)).abs();
        assert!(
            diff < 1e-6,
            "parabolic should approximate circular: diff={diff}"
        );
    }

    #[test]
    fn johann_symmetric_has_no_bragg() {
        let s = JohannCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        assert!(s.local_n_bragg(0.0, 10.0).is_none());
    }

    #[test]
    fn johann_asymmetric_has_bragg() {
        let s = JohannCylinderSurface::new(1000.0, CrossSection::Circular, 0.01);
        assert!(s.local_n_bragg(0.0, 10.0).is_some());
    }

    #[test]
    fn johansson_has_bragg_normal() {
        let s = JohanssonCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        let bragg = s.local_n_bragg(0.0, 10.0);
        assert!(bragg.is_some());
        let [nx, ny, nz] = bragg.unwrap();
        assert!(nx.abs() < 1e-15);
        let norm = (ny * ny + nz * nz).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn johansson_surface_matches_johann() {
        let jo = JohannCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        let js = JohanssonCylinderSurface::new(1000.0, CrossSection::Circular, 0.0);
        let z1 = jo.local_z(0.0, 50.0);
        let z2 = js.local_z(0.0, 50.0);
        assert!((z1 - z2).abs() < 1e-12, "surface shape should be identical");
    }
}
