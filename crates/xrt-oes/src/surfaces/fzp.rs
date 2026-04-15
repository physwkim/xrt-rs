//! Fresnel Zone Plate (FZP) surface.
//!
//! A flat surface with a radially-varying grating vector.
//! Zone radii: r_n = sqrt(n · f · λ)
//! Local grating density: ρ(r) = r / (f · λ)
//! Grating vector: local_g → -pos/|pos| × ρ(r)
//!
//! Ported from oes.py FZP surface logic.

use crate::surface::Surface;

/// Fresnel Zone Plate surface.
#[derive(Debug, Clone, Copy)]
pub struct FzpSurface {
    /// Focal length [mm]
    pub focus: f64,
    /// Design wavelength [Å]
    pub wavelength: f64,
    /// Number of zones (for aperture limit)
    pub n_zones: usize,
}

impl FzpSurface {
    pub fn new(focus: f64, wavelength: f64, n_zones: usize) -> Self {
        Self {
            focus,
            wavelength,
            n_zones,
        }
    }

    /// Outer zone radius [mm].
    pub fn outer_radius(&self) -> f64 {
        // r_n = sqrt(n * f * λ), with f in mm and λ in Å → convert λ to mm
        let lambda_mm = self.wavelength * 1e-7; // Å → mm
        (self.n_zones as f64 * self.focus * lambda_mm).sqrt()
    }

    /// Local grating density ρ(r) [lines/mm] at radial distance r [mm].
    fn local_rho(&self, r: f64) -> f64 {
        let lambda_mm = self.wavelength * 1e-7;
        r / (self.focus * lambda_mm)
    }
}

impl Surface for FzpSurface {
    fn local_z(&self, _x: f64, _y: f64) -> f64 {
        // FZP is a flat surface
        0.0
    }

    fn local_n(&self, _x: f64, _y: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }

    fn local_g(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        let r = (x * x + y * y).sqrt();
        if r < 1e-30 {
            return Some([0.0, 0.0, 0.0]);
        }

        let rho = self.local_rho(r);

        // Grating vector points radially inward: -pos/|pos| × ρ
        Some([-x / r * rho, -y / r * rho, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fzp_flat_z() {
        let s = FzpSurface::new(100.0, 1.0, 1000);
        assert_eq!(s.local_z(5.0, 3.0), 0.0);
    }

    #[test]
    fn fzp_flat_normal() {
        let s = FzpSurface::new(100.0, 1.0, 1000);
        let n = s.local_n(5.0, 3.0);
        assert!((n[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn fzp_g_at_origin() {
        let s = FzpSurface::new(100.0, 1.0, 1000);
        let g = s.local_g(0.0, 0.0).unwrap();
        assert!(g[0].abs() < 1e-15);
        assert!(g[1].abs() < 1e-15);
    }

    #[test]
    fn fzp_g_radial() {
        let s = FzpSurface::new(100.0, 1.0, 1000);
        // At (1.0, 0.0): grating vector should point in -x direction
        let g = s.local_g(1.0, 0.0).unwrap();
        assert!(g[0] < 0.0, "gx = {}", g[0]);
        assert!(g[1].abs() < 1e-15);
    }

    #[test]
    fn fzp_outer_radius() {
        let s = FzpSurface::new(100.0, 1.0, 1000);
        let r = s.outer_radius();
        // r = sqrt(1000 * 100 * 1e-7) = sqrt(0.01) = 0.1 mm
        assert!((r - 0.1).abs() < 1e-10, "outer radius = {r}, expected 0.1");
    }

    #[test]
    fn fzp_g_magnitude_increases_with_r() {
        let s = FzpSurface::new(100.0, 1.0, 1000);
        let g1 = s.local_g(0.5, 0.0).unwrap();
        let g2 = s.local_g(1.0, 0.0).unwrap();
        let mag1 = (g1[0] * g1[0] + g1[1] * g1[1]).sqrt();
        let mag2 = (g2[0] * g2[0] + g2[1] * g2[1]).sqrt();
        assert!(
            mag2 > mag1,
            "mag at r=1 ({mag2}) should > mag at r=0.5 ({mag1})"
        );
    }

    #[test]
    fn fzp_local_g_at_various_positions() {
        let fzp = FzpSurface::new(1000.0, 0.01, 1000);
        // At different radii, grating vector should point radially
        for r in [0.1, 0.5, 1.0, 2.0] {
            let g = fzp.local_g(r, 0.0);
            if let Some([gx, gy, gz]) = g {
                assert!(
                    gx.is_finite() && gy.is_finite() && gz.is_finite(),
                    "FZP g at r={r} should be finite"
                );
                // Grating vector should have radial component
                let g_mag = (gx * gx + gy * gy + gz * gz).sqrt();
                assert!(g_mag > 0.0, "FZP g magnitude should be > 0 at r={r}");
            }
        }
    }
}
