//! General Fresnel Zone Plate in the 0YZ plane.

use crate::surface::Surface;

/// General FZP oriented in the 0YZ plane.
///
/// Unlike NormalFZP which is oriented in XY, this FZP has zones
/// defined in the YZ plane with the optical axis along X.
#[derive(Debug, Clone, Copy)]
pub struct GeneralFzpSurface {
    /// Focal length [mm]
    pub focus: f64,
    /// Zone count (determines outer radius)
    pub n_zones: usize,
    /// Wavelength [mm]
    pub wavelength: f64,
}

impl GeneralFzpSurface {
    pub fn new(focus: f64, wavelength: f64, n_zones: usize) -> Self {
        Self {
            focus,
            wavelength,
            n_zones,
        }
    }

    /// Outer radius of the FZP [mm]
    pub fn outer_radius(&self) -> f64 {
        (self.n_zones as f64 * self.wavelength * self.focus).sqrt()
    }

    /// Local zone density (lines/mm) at radius rho
    fn local_rho(&self, rho: f64) -> f64 {
        if rho < 1e-30 {
            return 0.0;
        }
        rho / (self.wavelength * self.focus)
    }
}

impl Surface for GeneralFzpSurface {
    fn local_z(&self, _x: f64, _y: f64) -> f64 {
        0.0 // FZP is flat
    }

    fn local_n(&self, _x: f64, _y: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }

    fn local_g(&self, _x: f64, y: f64) -> Option<[f64; 3]> {
        // Grating vector is radial in YZ, pointing along Y
        let rho = y.abs();
        let dens = self.local_rho(rho);
        if dens < 1e-30 {
            return None;
        }
        let sign = if y >= 0.0 { -1.0 } else { 1.0 };
        Some([0.0, sign * dens, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_fzp_flat() {
        let f = GeneralFzpSurface::new(1000.0, 1e-6, 1000);
        assert_eq!(f.local_z(5.0, 10.0), 0.0);
    }

    #[test]
    fn general_fzp_grating_vector() {
        let f = GeneralFzpSurface::new(1000.0, 1e-6, 1000);
        let g = f.local_g(0.0, 1.0);
        assert!(g.is_some());
        let gv = g.unwrap();
        assert!(gv[1].abs() > 0.0, "should have nonzero gy");
    }
}
