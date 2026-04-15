//! Vertically collimating mirror (VCM) and mechanical support wrappers.
//!
//! VCM is functionally identical to BentFlatSurface for ray tracing.
//! The mechanical support classes (MirrorOnTripod, DCMOnTripod) are
//! parameterization helpers that compute pitch/roll from actuator positions.

use super::bent_flat::BentFlatSurface;

/// Vertically collimating mirror.
///
/// Functionally identical to BentFlatSurface. The name distinguishes
/// it from VFM (vertically focusing mirror) in beamline context.
pub type VcmSurface = BentFlatSurface;

/// Mirror on tripod with two x-stages.
///
/// This is a mechanical parameterization wrapper: given actuator positions,
/// it computes the effective pitch, roll, and yaw for a flat/bent mirror.
/// For ray tracing, the underlying surface is what matters.
#[derive(Debug, Clone, Copy)]
pub struct TripodParams {
    /// Jack 1 position [mm]
    pub jack1: f64,
    /// Jack 2 position [mm]
    pub jack2: f64,
    /// Jack 3 position [mm]
    pub jack3: f64,
    /// Distance between jacks along beam [mm]
    pub jack_sep_y: f64,
    /// Distance between jacks across beam [mm]
    pub jack_sep_x: f64,
}

impl TripodParams {
    pub fn new(jack1: f64, jack2: f64, jack3: f64, jack_sep_y: f64, jack_sep_x: f64) -> Self {
        Self {
            jack1,
            jack2,
            jack3,
            jack_sep_y,
            jack_sep_x,
        }
    }

    /// Compute pitch angle from jack positions [rad]
    pub fn pitch(&self) -> f64 {
        ((self.jack2 - self.jack1) / self.jack_sep_y).atan()
    }

    /// Compute roll angle from jack positions [rad]
    pub fn roll(&self) -> f64 {
        let mean_12 = (self.jack1 + self.jack2) / 2.0;
        ((self.jack3 - mean_12) / self.jack_sep_x).atan()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::Surface;

    #[test]
    fn vcm_is_bent_flat() {
        let vcm = VcmSurface::new(10000.0, 0.0);
        assert!(vcm.local_z(0.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn tripod_zero_pitch() {
        let t = TripodParams::new(0.0, 0.0, 0.0, 1000.0, 50.0);
        assert!(t.pitch().abs() < 1e-15);
        assert!(t.roll().abs() < 1e-15);
    }

    #[test]
    fn tripod_small_pitch() {
        let t = TripodParams::new(0.0, 1.0, 0.0, 1000.0, 50.0);
        assert!((t.pitch() - 0.001).abs() < 1e-6);
    }
}
