//! DCM mechanical variants.
//!
//! DCMOnTripodWithOneXStage and DCMwithSagittalFocusing are
//! DCM configurations with specific mechanical/crystal setups.

/// DCM on tripod with one x-stage.
///
/// This is a mechanical parameterization of the standard DCM.
/// The ray tracing behavior is identical to `Beamline::add_dcm()`.
/// The tripod parameters determine the pitch angle.
#[derive(Debug, Clone, Copy)]
pub struct DcmOnTripodConfig {
    pub bragg_angle: f64,
    pub jack1: f64,
    pub jack2: f64,
    pub jack3: f64,
    pub x_stage: f64,
}

/// DCM with sagittal focusing.
///
/// Uses a sagittally bent second crystal for horizontal focusing.
/// Crystal 1: flat, Crystal 2: cylindrically bent (sagittal radius Rs).
#[derive(Debug, Clone, Copy)]
pub struct DcmSagittalConfig {
    pub bragg_angle: f64,
    pub gap: f64,
    /// Sagittal bending radius of crystal 2 [mm]
    pub rs_crystal2: f64,
}

impl DcmSagittalConfig {
    pub fn new(bragg_angle: f64, gap: f64, rs_crystal2: f64) -> Self {
        Self {
            bragg_angle,
            gap,
            rs_crystal2,
        }
    }
}
