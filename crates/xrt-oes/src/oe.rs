//! Optical element parameters and reflect() entry point.
//!
//! Ported from oes_base.py:1064-1181.

use xrt_core::beam::{Beam, RayState};
use xrt_core::transforms::{RotationParams, rotate_beam};
use xrt_math::rootfind::RootFindConfig;

use crate::aperture::Aperture;
use crate::reflect::{self, DeflectionMode, RayResult};
use crate::surface::Surface;

/// Parameters for an optical element.
#[derive(Debug, Clone)]
pub struct OeParams {
    /// Center position in global coordinates [mm]
    pub center: [f64; 3],
    /// Pitch angle [rad]
    pub pitch: f64,
    /// Roll angle [rad]
    pub roll: f64,
    /// Yaw angle [rad]
    pub yaw: f64,
    /// Position roll: rotation around beam axis before OE orientation [rad].
    /// Equivalent to XRT Python's `positionRoll`. Applied as Ry rotation
    /// before the standard pitch/roll/yaw transform.
    pub position_roll: f64,
    /// Aperture definition
    pub aperture: Aperture,
    /// Normal inversion (1 for vacuum→surface, -1 for surface→vacuum)
    pub invert_normal: i32,
    /// Deflection mode
    pub mode: DeflectionMode,
    /// Root-finding configuration
    pub config: RootFindConfig,
}

impl Default for OeParams {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            pitch: 0.0,
            roll: 0.0,
            yaw: 0.0,
            position_roll: 0.0,
            aperture: Aperture::default(),
            invert_normal: 1,
            mode: DeflectionMode::Reflect,
            config: RootFindConfig::default(),
        }
    }
}

impl OeParams {
    /// Apply forward position_roll rotation to beam (global → local).
    pub fn apply_position_roll_fwd(&self, beam: &mut Beam, indices: &[usize]) {
        if self.position_roll != 0.0 {
            let pr = RotationParams::default_sequence(0.0, -self.position_roll, 0.0);
            rotate_beam(beam, Some(indices), &pr, false, false);
        }
    }

    /// Apply inverse position_roll rotation to beam (local → global).
    pub fn apply_position_roll_inv(&self, beam: &mut Beam, indices: &[usize]) {
        if self.position_roll != 0.0 {
            let pr = RotationParams::default_sequence(0.0, self.position_roll, 0.0);
            rotate_beam(beam, Some(indices), &pr, false, false);
        }
    }
}

/// A complete optical element: surface + parameters.
pub struct OpticalElement<S: Surface> {
    pub surface: S,
    pub params: OeParams,
}

impl<S: Surface> OpticalElement<S> {
    pub fn new(surface: S, params: OeParams) -> Self {
        Self { surface, params }
    }

    /// Reflect a beam off this optical element.
    ///
    /// 1. Select good rays
    /// 2. Transform to local coordinates
    /// 3. Find intersection + reflect
    /// 4. Transform back to global coordinates
    ///
    /// Returns the reflected beam (modifies in place).
    pub fn reflect(&self, beam: &mut Beam) -> Vec<RayResult> {
        // Find good rays
        let good: Vec<usize> = (0..beam.nrays())
            .filter(|&i| beam.state[i] == 1)
            .collect();

        if good.is_empty() {
            return vec![];
        }

        // Transform to local coordinates (apply OE rotation)
        let rotation = RotationParams::default_sequence(
            -self.params.pitch,
            -(self.params.roll + self.params.position_roll),
            -self.params.yaw,
        );

        // Translate to OE center
        for &i in &good {
            beam.x[i] -= self.params.center[0];
            beam.y[i] -= self.params.center[1];
            beam.z[i] -= self.params.center[2];
        }

        // Rotate beam to local frame
        rotate_beam(beam, Some(&good), &rotation, false, false);

        // Core reflection
        let phys_x = Some(self.params.aperture.phys_x);
        let phys_y = Some(self.params.aperture.phys_y);

        let results = reflect::reflect_local(
            &self.surface,
            beam,
            &good,
            &self.params.aperture,
            self.params.invert_normal,
            self.params.mode,
            phys_x,
            phys_y,
            &self.params.config,
        );

        // Apply results to beam
        reflect::apply_results(beam, &good, &results);

        // Rotate back to global frame (inverse rotation — reversed sequence)
        let inv_rotation = RotationParams::inverse_sequence(
            self.params.pitch,
            self.params.roll + self.params.position_roll,
            self.params.yaw,
        );

        let good_after: Vec<usize> = good
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.state == RayState::Good || r.state == RayState::Over)
            .map(|(&i, _)| i)
            .collect();

        rotate_beam(beam, Some(&good_after), &inv_rotation, false, false);

        // Translate back
        for &i in &good_after {
            beam.x[i] += self.params.center[0];
            beam.y[i] += self.params.center[1];
            beam.z[i] += self.params.center[2];
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::flat::FlatSurface;

    #[test]
    fn optical_element_reflect() {
        let oe = OpticalElement::new(
            FlatSurface,
            OeParams {
                center: [0.0, 0.0, 0.0],
                pitch: 0.01, // 10 mrad pitch
                ..Default::default()
            },
        );

        let mut beam = Beam::new(5);
        for i in 0..5 {
            beam.y[i] = -1000.0;
            beam.z[i] = 0.0;
            beam.a[i] = 0.0;
            beam.b[i] = 1.0;
            beam.c[i] = 0.0;
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
        }

        let results = oe.reflect(&mut beam);
        assert_eq!(results.len(), 5);

        // After reflection off a pitched mirror, rays should have a c component
        let good_count = results.iter().filter(|r| r.state == RayState::Good).count();
        assert!(good_count > 0, "no good rays after reflection");
    }
}
