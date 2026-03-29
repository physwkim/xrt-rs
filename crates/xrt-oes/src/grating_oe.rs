//! GratingOpticalElement: grating surface with diffraction order and optional material.
//!
//! Combines a grating surface (BlazedGrating, LaminarGrating, FZP) with
//! a diffraction order and optional material coating for amplitude calculation.

use ndarray::Array1;

use xrt_core::beam::{Beam, RayState};
use xrt_core::transforms::{RotationParams, rotate_beam};
use xrt_materials::material::Material;

use crate::oe::OeParams;
use crate::reflect::{self, DeflectionMode, RayResult};
use crate::surface::Surface;

/// A grating optical element with diffraction order.
pub struct GratingOpticalElement<S: Surface> {
    pub surface: S,
    pub params: OeParams,
    pub order: i32,
    pub material: Option<Material>,
}

impl<S: Surface> GratingOpticalElement<S> {
    /// Create a grating OE with a given diffraction order.
    pub fn new(surface: S, params: OeParams, order: i32) -> Self {
        Self {
            surface,
            params,
            order,
            material: None,
        }
    }

    /// Add a material coating for Fresnel amplitude calculation.
    pub fn with_material(mut self, material: Material) -> Self {
        self.material = Some(material);
        self
    }

    /// Reflect/diffract a beam off this grating.
    pub fn reflect(&self, beam: &mut Beam) -> Vec<RayResult> {
        let good: Vec<usize> = (0..beam.nrays())
            .filter(|&i| beam.state[i] > 0)
            .collect();

        if good.is_empty() {
            return vec![];
        }

        // Transform to local
        let rotation = RotationParams::default_sequence(
            -self.params.pitch,
            -self.params.roll,
            -self.params.yaw,
        );
        for &i in &good {
            beam.x[i] -= self.params.center[0];
            beam.y[i] -= self.params.center[1];
            beam.z[i] -= self.params.center[2];
        }
        rotate_beam(beam, Some(&good), &rotation, false, false);

        // Grating reflection with diffraction order
        let results = reflect::reflect_local(
            &self.surface,
            beam,
            &good,
            &self.params.aperture,
            self.params.invert_normal,
            DeflectionMode::Grating { order: self.order },
            Some(self.params.aperture.phys_x),
            Some(self.params.aperture.phys_y),
            &self.params.config,
        );

        reflect::apply_results(beam, &good, &results);

        // Apply material amplitude if present
        let good_after: Vec<usize> = good
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.state == RayState::Good)
            .map(|(&i, _)| i)
            .collect();

        if let Some(ref material) = self.material {
            if !good_after.is_empty() {
                let n = good_after.len();
                let mut bidn_arr = Array1::<f64>::zeros(n);
                let mut energy_arr = Array1::<f64>::zeros(n);
                for (j, &i) in good_after.iter().enumerate() {
                    let ri = good.iter().position(|&g| g == i).unwrap_or(0);
                    bidn_arr[j] = results[ri].beam_in_dot_normal;
                    energy_arr[j] = beam.e[i];
                }
                if let Ok(amp) = material.get_amplitude(&energy_arr, &bidn_arr, true) {
                    reflect::apply_material_amplitude(
                        beam,
                        &good_after,
                        amp.rs.as_slice().unwrap(),
                        amp.rp.as_slice().unwrap(),
                    );
                }
            }
        }

        // Transform back to global
        let inv_rotation = RotationParams::default_sequence(
            self.params.pitch,
            self.params.roll,
            self.params.yaw,
        );
        let alive: Vec<usize> = good
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.state == RayState::Good || r.state == RayState::Over)
            .map(|(&i, _)| i)
            .collect();
        rotate_beam(beam, Some(&alive), &inv_rotation, false, false);
        for &i in &alive {
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
    use crate::beamline::OeParamsBuilder;
    use crate::surfaces::grating::BlazedGrating;

    #[test]
    fn grating_oe_create() {
        let g = BlazedGrating::new(600.0, 0.02, 0.5);
        let oe = GratingOpticalElement::new(
            g,
            OeParamsBuilder::new().pitch(0.01).build(),
            1,
        );
        assert_eq!(oe.order, 1);
        assert!(oe.material.is_none());
    }

    #[test]
    fn grating_oe_reflect() {
        let g = BlazedGrating::new(600.0, 0.02, 0.5);
        let oe = GratingOpticalElement::new(
            g,
            OeParamsBuilder::new().pitch(0.01).build(),
            -1,
        );

        let mut beam = Beam::new(5);
        for i in 0..5 {
            beam.y[i] = -1000.0;
            beam.a[i] = 0.0;
            beam.b[i] = 1.0;
            beam.c[i] = 0.0;
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
        }

        let results = oe.reflect(&mut beam);
        assert_eq!(results.len(), 5);
    }
}
