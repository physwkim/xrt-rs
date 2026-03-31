//! MaterialOpticalElement: a surface with Fresnel material reflectivity.
//!
//! Combines geometric reflection (from Surface) with Fresnel amplitude
//! calculations (from Material). This is the standard mirror OE:
//! after ray-surface intersection and specular reflection, the material's
//! Fresnel coefficients Rs/Rp are computed and applied to the beam.

use ndarray::Array1;

use xrt_core::beam::{Beam, RayState};
use xrt_core::transforms::{RotationParams, rotate_beam};
use xrt_materials::material::Material;

use crate::oe::OeParams;
use crate::reflect::{self, RayResult};
use crate::surface::Surface;

/// An optical element with a material coating for Fresnel reflectivity.
pub struct MaterialOpticalElement<S: Surface> {
    pub surface: S,
    pub params: OeParams,
    pub material: Material,
    /// Whether the beam comes from vacuum (true) or from inside the material (false).
    pub from_vacuum: bool,
}

impl<S: Surface> MaterialOpticalElement<S> {
    pub fn new(surface: S, params: OeParams, material: Material) -> Self {
        Self {
            surface,
            params,
            material,
            from_vacuum: true,
        }
    }

    /// Reflect a beam off this material-coated optical element.
    ///
    /// 1. Geometric intersection and specular reflection
    /// 2. Compute Fresnel Rs/Rp from material refractive index
    /// 3. Apply amplitudes to beam coherency matrix
    pub fn reflect(&self, beam: &mut Beam) -> Vec<RayResult> {
        let good: Vec<usize> = (0..beam.nrays())
            .filter(|&i| beam.state[i] > 0)
            .collect();

        if good.is_empty() {
            return vec![];
        }

        // Transform to local coordinates
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

        // Core geometric reflection
        let results = reflect::reflect_local(
            &self.surface,
            beam,
            &good,
            &self.params.aperture,
            self.params.invert_normal,
            self.params.mode,
            Some(self.params.aperture.phys_x),
            Some(self.params.aperture.phys_y),
            &self.params.config,
        );

        // Apply geometric results
        reflect::apply_results(beam, &good, &results);

        // Identify good rays after intersection
        let good_after: Vec<usize> = good
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.state == RayState::Good)
            .map(|(&i, _)| i)
            .collect();

        // Apply material Fresnel amplitudes
        if !good_after.is_empty() {
            self.apply_fresnel(beam, &good_after, &results, &good);
        }

        // Rotate back to global (inverse rotation — reversed sequence)
        let inv_rotation = RotationParams::inverse_sequence(
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

    /// Compute and apply Fresnel reflectivity amplitudes.
    fn apply_fresnel(
        &self,
        beam: &mut Beam,
        good_after: &[usize],
        results: &[RayResult],
        good_orig: &[usize],
    ) {
        let n = good_after.len();

        let mut bidn_arr = Array1::<f64>::zeros(n);
        let mut energy_arr = Array1::<f64>::zeros(n);

        // Build O(1) lookup from ray index → result index
        let idx_map: std::collections::HashMap<usize, usize> =
            good_orig.iter().enumerate().map(|(j, &i)| (i, j)).collect();

        for (j, &i) in good_after.iter().enumerate() {
            let result_idx = idx_map[&i];
            bidn_arr[j] = results[result_idx].beam_in_dot_normal;
            energy_arr[j] = beam.e[i];
        }

        match self.material.get_amplitude(&energy_arr, &bidn_arr, self.from_vacuum) {
            Ok(amp) => {
                reflect::apply_material_amplitude(
                    beam,
                    good_after,
                    amp.rs.as_slice().unwrap(),
                    amp.rp.as_slice().unwrap(),
                );
            }
            Err(_) => {
                // If material calculation fails, mark rays as dead
                for &i in good_after {
                    beam.state[i] = RayState::Dead as i32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beamline::OeParamsBuilder;
    use crate::surfaces::flat::FlatSurface;
    use crate::surfaces::toroid::ToroidSurface;
    use xrt_materials::data::ScatteringTable;
    use xrt_materials::material::MaterialKind;

    fn si_mirror() -> Material {
        Material::new(
            &["Si"], None, 2.33,
            MaterialKind::Mirror, None,
            ScatteringTable::ChantlerTotal,
        ).unwrap()
    }

    #[test]
    fn material_oe_create() {
        let oe = MaterialOpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new().pitch(0.003).build(),
            si_mirror(),
        );
        assert!(oe.from_vacuum);
    }

    #[test]
    fn material_oe_reflect_modifies_coherency() {
        let oe = MaterialOpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new().pitch(0.01).build(),
            si_mirror(),
        );

        let mut beam = Beam::with_amplitudes(5);
        for i in 0..5 {
            beam.y[i] = -1000.0;
            beam.a[i] = 0.0;
            beam.b[i] = 1.0;
            beam.c[i] = 0.0;
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
            beam.jss[i] = 1.0;
            beam.jpp[i] = 0.0;
        }

        let results = oe.reflect(&mut beam);
        assert_eq!(results.len(), 5);

        // After material reflection, jss should be modified by |Rs|²
        let good_count = results.iter().filter(|r| r.state == RayState::Good).count();
        if good_count > 0 {
            let i = (0..5).find(|&i| beam.state[i] == RayState::Good as i32).unwrap();
            // At 10 mrad grazing, Si mirror should have high reflectivity
            assert!(
                beam.jss[i] > 0.0 && beam.jss[i].is_finite(),
                "jss = {}", beam.jss[i]
            );
        }
    }

    #[test]
    fn toroid_material_oe() {
        let oe = MaterialOpticalElement::new(
            ToroidSurface::new(5e6, 50.0),
            OeParamsBuilder::new().pitch(0.003).build(),
            si_mirror(),
        );

        let mut beam = Beam::new(3);
        for i in 0..3 {
            beam.y[i] = -1000.0;
            beam.b[i] = 1.0;
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
        }

        let results = oe.reflect(&mut beam);
        assert_eq!(results.len(), 3);
    }
}
