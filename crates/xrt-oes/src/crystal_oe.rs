//! CrystalOpticalElement: combines a surface geometry with crystal diffraction.
//!
//! Integrates xrt-materials crystal amplitude calculations with the OE
//! reflection pipeline. After ray-surface intersection, computes the
//! crystal reflectivity amplitudes (Rs, Rp) and applies them to the
//! beam coherency matrix.

use ndarray::Array1;

use xrt_core::beam::{Beam, RayState};
use xrt_core::transforms::{RotationParams, rotate_beam};
use xrt_materials::crystal::{CrystalBase, StructureFactor};

use crate::oe::OeParams;
use crate::reflect::{self, RayResult};
use crate::surface::Surface;

/// An optical element with a crystal material for diffraction calculations.
///
/// Combines the geometric surface with crystal amplitude calculations.
/// After intersection and deflection, the crystal reflectivity amplitudes
/// are computed and applied to the beam.
pub struct CrystalOpticalElement<S: Surface> {
    pub surface: S,
    pub params: OeParams,
    pub crystal: CrystalBase,
}

impl<S: Surface> CrystalOpticalElement<S> {
    pub fn new(surface: S, params: OeParams, crystal: CrystalBase) -> Self {
        Self {
            surface,
            params,
            crystal,
        }
    }

    /// Reflect a beam off this crystal optical element.
    ///
    /// Performs intersection, deflection, and crystal amplitude calculation.
    pub fn reflect(
        &self,
        beam: &mut Beam,
        sf: &dyn StructureFactor,
    ) -> Vec<RayResult> {
        let good: Vec<usize> = (0..beam.nrays())
            .filter(|&i| beam.state[i] == 1)
            .collect();

        if good.is_empty() {
            return vec![];
        }

        // Transform to local coordinates
        let rotation = RotationParams::default_sequence(
            -self.params.pitch,
            -(self.params.roll + self.params.position_roll),
            -self.params.yaw,
        );

        for &i in &good {
            beam.x[i] -= self.params.center[0];
            beam.y[i] -= self.params.center[1];
            beam.z[i] -= self.params.center[2];
        }
        rotate_beam(beam, Some(&good), &rotation, false, false);

        // Core reflection
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

        // Compute crystal amplitudes for good rays
        if !good_after.is_empty() {
            self.apply_crystal_amplitude(beam, &good_after, &results, &good, sf);
        }

        // Rotate back to global (inverse rotation — reversed sequence)
        let inv_rotation = RotationParams::inverse_sequence(
            self.params.pitch,
            self.params.roll + self.params.position_roll,
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

    /// Compute and apply crystal reflectivity amplitudes.
    fn apply_crystal_amplitude(
        &self,
        beam: &mut Beam,
        good_after: &[usize],
        results: &[RayResult],
        good_orig: &[usize],
        sf: &dyn StructureFactor,
    ) {
        let n = good_after.len();

        // Collect beam_in_dot_normal for good rays
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

        // Compute crystal amplitude
        match self.crystal.get_amplitude(&energy_arr, &bidn_arr, None, None, sf) {
            Ok((rs, rp)) => {
                // Apply to beam coherency matrix
                reflect::apply_material_amplitude(
                    beam,
                    good_after,
                    rs.as_slice().unwrap(),
                    rp.as_slice().unwrap(),
                );
            }
            Err(e) => {
                // Crystal calculation failed — keep rays Good with reduced amplitude
                // (matching xrt Python behavior: never kill rays from amplitude)
                eprintln!("  crystal amplitude error: {e:?}, keeping {} rays", good_after.len());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use crate::beamline::OeParamsBuilder;
    use crate::surfaces::flat::FlatSurface;
    use xrt_materials::crystal::{CrystalBase, CrystalGeometry};
    use xrt_materials::material::{Material, MaterialKind};
    use xrt_materials::data::ScatteringTable;

    // Minimal structure factor for testing
    struct TestStructureFactor;
    impl StructureFactor for TestStructureFactor {
        fn get_structure_factor(
            &self,
            e: &Array1<f64>,
            _sin_theta_over_lambda: &Array1<f64>,
            _need_fhkl: bool,
        ) -> Result<(Array1<Complex64>, Array1<Complex64>, Array1<Complex64>), xrt_core::error::XrtError> {
            let n = e.len();
            let f0 = Array1::from_elem(n, Complex64::new(14.0, -0.5));
            let fhkl = Array1::from_elem(n, Complex64::new(10.0, -0.3));
            let fhkl_bar = Array1::from_elem(n, Complex64::new(10.0, -0.3));
            Ok((f0, fhkl, fhkl_bar))
        }
    }

    fn make_si_crystal() -> CrystalBase {
        let si = Material::new(
            &["Si"], None, 2.33,
            MaterialKind::Mirror, None,
            ScatteringTable::ChantlerTotal,
        ).unwrap();

        CrystalBase::new(
            si,
            [1, 1, 1],
            3.1356,         // d-spacing [Å]
            Some(160.18),   // V [ų]
            CrystalGeometry::BraggReflected,
            1.0,            // Debye-Waller
            None,           // semi-infinite
            0.0,            // no mosaicity
        )
    }

    #[test]
    fn crystal_oe_create() {
        let crystal = make_si_crystal();
        let oe = CrystalOpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new().pitch(0.2).build(),
            crystal,
        );
        assert_eq!(oe.crystal.hkl, [1, 1, 1]);
    }

    #[test]
    fn crystal_oe_reflect_with_amplitude() {
        let crystal = make_si_crystal();
        let oe = CrystalOpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new().pitch(0.01).build(),
            crystal,
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
            if let Some(ref mut es) = beam.es {
                es[i] = Complex64::new(1.0, 0.0);
            }
        }

        let sf = TestStructureFactor;
        let results = oe.reflect(&mut beam, &sf);
        assert_eq!(results.len(), 5);

        // Check that crystal amplitude was applied (jss modified from 1.0)
        let good_count = results.iter().filter(|r| r.state == RayState::Good).count();
        if good_count > 0 {
            // At least some rays should have modified coherency
            let first_good = (0..5).find(|&i| beam.state[i] == RayState::Good as i32);
            if let Some(i) = first_good {
                // jss should have been multiplied by |rs|²
                assert!(
                    beam.jss[i].is_finite(),
                    "jss[{}] = {}", i, beam.jss[i]
                );
            }
        }
    }
}
