//! High-level beamline orchestration.
//!
//! Chains optical elements into a sequential pipeline and propagates
//! a beam through them. Provides a fluent API for common beamline setups.
//!
//! # Example
//! ```rust,ignore
//! let results = Beamline::new()
//!     .add("M1", OpticalElement::new(toroid, oe_params))
//!     .add("Grating", OpticalElement::new(grating, grating_params))
//!     .propagate(&mut beam);
//! ```

use crate::core::beam::{Beam, RayState};

use crate::materials::crystal::StructureFactor;

use crate::oes::crystal_oe::CrystalOpticalElement;
use crate::oes::grating_oe::GratingOpticalElement;
use crate::oes::material_oe::MaterialOpticalElement;
use crate::oes::oe::{OeParams, OpticalElement};
use crate::oes::param_oe::ParametricOpticalElement;
use crate::oes::reflect::{DeflectionMode, RayResult};
use crate::oes::surface::{ParametricSurface, Surface};

/// Result of propagating through one OE.
#[derive(Debug)]
pub struct OeOutput {
    /// Name of the optical element
    pub name: String,
    /// Per-ray results
    pub results: Vec<RayResult>,
    /// Number of good rays after this element
    pub good_count: usize,
    /// Number of lost rays at this element
    pub lost_count: usize,
}

/// Statistics for a complete beamline propagation.
#[derive(Debug)]
pub struct BeamlineOutput {
    /// Results per optical element
    pub elements: Vec<OeOutput>,
    /// Total good rays at the end
    pub final_good_count: usize,
    /// Total rays at start
    pub initial_count: usize,
}

impl BeamlineOutput {
    /// Overall transmission efficiency.
    pub fn efficiency(&self) -> f64 {
        if self.initial_count == 0 {
            return 0.0;
        }
        self.final_good_count as f64 / self.initial_count as f64
    }
}

impl std::fmt::Display for BeamlineOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Beamline: {} elements, {} → {} rays ({:.1}%)",
            self.elements.len(),
            self.initial_count,
            self.final_good_count,
            self.efficiency() * 100.0,
        )?;
        for el in &self.elements {
            if el.results.is_empty() {
                writeln!(f, "  {:20} (drift)", el.name)?;
            } else {
                writeln!(
                    f,
                    "  {:20} good={:6}  lost={:4}",
                    el.name, el.good_count, el.lost_count
                )?;
            }
        }
        Ok(())
    }
}

/// A trait-object wrapper for optical elements, enabling heterogeneous collections.
trait OeReflector: Send + Sync {
    fn reflect_beam(&self, beam: &mut Beam) -> Vec<RayResult>;
}

impl<S: Surface> OeReflector for OpticalElement<S> {
    fn reflect_beam(&self, beam: &mut Beam) -> Vec<RayResult> {
        self.reflect(beam)
    }
}

impl<S: Surface> OeReflector for MaterialOpticalElement<S> {
    fn reflect_beam(&self, beam: &mut Beam) -> Vec<RayResult> {
        self.reflect(beam)
    }
}

impl<S: Surface> OeReflector for GratingOpticalElement<S> {
    fn reflect_beam(&self, beam: &mut Beam) -> Vec<RayResult> {
        self.reflect(beam)
    }
}

/// A free-space drift element.
struct DriftElement {
    distance: f64,
}

impl OeReflector for DriftElement {
    fn reflect_beam(&self, beam: &mut Beam) -> Vec<RayResult> {
        beam.propagate(self.distance);
        vec![] // no per-ray results for drift
    }
}

impl<P: ParametricSurface> OeReflector for ParametricOpticalElement<P> {
    fn reflect_beam(&self, beam: &mut Beam) -> Vec<RayResult> {
        self.reflect(beam)
    }
}

/// Wrapper that bundles a CrystalOpticalElement with its StructureFactor.
struct CrystalOeWithSf<S: Surface> {
    oe: CrystalOpticalElement<S>,
    sf: Box<dyn StructureFactor + Send + Sync>,
}

impl<S: Surface> OeReflector for CrystalOeWithSf<S> {
    fn reflect_beam(&self, beam: &mut Beam) -> Vec<RayResult> {
        self.oe.reflect(beam, self.sf.as_ref())
    }
}

/// Named optical element in a beamline.
struct NamedOe {
    name: String,
    element: Box<dyn OeReflector>,
}

/// A beamline: an ordered sequence of optical elements.
///
/// Propagates a beam through each element in order.
pub struct Beamline {
    elements: Vec<NamedOe>,
}

impl Beamline {
    /// Create an empty beamline.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Add a Surface optical element to the beamline.
    pub fn add<S: Surface + 'static>(mut self, name: &str, oe: OpticalElement<S>) -> Self {
        self.elements.push(NamedOe {
            name: name.to_string(),
            element: Box::new(oe),
        });
        self
    }

    /// Add a ParametricSurface optical element to the beamline.
    pub fn add_parametric<P: ParametricSurface + 'static>(
        mut self,
        name: &str,
        oe: ParametricOpticalElement<P>,
    ) -> Self {
        self.elements.push(NamedOe {
            name: name.to_string(),
            element: Box::new(oe),
        });
        self
    }

    /// Add a material-coated optical element (Fresnel reflectivity).
    pub fn add_material<S: Surface + 'static>(
        mut self,
        name: &str,
        oe: MaterialOpticalElement<S>,
    ) -> Self {
        self.elements.push(NamedOe {
            name: name.to_string(),
            element: Box::new(oe),
        });
        self
    }

    /// Add a grating optical element to the beamline.
    pub fn add_grating<S: Surface + 'static>(
        mut self,
        name: &str,
        oe: GratingOpticalElement<S>,
    ) -> Self {
        self.elements.push(NamedOe {
            name: name.to_string(),
            element: Box::new(oe),
        });
        self
    }

    /// Add a free-space drift (propagate rays by distance [mm]).
    pub fn drift(mut self, distance: f64) -> Self {
        self.elements.push(NamedOe {
            name: format!("drift_{:.0}mm", distance),
            element: Box::new(DriftElement { distance }),
        });
        self
    }

    /// Add a crystal optical element with its structure factor to the beamline.
    pub fn add_crystal<S: Surface + 'static>(
        mut self,
        name: &str,
        oe: CrystalOpticalElement<S>,
        sf: Box<dyn StructureFactor + Send + Sync>,
    ) -> Self {
        self.elements.push(NamedOe {
            name: name.to_string(),
            element: Box::new(CrystalOeWithSf { oe, sf }),
        });
        self
    }

    /// Add a compound refractive lens (CRL) -- multiple paraboloid lenses in sequence.
    ///
    /// Each lens is a `MaterialOpticalElement` with `ParaboloidLensSurface` and
    /// `DeflectionMode::Refract`. Lenses are separated by `spacing` mm.
    ///
    /// # Arguments
    /// * `name_prefix` - Name prefix for each lens (suffixed with `_0`, `_1`, ...)
    /// * `n_lenses` - Number of individual lenses
    /// * `focus` - Focal length per lens [mm]
    /// * `z_max` - Optional maximum sag (clipping height) [mm]
    /// * `material` - Lens material (e.g. Be, Al)
    /// * `n1_over_n2` - Refractive index ratio n1/n2 for Snell's law
    /// * `spacing` - Distance between successive lenses [mm]
    pub fn add_crl(
        mut self,
        name_prefix: &str,
        n_lenses: usize,
        focus: f64,
        z_max: Option<f64>,
        material: crate::materials::material::Material,
        n1_over_n2: f64,
        spacing: f64,
    ) -> Self {
        use crate::oes::surfaces::lens::ParaboloidLensSurface;

        for i in 0..n_lenses {
            let lens_name = format!("{name_prefix}_{i}");
            let lens = MaterialOpticalElement::new(
                ParaboloidLensSurface::new(focus, z_max),
                OeParamsBuilder::new()
                    .mode(DeflectionMode::Refract { n1_over_n2 })
                    .build(),
                material.clone(),
            );
            self.elements.push(NamedOe {
                name: lens_name,
                element: Box::new(lens),
            });
            if i < n_lenses - 1 {
                self = self.drift(spacing);
            }
        }
        self
    }

    /// Add a double crystal monochromator (DCM).
    ///
    /// Two crystal reflections separated by a drift gap.
    /// Crystals are named `"{name_prefix}_1"` and `"{name_prefix}_2"`.
    ///
    /// # Arguments
    /// * `name_prefix` - Name prefix (e.g. "DCM" → "DCM_1", "DCM_2")
    /// * `crystal1` - First crystal optical element
    /// * `sf1` - Structure factor for crystal 1
    /// * `crystal2` - Second crystal optical element
    /// * `sf2` - Structure factor for crystal 2
    /// * `gap` - Drift distance between the two crystals [mm]
    pub fn add_dcm<S: Surface + 'static>(
        self,
        name_prefix: &str,
        crystal1: CrystalOpticalElement<S>,
        sf1: Box<dyn StructureFactor + Send + Sync>,
        crystal2: CrystalOpticalElement<S>,
        sf2: Box<dyn StructureFactor + Send + Sync>,
        gap: f64,
    ) -> Self {
        let n1 = format!("{name_prefix}_1");
        let n2 = format!("{name_prefix}_2");
        self.add_crystal(&n1, crystal1, sf1)
            .drift(gap)
            .add_crystal(&n2, crystal2, sf2)
    }

    /// Add a double paraboloid lens (bi-concave, front + back).
    ///
    /// Two paraboloid refracting surfaces separated by the lens thickness.
    /// The front surface refracts vacuum → material (`n1_over_n2`),
    /// the back surface refracts material → vacuum (`1.0 / n1_over_n2`).
    ///
    /// # Arguments
    /// * `name_prefix` - Name prefix (e.g. "BiLens" → "BiLens_front", "BiLens_back")
    /// * `focus` - Focal length per surface [mm]
    /// * `z_max` - Optional maximum sag (clipping height) [mm]
    /// * `material` - Lens material
    /// * `n1_over_n2` - Refractive index ratio n1/n2 for the front surface
    /// * `thickness` - Distance between front and back surfaces [mm]
    pub fn add_double_lens(
        mut self,
        name_prefix: &str,
        focus: f64,
        z_max: Option<f64>,
        material: crate::materials::material::Material,
        n1_over_n2: f64,
        thickness: f64,
    ) -> Self {
        use crate::oes::surfaces::lens::ParaboloidLensSurface;

        let front_name = format!("{name_prefix}_front");
        let back_name = format!("{name_prefix}_back");

        let front = MaterialOpticalElement::new(
            ParaboloidLensSurface::new(focus, z_max),
            OeParamsBuilder::new()
                .mode(DeflectionMode::Refract { n1_over_n2 })
                .build(),
            material.clone(),
        );
        self.elements.push(NamedOe {
            name: front_name,
            element: Box::new(front),
        });

        self = self.drift(thickness);

        let back = MaterialOpticalElement::new(
            ParaboloidLensSurface::new(focus, z_max),
            OeParamsBuilder::new()
                .mode(DeflectionMode::Refract {
                    n1_over_n2: 1.0 / n1_over_n2,
                })
                .build(),
            material,
        );
        self.elements.push(NamedOe {
            name: back_name,
            element: Box::new(back),
        });

        self
    }

    /// Add a double parabolic cylinder lens (focuses in x only).
    pub fn add_double_cylinder_lens(
        self,
        name_prefix: &str,
        focus: f64,
        z_max: Option<f64>,
        material: crate::materials::material::Material,
        n1_over_n2: f64,
        thickness: f64,
    ) -> Self {
        use crate::oes::material_oe::MaterialOpticalElement;
        use crate::oes::reflect::DeflectionMode;
        use crate::oes::surfaces::cylinder_lens::ParabolicCylinderLensSurface;

        let front_name = format!("{name_prefix}_front");
        let back_name = format!("{name_prefix}_back");

        let front = MaterialOpticalElement::new(
            ParabolicCylinderLensSurface::new(focus, z_max),
            OeParamsBuilder::new()
                .mode(DeflectionMode::Refract { n1_over_n2 })
                .build(),
            material.clone(),
        );
        let back = MaterialOpticalElement::new(
            ParabolicCylinderLensSurface::new(focus, z_max),
            OeParamsBuilder::new()
                .mode(DeflectionMode::Refract {
                    n1_over_n2: 1.0 / n1_over_n2,
                })
                .build(),
            material,
        );

        self.add_material(Box::leak(front_name.into_boxed_str()), front)
            .drift(thickness)
            .add_material(Box::leak(back_name.into_boxed_str()), back)
    }

    /// Number of optical elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Whether the beamline is empty.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Propagate a beam through all optical elements in order.
    ///
    /// Returns statistics for each element.
    pub fn propagate(&self, beam: &mut Beam) -> BeamlineOutput {
        let initial_count = beam.nrays();
        let mut outputs = Vec::with_capacity(self.elements.len());

        for named_oe in &self.elements {
            let good_before: usize = beam
                .state
                .iter()
                .filter(|&&s| s == RayState::Good as i32)
                .count();

            let results = named_oe.element.reflect_beam(beam);

            let good_after: usize = beam
                .state
                .iter()
                .filter(|&&s| s == RayState::Good as i32)
                .count();

            let lost = good_before.saturating_sub(good_after);

            outputs.push(OeOutput {
                name: named_oe.name.clone(),
                results,
                good_count: good_after,
                lost_count: lost,
            });
        }

        let final_good_count = beam
            .state
            .iter()
            .filter(|&&s| s == RayState::Good as i32)
            .count();

        BeamlineOutput {
            elements: outputs,
            final_good_count,
            initial_count,
        }
    }
}

impl Default for Beamline {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for OeParams with fluent API.
pub struct OeParamsBuilder {
    params: OeParams,
}

impl OeParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: OeParams::default(),
        }
    }

    pub fn center(mut self, x: f64, y: f64, z: f64) -> Self {
        self.params.center = [x, y, z];
        self
    }

    pub fn pitch(mut self, rad: f64) -> Self {
        self.params.pitch = rad;
        self
    }

    pub fn roll(mut self, rad: f64) -> Self {
        self.params.roll = rad;
        self
    }

    pub fn yaw(mut self, rad: f64) -> Self {
        self.params.yaw = rad;
        self
    }

    pub fn invert_normal(mut self) -> Self {
        self.params.invert_normal = -1;
        self
    }

    pub fn mode(mut self, mode: DeflectionMode) -> Self {
        self.params.mode = mode;
        self
    }

    pub fn build(self) -> OeParams {
        self.params
    }
}

impl Default for OeParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oes::surfaces::flat::FlatSurface;

    fn make_beam(n: usize) -> Beam {
        let mut beam = Beam::new(n);
        for i in 0..n {
            beam.y[i] = -1000.0;
            beam.a[i] = 0.0;
            beam.b[i] = 1.0;
            beam.c[i] = 0.0;
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
        }
        beam
    }

    #[test]
    fn empty_beamline() {
        let bl = Beamline::new();
        assert!(bl.is_empty());
        let mut beam = make_beam(10);
        let output = bl.propagate(&mut beam);
        assert_eq!(output.initial_count, 10);
        assert_eq!(output.final_good_count, 10);
        assert!((output.efficiency() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn single_flat_mirror() {
        let oe = OpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new()
                .center(0.0, 0.0, 0.0)
                .pitch(0.01)
                .build(),
        );
        let bl = Beamline::new().add("M1", oe);
        assert_eq!(bl.len(), 1);

        let mut beam = make_beam(20);
        let output = bl.propagate(&mut beam);
        assert_eq!(output.elements.len(), 1);
        assert_eq!(output.elements[0].name, "M1");
        assert!(output.final_good_count > 0);
    }

    #[test]
    fn two_mirrors() {
        let m1 = OpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new()
                .center(0.0, 0.0, 0.0)
                .pitch(0.01)
                .build(),
        );
        let m2 = OpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new()
                .center(0.0, 2000.0, 0.0)
                .pitch(-0.01)
                .build(),
        );
        let bl = Beamline::new().add("M1", m1).add("M2", m2);
        assert_eq!(bl.len(), 2);

        let mut beam = make_beam(20);
        let output = bl.propagate(&mut beam);
        assert_eq!(output.elements.len(), 2);
    }

    #[test]
    fn oe_params_builder() {
        let params = OeParamsBuilder::new()
            .center(100.0, 5000.0, 0.0)
            .pitch(0.005)
            .roll(0.001)
            .mode(DeflectionMode::Grating { order: 1 })
            .build();

        assert!((params.center[0] - 100.0).abs() < 1e-15);
        assert!((params.pitch - 0.005).abs() < 1e-15);
        assert_eq!(params.mode, DeflectionMode::Grating { order: 1 });
    }

    #[test]
    fn efficiency_calculation() {
        let output = BeamlineOutput {
            elements: vec![],
            final_good_count: 80,
            initial_count: 100,
        };
        assert!((output.efficiency() - 0.8).abs() < 1e-15);
    }

    #[test]
    fn beamline_with_parametric() {
        use crate::oes::surfaces::elliptical::EllipticalSurface;

        let ellipse = EllipticalSurface::from_pq(10_000.0, 5_000.0, 0.003);
        let oe =
            ParametricOpticalElement::new(ellipse, OeParamsBuilder::new().pitch(0.003).build());

        let bl = Beamline::new()
            .add(
                "M1",
                OpticalElement::new(FlatSurface, OeParamsBuilder::new().pitch(0.01).build()),
            )
            .add_parametric("Elliptical", oe);

        assert_eq!(bl.len(), 2);

        let mut beam = make_beam(10);
        let output = bl.propagate(&mut beam);
        assert_eq!(output.elements.len(), 2);
        assert_eq!(output.elements[0].name, "M1");
        assert_eq!(output.elements[1].name, "Elliptical");
    }

    #[test]
    fn beamline_with_material_and_drift() {
        use crate::materials::data::ScatteringTable;
        use crate::materials::material::{Material, MaterialKind};

        let si = Material::new(
            &["Si"],
            None,
            2.33,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let m1 = crate::oes::material_oe::MaterialOpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new().pitch(0.01).build(),
            si,
        );

        let bl = Beamline::new()
            .add_material("Si_Mirror", m1)
            .drift(2000.0)
            .add(
                "Screen_Mirror",
                OpticalElement::new(FlatSurface, OeParamsBuilder::new().build()),
            );

        assert_eq!(bl.len(), 3);

        let mut beam = make_beam(10);
        let output = bl.propagate(&mut beam);
        assert_eq!(output.elements.len(), 3);
        assert_eq!(output.elements[0].name, "Si_Mirror");
        assert!(output.elements[1].name.contains("drift"));
    }

    #[test]
    fn beamline_with_crystal() {
        use crate::materials::crystal::{CrystalBase, CrystalGeometry, StructureFactor as SF};
        use crate::materials::data::ScatteringTable;
        use crate::materials::material::{Material, MaterialKind};
        use ndarray::Array1;
        use num_complex::Complex64;

        struct TestSf;
        impl SF for TestSf {
            fn get_structure_factor(
                &self,
                e: &Array1<f64>,
                _stol: &Array1<f64>,
                _need_fhkl: bool,
            ) -> Result<
                (Array1<Complex64>, Array1<Complex64>, Array1<Complex64>),
                crate::core::error::XrtError,
            > {
                let n = e.len();
                Ok((
                    Array1::from_elem(n, Complex64::new(14.0, -0.5)),
                    Array1::from_elem(n, Complex64::new(10.0, -0.3)),
                    Array1::from_elem(n, Complex64::new(10.0, -0.3)),
                ))
            }
        }

        let si = Material::new(
            &["Si"],
            None,
            2.33,
            MaterialKind::Mirror,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .unwrap();

        let crystal = CrystalBase::new(
            si,
            [1, 1, 1],
            3.1356,
            Some(160.18),
            CrystalGeometry::BraggReflected,
            1.0,
            None,
            0.0,
        );

        let crystal_oe = crate::oes::crystal_oe::CrystalOpticalElement::new(
            FlatSurface,
            OeParamsBuilder::new().pitch(0.2).build(),
            crystal,
        );

        let bl = Beamline::new().add_crystal("Si111", crystal_oe, Box::new(TestSf));

        assert_eq!(bl.len(), 1);

        let mut beam = make_beam(5);
        let output = bl.propagate(&mut beam);
        assert_eq!(output.elements.len(), 1);
        assert_eq!(output.elements[0].name, "Si111");
    }
}
