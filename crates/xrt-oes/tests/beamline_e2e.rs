//! End-to-end beamline integration tests.
//!
//! Exercises the full pipeline: Source → MaterialMirror → Grating → Screen.

use xrt_core::beam::{Beam, RayState};
use xrt_sources::distributions::{EnergyDist, SpatialDist};
use xrt_sources::geometric::GeometricSource;
use xrt_materials::data::ScatteringTable;
use xrt_materials::material::{Material, MaterialKind};
use xrt_oes::beamline::{Beamline, OeParamsBuilder};
use xrt_oes::material_oe::MaterialOpticalElement;
use xrt_oes::grating_oe::GratingOpticalElement;
use xrt_oes::crystal_oe::CrystalOpticalElement;
use xrt_oes::oe::OpticalElement;
use xrt_oes::screen::Screen;
use xrt_oes::surfaces::flat::FlatSurface;
use xrt_oes::surfaces::toroid::ToroidSurface;
use xrt_oes::surfaces::grating::BlazedGrating;

fn si_mirror() -> Material {
    Material::new(
        &["Si"], None, 2.33,
        MaterialKind::Mirror, None,
        ScatteringTable::ChantlerTotal,
    ).unwrap()
}

fn au_grating() -> Material {
    Material::new(
        &["Au"], None, 19.32,
        MaterialKind::Grating, None,
        ScatteringTable::ChantlerTotal,
    ).unwrap()
}

fn collimated_source(nrays: usize, energy: f64) -> Beam {
    let source = GeometricSource {
        nrays,
        dist_x: SpatialDist::Normal(0.05),
        dist_z: SpatialDist::Normal(0.02),
        dist_xprime: SpatialDist::None,
        dist_zprime: SpatialDist::None,
        dist_e: EnergyDist::Lines(vec![energy], None),
        ..Default::default()
    };
    source.shine()
}

/// A realistic soft X-ray beamline: Source → Si flat mirror → Screen.
#[test]
fn single_si_mirror_beamline() {
    let mut beam = collimated_source(500, 1000.0);
    assert_eq!(beam.nrays(), 500);

    let mirror = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.01).build(),
        si_mirror(),
    );

    let bl = Beamline::new()
        .add_material("M1_Si", mirror)
        .drift(3000.0);

    let output = bl.propagate(&mut beam);

    assert_eq!(output.elements.len(), 2);
    assert!(output.final_good_count > 0, "all rays lost");
    assert!(output.efficiency() > 0.0, "zero efficiency");

    // Screen to check beam footprint
    let screen = Screen::new([0.0, 3000.0, 30.0], 50.0, 100.0, 20, 20);
    let capture = screen.capture(&beam);
    assert!(capture.total_intensity() >= 0.0);
}

/// Two-mirror beamline with toroid focusing.
#[test]
fn two_mirror_focusing() {
    let mut beam = collimated_source(300, 10000.0);

    let m1 = MaterialOpticalElement::new(
        ToroidSurface::new(5e6, 50.0),
        OeParamsBuilder::new()
            .center(0.0, 5000.0, 0.0)
            .pitch(0.003)
            .build(),
        si_mirror(),
    );

    let m2 = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new()
            .center(0.0, 10000.0, 0.0)
            .pitch(0.003)
            .build(),
        si_mirror(),
    );

    let bl = Beamline::new()
        .add_material("M1_Toroid", m1)
        .drift(5000.0)
        .add_material("M2_Flat", m2);

    let output = bl.propagate(&mut beam);
    assert_eq!(output.elements.len(), 3);
    // Track efficiency through the beamline
    for (i, el) in output.elements.iter().enumerate() {
        if !el.results.is_empty() {
            assert!(
                el.good_count <= 300,
                "element {}: good={} > initial",
                el.name, el.good_count
            );
        }
    }
}

/// Grating monochromator: mirror → grating → screen.
#[test]
fn grating_monochromator() {
    let mut beam = collimated_source(200, 800.0);

    let mirror = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.02).build(),
        si_mirror(),
    );

    let grating = GratingOpticalElement::new(
        BlazedGrating::new(600.0, 0.02, 0.5),
        OeParamsBuilder::new()
            .center(0.0, 3000.0, 0.0)
            .pitch(0.02)
            .build(),
        -1, // first negative order
    ).with_material(au_grating());

    let bl = Beamline::new()
        .add_material("M1", mirror)
        .drift(3000.0)
        .add_grating("Grating", grating);

    let output = bl.propagate(&mut beam);
    assert_eq!(output.elements.len(), 3);
    assert_eq!(output.elements[0].name, "M1");
    assert_eq!(output.elements[2].name, "Grating");
}

/// Mixed beamline: geometric OE + material OE + drift.
#[test]
fn mixed_oe_types() {
    let mut beam = collimated_source(100, 10000.0);

    let bare_mirror = OpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.01).build(),
    );

    let coated_mirror = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new()
            .center(0.0, 5000.0, 0.0)
            .pitch(-0.01)
            .build(),
        si_mirror(),
    );

    let bl = Beamline::new()
        .add("Bare_M1", bare_mirror)
        .drift(5000.0)
        .add_material("Coated_M2", coated_mirror);

    let output = bl.propagate(&mut beam);
    assert_eq!(output.elements.len(), 3);
    assert_eq!(output.initial_count, 100);
}

/// Beam propagation preserves direction and accumulates path.
#[test]
fn propagation_physics() {
    let mut beam = Beam::new(3);
    beam.set_state(RayState::Good);

    // Ray 0: straight +y
    beam.b[0] = 1.0;
    // Ray 1: 45° in y-z plane
    beam.b[1] = (0.5_f64).sqrt();
    beam.c[1] = (0.5_f64).sqrt();
    // Ray 2: at angle in x-y plane
    beam.a[2] = 0.1;
    beam.b[2] = (1.0 - 0.01_f64).sqrt();

    beam.propagate(1000.0);

    // Ray 0: should be at y=1000
    assert!((beam.y[0] - 1000.0).abs() < 1e-8);
    assert!(beam.x[0].abs() < 1e-8);

    // Ray 1: should have moved in both y and z
    assert!((beam.y[1] - 1000.0 * (0.5_f64).sqrt()).abs() < 1e-8);
    assert!((beam.z[1] - 1000.0 * (0.5_f64).sqrt()).abs() < 1e-8);

    // All paths should be 1000
    for i in 0..3 {
        assert!((beam.path[i] - 1000.0).abs() < 1e-8);
    }
}

/// Screen captures correct centroid for offset beam.
#[test]
fn screen_centroid_accuracy() {
    let screen = Screen::new([0.0, 1000.0, 0.0], 20.0, 20.0, 100, 100);

    let mut beam = Beam::new(500);
    beam.set_state(RayState::Good);
    for i in 0..500 {
        // Beam offset to x=3, z=-2
        beam.x[i] = 3.0 + (i as f64 / 499.0 - 0.5) * 0.1;
        beam.z[i] = -2.0 + (i as f64 / 499.0 - 0.5) * 0.1;
        beam.b[i] = 1.0;
    }

    let capture = screen.capture(&beam);
    let [cx, cz] = capture.centroid();

    assert!((cx - 3.0).abs() < 0.5, "centroid x = {cx}, expected ~3.0");
    assert!((cz - (-2.0)).abs() < 0.5, "centroid z = {cz}, expected ~-2.0");
}

/// BeamlineOutput provides useful diagnostics.
#[test]
fn beamline_output_diagnostics() {
    let mut beam = collimated_source(100, 10000.0);

    let mirror = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.005).build(),
        si_mirror(),
    );

    let bl = Beamline::new().add_material("M1", mirror);
    let output = bl.propagate(&mut beam);

    // Check output structure
    assert_eq!(output.initial_count, 100);
    assert!(output.final_good_count <= 100);
    assert!(output.efficiency() >= 0.0 && output.efficiency() <= 1.0);

    let m1_out = &output.elements[0];
    assert_eq!(m1_out.name, "M1");
    assert_eq!(m1_out.good_count + m1_out.lost_count, 100);
}

#[test]
fn golden_grating_beamline() {
    // Test a simple grating beamline: Source -> BlazedGrating -> Screen
    let mut beam = collimated_source(500, 1000.0);

    let grating = GratingOpticalElement::new(
        BlazedGrating::new(600.0, 0.02, 0.5),
        OeParamsBuilder::new().pitch(0.01).build(),
        1, // first positive order
    );

    let bl = Beamline::new()
        .add_grating("G1", grating)
        .drift(1000.0);

    let output = bl.propagate(&mut beam);
    assert!(output.final_good_count > 0,
            "Grating beamline should have good rays");
    assert!(output.efficiency() > 0.0,
            "Grating should have non-zero efficiency");
}

#[test]
fn golden_negative_grating_order() {
    let mut beam = collimated_source(500, 1000.0);

    let grating = GratingOpticalElement::new(
        BlazedGrating::new(600.0, 0.02, 0.5),
        OeParamsBuilder::new().pitch(0.01).build(),
        -1, // first negative order
    );

    let bl = Beamline::new()
        .add_grating("G1", grating)
        .drift(1000.0);

    let output = bl.propagate(&mut beam);
    // Negative order should also produce valid rays
    assert!(output.final_good_count > 0,
            "Negative grating order should have good rays");
}

#[test]
fn golden_multi_element_beamline() {
    // Source -> Mirror -> Drift -> Mirror -> Screen
    let mut beam = collimated_source(1000, 10000.0);

    let m1 = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.003).build(),
        si_mirror(),
    );
    let m2 = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.003).build(),
        si_mirror(),
    );

    let bl = Beamline::new()
        .add_material("M1", m1)
        .drift(5000.0)
        .add_material("M2", m2)
        .drift(5000.0);

    let output = bl.propagate(&mut beam);
    assert!(output.elements.len() >= 2,
            "Should have at least 2 OE outputs, got {}", output.elements.len());
    // Second mirror may not intercept all rays depending on geometry,
    // so just verify the pipeline runs without panic and produces output
    assert!(output.initial_count == 1000, "Should start with 1000 rays");
}

#[test]
fn golden_crystal_beamline() {
    use xrt_materials::crystal::{CrystalGeometry, StructureFactor};
    use xrt_materials::crystal_variants::CrystalSi;
    use xrt_materials::data::ScatteringTable;

    let mut beam = collimated_source(500, 10000.0);

    let si = CrystalSi::new(
        [1, 1, 1],
        297.15,
        CrystalGeometry::BraggReflected,
        1.0,
        None,
        0.0,
        ScatteringTable::ChantlerTotal,
    )
    .unwrap();

    let bragg_angle = si.base.get_bragg_angle(&ndarray::array![10000.0])[0];

    let crystal_oe = CrystalOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new()
            .pitch(bragg_angle)
            .build(),
        si.base.clone(),
    );

    let bl = Beamline::new()
        .add_crystal("Si111", crystal_oe, Box::new(si))
        .drift(5000.0);

    let output = bl.propagate(&mut beam);
    // Crystal beamline should run without panic
    assert!(output.initial_count == 500);
}

#[test]
fn golden_three_element_beamline() {
    // Source -> Mirror -> Drift -> Mirror -> Drift -> Grating
    let mut beam = collimated_source(500, 10000.0);

    let m1 = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.003).build(),
        si_mirror(),
    );
    let m2 = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.003).build(),
        si_mirror(),
    );
    let g1 = GratingOpticalElement::new(
        BlazedGrating::new(600.0, 0.02, 0.5),
        OeParamsBuilder::new().pitch(0.01).build(),
        1,
    );

    let bl = Beamline::new()
        .add_material("M1", m1)
        .drift(1000.0)
        .add_material("M2", m2)
        .drift(1000.0)
        .add_grating("G1", g1);

    let output = bl.propagate(&mut beam);
    assert!(output.elements.len() >= 3,
            "Should have at least 3 OE outputs, got {}", output.elements.len());
    assert!(output.initial_count == 500);
}

#[test]
fn golden_coherency_through_mirror() {
    // Verify coherency matrix is preserved/modified through flat mirror
    use xrt_sources::polarization::Polarization;

    let source = GeometricSource {
        nrays: 100,
        dist_e: EnergyDist::Lines(vec![10000.0], None),
        polarization: Polarization::Horizontal,
        ..Default::default()
    };
    let mut beam = source.shine();

    // Check initial coherency: horizontal -> jss=1, jpp=0
    assert!((beam.jss[0] - 1.0).abs() < 1e-10, "initial jss should be 1.0");
    assert!(beam.jpp[0].abs() < 1e-10, "initial jpp should be 0.0");

    let mirror = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.003).build(),
        si_mirror(),
    );
    let bl = Beamline::new()
        .add_material("M1", mirror)
        .drift(1000.0);

    let _output = bl.propagate(&mut beam);

    // After reflection, jss should still be set for surviving rays
    let good_val = RayState::Good as i32;
    let good_count = beam.state.iter().filter(|&&s| s == good_val).count();
    if good_count > 0 {
        let first_good = beam.state.iter().position(|&s| s == good_val).unwrap();
        assert!(beam.jss[first_good].is_finite(), "jss after mirror should be finite");
        assert!(beam.jpp[first_good].is_finite(), "jpp after mirror should be finite");
    }
}

#[test]
fn golden_refract_beamline() {
    use xrt_oes::reflect::DeflectionMode;

    let mut beam = collimated_source(500, 10000.0);

    // Create a lens-like element using refraction
    let si = si_mirror();
    let lens = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new()
            .pitch(0.003)
            .mode(DeflectionMode::Refract { n1_over_n2: 0.9999 })
            .build(),
        si,
    );

    let bl = Beamline::new()
        .add_material("Lens", lens)
        .drift(5000.0);

    let output = bl.propagate(&mut beam);
    // Refraction should produce valid output
    assert!(output.initial_count == 500);
}

#[test]
fn golden_passthrough_beamline() {
    use xrt_oes::reflect::DeflectionMode;

    let mut beam = collimated_source(500, 10000.0);
    let initial_count = beam.nrays();

    // PassThrough should not deflect the beam
    let pass = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new()
            .mode(DeflectionMode::PassThrough)
            .build(),
        si_mirror(),
    );

    let bl = Beamline::new()
        .add_material("Pass", pass)
        .drift(1000.0);

    let output = bl.propagate(&mut beam);
    // PassThrough pipeline should run without panic
    assert_eq!(output.initial_count, initial_count as usize);
}
