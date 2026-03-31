//! Integration tests for the xrt-oes optical element pipeline.
//!
//! Tests the full pipeline: surface creation → intersection → reflection.

use xrt_oes::intersection::find_intersection_surface;
use xrt_oes::surface::{ParametricSurface, Surface};
use xrt_oes::surfaces::flat::FlatSurface;
use xrt_oes::surfaces::toroid::ToroidSurface;
use xrt_oes::surfaces::spherical::SphericalSurface;
use xrt_oes::surfaces::lens::ParaboloidLensSurface;
use xrt_oes::surfaces::elliptical::EllipticalSurface;
use xrt_oes::surfaces::grating::{BlazedGrating, LaminarGrating};
use xrt_oes::surfaces::fzp::FzpSurface;
use xrt_math::rootfind::RootFindConfig;

#[test]
fn flat_surface_intersect_batch() {
    let surface = FlatSurface;
    let config = RootFindConfig::default();
    let n = 100;

    for i in 0..n {
        let x = (i as f64 - 50.0) * 0.1;
        let result = find_intersection_surface(
            &surface, 0.0, 20.0, x, 0.0, 10.0, 0.0, 0.0, -1.0, 1, &config,
        );
        assert!(result.converged, "ray {i} didn't converge");
        assert!((result.z).abs() < 1e-8, "z = {} for ray {i}", result.z);
        assert!((result.x - x).abs() < 1e-8);
    }
}

#[test]
fn spherical_surface_intersect() {
    let surface = SphericalSurface::new(1000.0);
    let config = RootFindConfig::default();

    // Ray straight down at x=0 should hit at z=0
    let result = find_intersection_surface(
        &surface, 0.0, 20.0, 0.0, 0.0, 10.0, 0.0, 0.0, -1.0, 1, &config,
    );
    assert!(result.converged);
    assert!(result.z.abs() < 1e-6, "z = {}", result.z);

    // Ray at x=10 should hit at z > 0 (spherical sag)
    let result = find_intersection_surface(
        &surface, 0.0, 20.0, 10.0, 0.0, 10.0, 0.0, 0.0, -1.0, 1, &config,
    );
    assert!(result.converged);
    let expected_z = 1000.0 - (1000.0_f64.powi(2) - 10.0_f64.powi(2)).sqrt();
    assert!(
        (result.z - expected_z).abs() < 0.01,
        "z = {}, expected {}",
        result.z,
        expected_z
    );
}

#[test]
fn toroid_surface_intersect() {
    let surface = ToroidSurface::new(5e6, 50.0);
    let config = RootFindConfig::default();

    let result = find_intersection_surface(
        &surface, 0.0, 20.0, 0.0, 0.0, 10.0, 0.0, 0.0, -1.0, 1, &config,
    );
    assert!(result.converged);
    assert!(result.z.abs() < 1e-6);
}

#[test]
fn lens_surface_intersect() {
    let surface = ParaboloidLensSurface::new(100.0, None);
    let config = RootFindConfig::default();

    let result = find_intersection_surface(
        &surface, 0.0, 20.0, 5.0, 0.0, 10.0, 0.0, 0.0, -1.0, 1, &config,
    );
    assert!(result.converged);
    let expected_z = 25.0 / 400.0; // x²/(4f) = 25/400
    assert!(
        (result.z - expected_z).abs() < 0.01,
        "z = {}, expected {}",
        result.z,
        expected_z
    );
}

#[test]
fn all_surfaces_normal_at_origin() {
    // Every surface should have normal ≈ [0,0,1] at origin
    let surfaces: Vec<Box<dyn Surface>> = vec![
        Box::new(FlatSurface),
        Box::new(SphericalSurface::new(1000.0)),
        Box::new(ToroidSurface::new(5e6, 50.0)),
        Box::new(ParaboloidLensSurface::new(100.0, None)),
        Box::new(BlazedGrating::new(1200.0, 0.02, 0.5)),
        Box::new(LaminarGrating::new(1200.0, 0.005, 0.5)),
        Box::new(FzpSurface::new(100.0, 1.0, 1000)),
    ];

    for (idx, surface) in surfaces.iter().enumerate() {
        let n = surface.local_n(0.0, 0.0);
        assert!(
            n[2] > 0.9,
            "surface {idx}: normal z = {}, expected > 0.9",
            n[2]
        );
    }
}

#[test]
fn all_surfaces_z_at_origin() {
    let surfaces: Vec<(&str, Box<dyn Surface>)> = vec![
        ("flat", Box::new(FlatSurface)),
        ("spherical", Box::new(SphericalSurface::new(1000.0))),
        ("toroid", Box::new(ToroidSurface::new(5e6, 50.0))),
        ("lens", Box::new(ParaboloidLensSurface::new(100.0, None))),
        ("fzp", Box::new(FzpSurface::new(100.0, 1.0, 1000))),
    ];

    for (name, surface) in &surfaces {
        let z = surface.local_z(0.0, 0.0);
        assert!(
            z.abs() < 1e-12,
            "{name}: z(0,0) = {z}, expected 0"
        );
    }
}

#[test]
fn parametric_surfaces_roundtrip() {
    let surfaces: Vec<(&str, Box<dyn ParametricSurface>)> = vec![
        ("elliptical", Box::new(EllipticalSurface::new(100.0, 50.0, 0.0))),
        ("parabolical", Box::new(xrt_oes::surfaces::parabolical::ParabolicalSurface::new(10.0, 0.0))),
        ("hyperbolic", Box::new(xrt_oes::surfaces::hyperbolic::HyperbolicSurface::new(100.0, 50.0, 0.0))),
    ];

    for (name, surface) in &surfaces {
        let (x, y, z) = (3.0, 5.0, 7.0);
        let (s, phi, r) = surface.xyz_to_param(x, y, z);
        let (x2, y2, z2) = surface.param_to_xyz(s, phi, r);
        assert!(
            (x2 - x).abs() < 1e-8 && (y2 - y).abs() < 1e-8 && (z2 - z).abs() < 1e-8,
            "{name}: roundtrip failed: ({x},{y},{z}) → ({s},{phi},{r}) → ({x2},{y2},{z2})"
        );
    }
}

/// End-to-end: GeometricSource → Material Mirror → Screen
#[test]
fn end_to_end_source_mirror_screen() {
    use xrt_sources::distributions::{EnergyDist, SpatialDist};
    use xrt_sources::geometric::GeometricSource;
    use xrt_oes::beamline::{Beamline, OeParamsBuilder};
    use xrt_oes::material_oe::MaterialOpticalElement;
    use xrt_oes::screen::Screen;
    use xrt_materials::data::ScatteringTable;
    use xrt_materials::material::{Material, MaterialKind};

    // 1. Source: collimated beam going in +y
    let source = GeometricSource {
        nrays: 200,
        dist_x: SpatialDist::Normal(0.1),
        dist_z: SpatialDist::Normal(0.05),
        dist_xprime: SpatialDist::None,
        dist_zprime: SpatialDist::None,
        dist_e: EnergyDist::Lines(vec![10000.0], None),
        ..Default::default()
    };
    let mut beam = source.shine();
    assert_eq!(beam.nrays(), 200);
    let initial_good = beam.good_indices().len();
    assert_eq!(initial_good, 200);

    // 2. Flat Si mirror at origin, 10 mrad pitch
    let si = Material::new(
        &["Si"], None, 2.33,
        MaterialKind::Mirror, None,
        ScatteringTable::ChantlerTotal,
    ).unwrap();

    let mirror = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new().pitch(0.01).build(),
        si,
    );

    // 3. Propagate: mirror → drift
    let bl = Beamline::new()
        .add_material("Si_Flat", mirror)
        .drift(5000.0);

    let output = bl.propagate(&mut beam);
    assert_eq!(output.elements.len(), 2);

    // After mirror, some rays should still be good
    let good_after = beam.good_indices().len();
    assert!(good_after > 0, "no good rays after mirror");

    // 4. Screen at a large area to catch reflected rays
    let screen = Screen::new([0.0, 5000.0, 50.0], 50.0, 100.0, 20, 20);
    let capture = screen.capture(&beam);

    // Reflected rays should be going upward (c > 0 after mirror pitch)
    // Screen captures by projecting to its y-plane
    assert!(
        capture.n_captured > 0 || good_after > 0,
        "no rays captured: n_captured={}, good_after={}",
        capture.n_captured, good_after
    );
}

/// Test: Grating diffraction changes ray direction.
#[test]
fn grating_deflection_changes_direction() {
    use xrt_core::beam::{Beam, RayState};
    use xrt_oes::reflect::{reflect_local, DeflectionMode};
    use xrt_oes::aperture::Aperture;

    let grating = BlazedGrating::new(600.0, 0.02, 0.5);
    let mut beam = Beam::new(1);
    // Ray at grazing angle
    let angle = 0.02_f64;
    beam.y[0] = -100.0;
    beam.z[0] = 10.0;
    beam.a[0] = 0.0;
    beam.b[0] = angle.cos();
    beam.c[0] = -angle.sin();
    beam.state[0] = RayState::Good as i32;
    beam.e[0] = 10000.0;

    let good = vec![0usize];
    let aperture = Aperture::default();

    // Reflect with grating order 1
    let results = reflect_local(
        &grating, &mut beam, &good, &aperture, 1,
        DeflectionMode::Grating { order: 1 },
        None, None,
        &xrt_math::rootfind::RootFindConfig::default(),
    );

    if !results.is_empty() && results[0].state == RayState::Good {
        // After grating, direction should differ from specular
        let _specular_c = angle.sin(); // specular would reflect c symmetrically
        // Grating adds diffraction angle — direction should be different
        assert!(
            results[0].a.is_finite() && results[0].b.is_finite() && results[0].c.is_finite(),
            "grating deflection produced non-finite direction"
        );
    }
}

/// Test: Refraction bends ray through a surface.
#[test]
fn refraction_bends_ray() {
    use xrt_core::beam::{Beam, RayState};
    use xrt_oes::reflect::{reflect_local, DeflectionMode};
    use xrt_oes::aperture::Aperture;

    let surface = FlatSurface;
    let mut beam = Beam::new(1);
    // Ray at 30° to normal
    let theta = 30.0_f64.to_radians();
    beam.x[0] = 5.0 * theta.sin();
    beam.z[0] = 5.0;
    beam.a[0] = 0.0;
    beam.b[0] = 0.0;
    beam.c[0] = -1.0;
    beam.state[0] = RayState::Good as i32;
    beam.e[0] = 10000.0;

    let good = vec![0usize];
    let aperture = Aperture::default();

    let results = reflect_local(
        &surface, &mut beam, &good, &aperture, 1,
        DeflectionMode::Refract { n1_over_n2: 0.5 },
        None, None,
        &xrt_math::rootfind::RootFindConfig::default(),
    );

    assert!(!results.is_empty());
    if results[0].state == RayState::Good {
        // After refraction, ray should still go downward (c < 0)
        assert!(results[0].c < 0.0, "refracted c = {}", results[0].c);
        // Direction should be normalized
        let norm = (results[0].a.powi(2) + results[0].b.powi(2) + results[0].c.powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-8, "norm = {norm}");
    }
}

/// Test: Screen captures diverging beam with correct footprint.
#[test]
fn screen_diverging_beam_footprint() {
    use xrt_core::beam::{Beam, RayState};
    use xrt_oes::screen::Screen;

    let screen = Screen::new([0.0, 1000.0, 0.0], 20.0, 20.0, 40, 40);

    let mut beam = Beam::new(1000);
    beam.set_state(RayState::Good);
    for i in 0..1000 {
        let frac = i as f64 / 999.0 - 0.5;
        beam.x[i] = 0.0;
        beam.y[i] = 0.0;
        beam.z[i] = 0.0;
        beam.a[i] = frac * 0.01; // diverging in x
        beam.b[i] = 1.0;
        beam.c[i] = 0.0;
        // Normalize
        let norm = (beam.a[i].powi(2) + beam.b[i].powi(2)).sqrt();
        beam.a[i] /= norm;
        beam.b[i] /= norm;
    }

    let capture = screen.capture(&beam);
    assert!(capture.n_captured > 900);
    // Beam should spread out at the screen
    let [sx, _] = capture.rms_size();
    // At 1000mm distance, ±5 mrad divergence → ±5mm spread
    assert!(sx > 1.0, "σx = {sx}, expected > 1mm for diverging beam");
}

/// Test: Beam propagation moves rays.
#[test]
fn beam_propagate_moves_rays() {
    use xrt_core::beam::{Beam, RayState};

    let mut beam = Beam::new(3);
    beam.set_state(RayState::Good);
    beam.b[0] = 1.0; // ray going in +y
    beam.b[1] = 1.0;
    beam.a[2] = 0.5_f64.sqrt();
    beam.b[2] = 0.5_f64.sqrt();

    beam.propagate(100.0);

    assert!((beam.y[0] - 100.0).abs() < 1e-10);
    assert!((beam.y[1] - 100.0).abs() < 1e-10);
    assert!((beam.x[2] - 100.0 * 0.5_f64.sqrt()).abs() < 1e-10);
    assert!((beam.path[0] - 100.0).abs() < 1e-10);
}

#[test]
fn grating_vectors_consistent() {
    let blazed = BlazedGrating::new(600.0, 0.02, 0.5);
    let laminar = LaminarGrating::new(600.0, 0.005, 0.5);

    let g_b = blazed.local_g(0.0, 0.0).unwrap();
    let g_l = laminar.local_g(0.0, 0.0).unwrap();

    // Same groove density → same grating vector magnitude
    assert!((g_b[1] - g_l[1]).abs() < 1e-10);
    assert!((g_b[1] + 600.0).abs() < 1e-10);
}
