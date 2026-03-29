//! End-to-end pipeline integration tests.
//!
//! Tests the full workflow: source → OE → diffraction screen.

use num_complex::Complex64;

use xrt_core::beam::RayState;
use xrt_sources::distributions::{EnergyDist, SpatialDist};
use xrt_sources::geometric::GeometricSource;
use xrt_waves::diffraction::{DiffractionRay, PixelPoint, diffraction_integral};
use xrt_waves::prepare_wave::{WaveParams, prepare_wave, beam_to_diffraction_rays};

/// Test: GeometricSource → flat mirror → diffraction screen.
#[test]
fn source_to_flat_mirror_to_screen() {
    // 1. Generate rays from a geometric source
    let source = GeometricSource {
        nrays: 200,
        dist_x: SpatialDist::Normal(0.1),
        dist_z: SpatialDist::Normal(0.1),
        dist_xprime: SpatialDist::Normal(1e-4),
        dist_zprime: SpatialDist::Normal(1e-4),
        dist_e: EnergyDist::Lines(vec![10000.0], None),
        ..Default::default()
    };
    let beam = source.shine();
    assert_eq!(beam.nrays(), 200);

    // 2. Create "reflected" rays on a flat mirror at y=1000
    let n = beam.nrays();
    let normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]; n];
    let good: Vec<usize> = (0..n)
        .filter(|&i| beam.state[i] == RayState::Good as i32)
        .collect();

    let rays = beam_to_diffraction_rays(&beam, &normals, &good);
    assert!(!rays.is_empty());

    // 3. Create diffraction screen
    let params = WaveParams::new([0.0, 2000.0, 0.0], 1.0, 1.0, 5, 5);
    let pixels = prepare_wave(&params);
    assert_eq!(pixels.len(), 25);

    // 4. Compute diffraction integral
    let results = diffraction_integral(&rays, &pixels);
    assert_eq!(results.len(), 25);

    // Should have non-zero field at center
    let center = &results[12]; // center pixel in 5x5 grid
    assert!(
        center.es.norm() > 0.0 || center.ep.norm() > 0.0,
        "zero field at screen center"
    );
}

/// Test: Manually constructed rays → diffraction with symmetry check.
#[test]
fn diffraction_symmetry() {
    // Symmetric source → symmetric diffraction pattern
    let n_rays = 51; // odd number for exact symmetry
    let mut rays = Vec::with_capacity(n_rays);
    for i in 0..n_rays {
        let x = (i as f64 - 25.0) * 0.02; // symmetric about x=0
        rays.push(DiffractionRay {
            x,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            nl: 1.0,
            es: Complex64::new(1.0, 0.0),
            ep: Complex64::new(0.0, 0.0),
            energy: 10000.0,
        });
    }

    // Symmetric pixel positions
    let pixels = vec![
        PixelPoint { x: -0.5, y: 1000.0, z: 0.0 },
        PixelPoint { x: 0.0, y: 1000.0, z: 0.0 },
        PixelPoint { x: 0.5, y: 1000.0, z: 0.0 },
    ];

    let results = diffraction_integral(&rays, &pixels);

    // |Es| at x=-0.5 should approximately equal |Es| at x=+0.5
    let left_intensity = results[0].es.norm_sqr();
    let right_intensity = results[2].es.norm_sqr();
    let ratio = if left_intensity > right_intensity {
        right_intensity / left_intensity
    } else {
        left_intensity / right_intensity
    };
    assert!(
        ratio > 0.9,
        "asymmetric: left={left_intensity}, right={right_intensity}"
    );

    // Center should have highest intensity
    let center_intensity = results[1].es.norm_sqr();
    assert!(center_intensity >= left_intensity * 0.9);
}

/// Test: Different source types can all generate beams.
#[test]
fn all_source_types_generate_beams() {
    // Geometric source
    let geo = GeometricSource {
        nrays: 50,
        ..Default::default()
    };
    let beam_geo = geo.shine();
    assert_eq!(beam_geo.nrays(), 50);
    assert!(beam_geo.state.iter().all(|&s| s == RayState::Good as i32));

    // BendingMagnet source
    let mut bm = xrt_sources::bending_magnet::BendingMagnet::new(
        3.0, 0.3, 1.0, 50, 5000.0, 15000.0, 0.002, 0.002,
    );
    let beam_bm = bm.shine();
    assert_eq!(beam_bm.nrays(), 50);

    // Wiggler source
    let mut wig = xrt_sources::wiggler::Wiggler::new(
        3.0, 0.3, 10.0, 80.0, 20, 50, 5000.0, 15000.0, 0.003, 0.003,
    );
    let beam_wig = wig.shine();
    assert_eq!(beam_wig.nrays(), 50);

    // Undulator source
    let mut und = xrt_sources::undulator::Undulator::new(
        3.0, 0.3, 0.0, 2.0, 30.0, 50, 50, 5000.0, 15000.0, 0.001, 0.001,
    );
    let beam_und = und.shine();
    assert_eq!(beam_und.nrays(), 50);

    // All beams should have normalized directions
    for (name, beam) in [
        ("geometric", &beam_geo),
        ("bm", &beam_bm),
        ("wiggler", &beam_wig),
        ("undulator", &beam_und),
    ] {
        for i in 0..beam.nrays() {
            let norm = (beam.a[i].powi(2) + beam.b[i].powi(2) + beam.c[i].powi(2)).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-8,
                "{name} ray {i}: direction norm = {norm}"
            );
        }
    }
}

/// Test: prepare_wave generates correct grid.
#[test]
fn prepare_wave_grid_properties() {
    let params = WaveParams::new([10.0, 5000.0, -2.0], 3.0, 1.5, 21, 11);
    let pixels = prepare_wave(&params);

    assert_eq!(pixels.len(), 21 * 11);

    // All y values should be at screen distance
    for p in &pixels {
        assert!((p.y - 5000.0).abs() < 1e-10);
    }

    // Check x range: center ± dx
    let x_vals: Vec<f64> = pixels.iter().map(|p| p.x).collect();
    let x_min = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((x_min - 7.0).abs() < 1e-10, "x_min = {x_min}"); // 10 - 3
    assert!((x_max - 13.0).abs() < 1e-10, "x_max = {x_max}"); // 10 + 3

    // Check z range
    let z_vals: Vec<f64> = pixels.iter().map(|p| p.z).collect();
    let z_min = z_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let z_max = z_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((z_min - (-3.5)).abs() < 1e-10, "z_min = {z_min}"); // -2 - 1.5
    assert!((z_max - (-0.5)).abs() < 1e-10, "z_max = {z_max}"); // -2 + 1.5
}

/// Test: BendingMagnet → diffraction screen produces finite results.
#[test]
fn bm_source_to_diffraction() {
    let mut bm = xrt_sources::bending_magnet::BendingMagnet::new(
        3.0, 0.3, 1.0, 100, 8000.0, 12000.0, 0.001, 0.001,
    );
    let beam = bm.shine();

    // Convert beam to diffraction rays (assuming flat mirror reflection)
    let normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]; beam.nrays()];
    let good: Vec<usize> = (0..beam.nrays()).collect();
    let rays = beam_to_diffraction_rays(&beam, &normals, &good);

    // Create screen
    let pixels = vec![
        PixelPoint { x: 0.0, y: 1000.0, z: 0.0 },
    ];
    let results = diffraction_integral(&rays, &pixels);

    assert_eq!(results.len(), 1);
    assert!(
        results[0].es.norm().is_finite(),
        "Es = {}",
        results[0].es
    );
}
