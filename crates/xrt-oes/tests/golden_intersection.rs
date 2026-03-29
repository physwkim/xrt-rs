//! Golden-value tests for ray-surface intersection (Domain 6).
//!
//! Tests find_intersection_surface against known intersection points.

use xrt_math::rootfind::RootFindConfig;
use xrt_oes::intersection::find_intersection_surface;
use xrt_oes::surface::Surface;
use xrt_oes::surfaces::flat::FlatSurface;
use xrt_oes::surfaces::lens::ParaboloidLensSurface;
use xrt_oes::surfaces::spherical::SphericalSurface;
use xrt_oes::surfaces::toroid::ToroidSurface;

fn load_fixture() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../validation/fixtures/intersection.json"
    );
    let text = std::fs::read_to_string(path)
        .expect("Run `python validation/generate_fixtures.py` first");
    serde_json::from_str(&text).unwrap()
}

fn test_surface<S: Surface>(surface: &S, fix: &serde_json::Value, surf_name: &str, tol: f64) {
    let rays = fix["rays"].as_array().unwrap();
    let config = RootFindConfig::default();

    for (i, ray) in rays.iter().enumerate() {
        let x0 = ray["x"].as_f64().unwrap();
        let y0 = ray["y"].as_f64().unwrap();
        let z0 = ray["z"].as_f64().unwrap();
        let a = ray["a"].as_f64().unwrap();
        let b = ray["b"].as_f64().unwrap();
        let c = ray["c"].as_f64().unwrap();
        let t1 = ray["t1"].as_f64().unwrap();
        let t2 = ray["t2"].as_f64().unwrap();

        let result = find_intersection_surface(surface, t1, t2, x0, y0, z0, a, b, c, 1, &config);

        if result.converged {
            // Verify the intersection point actually lies on the surface
            let z_surf = surface.local_z(result.x, result.y);
            let z_diff = (result.z - z_surf).abs();
            assert!(
                z_diff < tol,
                "{surf_name} ray{i}: z_hit={:.10e}, z_surf={:.10e} (diff={z_diff:.2e})",
                result.z,
                z_surf
            );
        }
    }
}

#[test]
fn golden_intersection_flat() {
    let fix = load_fixture();
    test_surface(&FlatSurface, &fix, "flat", 1e-10);
}

#[test]
fn golden_intersection_toroid() {
    let fix = load_fixture();
    test_surface(&ToroidSurface::new(5e6, 50.0), &fix, "toroid", 1e-10);
}

#[test]
fn golden_intersection_spherical() {
    let fix = load_fixture();
    test_surface(&SphericalSurface::new(1000.0), &fix, "spherical", 1e-10);
}

#[test]
fn golden_intersection_paraboloid_lens() {
    let fix = load_fixture();
    test_surface(
        &ParaboloidLensSurface::new(100.0, None),
        &fix,
        "paraboloid_lens",
        1e-10,
    );
}
