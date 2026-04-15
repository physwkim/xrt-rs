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
    let text =
        std::fs::read_to_string(path).expect("Run `python validation/generate_fixtures.py` first");
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

fn test_surface_numerical<S: Surface>(
    surface: &S,
    fix: &serde_json::Value,
    surf_name: &str,
    tol: f64,
) {
    let rays = fix["rays"].as_array().unwrap();
    let intersections = fix["surfaces"][surf_name]["intersections"]
        .as_array()
        .unwrap();
    let config = RootFindConfig::default();

    for (i, (ray, expected)) in rays.iter().zip(intersections.iter()).enumerate() {
        if expected.is_null() {
            continue;
        }
        let x0 = ray["x"].as_f64().unwrap();
        let y0 = ray["y"].as_f64().unwrap();
        let z0 = ray["z"].as_f64().unwrap();
        let a = ray["a"].as_f64().unwrap();
        let b = ray["b"].as_f64().unwrap();
        let c = ray["c"].as_f64().unwrap();
        let t1 = ray["t1"].as_f64().unwrap();
        let t2 = ray["t2"].as_f64().unwrap();

        let result = find_intersection_surface(surface, t1, t2, x0, y0, z0, a, b, c, 1, &config);
        assert!(result.converged, "{surf_name} ray{i}: did not converge");

        let exp_t = expected["t"].as_f64().unwrap();
        let exp_x = expected["x"].as_f64().unwrap();
        let exp_y = expected["y"].as_f64().unwrap();
        let exp_z = expected["z"].as_f64().unwrap();

        assert!(
            (result.t - exp_t).abs() < tol,
            "{surf_name} ray{i} t: {:.10e} != {exp_t:.10e}",
            result.t
        );
        assert!(
            (result.x - exp_x).abs() < tol,
            "{surf_name} ray{i} x: {:.10e} != {exp_x:.10e}",
            result.x
        );
        assert!(
            (result.y - exp_y).abs() < tol,
            "{surf_name} ray{i} y: {:.10e} != {exp_y:.10e}",
            result.y
        );
        assert!(
            (result.z - exp_z).abs() < tol,
            "{surf_name} ray{i} z: {:.10e} != {exp_z:.10e}",
            result.z
        );
    }
}

#[test]
fn golden_intersection_flat_numerical() {
    let fix = load_fixture();
    test_surface_numerical(&FlatSurface, &fix, "flat", 1e-8);
}

#[test]
fn golden_intersection_toroid_numerical() {
    let fix = load_fixture();
    test_surface_numerical(&ToroidSurface::new(5e6, 50.0), &fix, "toroid", 1e-8);
}

#[test]
fn golden_intersection_spherical_numerical() {
    let fix = load_fixture();
    test_surface_numerical(&SphericalSurface::new(1000.0), &fix, "spherical", 1e-8);
}

#[test]
fn golden_intersection_paraboloid_lens_numerical() {
    let fix = load_fixture();
    test_surface_numerical(
        &ParaboloidLensSurface::new(100.0, None),
        &fix,
        "paraboloid_lens",
        1e-8,
    );
}
