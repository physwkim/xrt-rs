//! Golden-value tests for surface geometry (Domain 5).
//!
//! Tests local_z and local_n for all surface types against Python XRT fixtures.

use xrt_oes::surface::Surface;
use xrt_oes::surfaces::flat::FlatSurface;
use xrt_oes::surfaces::grating::{BlazedGrating, LaminarGrating};
use xrt_oes::surfaces::lens::ParaboloidLensSurface;
use xrt_oes::surfaces::spherical::SphericalSurface;
use xrt_oes::surfaces::toroid::ToroidSurface;

fn load_fixture() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../validation/fixtures/surfaces.json"
    );
    let text =
        std::fs::read_to_string(path).expect("Run `python validation/generate_fixtures.py` first");
    serde_json::from_str(&text).unwrap()
}

fn make_surface(surf: &serde_json::Value) -> Box<dyn Surface> {
    let typ = surf["type"].as_str().unwrap();
    let params = &surf["params"];
    match typ {
        "flat" => Box::new(FlatSurface),
        "toroid" => Box::new(ToroidSurface::new(
            params["R"].as_f64().unwrap(),
            params["r"].as_f64().unwrap(),
        )),
        "spherical" => Box::new(SphericalSurface::new(params["R"].as_f64().unwrap())),
        "paraboloid_lens" => Box::new(ParaboloidLensSurface::new(
            params["focus"].as_f64().unwrap(),
            None,
        )),
        "blazed_grating" => Box::new(BlazedGrating::new(
            params["rho"].as_f64().unwrap(),
            params["blaze"].as_f64().unwrap(),
            params["antiBlaze"].as_f64().unwrap(),
        )),
        "laminar_grating" => Box::new(LaminarGrating::new(
            params["rho"].as_f64().unwrap(),
            params["depth"].as_f64().unwrap(),
            params["duty"].as_f64().unwrap(),
        )),
        other => panic!("unknown surface type: {other}"),
    }
}

#[test]
fn golden_surfaces_z() {
    let fix = load_fixture();
    let tol = 1e-12;

    for surf_data in fix["surfaces"].as_array().unwrap() {
        let typ = surf_data["type"].as_str().unwrap();
        let surface = make_surface(surf_data);

        for pt in surf_data["points"].as_array().unwrap() {
            let x = pt["x"].as_f64().unwrap();
            let y = pt["y"].as_f64().unwrap();
            let expected_z = pt["z"].as_f64().unwrap();

            let actual_z = surface.local_z(x, y);
            let diff = (actual_z - expected_z).abs();
            assert!(
                diff < tol,
                "{typ} local_z({x},{y}): rust={actual_z:.15e} != python={expected_z:.15e} (diff={diff:.2e})"
            );
        }
    }
}

#[test]
fn golden_surfaces_n() {
    let fix = load_fixture();
    let tol = 1e-12;

    for surf_data in fix["surfaces"].as_array().unwrap() {
        let typ = surf_data["type"].as_str().unwrap();
        let surface = make_surface(surf_data);

        for pt in surf_data["points"].as_array().unwrap() {
            let x = pt["x"].as_f64().unwrap();
            let y = pt["y"].as_f64().unwrap();
            let exp_nx = pt["nx"].as_f64().unwrap();
            let exp_ny = pt["ny"].as_f64().unwrap();
            let exp_nz = pt["nz"].as_f64().unwrap();

            let [nx, ny, nz] = surface.local_n(x, y);
            let dx = (nx - exp_nx).abs();
            let dy = (ny - exp_ny).abs();
            let dz = (nz - exp_nz).abs();

            assert!(
                dx < tol && dy < tol && dz < tol,
                "{typ} local_n({x},{y}): rust=[{nx:.15e},{ny:.15e},{nz:.15e}] != python=[{exp_nx:.15e},{exp_ny:.15e},{exp_nz:.15e}] (d=[{dx:.2e},{dy:.2e},{dz:.2e}])"
            );
        }
    }
}

#[test]
fn golden_blazed_non_boundary() {
    let fix = load_fixture();
    let pts = fix["blazed_non_boundary"].as_array().unwrap();
    let surface = BlazedGrating::new(600.0, 0.02, 0.5);

    for pt in pts {
        let x = pt["x"].as_f64().unwrap();
        let y = pt["y"].as_f64().unwrap();
        let exp_z = pt["z"].as_f64().unwrap();
        let exp_nx = pt["nx"].as_f64().unwrap();
        let exp_ny = pt["ny"].as_f64().unwrap();
        let exp_nz = pt["nz"].as_f64().unwrap();

        let z = surface.local_z(x, y);
        let [nx, ny, nz] = surface.local_n(x, y);

        let tol = 1e-12;
        assert!(
            (z - exp_z).abs() < tol,
            "blazed z({x},{y}): {z:.15e} != {exp_z:.15e}"
        );
        assert!(
            (nx - exp_nx).abs() < tol,
            "blazed nx({x},{y}): {nx:.15e} != {exp_nx:.15e}"
        );
        assert!(
            (ny - exp_ny).abs() < tol,
            "blazed ny({x},{y}): {ny:.15e} != {exp_ny:.15e}"
        );
        assert!(
            (nz - exp_nz).abs() < tol,
            "blazed nz({x},{y}): {nz:.15e} != {exp_nz:.15e}"
        );
    }
}

#[test]
fn golden_toroid_sagittal_boundary() {
    // Test toroid near x = r (sagittal radius boundary)
    let surface = ToroidSurface::new(5e6, 50.0);

    // x slightly less than r: should be valid
    let z = surface.local_z(49.9, 0.0);
    assert!(z.is_finite(), "z at x=49.9 should be finite");

    let [nx, ny, nz] = surface.local_n(49.9, 0.0);
    assert!(
        nx.is_finite() && ny.is_finite() && nz.is_finite(),
        "normal at x=49.9 should be finite"
    );

    // x = r exactly: may give large derivative
    let z_edge = surface.local_z(50.0, 0.0);
    assert!(z_edge.is_finite(), "z at x=r should be finite");
}

#[test]
fn golden_spherical_radius_boundary() {
    let surface = SphericalSurface::new(1000.0);

    // Near the boundary rho = R
    let z = surface.local_z(999.0, 0.0);
    assert!(z.is_finite(), "z near boundary should be finite");

    // Beyond boundary rho > R: should return R (clamped)
    let z_beyond = surface.local_z(1001.0, 0.0);
    assert!(z_beyond.is_finite(), "z beyond boundary should be finite");
    assert!(
        (z_beyond - 1000.0).abs() < 1e-6,
        "z beyond boundary should be R, got {z_beyond}"
    );
}
