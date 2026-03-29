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
    let text = std::fs::read_to_string(path)
        .expect("Run `python validation/generate_fixtures.py` first");
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
        "spherical" => Box::new(SphericalSurface::new(
            params["R"].as_f64().unwrap(),
        )),
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
