//! Golden-value tests for Kirchhoff diffraction integral (Domain 8).
//!
//! Tests the diffraction integral with a small deterministic input set.

use num_complex::Complex64;
use xrt_waves::diffraction::{diffraction_integral, DiffractionRay, PixelPoint};

fn load_fixture() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../validation/fixtures/diffraction_ref.json"
    );
    let text = std::fs::read_to_string(path)
        .expect("Run `python validation/generate_fixtures.py` first");
    serde_json::from_str(&text).unwrap()
}

#[test]
fn golden_diffraction_finite_results() {
    let fix = load_fixture();
    let tc = &fix["test_cases"][0];
    let ray_data = &tc["rays"];
    let pix_data = &tc["pixels"];

    let n_rays = ray_data["x"].as_array().unwrap().len();
    let rays: Vec<DiffractionRay> = (0..n_rays)
        .map(|i| DiffractionRay {
            x: ray_data["x"][i].as_f64().unwrap(),
            y: ray_data["y"][i].as_f64().unwrap(),
            z: ray_data["z"][i].as_f64().unwrap(),
            nx: ray_data["nx"][i].as_f64().unwrap(),
            ny: ray_data["ny"][i].as_f64().unwrap(),
            nz: ray_data["nz"][i].as_f64().unwrap(),
            nl: ray_data["nl"][i].as_f64().unwrap(),
            es: Complex64::new(
                ray_data["es_re"][i].as_f64().unwrap(),
                ray_data["es_im"][i].as_f64().unwrap(),
            ),
            ep: Complex64::new(
                ray_data["ep_re"][i].as_f64().unwrap(),
                ray_data["ep_im"][i].as_f64().unwrap(),
            ),
            energy: ray_data["energy"][i].as_f64().unwrap(),
        })
        .collect();

    let n_pix = pix_data["x"].as_array().unwrap().len();
    let pixels: Vec<PixelPoint> = (0..n_pix)
        .map(|i| PixelPoint {
            x: pix_data["x"][i].as_f64().unwrap(),
            y: pix_data["y"][i].as_f64().unwrap(),
            z: pix_data["z"][i].as_f64().unwrap(),
        })
        .collect();

    let results = diffraction_integral(&rays, &pixels);

    assert_eq!(results.len(), n_pix);
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.es.re.is_finite() && r.es.im.is_finite(),
            "pixel[{i}] es = {} is not finite",
            r.es
        );
        assert!(
            r.ep.re.is_finite() && r.ep.im.is_finite(),
            "pixel[{i}] ep = {} is not finite",
            r.ep
        );
        // With non-zero input, output should be non-zero
        assert!(
            r.es.norm() > 1e-30,
            "pixel[{i}] es norm = {} is suspiciously small",
            r.es.norm()
        );
    }
}

#[test]
fn golden_diffraction_symmetry() {
    // Symmetric rays should produce symmetric pixel intensities
    let rays = vec![
        DiffractionRay {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            nl: 1.0,
            es: Complex64::new(1.0, 0.0),
            ep: Complex64::new(1.0, 0.0),
            energy: 10000.0,
        },
    ];

    let pixels = vec![
        PixelPoint { x: -0.01, y: 0.0, z: 1000.0 },
        PixelPoint { x: 0.01, y: 0.0, z: 1000.0 },
    ];

    let results = diffraction_integral(&rays, &pixels);

    let es_diff = (results[0].es.norm() - results[1].es.norm()).abs();
    assert!(
        es_diff < 1e-10,
        "symmetric pixels should have same |es|: {:.10e} vs {:.10e}",
        results[0].es.norm(),
        results[1].es.norm()
    );
}
