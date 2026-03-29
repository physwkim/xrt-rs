//! Golden-value tests for crystal diffraction (Domain 4).
//!
//! Tests Bragg angle and crystal amplitude against Python XRT reference.

use ndarray::{array, Array1};

use xrt_materials::crystal::CrystalGeometry;
use xrt_materials::crystal_variants::CrystalSi;
use xrt_materials::data::ScatteringTable;

fn load_fixture(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/../../validation/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Run generate_fixtures.py first: {path}"));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn golden_si111_bragg_angle() {
    let fix = load_fixture("crystal_si111.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().contains("bragg_angle"))
        .unwrap();

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

    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = tc["theta_b_rad"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let theta = si.base.get_bragg_angle(&e_arr);

    for (i, energy) in energies.iter().enumerate() {
        let diff = (theta[i] - expected[i]).abs();
        assert!(
            diff < 1e-12,
            "theta_B(E={energy}): {:.15e} != {:.15e} (diff={diff:.2e})",
            theta[i],
            expected[i]
        );
    }
}

#[test]
fn golden_si220_bragg_angle() {
    let fix = load_fixture("crystal_si220.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().contains("bragg_angle"))
        .unwrap();

    let si = CrystalSi::new(
        [2, 2, 0],
        297.15,
        CrystalGeometry::BraggReflected,
        1.0,
        None,
        0.0,
        ScatteringTable::ChantlerTotal,
    )
    .unwrap();

    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = tc["theta_b_rad"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let theta = si.base.get_bragg_angle(&e_arr);

    for (i, energy) in energies.iter().enumerate() {
        let diff = (theta[i] - expected[i]).abs();
        assert!(
            diff < 1e-12,
            "theta_B(E={energy}): {:.15e} != {:.15e} (diff={diff:.2e})",
            theta[i],
            expected[i]
        );
    }
}

#[test]
fn golden_si111_amplitude() {
    let fix = load_fixture("crystal_si111.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().contains("amplitude"))
        .unwrap();

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

    let energy = tc["energy_ev"].as_f64().unwrap();
    let bidn = tc["beam_in_dot_normal"].as_f64().unwrap();
    let e_arr = array![energy];
    let bidn_arr = array![bidn];

    let (rs, rp) = si
        .base
        .get_amplitude(&e_arr, &bidn_arr, None, None, &si)
        .unwrap();

    let exp_rs_re = tc["rs_real"].as_f64().unwrap();
    let exp_rs_im = tc["rs_imag"].as_f64().unwrap();
    let exp_rp_re = tc["rp_real"].as_f64().unwrap();
    let exp_rp_im = tc["rp_imag"].as_f64().unwrap();

    let tol = 1e-6;
    assert!(
        (rs[0].re - exp_rs_re).abs() < tol,
        "rs.re: {:.10e} != {exp_rs_re:.10e}",
        rs[0].re
    );
    assert!(
        (rs[0].im - exp_rs_im).abs() < tol,
        "rs.im: {:.10e} != {exp_rs_im:.10e}",
        rs[0].im
    );
    assert!(
        (rp[0].re - exp_rp_re).abs() < tol,
        "rp.re: {:.10e} != {exp_rp_re:.10e}",
        rp[0].re
    );
    assert!(
        (rp[0].im - exp_rp_im).abs() < tol,
        "rp.im: {:.10e} != {exp_rp_im:.10e}",
        rp[0].im
    );
}
