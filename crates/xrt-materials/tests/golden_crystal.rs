//! Golden-value tests for crystal diffraction (Domain 4).
//!
//! Tests Bragg angle and crystal amplitude against Python XRT reference.

use ndarray::{array, Array1};
use num_complex::Complex64;

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

#[test]
fn golden_si220_amplitude() {
    let fix = load_fixture("crystal_si220.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().contains("amplitude"))
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

    let energy = tc["energy_ev"].as_f64().unwrap();
    let bidn = tc["beam_in_dot_normal"].as_f64().unwrap();
    let e_arr = array![energy];
    let bidn_arr = array![bidn];

    let (rs, rp) = si
        .base
        .get_amplitude(&e_arr, &bidn_arr, None, None, &si)
        .unwrap();

    let tol = 1e-6;
    assert!(
        (rs[0].re - tc["rs_real"].as_f64().unwrap()).abs() < tol,
        "rs.re: {:.10e} != {:.10e}",
        rs[0].re,
        tc["rs_real"].as_f64().unwrap()
    );
    assert!(
        (rs[0].im - tc["rs_imag"].as_f64().unwrap()).abs() < tol,
        "rs.im: {:.10e} != {:.10e}",
        rs[0].im,
        tc["rs_imag"].as_f64().unwrap()
    );
    assert!(
        (rp[0].re - tc["rp_real"].as_f64().unwrap()).abs() < tol,
        "rp.re: {:.10e} != {:.10e}",
        rp[0].re,
        tc["rp_real"].as_f64().unwrap()
    );
    assert!(
        (rp[0].im - tc["rp_imag"].as_f64().unwrap()).abs() < tol,
        "rp.im: {:.10e} != {:.10e}",
        rp[0].im,
        tc["rp_imag"].as_f64().unwrap()
    );
}

fn test_chi(fixture_name: &str, hkl: [i32; 3]) {
    let fix = load_fixture(fixture_name);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().contains("_chi"))
        .unwrap();

    let si = CrystalSi::new(
        hkl,
        297.15,
        CrystalGeometry::BraggReflected,
        1.0,
        None,
        0.0,
        ScatteringTable::ChantlerTotal,
    )
    .unwrap();

    let tol = 1e-10;
    for chi_val in tc["chi_values"].as_array().unwrap() {
        let energy = chi_val["energy_ev"].as_f64().unwrap();
        let stol = chi_val["stol"].as_f64().unwrap();

        let e_arr = array![energy];
        let stol_arr = array![stol];
        let result = si.base.get_f_chi(&e_arr, &stol_arr, &si).unwrap();

        let exp_chi0 = Complex64::new(
            chi_val["chi0_re"].as_f64().unwrap(),
            chi_val["chi0_im"].as_f64().unwrap(),
        );
        let exp_chih = Complex64::new(
            chi_val["chih_re"].as_f64().unwrap(),
            chi_val["chih_im"].as_f64().unwrap(),
        );
        let exp_chih_bar = Complex64::new(
            chi_val["chih_bar_re"].as_f64().unwrap(),
            chi_val["chih_bar_im"].as_f64().unwrap(),
        );

        assert!(
            (result.chi0[0].re - exp_chi0.re).abs() < tol
                && (result.chi0[0].im - exp_chi0.im).abs() < tol,
            "chi0(E={energy}): {:?} != {:?}",
            result.chi0[0],
            exp_chi0
        );
        assert!(
            (result.chih[0].re - exp_chih.re).abs() < tol
                && (result.chih[0].im - exp_chih.im).abs() < tol,
            "chih(E={energy}): {:?} != {:?}",
            result.chih[0],
            exp_chih
        );
        assert!(
            (result.chih_bar[0].re - exp_chih_bar.re).abs() < tol
                && (result.chih_bar[0].im - exp_chih_bar.im).abs() < tol,
            "chih_bar(E={energy}): {:?} != {:?}",
            result.chih_bar[0],
            exp_chih_bar
        );
    }
}

#[test]
fn golden_si111_chi() {
    test_chi("crystal_si111.json", [1, 1, 1]);
}

#[test]
fn golden_si220_chi() {
    test_chi("crystal_si220.json", [2, 2, 0]);
}

fn test_rocking_mini(fixture_name: &str, hkl: [i32; 3]) {
    let fix = load_fixture(fixture_name);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().contains("rocking_mini"))
        .unwrap();

    let si = CrystalSi::new(
        hkl,
        297.15,
        CrystalGeometry::BraggReflected,
        1.0,
        None,
        0.0,
        ScatteringTable::ChantlerTotal,
    )
    .unwrap();

    let energy = tc["energy_ev"].as_f64().unwrap();
    let e_arr = array![energy];
    let bidn_arr = tc["beam_in_dot_normal"].as_array().unwrap();
    let exp_rs = tc["rs"].as_array().unwrap();
    let exp_rp = tc["rp"].as_array().unwrap();

    let tol = 1e-6;
    for (i, bidn_val) in bidn_arr.iter().enumerate() {
        let bidn = bidn_val.as_f64().unwrap();
        let b_arr = array![bidn];
        let (rs, rp) = si
            .base
            .get_amplitude(&e_arr, &b_arr, None, None, &si)
            .unwrap();

        let exp_rs_re = exp_rs[i]["re"].as_f64().unwrap();
        let exp_rs_im = exp_rs[i]["im"].as_f64().unwrap();
        let exp_rp_re = exp_rp[i]["re"].as_f64().unwrap();
        let exp_rp_im = exp_rp[i]["im"].as_f64().unwrap();

        assert!(
            (rs[0].re - exp_rs_re).abs() < tol && (rs[0].im - exp_rs_im).abs() < tol,
            "rocking[{i}] rs: ({:.6e},{:.6e}) != ({exp_rs_re:.6e},{exp_rs_im:.6e})",
            rs[0].re,
            rs[0].im
        );
        assert!(
            (rp[0].re - exp_rp_re).abs() < tol && (rp[0].im - exp_rp_im).abs() < tol,
            "rocking[{i}] rp: ({:.6e},{:.6e}) != ({exp_rp_re:.6e},{exp_rp_im:.6e})",
            rp[0].re,
            rp[0].im
        );
    }
}

#[test]
fn golden_si111_rocking() {
    test_rocking_mini("crystal_si111.json", [1, 1, 1]);
}

#[test]
fn golden_si220_rocking() {
    test_rocking_mini("crystal_si220.json", [2, 2, 0]);
}
