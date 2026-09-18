//! Golden-value tests for material optics (Domains 2-3).
//!
//! Tests scattering factors (f0, f1/f2), refractive index, and absorption
//! against Python XRT reference values in JSON fixtures.

use ndarray::{Array1, array};

use xrt_materials::data::ScatteringTable;
use xrt_materials::elements::Element;
use xrt_materials::material::{Material, MaterialKind};

fn load_fixture(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/../../validation/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Run generate_fixtures.py first: {path}"));
    serde_json::from_str(&text).unwrap()
}

// ── Domain 2: Scattering ────────────────────────────────────────────────────

#[test]
fn golden_f0_si() {
    let fix = load_fixture("scattering_si.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_f0"))
        .unwrap();

    let elem = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
    let qs: Vec<f64> = tc["q_over_4pi"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = tc["f0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    for (q, exp) in qs.iter().zip(expected.iter()) {
        let actual = elem.get_f0_scalar(*q);
        let diff = (actual - exp).abs();
        assert!(
            diff < 1e-14,
            "f0(q={q}): rust={actual:.15e} != python={exp:.15e} (diff={diff:.2e})"
        );
    }
}

#[test]
fn golden_f0_au() {
    let fix = load_fixture("scattering_au.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_f0"))
        .unwrap();

    let elem = Element::new("Au", ScatteringTable::ChantlerTotal).unwrap();
    let qs: Vec<f64> = tc["q_over_4pi"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = tc["f0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    for (q, exp) in qs.iter().zip(expected.iter()) {
        let actual = elem.get_f0_scalar(*q);
        let diff = (actual - exp).abs();
        assert!(
            diff < 1e-14,
            "f0(q={q}): rust={actual:.15e} != python={exp:.15e} (diff={diff:.2e})"
        );
    }
}

#[test]
fn golden_f1f2_si() {
    let fix = load_fixture("scattering_si.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_f1f2"))
        .unwrap();

    let elem = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f1: Vec<f64> = tc["f1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f2: Vec<f64> = tc["f2"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let f1f2 = elem.get_f1f2(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff_f1 = (f1f2[i].re - exp_f1[i]).abs();
        let diff_f2 = (f1f2[i].im - exp_f2[i]).abs();
        assert!(
            diff_f1 < 1e-12,
            "f1(E={energy}): {:.15e} != {:.15e} (diff={diff_f1:.2e})",
            f1f2[i].re,
            exp_f1[i]
        );
        assert!(
            diff_f2 < 1e-12,
            "f2(E={energy}): {:.15e} != {:.15e} (diff={diff_f2:.2e})",
            f1f2[i].im,
            exp_f2[i]
        );
    }
}

// ── Domain 3: Material optics ───────────────────────────────────────────────

fn make_material(fix: &serde_json::Value) -> Material {
    let elems: Vec<&str> = fix["elements"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let quantities: Vec<f64> = fix["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let rho = fix["rho"].as_f64().unwrap();
    Material::new(
        &elems,
        Some(&quantities),
        rho,
        MaterialKind::Mirror,
        None,
        ScatteringTable::ChantlerTotal,
    )
    .unwrap()
}

#[test]
fn golden_refractive_index_si() {
    let fix = load_fixture("material_si.json");
    let mat = make_material(&fix);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_n"))
        .unwrap();

    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_re: Vec<f64> = tc["n_real"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_im: Vec<f64> = tc["n_imag"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let n = mat.get_refractive_index(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff_re = (n[i].re - exp_re[i]).abs();
        let diff_im = (n[i].im - exp_im[i]).abs();
        assert!(
            diff_re < 1e-10,
            "n.re(E={energy}): {:.17e} != {:.17e} (diff={diff_re:.2e})",
            n[i].re,
            exp_re[i]
        );
        assert!(
            diff_im < 1e-10,
            "n.im(E={energy}): {:.17e} != {:.17e} (diff={diff_im:.2e})",
            n[i].im,
            exp_im[i]
        );
    }
}

#[test]
fn golden_absorption_si() {
    let fix = load_fixture("material_si.json");
    let mat = make_material(&fix);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_mu"))
        .unwrap();

    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_mu: Vec<f64> = tc["mu_cm_inv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let mu = mat.get_absorption_coefficient(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff = (mu[i] - exp_mu[i]).abs();
        assert!(
            diff < 1e-6,
            "mu(E={energy}): {:.10e} != {:.10e} (diff={diff:.2e})",
            mu[i],
            exp_mu[i]
        );
    }
}

#[test]
fn golden_fresnel_si() {
    let fix = load_fixture("material_si.json");
    let mat = make_material(&fix);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_fresnel"))
        .unwrap();

    let e_arr = array![tc["energy_ev"].as_f64().unwrap()];

    for amp in tc["amplitudes"].as_array().unwrap() {
        let sin_theta = amp["sin_theta"].as_f64().unwrap();
        let bidn = array![sin_theta];
        let result = mat.get_amplitude(&e_arr, &bidn, true).unwrap();

        let exp_rs_re = amp["rs_real"].as_f64().unwrap();
        let exp_rs_im = amp["rs_imag"].as_f64().unwrap();
        let exp_rp_re = amp["rp_real"].as_f64().unwrap();
        let exp_rp_im = amp["rp_imag"].as_f64().unwrap();

        let tol = 1e-8;
        let d_rs_re = (result.rs[0].re - exp_rs_re).abs();
        let d_rs_im = (result.rs[0].im - exp_rs_im).abs();
        let d_rp_re = (result.rp[0].re - exp_rp_re).abs();
        let d_rp_im = (result.rp[0].im - exp_rp_im).abs();

        assert!(
            d_rs_re < tol,
            "rs.re(θ={sin_theta}): {:.10e} != {exp_rs_re:.10e} (diff={d_rs_re:.2e})",
            result.rs[0].re
        );
        assert!(
            d_rs_im < tol,
            "rs.im(θ={sin_theta}): {:.10e} != {exp_rs_im:.10e} (diff={d_rs_im:.2e})",
            result.rs[0].im
        );
        assert!(
            d_rp_re < tol,
            "rp.re(θ={sin_theta}): {:.10e} != {exp_rp_re:.10e} (diff={d_rp_re:.2e})",
            result.rp[0].re
        );
        assert!(
            d_rp_im < tol,
            "rp.im(θ={sin_theta}): {:.10e} != {exp_rp_im:.10e} (diff={d_rp_im:.2e})",
            result.rp[0].im
        );
    }
}

#[test]
fn golden_f1f2_au() {
    let fix = load_fixture("scattering_au.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_f1f2"))
        .unwrap();

    let elem = Element::new("Au", ScatteringTable::ChantlerTotal).unwrap();
    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f1: Vec<f64> = tc["f1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f2: Vec<f64> = tc["f2"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let f1f2 = elem.get_f1f2(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff_f1 = (f1f2[i].re - exp_f1[i]).abs();
        let diff_f2 = (f1f2[i].im - exp_f2[i]).abs();
        assert!(
            diff_f1 < 1e-12,
            "Au f1(E={energy}): {:.15e} != {:.15e} (diff={diff_f1:.2e})",
            f1f2[i].re,
            exp_f1[i]
        );
        assert!(
            diff_f2 < 1e-12,
            "Au f2(E={energy}): {:.15e} != {:.15e} (diff={diff_f2:.2e})",
            f1f2[i].im,
            exp_f2[i]
        );
    }
}

// ── Scattering: W, O ──────────────────────────────────────────────────────

fn test_f0(fixture_name: &str, elem_name: &str) {
    let fix = load_fixture(fixture_name);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_f0"))
        .unwrap();

    let elem = Element::new(elem_name, ScatteringTable::ChantlerTotal).unwrap();
    let qs: Vec<f64> = tc["q_over_4pi"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = tc["f0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    for (q, exp) in qs.iter().zip(expected.iter()) {
        let actual = elem.get_f0_scalar(*q);
        let diff = (actual - exp).abs();
        // Use relative tolerance: 1e-14 for small values, scale with magnitude
        let tol = 1e-14_f64.max(exp.abs() * 1e-14);
        assert!(
            diff < tol,
            "{elem_name} f0(q={q}): {actual:.15e} != {exp:.15e} (diff={diff:.2e}, tol={tol:.2e})"
        );
    }
}

fn test_f1f2(fixture_name: &str, elem_name: &str) {
    let fix = load_fixture(fixture_name);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_f1f2"))
        .unwrap();

    let elem = Element::new(elem_name, ScatteringTable::ChantlerTotal).unwrap();
    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f1: Vec<f64> = tc["f1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f2: Vec<f64> = tc["f2"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let f1f2 = elem.get_f1f2(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff_f1 = (f1f2[i].re - exp_f1[i]).abs();
        let diff_f2 = (f1f2[i].im - exp_f2[i]).abs();
        assert!(
            diff_f1 < 1e-12,
            "{elem_name} f1(E={energy}): {:.15e} != {:.15e} (diff={diff_f1:.2e})",
            f1f2[i].re,
            exp_f1[i]
        );
        assert!(
            diff_f2 < 1e-12,
            "{elem_name} f2(E={energy}): {:.15e} != {:.15e} (diff={diff_f2:.2e})",
            f1f2[i].im,
            exp_f2[i]
        );
    }
}

#[test]
fn golden_f0_w() {
    test_f0("scattering_w.json", "W");
}

#[test]
fn golden_f0_o() {
    test_f0("scattering_o.json", "O");
}

#[test]
fn golden_f1f2_w() {
    test_f1f2("scattering_w.json", "W");
}

#[test]
fn golden_f1f2_o() {
    test_f1f2("scattering_o.json", "O");
}

#[test]
fn golden_f1f2_edge_si() {
    let fix = load_fixture("scattering_si.json");
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().contains("edge"))
        .unwrap();

    let elem = Element::new("Si", ScatteringTable::ChantlerTotal).unwrap();
    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f1: Vec<f64> = tc["f1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_f2: Vec<f64> = tc["f2"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let f1f2 = elem.get_f1f2(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff_f1 = (f1f2[i].re - exp_f1[i]).abs();
        let diff_f2 = (f1f2[i].im - exp_f2[i]).abs();
        assert!(
            diff_f1 < 1e-12,
            "Si edge f1(E={energy}): {:.15e} != {:.15e} (diff={diff_f1:.2e})",
            f1f2[i].re,
            exp_f1[i]
        );
        assert!(
            diff_f2 < 1e-12,
            "Si edge f2(E={energy}): {:.15e} != {:.15e} (diff={diff_f2:.2e})",
            f1f2[i].im,
            exp_f2[i]
        );
    }
}

// ── Material optics: Au, SiO2 ─────────────────────────────────────────────

fn test_refractive_index(fixture_name: &str) {
    let fix = load_fixture(fixture_name);
    let mat = make_material(&fix);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_n"))
        .unwrap();

    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_re: Vec<f64> = tc["n_real"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_im: Vec<f64> = tc["n_imag"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let n = mat.get_refractive_index(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff_re = (n[i].re - exp_re[i]).abs();
        let diff_im = (n[i].im - exp_im[i]).abs();
        assert!(
            diff_re < 1e-10,
            "n.re(E={energy}): {:.17e} != {:.17e} (diff={diff_re:.2e})",
            n[i].re,
            exp_re[i]
        );
        assert!(
            diff_im < 1e-10,
            "n.im(E={energy}): {:.17e} != {:.17e} (diff={diff_im:.2e})",
            n[i].im,
            exp_im[i]
        );
    }
}

fn test_absorption(fixture_name: &str) {
    let fix = load_fixture(fixture_name);
    let mat = make_material(&fix);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_mu"))
        .unwrap();

    let energies: Vec<f64> = tc["energies_ev"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_mu: Vec<f64> = tc["mu_cm_inv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let e_arr = Array1::from_vec(energies.clone());
    let mu = mat.get_absorption_coefficient(&e_arr).unwrap();

    for (i, energy) in energies.iter().enumerate() {
        let diff = (mu[i] - exp_mu[i]).abs();
        assert!(
            diff < 1e-6,
            "mu(E={energy}): {:.10e} != {:.10e} (diff={diff:.2e})",
            mu[i],
            exp_mu[i]
        );
    }
}

fn test_fresnel(fixture_name: &str) {
    let fix = load_fixture(fixture_name);
    let mat = make_material(&fix);
    let tc = fix["test_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str().unwrap().ends_with("_fresnel"))
        .unwrap();

    let e_arr = array![tc["energy_ev"].as_f64().unwrap()];

    for amp in tc["amplitudes"].as_array().unwrap() {
        let sin_theta = amp["sin_theta"].as_f64().unwrap();
        let bidn = array![sin_theta];
        let result = mat.get_amplitude(&e_arr, &bidn, true).unwrap();

        let tol = 1e-8;
        let d_rs_re = (result.rs[0].re - amp["rs_real"].as_f64().unwrap()).abs();
        let d_rs_im = (result.rs[0].im - amp["rs_imag"].as_f64().unwrap()).abs();
        let d_rp_re = (result.rp[0].re - amp["rp_real"].as_f64().unwrap()).abs();
        let d_rp_im = (result.rp[0].im - amp["rp_imag"].as_f64().unwrap()).abs();

        assert!(d_rs_re < tol, "rs.re(θ={sin_theta}): diff={d_rs_re:.2e}");
        assert!(d_rs_im < tol, "rs.im(θ={sin_theta}): diff={d_rs_im:.2e}");
        assert!(d_rp_re < tol, "rp.re(θ={sin_theta}): diff={d_rp_re:.2e}");
        assert!(d_rp_im < tol, "rp.im(θ={sin_theta}): diff={d_rp_im:.2e}");
    }
}

#[test]
fn golden_refractive_index_au() {
    test_refractive_index("material_au.json");
}

#[test]
fn golden_refractive_index_sio2() {
    test_refractive_index("material_sio2.json");
}

#[test]
fn golden_absorption_au() {
    test_absorption("material_au.json");
}

#[test]
fn golden_absorption_sio2() {
    test_absorption("material_sio2.json");
}

#[test]
fn golden_fresnel_au() {
    test_fresnel("material_au.json");
}

#[test]
fn golden_fresnel_sio2() {
    test_fresnel("material_sio2.json");
}
