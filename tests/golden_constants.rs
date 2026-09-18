//! Golden-value tests for physical constants (Domain 1).
//!
//! Reads constants from validation/fixtures/constants.json and compares
//! against xrt_rs::core::consts values. Tolerance: 1e-15 (identical literals).

use xrt_rs::core::consts;

fn load_fixture() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/validation/fixtures/constants.json"
    );
    let text =
        std::fs::read_to_string(path).expect("Run `python validation/generate_fixtures.py` first");
    serde_json::from_str(&text).unwrap()
}

macro_rules! check_const {
    ($fixture:expr, $name:expr, $rust_val:expr, $tol:expr) => {
        let expected = $fixture["constants"][$name]
            .as_f64()
            .unwrap_or_else(|| panic!("missing constant: {}", $name));
        let actual: f64 = $rust_val;
        let diff = (actual - expected).abs();
        assert!(
            diff <= $tol,
            "{}: rust={:.17e} != python={:.17e} (diff={:.2e}, tol={:.2e})",
            $name,
            actual,
            expected,
            diff,
            $tol
        );
    };
}

#[test]
fn golden_constants() {
    let fix = load_fixture();
    let tol = 1e-15;

    // Fundamental constants: should be identical literals
    check_const!(fix, "PI", consts::PI, tol);
    check_const!(fix, "PI2", consts::PI2, tol);
    check_const!(fix, "SIE0", consts::SIE0, tol);
    check_const!(fix, "C", consts::C, tol);
    check_const!(fix, "M0", consts::M0, tol);
    check_const!(fix, "SIM0", consts::SIM0, tol);
    check_const!(fix, "M0C2", consts::M0C2, tol);
    check_const!(fix, "HPLANCK", consts::HPLANCK, tol);
    check_const!(fix, "EV2ERG", consts::EV2ERG, tol);
    check_const!(fix, "SIHPLANCK", consts::SIHPLANCK, tol);
    check_const!(fix, "R0", consts::R0, tol);
    check_const!(fix, "AVOGADRO", consts::AVOGADRO, tol);

    // Derived constants: may differ slightly due to derivation order / precision
    check_const!(fix, "E0", consts::E0, 1e-10);
    check_const!(fix, "SIC", consts::SIC, 1e-10);
    check_const!(fix, "FINE_STR", consts::FINE_STR, 1e-8);
    check_const!(fix, "K2B", consts::K2B, 1e-2); // different constant sets
    check_const!(fix, "EMC", consts::EMC, 1e-8);
    check_const!(fix, "E2W", consts::E2W, 1e6); // large value, relative match
    check_const!(fix, "CH", consts::CH, 1e-8);
    check_const!(fix, "CHBAR", consts::CHBAR, 1e-8);
}
