//! Example: Soft X-ray grating monochromator.
//!
//! Demonstrates: BendingMagnet → flat mirror → blazed grating → exit slit screen
//!
//! Run: cargo run --example grating_monochromator

use xrt_rs::materials::data::ScatteringTable;
use xrt_rs::materials::material::{Material, MaterialKind};
use xrt_rs::oes::beamline::{Beamline, OeParamsBuilder};
use xrt_rs::oes::grating_oe::GratingOpticalElement;
use xrt_rs::oes::material_oe::MaterialOpticalElement;
use xrt_rs::oes::screen::Screen;
use xrt_rs::oes::surfaces::flat::FlatSurface;
use xrt_rs::oes::surfaces::grating::BlazedGrating;
use xrt_rs::sources::bending_magnet::BendingMagnet;

fn main() {
    println!("=== XRT-RS: Grating Monochromator ===\n");

    // 1. Bending magnet source: 3 GeV, 300 mA, 1 T
    let mut bm = BendingMagnet::new(
        3.0,   // 3 GeV electron energy
        0.3,   // 300 mA beam current
        1.0,   // 1 T magnetic field
        5_000, // 5000 rays
        200.0, 1500.0, // 200-1500 eV photon energy
        0.002, 0.002, // ±2 mrad angular acceptance
    );
    let mut beam = bm.shine();
    println!("BM source: {} rays, E=[200, 1500] eV", beam.nrays());

    // Energy statistics
    let e_mean: f64 = beam.e.iter().sum::<f64>() / beam.nrays() as f64;
    let e_min = beam.e.iter().cloned().fold(f64::INFINITY, f64::min);
    let e_max = beam.e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  E: mean={:.0} eV, range=[{:.0}, {:.0}] eV\n",
        e_mean, e_min, e_max
    );

    // 2. Materials
    let si = Material::new(
        &["Si"],
        None,
        2.33,
        MaterialKind::Mirror,
        None,
        ScatteringTable::ChantlerTotal,
    )
    .expect("Si");

    let au = Material::new(
        &["Au"],
        None,
        19.32,
        MaterialKind::Grating,
        None,
        ScatteringTable::ChantlerTotal,
    )
    .expect("Au");

    // 3. Pre-mirror (collimating, at origin)
    let m1 = MaterialOpticalElement::new(
        FlatSurface,
        OeParamsBuilder::new()
            .pitch(0.02) // 20 mrad grazing
            .build(),
        si,
    );

    // 4. Blazed grating: 600 l/mm, first order (at origin, same location)
    let grating = GratingOpticalElement::new(
        BlazedGrating::new(600.0, 0.015, 0.5), // 600 l/mm, 15 mrad blaze
        OeParamsBuilder::new().pitch(0.02).build(),
        -1, // first negative order
    )
    .with_material(au);

    // 5. Build beamline
    let bl = Beamline::new()
        .add_material("M1_Pre", m1)
        .add_grating("Grating_600", grating)
        .drift(3_000.0);

    let output = bl.propagate(&mut beam);

    // 6. Report
    println!("Beamline:");
    for el in &output.elements {
        if el.results.is_empty() {
            println!("  {:15} (drift)", el.name);
        } else {
            println!(
                "  {:15} good={:5}  lost={:4}",
                el.name, el.good_count, el.lost_count
            );
        }
    }
    println!("\nEfficiency: {:.1}%\n", output.efficiency() * 100.0);

    // 7. Exit slit screen — wide enough to catch diffracted beam
    let screen = Screen::new([0.0, 3_000.0, 0.0], 200.0, 500.0, 50, 50);
    let capture = screen.capture(&beam);

    println!("Exit slit screen:");
    println!(
        "  captured: {} rays (missed: {})",
        capture.n_captured, capture.n_missed
    );
    if capture.n_captured > 0 {
        let [cx, cz] = capture.centroid();
        let [sx, sz] = capture.rms_size();
        println!("  centroid: ({:.2}, {:.2}) mm", cx, cz);
        println!("  RMS: σx={:.3}mm, σz={:.3}mm", sx, sz);
    }

    // 8. Energy of surviving good rays
    let good = beam.good_indices();
    if !good.is_empty() {
        let good_e: Vec<f64> = good.iter().map(|&i| beam.e[i]).collect();
        let ge_mean: f64 = good_e.iter().sum::<f64>() / good_e.len() as f64;
        let ge_min = good_e.iter().cloned().fold(f64::INFINITY, f64::min);
        let ge_max = good_e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "\nGood rays energy: mean={:.0} eV, range=[{:.0}, {:.0}] eV",
            ge_mean, ge_min, ge_max
        );
    }
}
