//! Example: Two-mirror beamline with Si coating.
//!
//! Demonstrates: GeometricSource → toroid mirror → flat mirror → screen
//!
//! Run: cargo run --example mirror_beamline

use xrt_rs::materials::data::ScatteringTable;
use xrt_rs::materials::material::{Material, MaterialKind};
use xrt_rs::oes::beamline::{Beamline, OeParamsBuilder};
use xrt_rs::oes::material_oe::MaterialOpticalElement;
use xrt_rs::oes::screen::Screen;
use xrt_rs::oes::surfaces::toroid::ToroidSurface;
use xrt_rs::sources::distributions::{EnergyDist, SpatialDist};
use xrt_rs::sources::geometric::GeometricSource;

fn main() {
    println!("=== XRT-RS: Two-Mirror Beamline ===\n");

    // 1. Create a collimated X-ray source at 10 keV
    let source = GeometricSource {
        nrays: 10_000,
        dist_x: SpatialDist::Normal(0.2),       // 200 μm horizontal
        dist_z: SpatialDist::Normal(0.05),      // 50 μm vertical
        dist_xprime: SpatialDist::Normal(5e-5), // 50 μrad divergence
        dist_zprime: SpatialDist::Normal(2e-5), // 20 μrad divergence
        dist_e: EnergyDist::Lines(vec![10_000.0], None),
        ..Default::default()
    };
    let mut beam = source.shine();
    let stats = beam.statistics().unwrap();
    println!("Source: {}", stats);
    println!();

    // 2. Si mirror material
    let si = Material::new(
        &["Si"],
        None,
        2.33,
        MaterialKind::Mirror,
        None,
        ScatteringTable::ChantlerTotal,
    )
    .expect("Si material");

    // 3. Toroid focusing mirror at 5 mrad grazing (at origin)
    let m1 = MaterialOpticalElement::new(
        ToroidSurface::new(5e6, 50.0), // R=5km, r=50mm
        OeParamsBuilder::new()
            .pitch(0.005) // 5 mrad grazing
            .build(),
        si,
    );

    // 4. Build and run beamline: mirror → drift to screen
    let bl = Beamline::new()
        .add_material("M1_Toroid", m1)
        .drift(10_000.0);

    let output = bl.propagate(&mut beam);

    // 5. Report (using Display impl)
    println!("{output}");

    // 6. Screen — placed at a wide area to catch reflected beam
    // After 5 mrad pitch reflection, rays deflect by ~10 mrad in z
    let screen = Screen::new([0.0, 10_000.0, 100.0], 50.0, 200.0, 100, 100);
    let capture = screen.capture(&beam);

    println!("Screen at y=10m:");
    println!(
        "  captured: {} rays (missed: {})",
        capture.n_captured, capture.n_missed
    );
    if capture.n_captured > 0 {
        let [cx, cz] = capture.centroid();
        let [sx, sz] = capture.rms_size();
        println!("  centroid: ({:.3}, {:.3}) mm", cx, cz);
        println!("  RMS size: σx={:.3}mm, σz={:.3}mm", sx, sz);
        println!("  peak intensity: {:.2}", capture.peak_intensity());
        if let Some(fw) = capture.fwhm_x() {
            println!("  FWHM_x: {:.3} mm", fw);
        }
    }

    // 8. Export data (uncomment to write files)
    // let mut f = std::fs::File::create("screen_2d.tsv").unwrap();
    // capture.write_tsv(&mut f).unwrap();
    // let mut f = std::fs::File::create("beam.tsv").unwrap();
    // beam.write_tsv(&mut f).unwrap();
}
