//! The GPU undulator kernel must map the two deflection parameters to the same
//! axes as the CPU `Undulator::build_i_map`: `ky` (the vertical field) drives
//! the horizontal oscillation, `kx` the vertical one. This is also the only
//! test that compiles `shaders/undulator.wgsl`.

use xrt_gpu::context::GpuContext;
use xrt_gpu::undulator::{GpuObsPoint, undulator_gpu};

#[test]
fn a_planar_undulator_radiates_sigma_polarized_on_axis() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    let on_axis = vec![GpuObsPoint {
        energy: 1000.0,
        theta: 0.0,
        psi: 0.0,
        _pad: 0.0,
    }];

    // kx = 0, ky = 2: vertical field, horizontal oscillation, sigma on axis.
    let planar = undulator_gpu(&ctx, &on_axis, 0.0, 2.0, 30.0, 50, 5870.85, 0.3, 64, 0.0);
    let (s, p) = (planar.amp_s[0].norm(), planar.amp_p[0].norm());
    assert!(s > 0.0, "|amp_s| = {s}");
    assert!(
        p < 1e-6 * s,
        "|amp_p| = {p} should vanish on axis, |amp_s| = {s}"
    );

    // The same magnitudes exchanged describe an undulator rotated by 90°.
    let rotated = undulator_gpu(&ctx, &on_axis, 2.0, 0.0, 30.0, 50, 5870.85, 0.3, 64, 0.0);
    let (s, p) = (rotated.amp_s[0].norm(), rotated.amp_p[0].norm());
    assert!(p > 0.0, "|amp_p| = {p}");
    assert!(
        s < 1e-6 * p,
        "|amp_s| = {s} should vanish on axis, |amp_p| = {p}"
    );
}
