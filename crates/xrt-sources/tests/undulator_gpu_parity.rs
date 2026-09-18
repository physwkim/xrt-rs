//! The WGSL undulator kernel must reproduce the f64 CPU reference wherever the
//! spectrum carries flux: at the fundamental and at the odd harmonics. Needs
//! `--features gpu`.
//!
//! The shader is invoked directly rather than through `build_i_map_auto`, which
//! sends a map this short to the CPU and would make the comparison vacuous.
#![cfg(feature = "gpu")]

use xrt_gpu::undulator::GpuObsPoint;
use xrt_sources::undulator::Undulator;

#[test]
fn the_shader_reproduces_the_cpu_spectrum_at_every_odd_harmonic() {
    let Some(ctx) = xrt_gpu::context::GpuContext::shared() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    let u = Undulator::new(3.0, 0.3, 0.0, 2.0, 30.0, 50, 100, 100.0, 1e7, 1e-3, 1e-3);
    let e1 = u.fundamental_energy();

    // 0.95 sits on a sidelobe, the rest are the odd harmonics. The even ones
    // are left out: they are suppressed by 17 orders on axis, so in f32 they
    // are the shader's noise floor rather than a number to compare.
    let ratios = [0.95, 1.0, 3.0, 5.0, 7.0, 9.0];
    let energies: Vec<f64> = ratios.iter().map(|r| r * e1).collect();
    let zeros = vec![0.0; energies.len()];

    let (cpu, cpu_s, _) = u.build_i_map(&energies, &zeros, &zeros);

    let obs: Vec<GpuObsPoint> = energies
        .iter()
        .map(|&e| GpuObsPoint {
            energy: e as f32,
            theta: 0.0,
            psi: 0.0,
            _pad: 0.0,
        })
        .collect();
    let gpu = xrt_gpu::undulator::undulator_gpu(
        ctx,
        &obs,
        u.kx,
        u.ky,
        u.period,
        u.n_periods,
        u.params.gamma,
        u.params.beam_current,
        64,
        u.phase_deg,
    );

    for (k, r) in ratios.iter().enumerate() {
        let rel = (gpu.intensity[k] - cpu[k]) / cpu[k];
        assert!(
            rel.abs() < 1e-4,
            "I({r}·E₁): shader {} vs CPU {}, {rel:+.3e} apart",
            gpu.intensity[k],
            cpu[k]
        );
        let rel_s = (gpu.amp_s[k].norm() - cpu_s[k].norm()) / cpu_s[k].norm();
        assert!(
            rel_s.abs() < 1e-4,
            "|amp_s|({r}·E₁): shader {} vs CPU {}, {rel_s:+.3e} apart",
            gpu.amp_s[k].norm(),
            cpu_s[k].norm()
        );
    }
}

#[test]
fn a_map_shorter_than_the_dispatch_threshold_stays_in_f64() {
    // The CPU path is the reference, so below the threshold build_i_map_auto
    // must return exactly what build_i_map returns - not merely something
    // close, which is all the f32 shader could manage.
    let u = Undulator::new(3.0, 0.3, 0.0, 2.0, 30.0, 50, 100, 100.0, 1e7, 1e-3, 1e-3);
    let e1 = u.fundamental_energy();
    let energies = vec![e1, 3.0 * e1];
    let zeros = vec![0.0; energies.len()];

    let (cpu, cpu_s, cpu_p) = u.build_i_map(&energies, &zeros, &zeros);
    let (auto, auto_s, auto_p) = u.build_i_map_auto(&energies, &zeros, &zeros);

    assert_eq!(cpu, auto);
    assert_eq!(cpu_s, auto_s);
    assert_eq!(cpu_p, auto_p);
}
