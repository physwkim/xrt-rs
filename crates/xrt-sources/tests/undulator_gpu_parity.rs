//! The WGSL undulator kernel must reproduce the f64 CPU reference wherever the
//! spectrum carries flux: at the fundamental and at the odd harmonics. Needs
//! `--features gpu`.
#![cfg(feature = "gpu")]

use xrt_sources::undulator::Undulator;

#[test]
fn the_shader_reproduces_the_cpu_spectrum_at_every_odd_harmonic() {
    if xrt_gpu::context::GpuContext::shared().is_none() {
        eprintln!("no GPU adapter available; skipping");
        return;
    }

    let u = Undulator::new(3.0, 0.3, 0.0, 2.0, 30.0, 50, 100, 100.0, 1e7, 1e-3, 1e-3);
    let e1 = u.fundamental_energy();

    // 0.95 sits on a sidelobe, the rest are the odd harmonics. The even ones
    // are left out: they are suppressed by 17 orders on axis, so in f32 they
    // are the shader's noise floor rather than a number to compare.
    let ratios = [0.95, 1.0, 3.0, 5.0, 7.0, 9.0];
    let energies: Vec<f64> = ratios.iter().map(|r| r * e1).collect();
    let zeros = vec![0.0; energies.len()];

    let (cpu, cpu_s, _) = u.build_i_map(&energies, &zeros, &zeros);
    let (gpu, gpu_s, _) = u.build_i_map_auto(&energies, &zeros, &zeros);

    for (k, r) in ratios.iter().enumerate() {
        let rel = (gpu[k] - cpu[k]) / cpu[k];
        assert!(
            rel.abs() < 1e-4,
            "I({r}·E₁): shader {} vs CPU {}, {rel:+.3e} apart",
            gpu[k],
            cpu[k]
        );
        let rel_s = (gpu_s[k].norm() - cpu_s[k].norm()) / cpu_s[k].norm();
        assert!(
            rel_s.abs() < 1e-4,
            "|amp_s|({r}·E₁): shader {} vs CPU {}, {rel_s:+.3e} apart",
            gpu_s[k].norm(),
            cpu_s[k].norm()
        );
    }
}
