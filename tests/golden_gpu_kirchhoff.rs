//! GPU vs CPU parity tests for Kirchhoff diffraction integral.
//!
//! Verifies that the GPU (WGSL shader with phase reduction) and CPU
//! implementations produce numerically equivalent results.
//!
//! The GPU shader subtracts a per-pixel reference path before computing
//! sin/cos, keeping the f32 argument small. The base phase exp(i*k*ref)
//! is restored on the CPU side in f64.

#![cfg(feature = "gpu")]

use num_complex::Complex64;

use xrt_rs::gpu::context::GpuContext;
use xrt_rs::gpu::fallback::kirchhoff_auto;
use xrt_rs::gpu::kirchhoff::{GpuPixel, GpuRay, kirchhoff_gpu};

fn load_fixture() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/validation/fixtures/diffraction_ref.json"
    );
    let text =
        std::fs::read_to_string(path).expect("Run `python validation/generate_fixtures.py` first");
    serde_json::from_str(&text).unwrap()
}

/// Create fixture rays/pixels (path ≈ 1000mm, E=10keV → large phase).
/// This tests the phase reduction fix.
fn make_fixture_rays_pixels() -> (Vec<GpuRay>, Vec<GpuPixel>) {
    let fix = load_fixture();
    let tc = &fix["test_cases"][0];
    let rd = &tc["rays"];
    let pd = &tc["pixels"];

    let n_rays = rd["x"].as_array().unwrap().len();
    let rays: Vec<GpuRay> = (0..n_rays)
        .map(|i| GpuRay {
            x: rd["x"][i].as_f64().unwrap() as f32,
            y: rd["y"][i].as_f64().unwrap() as f32,
            z: rd["z"][i].as_f64().unwrap() as f32,
            nx: rd["nx"][i].as_f64().unwrap() as f32,
            ny: rd["ny"][i].as_f64().unwrap() as f32,
            nz: rd["nz"][i].as_f64().unwrap() as f32,
            nl: rd["nl"][i].as_f64().unwrap() as f32,
            energy: rd["energy"][i].as_f64().unwrap() as f32,
            es_re: rd["es_re"][i].as_f64().unwrap() as f32,
            es_im: rd["es_im"][i].as_f64().unwrap() as f32,
            ep_re: rd["ep_re"][i].as_f64().unwrap() as f32,
            ep_im: rd["ep_im"][i].as_f64().unwrap() as f32,
        })
        .collect();

    let n_pix = pd["x"].as_array().unwrap().len();
    let pixels: Vec<GpuPixel> = (0..n_pix)
        .map(|i| GpuPixel {
            x: pd["x"][i].as_f64().unwrap() as f32,
            y: pd["y"][i].as_f64().unwrap() as f32,
            z: pd["z"][i].as_f64().unwrap() as f32,
            _pad: 0.0,
        })
        .collect();

    (rays, pixels)
}

/// CPU-only Kirchhoff (f64 precision, no GPU) for comparison.
fn kirchhoff_cpu_f64(rays: &[GpuRay], pixels: &[GpuPixel]) -> Vec<(Complex64, Complex64)> {
    use std::f64::consts::PI;
    use xrt_rs::core::consts::CHBAR;

    pixels
        .iter()
        .map(|pixel| {
            let mut es = Complex64::new(0.0, 0.0);
            let mut ep = Complex64::new(0.0, 0.0);
            for ray in rays {
                let dx = pixel.x as f64 - ray.x as f64;
                let dy = pixel.y as f64 - ray.y as f64;
                let dz = pixel.z as f64 - ray.z as f64;
                let path = (dx * dx + dy * dy + dz * dz).sqrt();
                if path < 1e-20 {
                    continue;
                }
                let inv_path = 1.0 / path;
                let ns = (ray.nx as f64 * dx + ray.ny as f64 * dy + ray.nz as f64 * dz) * inv_path;
                let k = ray.energy as f64 / CHBAR * 1e7;
                let phase = k * path;
                let (sin_p, cos_p) = phase.sin_cos();
                let obliquity = ray.nl as f64 + ns;
                let amp = k / (4.0 * PI) * obliquity * inv_path;
                let u = Complex64::new(-amp * sin_p, amp * cos_p);
                es += Complex64::new(ray.es_re as f64, ray.es_im as f64) * u;
                ep += Complex64::new(ray.ep_re as f64, ray.ep_im as f64) * u;
            }
            (es, ep)
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_gpu_long_distance_vs_cpu() {
    // This is the KEY test: E=10keV, path=1000mm → phase ≈ 5e10.
    // Before the phase-reduction fix, GPU returned zeros.
    // After: GPU subtracts ref_path in f32 (delta_phase is small),
    // then CPU multiplies back exp(i*k*ref_path) in f64.
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };
    eprintln!("GPU: {}", ctx.adapter_name());

    let (rays, pixels) = make_fixture_rays_pixels();

    let gpu_results = kirchhoff_gpu(&ctx, &rays, &pixels);
    let cpu_results = kirchhoff_cpu_f64(&rays, &pixels);

    assert_eq!(gpu_results.len(), cpu_results.len());

    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        // GPU should now produce non-zero results
        assert!(
            gpu.0.norm() > 1e-10,
            "pixel[{i}] GPU Es is zero/tiny after phase reduction: {:.6e}",
            gpu.0
        );

        // Compare GPU vs CPU (tolerance accounts for f32 path subtraction)
        let es_diff = (gpu.0 - cpu.0).norm();
        let es_scale = cpu.0.norm().max(1e-30);
        let rel = es_diff / es_scale;
        // Phase reduction keeps delta_path in f32, giving ~1e-4 relative error
        assert!(
            rel < 1e-3,
            "pixel[{i}] Es: GPU={:.4e} vs CPU={:.4e} (rel={rel:.2e})",
            gpu.0,
            cpu.0
        );
    }
}

#[test]
fn golden_gpu_long_distance_vs_fixture() {
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let fix = load_fixture();
    let expected = fix["test_cases"][0]["expected"].as_array().unwrap();
    let (rays, pixels) = make_fixture_rays_pixels();

    let gpu_results = kirchhoff_gpu(&ctx, &rays, &pixels);

    for (i, (gpu, exp)) in gpu_results.iter().zip(expected.iter()).enumerate() {
        let exp_es = Complex64::new(
            exp["es_re"].as_f64().unwrap(),
            exp["es_im"].as_f64().unwrap(),
        );

        assert!(
            gpu.0.norm() > 1e-10,
            "pixel[{i}] GPU Es still zero: {:.6e}",
            gpu.0
        );

        let rel = (gpu.0 - exp_es).norm() / exp_es.norm().max(1e-30);
        assert!(
            rel < 1e-3,
            "pixel[{i}] Es: GPU={:.4e} vs fixture={exp_es:.4e} (rel={rel:.2e})",
            gpu.0
        );
    }
}

#[test]
fn golden_gpu_short_distance_vs_cpu() {
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let rays = vec![
        GpuRay {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            nl: 1.0,
            energy: 1000.0,
            es_re: 1.0,
            es_im: 0.0,
            ep_re: 0.5,
            ep_im: 0.1,
        },
        GpuRay {
            x: 0.01,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            nl: 1.0,
            energy: 1000.0,
            es_re: 0.8,
            es_im: -0.1,
            ep_re: 0.6,
            ep_im: 0.0,
        },
    ];
    let pixels = vec![
        GpuPixel {
            x: 0.0,
            y: 0.0,
            z: 0.1,
            _pad: 0.0,
        },
        GpuPixel {
            x: 0.005,
            y: 0.0,
            z: 0.1,
            _pad: 0.0,
        },
    ];

    let gpu = kirchhoff_gpu(&ctx, &rays, &pixels);
    let cpu = kirchhoff_cpu_f64(&rays, &pixels);

    let tol = 5e-2; // f32 vs f64
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let rel = (g.0 - c.0).norm() / c.0.norm().max(1e-30);
        assert!(rel < tol, "pixel[{i}] Es rel diff = {rel:.2e}");
    }
}

#[test]
fn golden_gpu_symmetry() {
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let rays = vec![GpuRay {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        nx: 0.0,
        ny: 0.0,
        nz: 1.0,
        nl: 1.0,
        energy: 10000.0,
        es_re: 1.0,
        es_im: 0.0,
        ep_re: 1.0,
        ep_im: 0.0,
    }];
    let pixels = vec![
        GpuPixel {
            x: -0.01,
            y: 0.0,
            z: 1000.0,
            _pad: 0.0,
        },
        GpuPixel {
            x: 0.01,
            y: 0.0,
            z: 1000.0,
            _pad: 0.0,
        },
    ];

    let results = kirchhoff_gpu(&ctx, &rays, &pixels);
    assert_eq!(results.len(), 2);

    // Both pixels should now produce non-zero results (phase reduction fix)
    assert!(
        results[0].0.norm() > 1e-10,
        "|Es[0]| too small: {:.4e}",
        results[0].0.norm()
    );
    assert!(
        results[1].0.norm() > 1e-10,
        "|Es[1]| too small: {:.4e}",
        results[1].0.norm()
    );

    // Symmetric inputs → same |Es|
    let diff = (results[0].0.norm() - results[1].0.norm()).abs();
    assert!(
        diff < 1e-4 * results[0].0.norm(),
        "symmetric pixels: |Es[0]|={:.6e} vs |Es[1]|={:.6e}",
        results[0].0.norm(),
        results[1].0.norm()
    );
}

#[test]
fn golden_gpu_auto_dispatch() {
    let (rays, pixels) = make_fixture_rays_pixels();
    let results = kirchhoff_auto(&rays, &pixels);

    assert_eq!(results.len(), pixels.len());
    for (i, (es, _ep)) in results.iter().enumerate() {
        assert!(es.re.is_finite(), "auto pixel[{i}] Es.re not finite");
        assert!(
            es.norm() > 1e-10,
            "auto pixel[{i}] Es too small: {:.4e}",
            es.norm()
        );
    }
}

#[test]
fn golden_gpu_extreme_distance() {
    // Test at 10000mm (10x the standard fixture distance)
    // Phase ≈ 5e11 — would be impossible without phase reduction
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let rays = vec![
        GpuRay {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            nl: 1.0,
            energy: 10000.0,
            es_re: 1.0,
            es_im: 0.0,
            ep_re: 0.5,
            ep_im: 0.0,
        },
        GpuRay {
            x: 0.01,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            nl: 1.0,
            energy: 10000.0,
            es_re: 0.8,
            es_im: 0.1,
            ep_re: 0.6,
            ep_im: -0.1,
        },
    ];
    let pixels = vec![GpuPixel {
        x: 0.0,
        y: 0.0,
        z: 10000.0,
        _pad: 0.0,
    }];

    let gpu = kirchhoff_gpu(&ctx, &rays, &pixels);
    let cpu = kirchhoff_cpu_f64(&rays, &pixels);

    // GPU should produce non-zero result even at extreme distance
    assert!(
        gpu[0].0.norm() > 1e-10,
        "GPU at 10000mm should be non-zero: {:.4e}",
        gpu[0].0.norm()
    );

    // Compare with CPU
    let rel = (gpu[0].0 - cpu[0].0).norm() / cpu[0].0.norm();
    assert!(rel < 0.05, "GPU vs CPU at 10000mm: rel_err={rel:.2e}");
}
