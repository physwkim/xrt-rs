//! GPU vs CPU parity tests for Kirchhoff diffraction integral.
//!
//! Verifies that the GPU (WGSL shader) and CPU implementations produce
//! numerically equivalent results within f32 precision bounds.

use num_complex::Complex64;

use xrt_gpu::context::GpuContext;
use xrt_gpu::kirchhoff::{kirchhoff_gpu, GpuPixel, GpuRay};
use xrt_gpu::fallback::kirchhoff_auto;

/// Create test rays/pixels with SHORT distances suitable for f32 precision.
///
/// The GPU shader uses f32, so phase = k * path must fit in f32 precision.
/// At E=10keV, k ≈ 5e7 mm⁻¹. For path=1mm, phase ≈ 5e7 which is borderline.
/// We use path ≈ 0.1mm for safe f32 sin/cos.
fn make_short_distance_rays_pixels() -> (Vec<GpuRay>, Vec<GpuPixel>) {
    let rays = vec![
        GpuRay {
            x: 0.0, y: 0.0, z: 0.0,
            nx: 0.0, ny: 0.0, nz: 1.0,
            nl: 1.0, energy: 1000.0, // 1 keV → smaller k
            es_re: 1.0, es_im: 0.0,
            ep_re: 0.5, ep_im: 0.1,
        },
        GpuRay {
            x: 0.01, y: 0.0, z: 0.0,
            nx: 0.0, ny: 0.0, nz: 1.0,
            nl: 1.0, energy: 1000.0,
            es_re: 0.8, es_im: -0.1,
            ep_re: 0.6, ep_im: 0.0,
        },
        GpuRay {
            x: -0.01, y: 0.0, z: 0.0,
            nx: 0.0, ny: 0.0, nz: 1.0,
            nl: 1.0, energy: 1000.0,
            es_re: 0.9, es_im: 0.05,
            ep_re: 0.7, ep_im: -0.05,
        },
    ];
    let pixels = vec![
        GpuPixel { x: -0.005, y: 0.0, z: 0.1, _pad: 0.0 }, // 0.1 mm away
        GpuPixel { x: 0.0,    y: 0.0, z: 0.1, _pad: 0.0 },
        GpuPixel { x: 0.005,  y: 0.0, z: 0.1, _pad: 0.0 },
    ];
    (rays, pixels)
}


#[test]
fn golden_gpu_kirchhoff_vs_cpu() {
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };
    eprintln!("GPU: {}", ctx.adapter_name());

    let (rays, pixels) = make_short_distance_rays_pixels();

    // GPU result
    let gpu_results = kirchhoff_gpu(&ctx, &rays, &pixels);

    // CPU fallback result (same data types, same formula but f64 precision)
    let cpu_results = kirchhoff_auto_cpu_only(&rays, &pixels);

    assert_eq!(gpu_results.len(), cpu_results.len());

    // f32 GPU vs f64 CPU: expect ~1e-3 relative tolerance
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        let (gpu_es, gpu_ep) = gpu;
        let (cpu_es, cpu_ep) = cpu;

        // Check GPU results are finite
        assert!(
            gpu_es.re.is_finite() && gpu_es.im.is_finite(),
            "GPU pixel[{i}] Es not finite: {gpu_es}"
        );
        assert!(
            gpu_ep.re.is_finite() && gpu_ep.im.is_finite(),
            "GPU pixel[{i}] Ep not finite: {gpu_ep}"
        );

        // Compare GPU (f32 shader) vs CPU (f64 with f32 input).
        // At 1keV and 0.1mm distance, phase ≈ 5e5 rad — f32 sin/cos
        // gives ~1-2% precision loss from range reduction.
        let tol = 5e-2;
        let es_diff = (gpu_es - cpu_es).norm();
        let es_scale = cpu_es.norm().max(1e-30);
        assert!(
            es_diff / es_scale < tol,
            "pixel[{i}] Es: GPU={gpu_es:.6e} vs CPU={cpu_es:.6e} (rel_diff={:.2e})",
            es_diff / es_scale
        );

        let ep_diff = (gpu_ep - cpu_ep).norm();
        let ep_scale = cpu_ep.norm().max(1e-30);
        assert!(
            ep_diff / ep_scale < tol,
            "pixel[{i}] Ep: GPU={gpu_ep:.6e} vs CPU={cpu_ep:.6e} (rel_diff={:.2e})",
            ep_diff / ep_scale
        );
    }
}

#[test]
fn golden_gpu_kirchhoff_nonzero_output() {
    // Verify GPU produces non-zero output for reasonable inputs
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let (rays, pixels) = make_short_distance_rays_pixels();
    let gpu_results = kirchhoff_gpu(&ctx, &rays, &pixels);

    assert_eq!(gpu_results.len(), pixels.len());
    for (i, (es, _ep)) in gpu_results.iter().enumerate() {
        assert!(
            es.norm() > 1e-10,
            "pixel[{i}] GPU Es too small: {es:.6e}"
        );
    }
}

#[test]
fn golden_gpu_kirchhoff_symmetry() {
    let ctx = match GpuContext::new() {
        Some(ctx) => ctx,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    // Single ray at origin, two symmetric pixels
    let rays = vec![GpuRay {
        x: 0.0, y: 0.0, z: 0.0,
        nx: 0.0, ny: 0.0, nz: 1.0,
        nl: 1.0, energy: 10000.0,
        es_re: 1.0, es_im: 0.0,
        ep_re: 1.0, ep_im: 0.0,
    }];
    let pixels = vec![
        GpuPixel { x: -0.01, y: 0.0, z: 1000.0, _pad: 0.0 },
        GpuPixel { x:  0.01, y: 0.0, z: 1000.0, _pad: 0.0 },
    ];

    let results = kirchhoff_gpu(&ctx, &rays, &pixels);
    assert_eq!(results.len(), 2);

    // Symmetric inputs → same |Es| at both pixels
    let diff = (results[0].0.norm() - results[1].0.norm()).abs();
    assert!(
        diff < 1e-5,
        "symmetric pixels: |Es[0]|={:.6e} vs |Es[1]|={:.6e}",
        results[0].0.norm(),
        results[1].0.norm()
    );
}

#[test]
fn golden_gpu_kirchhoff_auto_dispatch() {
    // Test that kirchhoff_auto picks GPU and produces valid results
    let (rays, pixels) = make_short_distance_rays_pixels();
    let results = kirchhoff_auto(&rays, &pixels);

    assert_eq!(results.len(), pixels.len());
    for (i, (es, _ep)) in results.iter().enumerate() {
        assert!(es.re.is_finite(), "auto pixel[{i}] Es.re not finite");
        assert!(es.norm() > 1e-10, "auto pixel[{i}] Es too small");
    }
}

/// CPU-only version for comparison (bypasses GPU)
fn kirchhoff_auto_cpu_only(
    rays: &[GpuRay],
    pixels: &[GpuPixel],
) -> Vec<(Complex64, Complex64)> {
    use std::f64::consts::PI;
    use xrt_core::consts::CHBAR;

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
                if path < 1e-20 { continue; }
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
