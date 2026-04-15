use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex64;

use xrt_waves::diffraction::{diffraction_integral, DiffractionRay, PixelPoint};

fn make_rays(n: usize) -> Vec<DiffractionRay> {
    (0..n)
        .map(|i| {
            let x = (i as f64 - n as f64 / 2.0) * 0.01;
            DiffractionRay {
                x,
                y: 0.0,
                z: 0.0,
                nx: 0.0,
                ny: 0.0,
                nz: 1.0,
                nl: 1.0,
                es: Complex64::new(1.0, 0.0),
                ep: Complex64::new(0.0, 0.0),
                energy: 10000.0,
            }
        })
        .collect()
}

fn make_pixels(n: usize) -> Vec<PixelPoint> {
    (0..n)
        .map(|i| {
            let x = (i as f64 - n as f64 / 2.0) * 0.02;
            PixelPoint {
                x,
                y: 1000.0,
                z: 0.0,
            }
        })
        .collect()
}

fn bench_diffraction_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("kirchhoff_cpu");

    // O(N_pixel × N_ray) — benchmark different sizes
    for (n_rays, n_pixels) in [(100, 50), (500, 100), (1000, 200), (5000, 100)] {
        group.bench_with_input(
            criterion::BenchmarkId::new(
                format!("{}rays_{}pix", n_rays, n_pixels),
                n_rays * n_pixels,
            ),
            &(n_rays, n_pixels),
            |b, &(nr, np)| {
                let rays = make_rays(nr);
                let pixels = make_pixels(np);
                b.iter(|| diffraction_integral(black_box(&rays), black_box(&pixels)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_diffraction_integral);
criterion_main!(benches);
