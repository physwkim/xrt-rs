use criterion::{black_box, criterion_group, criterion_main, Criterion};

use xrt_core::beam::{Beam, RayState};
use xrt_math::rootfind::RootFindConfig;
use xrt_oes::intersection::find_intersection_surface;
use xrt_oes::reflect::{reflect_local, DeflectionMode};
use xrt_oes::aperture::Aperture;
use xrt_oes::surfaces::flat::FlatSurface;
use xrt_oes::surfaces::toroid::ToroidSurface;
use xrt_oes::surfaces::spherical::SphericalSurface;

fn make_grazing_beam(n: usize) -> Beam {
    let mut beam = Beam::new(n);
    let angle = 0.01_f64; // 10 mrad grazing
    for i in 0..n {
        beam.x[i] = (i as f64 - n as f64 / 2.0) * 0.01;
        beam.y[i] = -100.0;
        beam.z[i] = 10.0;
        beam.a[i] = 0.0;
        beam.b[i] = angle.cos();
        beam.c[i] = -angle.sin();
        beam.state[i] = RayState::Good as i32;
        beam.e[i] = 10000.0;
    }
    beam
}

fn bench_single_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_intersection");
    let config = RootFindConfig::default();

    group.bench_function("flat", |b| {
        let surface = FlatSurface;
        b.iter(|| {
            find_intersection_surface(
                black_box(&surface),
                0.0, 200.0, 0.0, -100.0, 10.0, 0.0, 0.01_f64.cos(), -0.01_f64.sin(),
                1, &config,
            )
        });
    });

    group.bench_function("spherical", |b| {
        let surface = SphericalSurface::new(1000.0);
        b.iter(|| {
            find_intersection_surface(
                black_box(&surface),
                0.0, 200.0, 0.0, -100.0, 10.0, 0.0, 0.01_f64.cos(), -0.01_f64.sin(),
                1, &config,
            )
        });
    });

    group.bench_function("toroid", |b| {
        let surface = ToroidSurface::new(5e6, 50.0);
        b.iter(|| {
            find_intersection_surface(
                black_box(&surface),
                0.0, 200.0, 0.0, -100.0, 10.0, 0.0, 0.01_f64.cos(), -0.01_f64.sin(),
                1, &config,
            )
        });
    });

    group.finish();
}

fn bench_parallel_reflect(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_reflect");
    let config = RootFindConfig::default();
    let aperture = Aperture::default();

    for nrays in [100, 1_000, 10_000, 100_000] {
        group.bench_with_input(
            criterion::BenchmarkId::new("flat", nrays),
            &nrays,
            |b, &n| {
                let surface = FlatSurface;
                b.iter_batched(
                    || {
                        let beam = make_grazing_beam(n);
                        let good: Vec<usize> = (0..n).collect();
                        (beam, good)
                    },
                    |(mut beam, good)| {
                        reflect_local(
                            &surface,
                            black_box(&mut beam),
                            &good,
                            &aperture,
                            1,
                            DeflectionMode::Reflect,
                            None, None,
                            &config,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            criterion::BenchmarkId::new("toroid", nrays),
            &nrays,
            |b, &n| {
                let surface = ToroidSurface::new(5e6, 50.0);
                b.iter_batched(
                    || {
                        let beam = make_grazing_beam(n);
                        let good: Vec<usize> = (0..n).collect();
                        (beam, good)
                    },
                    |(mut beam, good)| {
                        reflect_local(
                            &surface,
                            black_box(&mut beam),
                            &good,
                            &aperture,
                            1,
                            DeflectionMode::Reflect,
                            None, None,
                            &config,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_single_intersection, bench_parallel_reflect);
criterion_main!(benches);
