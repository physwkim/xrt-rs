use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use xrt_core::beam::Beam;
use xrt_core::transforms::{RotationParams, rotate_beam, rotate_xyz};

fn bench_rotate_beam(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotate_beam");
    for nrays in [1_000, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            criterion::BenchmarkId::new("pitch_only", nrays),
            &nrays,
            |b, &n| {
                let mut beam = Beam::new(n);
                beam.z.fill(1.0);
                let params = RotationParams::default_sequence(0.1, 0.0, 0.0);
                b.iter(|| {
                    rotate_beam(black_box(&mut beam), None, &params, false, false);
                });
            },
        );
        group.bench_with_input(
            criterion::BenchmarkId::new("all_angles", nrays),
            &nrays,
            |b, &n| {
                let mut beam = Beam::new(n);
                beam.z.fill(1.0);
                let params = RotationParams::default_sequence(0.1, 0.05, 0.02);
                b.iter(|| {
                    rotate_beam(black_box(&mut beam), None, &params, false, false);
                });
            },
        );
    }
    group.finish();
}

fn bench_rotate_xyz(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotate_xyz");
    for nrays in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            criterion::BenchmarkId::new("yaw_90", nrays),
            &nrays,
            |b, &n| {
                let mut x = Array1::from_elem(n, 1.0);
                let mut y = Array1::zeros(n);
                let mut z = Array1::zeros(n);
                let params =
                    RotationParams::default_sequence(0.0, 0.0, std::f64::consts::FRAC_PI_2);
                b.iter(|| {
                    rotate_xyz(
                        black_box(&mut x),
                        black_box(&mut y),
                        black_box(&mut z),
                        None,
                        &params,
                    );
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_rotate_beam, bench_rotate_xyz);
criterion_main!(benches);
