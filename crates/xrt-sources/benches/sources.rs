use criterion::{Criterion, black_box, criterion_group, criterion_main};

use xrt_sources::bending_magnet::BendingMagnet;
use xrt_sources::distributions::{EnergyDist, SpatialDist};
use xrt_sources::geometric::GeometricSource;

fn bench_geometric_source(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_source");

    for nrays in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            criterion::BenchmarkId::new("shine", nrays),
            &nrays,
            |b, &n| {
                let source = GeometricSource {
                    nrays: n,
                    dist_x: SpatialDist::Normal(0.32),
                    dist_z: SpatialDist::Normal(0.018),
                    dist_xprime: SpatialDist::Normal(1e-3),
                    dist_zprime: SpatialDist::Normal(1e-4),
                    dist_e: EnergyDist::Lines(vec![10000.0], None),
                    ..Default::default()
                };
                b.iter(|| black_box(source.shine()));
            },
        );
    }
    group.finish();
}

fn bench_bending_magnet(c: &mut Criterion) {
    let mut group = c.benchmark_group("bending_magnet");

    for nrays in [100, 500, 1_000] {
        group.bench_with_input(
            criterion::BenchmarkId::new("shine", nrays),
            &nrays,
            |b, &n| {
                b.iter(|| {
                    let mut bm =
                        BendingMagnet::new(3.0, 0.3, 1.0, n, 5000.0, 15000.0, 0.002, 0.002);
                    black_box(bm.shine())
                });
            },
        );
    }
    group.finish();
}

fn bench_build_i_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_i_map");

    for n in [100, 1_000, 10_000] {
        group.bench_with_input(criterion::BenchmarkId::new("bm", n), &n, |b, &n| {
            let bm = BendingMagnet::new(3.0, 0.3, 1.0, 100, 5000.0, 15000.0, 0.002, 0.002);
            let energies: Vec<f64> = (0..n).map(|i| 5000.0 + i as f64 * 10.0).collect();
            let thetas = vec![0.0; n];
            let psis = vec![0.0; n];
            b.iter(|| black_box(bm.build_i_map(&energies, &thetas, &psis)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_geometric_source,
    bench_bending_magnet,
    bench_build_i_map
);
criterion_main!(benches);
