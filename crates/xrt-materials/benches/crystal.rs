use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use xrt_materials::crystal::CrystalGeometry;
use xrt_materials::crystal_variants::CrystalSi;
use xrt_materials::data::ScatteringTable;

fn bench_bragg_angle(c: &mut Criterion) {
    let si = CrystalSi::new(
        [1, 1, 1],
        297.15,
        CrystalGeometry::BraggReflected,
        1.0,
        None,
        0.0,
        ScatteringTable::ChantlerTotal,
    )
    .unwrap();

    let mut group = c.benchmark_group("bragg_angle");
    for n in [100, 1000, 10000] {
        let energies = Array1::linspace(5000.0, 30000.0, n);
        group.bench_with_input(BenchmarkId::new("n_energies", n), &energies, |b, e| {
            b.iter(|| si.base.get_bragg_angle(black_box(e)))
        });
    }
    group.finish();
}

fn bench_amplitude(c: &mut Criterion) {
    let si = CrystalSi::new(
        [1, 1, 1],
        297.15,
        CrystalGeometry::BraggReflected,
        1.0,
        None,
        0.0,
        ScatteringTable::ChantlerTotal,
    )
    .unwrap();

    let mut group = c.benchmark_group("crystal_amplitude");
    for n in [10, 100, 1000] {
        let energies = Array1::from_elem(n, 10000.0);
        let bidn = Array1::linspace(-0.195, -0.193, n);
        group.bench_with_input(
            BenchmarkId::new("n_angles", n),
            &(energies, bidn),
            |b, (e, bd)| {
                b.iter(|| {
                    si.base
                        .get_amplitude(black_box(e), black_box(bd), None, None, &si)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_bragg_angle, bench_amplitude);
criterion_main!(benches);
