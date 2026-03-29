# xrt-rs

Rust port of [XRT](https://github.com/kklmn/xrt) (X-Ray Tracer) — a Python library for synchrotron radiation ray tracing and wave propagation.

Targets 10–100x performance over Python via Rust + rayon + wgpu GPU compute.

## Crate Structure

| Crate | Description |
|-------|-------------|
| `xrt-core` | Physical constants, beam representation, coordinate transforms |
| `xrt-math` | Root finding, interpolation, numerical utilities |
| `xrt-materials` | Elements, scattering factors, materials, crystals, multilayers |
| `xrt-oes` | Optical elements: surfaces, intersection, deflection, beamline |
| `xrt-sources` | Geometric, bending magnet, wiggler, undulator sources |
| `xrt-waves` | CPU Kirchhoff diffraction integral |
| `xrt-gpu` | GPU compute shaders (wgpu/WGSL) for diffraction and undulator |
| `xrt-pytte` | Takagi-Taupin equation solver for crystal diffraction |
| `xrt-python` | PyO3 bindings for Python interop |

## GPU Kirchhoff Diffraction

The GPU implementation uses a **phase reduction technique** to maintain precision with f32 compute shaders, solving a fundamental limitation in GPU-based X-ray diffraction computation.

### The Problem

In X-ray diffraction, the Kirchhoff integral accumulates phase factors:

```
U = exp(i k r) / r,   where k = E / (ℏc) × 10⁷ mm⁻¹
```

At typical synchrotron energies and distances (E = 10 keV, r = 1000 mm), the phase `k r ≈ 5 × 10¹⁰` radians. GPU shaders operate in f32 (7 significant digits), so `sin(5 × 10¹⁰)` produces **zero** — the result is completely wrong.

This affects all GPU diffraction codes that use f32, including the original Python XRT OpenCL implementation.

### The Solution: Algebraic Phase Reduction

Instead of computing `sin(k × path)` directly, we decompose it:

```
phase = k × path = k × ref_path + k × delta_path
```

where `ref_path` is the distance from the ray centroid to the pixel and `delta_path = path - ref_path` is the small difference. The base phase `exp(i k ref_path)` is computed on the CPU in f64, while the GPU only needs `sin(k × delta_path)` where `delta_path ≈ 10⁻⁵ mm`.

The key challenge is that subtracting two nearly-equal f32 values `path - ref_path` (both ≈ 1000 mm) causes **catastrophic cancellation** — the result rounds to zero.

We solve this with an algebraic identity that avoids the subtraction entirely:

```
path² - ref² = (path + ref)(path - ref)

delta_path = (path² - ref²) / (path + ref)
```

Expanding `path² - ref²` in terms of small ray-centroid offsets `δ = centroid - ray`:

```
path² - ref² = δx(2px - cx - rx) + δy(2py - cy - ry) + δz(2pz - cz - rz)
```

All terms involve `δ ≈ 0.1 mm` (small), so f32 precision is preserved throughout.

### Result

| | xrt-rs GPU | CPU (f64) | XRT Python GPU (OpenCL) |
|---|---|---|---|
| Method | f32 + algebraic phase reduction | f64 direct | f32 direct `sin(k×path)` |
| E=10 keV, r=1000 mm | **0.01% error** | reference | **unusable (output = 0)** |
| Tolerance | 1×10⁻³ | — | N/A |

The GPU-CPU phase factor recombination is:

```
Es_final = exp(i k ref_path) × Es_gpu     ← f64 on CPU
                                 ↑
                     computed with delta_path in f32 on GPU
```

## Validation Framework

All Rust code is validated against Python XRT reference values.

```
validation/
├── generate_fixtures.py      # Generates JSON golden values from Python XRT
├── fixtures/                 # 16 JSON fixture files
├── test_*.py                 # 12 pytest modules (Python XRT ↔ Rust cross-comparison)
└── TODO_validation.md        # Remaining coverage gaps
```

**Test counts:** 56 Rust golden tests, 72 Python pytest (pass).

### Running

```bash
# 1. Generate fixtures from Python XRT
python validation/generate_fixtures.py

# 2. Rust golden tests
cargo test --workspace

# 3. PyO3 build + Python cross-comparison
cd crates/xrt-python && maturin develop --release
pytest validation/ -v
```

### Domains Covered

| Domain | Rust Golden | Python pytest |
|--------|------------|---------------|
| Physical constants | 1 | 1 |
| Scattering (f0, f1/f2) | 8 (Si, Au, W, O + edges) | 12 |
| Material optics (n, μ, Fresnel) | 9 (Si, Au, SiO₂) | 9 |
| Crystal diffraction (Bragg, chi, Darwin, Laue) | 14 | 4 |
| Surface geometry (6 types) | 2 | 8 |
| Ray-surface intersection (4 types) | 8 | 6 |
| Takagi-Taupin rocking curves | 3 | 6 |
| Kirchhoff diffraction | 3 | 2 |
| GPU Kirchhoff (M4 Pro Metal) | 5 | — |
| Geometric + synchrotron sources | — | 8 |
| Multilayer reflectivity + roughness | — | 5 |
| Parametric surfaces | — | 4 |
| Beamline E2E pipeline | 3 | — |

## License

MIT
