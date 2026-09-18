# xrt-rs

A unified X-ray optics simulation engine in Rust, combining the capabilities of three established codes — [XRT](https://github.com/kklmn/xrt), [Shadow3](https://github.com/oasys-kit/shadow3), and [SRW](https://github.com/ochubar/SRW) — with GPU acceleration via wgpu.

**22K LOC Rust** | **42 surface types** | **47 XRT OE equivalents** | **94 cross-validated tests**

## Features

### Sources
- Geometric source (Gaussian, flat, annulus, point, weighted multi-line)
- Bending magnet, wiggler, undulator (with GPU acceleration)
- Full polarization: horizontal, vertical, ±45°, circular, custom

### Optical Elements
- **Mirrors**: flat, spherical, toroidal, elliptical, parabolic, hyperbolic, conical, cylindrical, bent flat, VFM/VCM
- **Crystals**: Si, Ge, arbitrary unit cell — Bragg/Laue reflected/transmitted, mosaicity convolution, Debye-Waller
- **Gratings**: blazed, laminar, VLS, holographic — with blaze efficiency (sinc²)
- **Multilayers**: Parratt recursion, Nevot-Croce roughness, reflected/transmitted
- **Lenses**: paraboloid, cylinder, CRL (compound refractive lens), double lens
- **Special**: FZP, diced optics, DCM (double crystal monochromator), surface error maps, mesh import
- **47 surface types** — full parity with XRT Python + Shadow3 extensions

### Physics
- Kirchhoff diffraction integral (CPU with Kahan summation + GPU with phase reduction)
- Fresnel 2D wavefront propagation
- Takagi-Taupin dynamical diffraction solver
- Fresnel reflectivity (s/p polarization, coherency matrix tracking)
- Optical path length tracking
- Debye-Waller surface roughness model

### Performance
- **Rayon** parallel ray tracing across all cores
- **wgpu GPU** compute shaders (Metal/Vulkan/DX12) for Kirchhoff diffraction and undulator radiation
- ~2x faster than XRT Python for bending magnet source generation
- 0 `unsafe`, 0 clippy warnings

## Module Structure

One published crate, `xrt-rs`, with a module per subsystem:

```
xrt_rs::
├── core        Physical constants, beam, transforms, optical path
├── math        Root finding, interpolation
├── materials   Elements, scattering (f0/f1/f2), materials, crystals, multilayers
├── oes         42 surface types, 5 OE wrappers, beamline pipeline
├── sources     Geometric, BM, wiggler, undulator
├── waves       Kirchhoff diffraction (CPU), Fresnel propagation
├── pytte       Takagi-Taupin ODE solver
└── gpu         wgpu/WGSL shaders for diffraction + undulator ("gpu" feature)

crates/xrt-python   PyO3 bindings, published to PyPI as the xrt-rs wheel
```

The `gpu` feature is off by default; without it the auto-dispatching entry
points run their f64 CPU paths.

## GPU Phase Reduction

The GPU Kirchhoff shader solves a fundamental f32 precision problem that affects all GPU diffraction codes.

**Problem:** At E=10 keV, path=1000 mm, the phase `k×path ≈ 5×10¹⁰`. GPU f32 `sin()` returns zero — XRT Python's OpenCL has this same issue.

**Solution:** Algebraic phase reduction avoids catastrophic cancellation:

```
delta_path = (path² - ref²) / (path + ref)
           = Σ δᵢ(2pᵢ - cᵢ - rᵢ) / (path + ref)
```

where `δ = centroid - ray ≈ 0.1 mm` (small, f32-safe). The base phase `exp(i·k·ref_path)` is restored on the CPU in f64.

| | xrt-rs GPU | CPU f64 | XRT Python GPU |
|---|---|---|---|
| E=10 keV, 1000 mm | **0.01% error** | reference | **output = 0** |

## Validation

Cross-validated against **three independent codes**:

| Source | Tests | Domains |
|--------|-------|---------|
| **XRT Python** | 84 | Scattering, materials, crystals, surfaces, intersection, diffraction, TT, multilayer, sources |
| **Shadow3 Fortran** | 4 | Source statistics, conic surfaces, spherical mirror, direction cosines |
| **SRW C++** | 6 | Undulator spectrum, Gaussian beam propagation, thin lens focusing |
| **Total** | **94 pass** | |

Plus **325+ Rust unit/golden tests** covering all public APIs.

### Running Tests

```bash
# Rust tests
cargo test --workspace

# Generate Python XRT fixtures
/path/to/xrt-env/python validation/generate_fixtures.py

# Generate Shadow3 fixtures
/path/to/shadow3-env/python validation/generate_shadow3_fixtures.py

# Generate SRW fixtures
/path/to/xrt-env/python validation/generate_srw_fixtures.py

# PyO3 build + cross-comparison
cd crates/xrt-python && maturin develop --release
pytest validation/ -v
```

## Bugs Found During Validation

| Bug | Location | Impact | Fix |
|-----|----------|--------|-----|
| GPU f32 phase overflow | `kirchhoff.wgsl` | Output = 0 at long distances | Algebraic phase reduction |
| Darwin width `.re` vs `.norm()` | `crystal.rs` | 50× error in Darwin width | Use complex modulus |
| XRT OpenCL ACCELERATOR query | `myopencl.py` | Crash on Apple Silicon | Catch `LogicError` |
| Fixture E₁ unit (mm vs cm) | `generate_fixtures.py` | 10× wrong undulator energy | Fix period unit |

## Examples

```rust
use xrt_rs::sources::geometric::GeometricSource;
use xrt_rs::oes::beamline::{Beamline, OeParamsBuilder};
use xrt_rs::oes::material_oe::MaterialOpticalElement;
use xrt_rs::oes::surfaces::flat::FlatSurface;
use xrt_rs::materials::material::{Material, MaterialKind};
use xrt_rs::materials::data::ScatteringTable;

// Create source
let mut beam = GeometricSource { nrays: 10000, ..Default::default() }.shine();

// Create Si mirror
let si = Material::new(&["Si"], None, 2.33, MaterialKind::Mirror, None,
    ScatteringTable::ChantlerTotal).unwrap();
let mirror = MaterialOpticalElement::new(
    FlatSurface,
    OeParamsBuilder::new().pitch(0.003).build(),
    si,
);

// Trace
let bl = Beamline::new()
    .add_material("M1", mirror)
    .drift(5000.0);
let output = bl.propagate(&mut beam);
println!("Efficiency: {:.1}%", output.efficiency() * 100.0);
```

## Data

Built-in scattering factor tables (no external dependencies):

| File | Size | Content |
|------|------|---------|
| `f0_xop.dat` | 82K | f0 Gaussian coefficients (all elements) |
| `AtomicData.dat` | 6.8K | Atomic masses, densities |
| `Chantler.npz` | 655K | f1/f2, 11 eV – 405 keV |
| `Henke.npz` | 551K | f1/f2, 10 eV – 30 keV |
| `BrCo.npz` | 566K | f1/f2, Brennan & Cowan |

## License

MIT
