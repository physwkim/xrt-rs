# Changelog

All notable changes to xrt-rs are documented in this file.

## [0.2.0] — 2026-03-31

### Fixed
- **Inverse rotation sequence** in all 5 OE types (oe, material_oe, crystal_oe,
  param_oe, grating_oe). The inverse transform now uses reversed `[X,Y,Z]`
  sequence instead of the forward `[Z,Y,X]`. This caused ~28 mm centroid
  offsets when OEs had large roll angles (e.g. HFM with roll = π/2).
- **DCM crystal 2 fine pitch sign**: changed from `bragg + fine_pitch` to
  `bragg - fine_pitch`. The 2nd crystal faces downward, so a positive
  fine_pitch must reduce the effective Bragg angle.
- **O(n²) index lookup** in `apply_fresnel()` and `apply_crystal_amplitude()`.
  Replaced per-ray `Vec::position()` linear search with `HashMap` O(1) lookup.
  For 50k rays this eliminated ~2.5 billion comparisons (1294 ms → 26 ms).
- **Undulator `harmonic_peaks_detected` test**: fixed array length mismatch
  between `energies` (200) and `thetas`/`psis` (1), and narrowed energy scan
  to ±30% around E₁ for reliable peak detection.
- **`assert!(... || true)` in scan.rs**: replaced no-op assertion with
  meaningful `assert!(cs[1].c0.re.is_finite())`.

### Added
- **CrystalSi and DCM Python bindings** (`py_crystal.rs`): `CrystalSi` class
  with `get_bragg_angle()`, and `DCM` class with `double_reflect()` and
  `cryst2_fine_pitch` setter for beamline simulation.
- **FlatMirror pitch/roll setters**: runtime adjustment for RL alignment.
- **`RotationParams::inverse_sequence()`**: reversed rotation sequence
  constructor for correct inverse transforms.
- **`check_equal_lengths()` helper**: input length validation for all Python
  batch functions (`find_intersection_rs`, `find_intersection_parametric_rs`,
  `diffraction_integral_rs`, `tt_solve_rs`, `multilayer_amplitude_rs`).
  Mismatched lengths now raise `ValueError` instead of panicking.
- **Monochromatic fast path** in `Element::get_f1f2()`: skips per-ray binary
  search when all rays share the same energy.
- **Uniform stol fast path** in `CrystalFcc::get_structure_factor()`: computes
  `f0_scalar` once when all `sin_theta/lambda` values are equal.
- **`position_roll` field** on `OeParams` with forward/inverse apply methods.

### Changed
- `cargo clippy --workspace --all-targets` now produces **0 warnings**.
- Removed dead-code `JohannCylinder` and `Hyperbolic` variants from Python
  surface enums (no corresponding Py\* classes existed).
- Updated `PLAN_remaining.md`: marked mosaicity as implemented.

### Performance
- Full beamline trace (source + DCM + VFM + HFM, 50k rays):
  **26 ms** (xrt-rs) vs **52 ms** (xrt Python) — **2× faster**.
- Previously 1294 ms due to O(n²) bug — **50× improvement** from v0.1.0.

---

## [0.1.0] — 2026-03-30

Initial release of xrt-rs: a Rust reimplementation of the XRT X-ray tracing
engine with Python bindings via PyO3.

### Core engine
- 9-crate workspace: xrt-core, xrt-math, xrt-materials, xrt-oes, xrt-sources,
  xrt-waves, xrt-pytte, xrt-gpu, xrt-python.
- 47/47 OE surface types matching Python XRT parity.
- Dynamical diffraction (Belyakov & Dmitrienko) for crystal optics.
- Takagi-Taupin ODE solver for strained crystals.
- GPU-accelerated Kirchhoff diffraction with algebraic phase reduction.
- Multilayer reflectivity via Parratt recursion with Nevot-Croce roughness.

### Materials
- CrystalSi, CrystalDiamond, CrystalFcc, CrystalFromCell variants.
- Temperature-dependent Si lattice (Swenson 1983).
- Chantler, Henke, Brennan-Cowan scattering factor tables.
- Fresnel reflectivity for mirrors, plates, lenses, gratings.

### Sources
- GeometricSource with configurable spatial/angular/energy distributions.
- BendingMagnet, Wiggler, Undulator synchrotron sources.
- Full polarization support (horizontal, vertical, circular, ±45°).

### Python bindings
- 10 low-level functions + 15 high-level Py\* classes.
- Drop-in compatible API with XRT Python conventions.

### Validation
- 443 Rust unit tests, 123 Python cross-validation tests.
- Cross-compared against XRT Python, Shadow3, and SRW.
- ~98% API coverage.

### GUI
- XRT-compatible beamline simulator GUI with 3D viewer.
- 4-panel beam profile (ImageView) and energy colormap.
