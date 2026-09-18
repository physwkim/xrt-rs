//! Beam container: Structure-of-Arrays layout for ray data.
//!
//! Ported from `xrt/backends/raycing/sources_beams.py`.

use std::io::Write;

use ndarray::Array1;
use num_complex::Complex64;

use crate::consts::DEFAULT_ENERGY;

/// Ray state flags (matches Python XRT conventions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RayState {
    /// Not yet traced
    Undefined = 0,
    /// Good (alive) ray
    Good = 1,
    /// Ray missed the optical element (out of acceptance)
    Out = 2,
    /// Ray absorbed or over aperture
    Over = 3,
    /// Ray is dead (energy below threshold, etc.)
    Dead = -1,
}

impl RayState {
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => Self::Undefined,
            1 => Self::Good,
            2 => Self::Out,
            3 => Self::Over,
            -1 => Self::Dead,
            _ => Self::Undefined,
        }
    }
}

/// Field amplitudes of every ray, allocated together.
///
/// Python creates `Es` and `Ep` in one step under `withAmplitudes`
/// (`sources_beams.py`) and gates `Ep` on `Es` when concatenating, so one
/// without the other is not a state the original can reach. Grouping them
/// makes that combination unrepresentable here.
#[derive(Debug, Clone)]
pub struct Amplitudes {
    pub es: Array1<Complex64>,
    pub ep: Array1<Complex64>,
}

/// Elevation of every ray above the optical surface, allocated together.
#[derive(Debug, Clone)]
pub struct Elevation {
    pub d: Array1<f64>,
    pub x: Array1<f64>,
    pub y: Array1<f64>,
    pub z: Array1<f64>,
}

/// Parametric surface coordinates of every ray, allocated together.
#[derive(Debug, Clone)]
pub struct Parametric {
    pub s: Array1<f64>,
    pub phi: Array1<f64>,
    pub r: Array1<f64>,
}

/// Rays `indices` of `arr`, in the given order.
fn pick_arr<T: Copy>(arr: &Array1<T>, indices: &[usize]) -> Array1<T> {
    Array1::from_iter(indices.iter().map(|&i| arr[i]))
}

/// `a` followed by `b`.
fn concat_arr<T: Clone>(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
    Array1::from_iter(a.iter().chain(b.iter()).cloned())
}

/// Concatenate two optional arrays, keeping the field only if both carry it.
fn concat_opt<T: Clone>(a: Option<&Array1<T>>, b: Option<&Array1<T>>) -> Option<Array1<T>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(concat_arr(a, b)),
        _ => None,
    }
}

impl Amplitudes {
    fn zeros(nrays: usize) -> Self {
        Self {
            es: Array1::from_elem(nrays, Complex64::new(0.0, 0.0)),
            ep: Array1::from_elem(nrays, Complex64::new(0.0, 0.0)),
        }
    }

    fn pick(&self, indices: &[usize]) -> Self {
        Self {
            es: pick_arr(&self.es, indices),
            ep: pick_arr(&self.ep, indices),
        }
    }

    fn concat(a: Option<&Self>, b: Option<&Self>) -> Option<Self> {
        let (a, b) = (a?, b?);
        Some(Self {
            es: concat_arr(&a.es, &b.es),
            ep: concat_arr(&a.ep, &b.ep),
        })
    }
}

impl Elevation {
    fn zeros(nrays: usize) -> Self {
        Self {
            d: Array1::zeros(nrays),
            x: Array1::zeros(nrays),
            y: Array1::zeros(nrays),
            z: Array1::zeros(nrays),
        }
    }

    fn pick(&self, indices: &[usize]) -> Self {
        Self {
            d: pick_arr(&self.d, indices),
            x: pick_arr(&self.x, indices),
            y: pick_arr(&self.y, indices),
            z: pick_arr(&self.z, indices),
        }
    }

    fn concat(a: Option<&Self>, b: Option<&Self>) -> Option<Self> {
        let (a, b) = (a?, b?);
        Some(Self {
            d: concat_arr(&a.d, &b.d),
            x: concat_arr(&a.x, &b.x),
            y: concat_arr(&a.y, &b.y),
            z: concat_arr(&a.z, &b.z),
        })
    }
}

impl Parametric {
    fn zeros(nrays: usize) -> Self {
        Self {
            s: Array1::zeros(nrays),
            phi: Array1::zeros(nrays),
            r: Array1::zeros(nrays),
        }
    }

    fn pick(&self, indices: &[usize]) -> Self {
        Self {
            s: pick_arr(&self.s, indices),
            phi: pick_arr(&self.phi, indices),
            r: pick_arr(&self.r, indices),
        }
    }

    fn concat(a: Option<&Self>, b: Option<&Self>) -> Option<Self> {
        let (a, b) = (a?, b?);
        Some(Self {
            s: concat_arr(&a.s, &b.s),
            phi: concat_arr(&a.phi, &b.phi),
            r: concat_arr(&a.r, &b.r),
        })
    }
}

/// SoA container for a bundle of rays.
///
/// Field layout mirrors Python XRT's `Beam` class for numpy zero-copy
/// compatibility via PyO3 and cache-friendly SIMD access.
///
/// # Invariant
///
/// Every optional per-ray field that is present has length [`Beam::nrays()`],
/// and concatenation keeps a field present only when both operands carry it.
/// The optional fields are therefore private: they are allocated exclusively
/// by the `ensure_*` methods, which take the length from `nrays()`, and
/// [`Beam::concatenate`] drops any field the other beam lacks rather than
/// leaving a short array behind.
#[derive(Debug, Clone)]
pub struct Beam {
    // ── Position ────────────────────────────────────────────────────────
    pub x: Array1<f64>,
    pub y: Array1<f64>,
    pub z: Array1<f64>,

    // ── Direction cosines (normalized) ─────────────────────────────────
    pub a: Array1<f64>,
    pub b: Array1<f64>,
    pub c: Array1<f64>,

    // ── State & scalar per-ray fields ──────────────────────────────────
    pub state: Array1<i32>,
    pub e: Array1<f64>,
    pub path: Array1<f64>,

    // ── Coherency matrix ───────────────────────────────────────────────
    pub jss: Array1<f64>,
    pub jpp: Array1<f64>,
    pub jsp: Array1<Complex64>,

    // ── Optional field amplitudes ──────────────────────────────────────
    amplitudes: Option<Amplitudes>,

    // ── Source parameters (scalars) ────────────────────────────────────
    pub source_sigma_x: f64,
    pub source_sigma_z: f64,
    pub filament_dx: f64,
    pub filament_dz: f64,
    pub filament_dtheta: f64,
    pub filament_dpsi: f64,
    pub filament_dgamma: f64,

    // ── Optional per-ray fields ────────────────────────────────────────
    theta: Option<Array1<f64>>,
    order: Option<Array1<i32>>,
    n_refl: Option<Array1<i32>>,

    // ── Elevation tracking ─────────────────────────────────────────────
    elevation: Option<Elevation>,

    // ── Parametric surface coordinates ─────────────────────────────────
    parametric: Option<Parametric>,
}

/// Summary statistics for good rays in a beam.
#[derive(Debug, Clone)]
pub struct BeamStatistics {
    /// Number of good rays
    pub n_good: usize,
    /// Total number of rays
    pub n_total: usize,
    /// Mean horizontal position [mm]
    pub mean_x: f64,
    /// Mean vertical position [mm]
    pub mean_z: f64,
    /// RMS horizontal beam size [mm]
    pub sigma_x: f64,
    /// RMS vertical beam size [mm]
    pub sigma_z: f64,
    /// RMS horizontal divergence [rad]
    pub sigma_xprime: f64,
    /// RMS vertical divergence [rad]
    pub sigma_zprime: f64,
    /// Mean photon energy [eV]
    pub mean_energy: f64,
    /// Minimum energy [eV]
    pub e_min: f64,
    /// Maximum energy [eV]
    pub e_max: f64,
}

impl std::fmt::Display for BeamStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Beam: {}/{} good rays ({:.1}%)",
            self.n_good,
            self.n_total,
            100.0 * self.n_good as f64 / self.n_total.max(1) as f64
        )?;
        writeln!(
            f,
            "  position:   x={:.3}±{:.3}mm, z={:.3}±{:.3}mm",
            self.mean_x, self.sigma_x, self.mean_z, self.sigma_z
        )?;
        writeln!(
            f,
            "  divergence: x'={:.1}μrad, z'={:.1}μrad",
            self.sigma_xprime * 1e6,
            self.sigma_zprime * 1e6
        )?;
        write!(
            f,
            "  energy:     {:.1} eV [{:.1}, {:.1}]",
            self.mean_energy, self.e_min, self.e_max
        )
    }
}

impl Beam {
    /// Create a new beam with `nrays` rays, all initialized to defaults.
    ///
    /// Matches Python: `Beam(nrays=nrays)` with `copyFrom=None, xyzOnly=False`.
    pub fn new(nrays: usize) -> Self {
        Self {
            x: Array1::zeros(nrays),
            y: Array1::zeros(nrays),
            z: Array1::zeros(nrays),
            a: Array1::zeros(nrays),
            b: Array1::ones(nrays), // default direction: along +y
            c: Array1::zeros(nrays),
            state: Array1::zeros(nrays),
            e: Array1::from_elem(nrays, DEFAULT_ENERGY),
            path: Array1::zeros(nrays),
            jss: Array1::ones(nrays),
            jpp: Array1::zeros(nrays),
            jsp: Array1::from_elem(nrays, Complex64::new(0.0, 0.0)),
            amplitudes: None,
            source_sigma_x: 0.0,
            source_sigma_z: 0.0,
            filament_dx: 0.0,
            filament_dz: 0.0,
            filament_dtheta: 0.0,
            filament_dpsi: 0.0,
            filament_dgamma: 0.0,
            theta: None,
            order: None,
            n_refl: None,
            elevation: None,
            parametric: None,
        }
    }

    /// Create a new beam with field amplitudes allocated.
    pub fn with_amplitudes(nrays: usize) -> Self {
        let mut beam = Self::new(nrays);
        beam.ensure_amplitudes();
        beam
    }

    /// Create a minimal beam with only position arrays (xyz).
    pub fn xyz_only(nrays: usize) -> Self {
        Self {
            x: Array1::zeros(nrays),
            y: Array1::zeros(nrays),
            z: Array1::zeros(nrays),
            a: Array1::zeros(0),
            b: Array1::zeros(0),
            c: Array1::zeros(0),
            state: Array1::zeros(0),
            e: Array1::zeros(0),
            path: Array1::zeros(0),
            jss: Array1::zeros(0),
            jpp: Array1::zeros(0),
            jsp: Array1::from_elem(0, Complex64::new(0.0, 0.0)),
            amplitudes: None,
            source_sigma_x: 0.0,
            source_sigma_z: 0.0,
            filament_dx: 0.0,
            filament_dz: 0.0,
            filament_dtheta: 0.0,
            filament_dpsi: 0.0,
            filament_dgamma: 0.0,
            theta: None,
            order: None,
            n_refl: None,
            elevation: None,
            parametric: None,
        }
    }

    /// Number of rays in this beam.
    pub fn nrays(&self) -> usize {
        self.x.len()
    }

    /// Field amplitudes, if this beam carries them.
    pub fn amplitudes(&self) -> Option<&Amplitudes> {
        self.amplitudes.as_ref()
    }

    /// Field amplitudes for in-place modification, if this beam carries them.
    pub fn amplitudes_mut(&mut self) -> Option<&mut Amplitudes> {
        self.amplitudes.as_mut()
    }

    /// Allocate the field amplitudes if absent, and return them.
    pub fn ensure_amplitudes(&mut self) -> &mut Amplitudes {
        let nrays = self.nrays();
        self.amplitudes
            .get_or_insert_with(|| Amplitudes::zeros(nrays))
    }

    /// Elevation above the optical surface, if this beam carries it.
    pub fn elevation(&self) -> Option<&Elevation> {
        self.elevation.as_ref()
    }

    /// Elevation for in-place modification, if this beam carries it.
    pub fn elevation_mut(&mut self) -> Option<&mut Elevation> {
        self.elevation.as_mut()
    }

    /// Allocate the elevation arrays if absent, and return them.
    pub fn ensure_elevation(&mut self) -> &mut Elevation {
        let nrays = self.nrays();
        self.elevation
            .get_or_insert_with(|| Elevation::zeros(nrays))
    }

    /// Parametric surface coordinates, if this beam carries them.
    pub fn parametric(&self) -> Option<&Parametric> {
        self.parametric.as_ref()
    }

    /// Parametric coordinates for in-place modification, if present.
    pub fn parametric_mut(&mut self) -> Option<&mut Parametric> {
        self.parametric.as_mut()
    }

    /// Allocate the parametric coordinates if absent, and return them.
    pub fn ensure_parametric(&mut self) -> &mut Parametric {
        let nrays = self.nrays();
        self.parametric
            .get_or_insert_with(|| Parametric::zeros(nrays))
    }

    /// Incidence angle of every ray, if this beam carries it.
    pub fn theta(&self) -> Option<&Array1<f64>> {
        self.theta.as_ref()
    }

    /// Allocate the incidence angles if absent, and return them.
    pub fn ensure_theta(&mut self) -> &mut Array1<f64> {
        let nrays = self.nrays();
        self.theta.get_or_insert_with(|| Array1::zeros(nrays))
    }

    /// Diffraction order of every ray, if this beam carries it.
    pub fn order(&self) -> Option<&Array1<i32>> {
        self.order.as_ref()
    }

    /// Allocate the diffraction orders if absent, and return them.
    pub fn ensure_order(&mut self) -> &mut Array1<i32> {
        let nrays = self.nrays();
        self.order.get_or_insert_with(|| Array1::zeros(nrays))
    }

    /// Reflection count of every ray, if this beam carries it.
    pub fn n_refl(&self) -> Option<&Array1<i32>> {
        self.n_refl.as_ref()
    }

    /// Allocate the reflection counts if absent, and return them.
    pub fn ensure_n_refl(&mut self) -> &mut Array1<i32> {
        let nrays = self.nrays();
        self.n_refl.get_or_insert_with(|| Array1::zeros(nrays))
    }

    /// Set all ray states to a given value.
    pub fn set_state(&mut self, state: RayState) {
        self.state.fill(state as i32);
    }

    /// Filter beam to only include rays at the given indices.
    pub fn filter_by_index(&self, indices: &[usize]) -> Self {
        let pick = |arr: &Array1<f64>| pick_arr(arr, indices);

        Self {
            x: pick(&self.x),
            y: pick(&self.y),
            z: pick(&self.z),
            a: pick(&self.a),
            b: pick(&self.b),
            c: pick(&self.c),
            state: pick_arr(&self.state, indices),
            e: pick(&self.e),
            path: pick(&self.path),
            jss: pick(&self.jss),
            jpp: pick(&self.jpp),
            jsp: pick_arr(&self.jsp, indices),
            amplitudes: self.amplitudes.as_ref().map(|amp| amp.pick(indices)),
            source_sigma_x: self.source_sigma_x,
            source_sigma_z: self.source_sigma_z,
            filament_dx: self.filament_dx,
            filament_dz: self.filament_dz,
            filament_dtheta: self.filament_dtheta,
            filament_dpsi: self.filament_dpsi,
            filament_dgamma: self.filament_dgamma,
            theta: self.theta.as_ref().map(&pick),
            order: self.order.as_ref().map(|arr| pick_arr(arr, indices)),
            n_refl: self.n_refl.as_ref().map(|arr| pick_arr(arr, indices)),
            elevation: self.elevation.as_ref().map(|el| el.pick(indices)),
            parametric: self.parametric.as_ref().map(|par| par.pick(indices)),
        }
    }

    /// Filter to only "good" rays (state == 1).
    pub fn filter_good(&self) -> Self {
        let indices: Vec<usize> = self
            .state
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s == RayState::Good as i32)
            .map(|(i, _)| i)
            .collect();
        self.filter_by_index(&indices)
    }

    /// Propagate good rays through free space by a given distance [mm].
    ///
    /// Advances each ray's position along its direction vector and
    /// accumulates the path length.
    pub fn propagate(&mut self, distance: f64) {
        for i in 0..self.nrays() {
            if self.state[i] == RayState::Good as i32 {
                self.x[i] += self.a[i] * distance;
                self.y[i] += self.b[i] * distance;
                self.z[i] += self.c[i] * distance;
                self.path[i] += distance;
            }
        }
    }

    /// Propagate selected rays through free space.
    pub fn propagate_indices(&mut self, distance: f64, indices: &[usize]) {
        for &i in indices {
            self.x[i] += self.a[i] * distance;
            self.y[i] += self.b[i] * distance;
            self.z[i] += self.c[i] * distance;
            self.path[i] += distance;
        }
    }

    /// Get indices of all good rays.
    pub fn good_indices(&self) -> Vec<usize> {
        self.state
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s == RayState::Good as i32)
            .map(|(i, _)| i)
            .collect()
    }

    /// Kill rays outside an energy range (mark as Dead).
    ///
    /// Useful for simulating energy slits / monochromators.
    pub fn filter_energy(&mut self, e_min: f64, e_max: f64) -> usize {
        let mut killed = 0;
        for i in 0..self.nrays() {
            if self.state[i] == RayState::Good as i32 && (self.e[i] < e_min || self.e[i] > e_max) {
                self.state[i] = RayState::Dead as i32;
                killed += 1;
            }
        }
        killed
    }

    /// Get (x, z) footprint positions of good rays.
    pub fn footprint(&self) -> (Vec<f64>, Vec<f64>) {
        let good = self.good_indices();
        let x: Vec<f64> = good.iter().map(|&i| self.x[i]).collect();
        let z: Vec<f64> = good.iter().map(|&i| self.z[i]).collect();
        (x, z)
    }

    /// Write good rays to TSV format.
    ///
    /// Columns: x, y, z, a, b, c, energy, state, jss, jpp, path
    pub fn write_tsv<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "# x\ty\tz\ta\tb\tc\tenergy\tstate\tjss\tjpp\tpath")?;
        for i in 0..self.nrays() {
            if self.state[i] == RayState::Good as i32 {
                writeln!(
                    w,
                    "{:.6}\t{:.6}\t{:.6}\t{:.9}\t{:.9}\t{:.9}\t{:.2}\t{}\t{:.6}\t{:.6}\t{:.6}",
                    self.x[i],
                    self.y[i],
                    self.z[i],
                    self.a[i],
                    self.b[i],
                    self.c[i],
                    self.e[i],
                    self.state[i],
                    self.jss[i],
                    self.jpp[i],
                    self.path[i],
                )?;
            }
        }
        Ok(())
    }

    /// Compute statistics for good rays.
    ///
    /// Returns `None` if there are no good rays.
    pub fn statistics(&self) -> Option<BeamStatistics> {
        let good = self.good_indices();
        if good.is_empty() {
            return None;
        }
        let n = good.len() as f64;

        let mean_x = good.iter().map(|&i| self.x[i]).sum::<f64>() / n;
        let mean_z = good.iter().map(|&i| self.z[i]).sum::<f64>() / n;
        let mean_a = good.iter().map(|&i| self.a[i]).sum::<f64>() / n;
        let mean_c = good.iter().map(|&i| self.c[i]).sum::<f64>() / n;
        let mean_e = good.iter().map(|&i| self.e[i]).sum::<f64>() / n;

        let sigma_x = (good
            .iter()
            .map(|&i| (self.x[i] - mean_x).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let sigma_z = (good
            .iter()
            .map(|&i| (self.z[i] - mean_z).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let sigma_xp = (good
            .iter()
            .map(|&i| (self.a[i] - mean_a).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let sigma_zp = (good
            .iter()
            .map(|&i| (self.c[i] - mean_c).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();

        let e_min = good
            .iter()
            .map(|&i| self.e[i])
            .fold(f64::INFINITY, f64::min);
        let e_max = good
            .iter()
            .map(|&i| self.e[i])
            .fold(f64::NEG_INFINITY, f64::max);

        Some(BeamStatistics {
            n_good: good.len(),
            n_total: self.nrays(),
            mean_x,
            mean_z,
            sigma_x,
            sigma_z,
            sigma_xprime: sigma_xp,
            sigma_zprime: sigma_zp,
            mean_energy: mean_e,
            e_min,
            e_max,
        })
    }

    /// Concatenate another beam onto this one.
    ///
    /// An optional field survives only when both beams carry it, matching
    /// Python's `hasattr(self, X) and hasattr(beam, X)` gate. Python leaves
    /// its own stale array in place when only `self` has the field, which is
    /// what later makes its `filter_by_index` raise `IndexError`; dropping the
    /// field keeps `present ⟹ len == nrays()` true here.
    pub fn concatenate(&mut self, other: &Beam) {
        macro_rules! concat_mandatory {
            ($field:ident) => {
                self.$field = concat_arr(&self.$field, &other.$field);
            };
        }

        concat_mandatory!(x);
        concat_mandatory!(y);
        concat_mandatory!(z);
        concat_mandatory!(a);
        concat_mandatory!(b);
        concat_mandatory!(c);
        concat_mandatory!(state);
        concat_mandatory!(e);
        concat_mandatory!(path);
        concat_mandatory!(jss);
        concat_mandatory!(jpp);
        concat_mandatory!(jsp);

        self.amplitudes = Amplitudes::concat(self.amplitudes.as_ref(), other.amplitudes.as_ref());
        self.elevation = Elevation::concat(self.elevation.as_ref(), other.elevation.as_ref());
        self.parametric = Parametric::concat(self.parametric.as_ref(), other.parametric.as_ref());
        self.theta = concat_opt(self.theta.as_ref(), other.theta.as_ref());
        self.order = concat_opt(self.order.as_ref(), other.order.as_ref());
        self.n_refl = concat_opt(self.n_refl.as_ref(), other.n_refl.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_beam_defaults() {
        let beam = Beam::new(100);
        assert_eq!(beam.nrays(), 100);
        assert_eq!(beam.x[0], 0.0);
        assert_eq!(beam.b[0], 1.0); // default direction +y
        assert_eq!(beam.e[0], DEFAULT_ENERGY);
        assert_eq!(beam.jss[0], 1.0);
        assert_eq!(beam.jpp[0], 0.0);
        assert!(beam.amplitudes().is_none());
    }

    #[test]
    fn test_with_amplitudes() {
        let beam = Beam::with_amplitudes(50);
        let amp = beam.amplitudes().expect("amplitudes were requested");
        assert_eq!(amp.es.len(), 50);
        assert_eq!(amp.ep.len(), 50);
    }

    /// Every optional field that is present is as long as the beam.
    fn assert_optional_lengths(beam: &Beam) {
        let n = beam.nrays();
        if let Some(amp) = beam.amplitudes() {
            assert_eq!(amp.es.len(), n, "es");
            assert_eq!(amp.ep.len(), n, "ep");
        }
        if let Some(el) = beam.elevation() {
            assert_eq!(el.d.len(), n, "elevation_d");
            assert_eq!(el.x.len(), n, "elevation_x");
            assert_eq!(el.y.len(), n, "elevation_y");
            assert_eq!(el.z.len(), n, "elevation_z");
        }
        if let Some(par) = beam.parametric() {
            assert_eq!(par.s.len(), n, "s");
            assert_eq!(par.phi.len(), n, "phi");
            assert_eq!(par.r.len(), n, "r");
        }
        if let Some(theta) = beam.theta() {
            assert_eq!(theta.len(), n, "theta");
        }
        if let Some(order) = beam.order() {
            assert_eq!(order.len(), n, "order");
        }
        if let Some(n_refl) = beam.n_refl() {
            assert_eq!(n_refl.len(), n, "n_refl");
        }
    }

    #[test]
    fn amplitudes_survive_concatenation_only_when_both_beams_carry_them() {
        // present ∥ present
        let mut both = Beam::with_amplitudes(3);
        both.concatenate(&Beam::with_amplitudes(2));
        assert_eq!(both.nrays(), 5);
        assert_eq!(both.amplitudes().expect("both carried them").es.len(), 5);
        assert_optional_lengths(&both);

        // present ∥ absent
        let mut left = Beam::with_amplitudes(3);
        left.concatenate(&Beam::new(2));
        assert_eq!(left.nrays(), 5);
        assert!(left.amplitudes().is_none());
        assert_optional_lengths(&left);

        // absent ∥ present
        let mut right = Beam::new(3);
        right.concatenate(&Beam::with_amplitudes(2));
        assert_eq!(right.nrays(), 5);
        assert!(right.amplitudes().is_none());
        assert_optional_lengths(&right);
    }

    #[test]
    fn parametric_survives_concatenation_only_when_both_beams_carry_it() {
        let mut parametric = Beam::new(3);
        parametric.ensure_parametric();
        assert_eq!(parametric.parametric().expect("just allocated").s.len(), 3);

        let mut mixed = parametric.clone();
        mixed.concatenate(&Beam::new(2));
        assert!(mixed.parametric().is_none());
        assert_optional_lengths(&mixed);

        let mut other = Beam::new(2);
        other.ensure_parametric();
        parametric.concatenate(&other);
        assert_eq!(parametric.parametric().expect("both carried it").r.len(), 5);
        assert_optional_lengths(&parametric);
    }

    #[test]
    fn a_dropped_field_can_be_reallocated_at_the_new_length() {
        let mut beam = Beam::with_amplitudes(3);
        beam.concatenate(&Beam::new(2));
        assert!(beam.amplitudes().is_none());
        assert_eq!(beam.ensure_amplitudes().es.len(), 5);
        assert_optional_lengths(&beam);
    }

    #[test]
    fn filtering_shortens_every_present_optional_field() {
        let mut beam = Beam::with_amplitudes(5);
        beam.ensure_parametric();
        beam.ensure_elevation();
        beam.ensure_theta();
        beam.ensure_order();
        beam.ensure_n_refl();
        beam.set_state(RayState::Good);
        beam.state[1] = RayState::Dead as i32;
        beam.state[3] = RayState::Out as i32;

        let good = beam.filter_good();
        assert_eq!(good.nrays(), 3);
        assert_optional_lengths(&good);
    }

    #[test]
    fn test_filter_good() {
        let mut beam = Beam::new(5);
        beam.state[0] = RayState::Good as i32;
        beam.state[1] = RayState::Out as i32;
        beam.state[2] = RayState::Good as i32;
        beam.state[3] = RayState::Dead as i32;
        beam.state[4] = RayState::Good as i32;

        let good = beam.filter_good();
        assert_eq!(good.nrays(), 3);
    }

    #[test]
    fn test_concatenate() {
        let mut beam1 = Beam::new(3);
        let beam2 = Beam::new(2);
        beam1.concatenate(&beam2);
        assert_eq!(beam1.nrays(), 5);
    }

    #[test]
    fn test_ray_state_roundtrip() {
        assert_eq!(RayState::from_i32(1), RayState::Good);
        assert_eq!(RayState::from_i32(-1), RayState::Dead);
        assert_eq!(RayState::from_i32(99), RayState::Undefined);
    }

    #[test]
    fn test_statistics() {
        let mut beam = Beam::new(100);
        beam.set_state(RayState::Good);
        for i in 0..100 {
            beam.b[i] = 1.0;
            beam.e[i] = 10000.0 + i as f64;
        }
        let stats = beam.statistics().unwrap();
        assert_eq!(stats.n_good, 100);
        assert!((stats.mean_energy - 10049.5).abs() < 0.1);
        assert!((stats.e_min - 10000.0).abs() < 0.1);
        assert!((stats.e_max - 10099.0).abs() < 0.1);
    }

    #[test]
    fn test_statistics_empty() {
        let beam = Beam::new(5);
        // No good rays → None
        assert!(beam.statistics().is_none());
    }

    #[test]
    fn test_statistics_display() {
        let mut beam = Beam::new(10);
        beam.set_state(RayState::Good);
        for i in 0..10 {
            beam.b[i] = 1.0;
            beam.e[i] = 10000.0;
        }
        let stats = beam.statistics().unwrap();
        let s = format!("{stats}");
        assert!(s.contains("10/10 good"));
    }

    #[test]
    fn test_filter_energy() {
        let mut beam = Beam::new(100);
        beam.set_state(RayState::Good);
        for i in 0..100 {
            beam.e[i] = 8000.0 + i as f64 * 40.0; // 8000..11960
        }
        let killed = beam.filter_energy(9000.0, 11000.0);
        assert!(killed > 0);
        // Remaining good rays should be in [9000, 11000]
        for i in 0..100 {
            if beam.state[i] == RayState::Good as i32 {
                assert!(beam.e[i] >= 9000.0 && beam.e[i] <= 11000.0);
            }
        }
    }

    #[test]
    fn test_write_tsv() {
        let mut beam = Beam::new(3);
        beam.set_state(RayState::Good);
        beam.b[0] = 1.0;
        beam.b[1] = 1.0;
        beam.b[2] = 1.0;
        beam.e[0] = 10000.0;
        beam.e[1] = 10000.0;
        beam.e[2] = 10000.0;
        beam.state[1] = RayState::Dead as i32;

        let mut buf = Vec::new();
        beam.write_tsv(&mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("# x\ty\tz"));
        // Only 2 good rays should be written
        let data_lines: Vec<&str> = output.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(data_lines.len(), 2);
    }

    #[test]
    fn test_footprint() {
        let mut beam = Beam::new(5);
        beam.set_state(RayState::Good);
        beam.x[0] = 1.0;
        beam.z[0] = 2.0;
        beam.state[1] = RayState::Dead as i32;
        let (x, _z) = beam.footprint();
        assert_eq!(x.len(), 4); // 4 good rays
        assert!((x[0] - 1.0).abs() < 1e-15);
    }
}
