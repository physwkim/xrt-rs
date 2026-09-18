//! Surface and ParametricSurface traits for optical elements.
//!
//! All surfaces must implement `Send + Sync` for rayon parallelization.

/// A surface defined by z = f(x, y) in local coordinates.
pub trait Surface: Send + Sync {
    /// Surface height z at local coordinates (x, y).
    fn local_z(&self, x: f64, y: f64) -> f64;

    /// Surface normal [nx, ny, nz] at local coordinates (x, y).
    /// Should be approximately [0, 0, 1] for a nearly-flat surface.
    fn local_n(&self, x: f64, y: f64) -> [f64; 3];

    /// Optional surface distortion (additive to local_z).
    fn local_z_distorted(&self, _x: f64, _y: f64) -> Option<f64> {
        None
    }

    /// Optional grating vector [gx, gy, gz] in 1/mm at (x, y).
    fn local_g(&self, _x: f64, _y: f64) -> Option<[f64; 3]> {
        None
    }

    /// Optional Bragg-plane normal for bent crystal surfaces.
    /// Returns `None` for non-crystal surfaces (default).
    /// For bent crystals, returns the normal to the diffracting atomic planes,
    /// which differs from the physical surface normal `local_n()`.
    fn local_n_bragg(&self, _x: f64, _y: f64) -> Option<[f64; 3]> {
        None
    }
}

/// Bending cross-section for bent crystal/mirror surfaces.
#[derive(Debug, Clone, Copy)]
pub enum CrossSection {
    /// Circular cross-section (exact)
    Circular,
    /// Parabolic approximation (valid for small sagitta)
    Parabolic,
}

/// A surface defined parametrically: r = f(s, φ).
pub trait ParametricSurface: Send + Sync {
    /// Surface radius r at parametric coordinates (s, φ).
    fn local_r(&self, s: f64, phi: f64) -> f64;

    /// Surface normal [nx, ny, nz] at parametric coordinates (s, φ).
    fn local_n(&self, s: f64, phi: f64) -> [f64; 3];

    /// Convert Cartesian (x, y, z) to parametric (s, φ, r).
    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64);

    /// Convert parametric (s, φ, r) to Cartesian (x, y, z).
    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64);

    /// Optional radial distortion (additive to local_r).
    fn local_r_distorted(&self, _s: f64, _phi: f64) -> Option<f64> {
        None
    }
}
