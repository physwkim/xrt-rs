//! Root-finding methods for ray-surface intersection.
//!
//! Ported from `xrt/backends/raycing/oes_base.py:831-946`.
//!
//! These methods operate per-ray (scalar), designed for use with rayon
//! parallel iteration rather than the Python vectorized array approach.

/// Configuration for root-finding.
#[derive(Debug, Clone, Copy)]
pub struct RootFindConfig {
    pub z_eps: f64,
    pub max_iter: usize,
}

impl Default for RootFindConfig {
    fn default() -> Self {
        Self {
            z_eps: 1e-12,
            max_iter: 100,
        }
    }
}

/// Result of a single-ray intersection search.
#[derive(Debug, Clone)]
pub struct IntersectionResult {
    /// Ray parameter at intersection
    pub t: f64,
    /// Local coordinates at intersection
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Residual dz at convergence
    pub dz: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Whether the root was found within tolerance
    pub converged: bool,
}

/// Trait for evaluating the surface distance function `dz`.
///
/// Given ray origin (x0, y0, z0) and direction (a, b, c), computes the
/// signed distance from the surface at parameter t:
///   point = (x0 + a*t, y0 + b*t, z0 + c*t)
///   dz = local_z(x_local, y_local) - z_local (adjusted by invert_normal)
///
/// Returns (dz, x_local, y_local, z_local).
pub trait SurfaceEval {
    fn find_dz(
        &self,
        t: f64,
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
        invert_normal: i32,
    ) -> (f64, f64, f64, f64);
}

/// Secant method for root finding (single ray).
///
/// Equivalent to Python `_use_my_method` from oes_base.py:831-857.
pub fn secant_method<S: SurfaceEval>(
    surface: &S,
    mut t1: f64,
    mut t2: f64,
    x0: f64,
    y0: f64,
    z0: f64,
    a: f64,
    b: f64,
    c: f64,
    invert_normal: i32,
    t_min: f64,
    t_max: f64,
    config: &RootFindConfig,
) -> IntersectionResult {
    let (mut dz1, _, _, _) = surface.find_dz(t1, x0, y0, z0, a, b, c, invert_normal);
    let (mut dz2, mut x2, mut y2, mut z2) = surface.find_dz(t2, x0, y0, z0, a, b, c, invert_normal);

    let mut numit = 2;
    while dz2.abs() > config.z_eps && numit < config.max_iter {
        let t = t1;
        let dz = dz1;
        t1 = t2;
        dz1 = dz2;

        // Secant step
        let denom = dz1 - dz;
        if denom.abs() < 1e-300 {
            break; // Avoid division by zero
        }
        t2 = t - (t1 - t) * dz / denom;

        // Clamp to [t_min, t_max]
        t2 = t2.clamp(t_min, t_max);

        let result = surface.find_dz(t2, x0, y0, z0, a, b, c, invert_normal);
        dz2 = result.0;
        x2 = result.1;
        y2 = result.2;
        z2 = result.3;

        // If same sign as dz1, swap back to keep bracketing
        if dz2.signum() == dz1.signum() {
            t1 = t;
            dz1 = dz;
        }

        numit += 1;
    }

    IntersectionResult {
        t: t2,
        x: x2,
        y: y2,
        z: z2,
        dz: dz2,
        iterations: numit,
        converged: dz2.abs() <= config.z_eps,
    }
}

/// Brent's method for root finding (single ray).
///
/// Equivalent to Python `_use_Brent_method` from oes_base.py:859-946.
/// Combines inverse quadratic interpolation, secant method, and bisection.
pub fn brent_method<S: SurfaceEval>(
    surface: &S,
    mut t1: f64, // a (bracket endpoint)
    mut t2: f64, // b (best guess)
    x0: f64,
    y0: f64,
    z0: f64,
    a: f64,
    b: f64,
    c: f64,
    invert_normal: i32,
    config: &RootFindConfig,
) -> IntersectionResult {
    let (mut dz1, _, _, _) = surface.find_dz(t1, x0, y0, z0, a, b, c, invert_normal);
    let (mut dz2, mut x2, mut y2, mut z2) = surface.find_dz(t2, x0, y0, z0, a, b, c, invert_normal);

    // Ensure |dz1| >= |dz2| (swap if needed)
    if dz1.abs() < dz2.abs() {
        std::mem::swap(&mut t1, &mut t2);
        std::mem::swap(&mut dz1, &mut dz2);
    }

    let mut t3 = t1; // c := a
    let mut dz3 = dz1;
    let mut t4 = 0.0; // d
    let mut mflag = true;
    let mut numit = 2;

    while dz2.abs() > config.z_eps && numit < config.max_iter {
        let (xa, xb, xc, xd) = (t1, t2, t3, t4);
        let (fa, fb, fc) = (dz1, dz2, dz3);

        // Compute trial point
        let xs = if (fa - fc).abs() > 1e-300 && (fb - fc).abs() > 1e-300 {
            // Inverse quadratic interpolation
            xa * fb * fc / ((fa - fb) * (fa - fc))
                + fa * xb * fc / ((fb - fa) * (fb - fc))
                + fa * fb * xc / ((fc - fa) * (fc - fb))
        } else {
            // Secant method
            xb - fb * (xb - xa) / (fb - fa)
        };

        // Brent's conditions for rejecting the trial point
        let mid = (3.0 * xa + xb) / 4.0;
        let cond1 = xs < mid.min(xb) || xs > mid.max(xb);
        let cond2 = mflag && (xs - xb).abs() >= (xb - xc).abs() / 2.0;
        let cond3 = !mflag && (xs - xb).abs() >= (xc - xd).abs() / 2.0;
        let cond4 = mflag && (xb - xc).abs() < config.z_eps;
        let cond5 = !mflag && (xc - xd).abs() < config.z_eps;

        let (xs, new_mflag) = if cond1 || cond2 || cond3 || cond4 || cond5 {
            ((xa + xb) / 2.0, true) // bisection
        } else {
            (xs, false)
        };

        let result = surface.find_dz(xs, x0, y0, z0, a, b, c, invert_normal);
        let fs = result.0;
        x2 = result.1;
        y2 = result.2;
        z2 = result.3;

        // Update brackets
        t4 = xc;
        t3 = xb;
        dz3 = fb;

        if (fa < 0.0 && fs > 0.0) || (fa > 0.0 && fs < 0.0) {
            t2 = xs;
            dz2 = fs;
        } else {
            t1 = xs;
            dz1 = fs;
        }

        // Ensure |dz1| >= |dz2|
        if dz1.abs() < dz2.abs() {
            std::mem::swap(&mut t1, &mut t2);
            std::mem::swap(&mut dz1, &mut dz2);
        }

        mflag = new_mflag;
        numit += 1;
    }

    IntersectionResult {
        t: t2,
        x: x2,
        y: y2,
        z: z2,
        dz: dz2,
        iterations: numit,
        converged: dz2.abs() <= config.z_eps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat surface at z = 0 for testing.
    struct FlatSurface;

    impl SurfaceEval for FlatSurface {
        fn find_dz(
            &self,
            t: f64,
            x0: f64,
            y0: f64,
            z0: f64,
            a: f64,
            b: f64,
            c: f64,
            _invert_normal: i32,
        ) -> (f64, f64, f64, f64) {
            let x = x0 + a * t;
            let y = y0 + b * t;
            let z = z0 + c * t;
            // dz = surface_z(x, y) - z = 0 - z = -z
            (-z, x, y, z)
        }
    }

    /// A spherical surface: z = R - sqrt(R^2 - x^2 - y^2)
    struct SphericalSurface {
        r: f64,
    }

    impl SurfaceEval for SphericalSurface {
        fn find_dz(
            &self,
            t: f64,
            x0: f64,
            y0: f64,
            z0: f64,
            a: f64,
            b: f64,
            c: f64,
            _invert_normal: i32,
        ) -> (f64, f64, f64, f64) {
            let x = x0 + a * t;
            let y = y0 + b * t;
            let z = z0 + c * t;
            let r2 = self.r * self.r - x * x - y * y;
            let surface_z = if r2 > 0.0 { self.r - r2.sqrt() } else { self.r };
            (surface_z - z, x, y, z)
        }
    }

    #[test]
    fn flat_surface_secant() {
        let surface = FlatSurface;
        let config = RootFindConfig::default();

        // Ray from (0, 0, 1) going down at 45°
        let result = secant_method(
            &surface, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0, 0.0, 10.0, &config,
        );

        assert!(result.converged);
        assert!((result.t - 1.0).abs() < 1e-10);
        assert!((result.z).abs() < 1e-10);
    }

    #[test]
    fn flat_surface_brent() {
        let surface = FlatSurface;
        let config = RootFindConfig::default();

        let result = brent_method(
            &surface, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0, &config,
        );

        assert!(result.converged);
        assert!((result.t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spherical_surface_brent() {
        let surface = SphericalSurface { r: 1000.0 };
        let config = RootFindConfig::default();

        // Ray from (0, 0, -10) going along +z
        let result = brent_method(
            &surface, 0.0, 20.0, 0.0, 0.0, -10.0, 0.0, 0.0, 1.0, 0, &config,
        );

        assert!(result.converged);
        // At x=0, y=0, surface_z = R - R = 0, so intersection at z=0
        assert!((result.z).abs() < 1e-10);
    }

    #[test]
    fn spherical_surface_off_axis() {
        let r = 1000.0;
        let surface = SphericalSurface { r };
        let config = RootFindConfig::default();

        // Ray from (10, 0, -10) going along +z
        let result = brent_method(
            &surface, 0.0, 20.0, 10.0, 0.0, -10.0, 0.0, 0.0, 1.0, 0, &config,
        );

        assert!(result.converged);
        // Expected z = R - sqrt(R^2 - 100)
        let expected_z = r - (r * r - 100.0).sqrt();
        assert!(
            (result.z - expected_z).abs() < 1e-8,
            "z = {}, expected {}",
            result.z,
            expected_z
        );
    }
}
