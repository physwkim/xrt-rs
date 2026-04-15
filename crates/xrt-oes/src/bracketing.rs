//! Bracketing: estimate t_min and t_max for ray-surface intersection.
//!
//! Ported from oes_base.py:1488-1552.

/// Default half-size of an optical element [mm].
pub const MAX_HALF_SIZE: f64 = 1e6;

/// Default maximum depth [mm].
pub const MAX_DEPTH: f64 = 1e4;

/// Small offset for bracketing.
pub const DT: f64 = 1e-6;

/// Estimate t_min and t_max for a single coordinate component.
///
/// Given position `xyz` and direction `abc` along one axis,
/// computes the t range that spans `[lim_min, lim_max]`.
#[inline]
fn set_t(xyz: f64, abc: f64, lim_min: f64, lim_max: f64) -> (f64, f64) {
    if abc.abs() < 1e-300 {
        // Ray is parallel to this axis — use full range
        return (-MAX_HALF_SIZE, MAX_HALF_SIZE);
    }
    if abc > 0.0 {
        let t_min = (lim_min - xyz) / abc - DT;
        let t_max = (lim_max - xyz) / abc + DT;
        (t_min, t_max)
    } else {
        let t_min = (lim_max - xyz) / abc - DT;
        let t_max = (lim_min - xyz) / abc + DT;
        (t_min, t_max)
    }
}

/// Estimate the bracketing interval [t_min, t_max] for a single ray.
///
/// Selects the axis with the largest direction component to get the
/// tightest bounds.
pub fn bracket_ray(
    x: f64,
    y: f64,
    z: f64,
    a: f64,
    b: f64,
    c: f64,
    phys_x: Option<[f64; 2]>,
    phys_y: Option<[f64; 2]>,
) -> (f64, f64) {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let abs_c = c.abs();
    let max_abc = abs_a.max(abs_b).max(abs_c);

    let (t_min, t_max) = if max_abc == abs_a {
        let lims = phys_x.unwrap_or([-MAX_HALF_SIZE, MAX_HALF_SIZE]);
        let lim_min = if lims[0] > f64::NEG_INFINITY {
            lims[0]
        } else {
            -MAX_HALF_SIZE
        };
        let lim_max = if lims[1] < f64::INFINITY {
            lims[1]
        } else {
            MAX_HALF_SIZE
        };
        set_t(x, a, lim_min, lim_max)
    } else if max_abc == abs_b {
        let lims = phys_y.unwrap_or([-MAX_HALF_SIZE, MAX_HALF_SIZE]);
        let lim_min = if lims[0] > f64::NEG_INFINITY {
            lims[0]
        } else {
            -MAX_HALF_SIZE
        };
        let lim_max = if lims[1] < f64::INFINITY {
            lims[1]
        } else {
            MAX_HALF_SIZE
        };
        set_t(y, b, lim_min, lim_max)
    } else {
        set_t(z, c, -MAX_DEPTH, MAX_DEPTH)
    };

    // Allow moderate backward search for rays near the surface.
    // The original -1e-6 was too restrictive for large pitch angles.
    // At DCM theta=14°, rays can start ~2mm past the surface and need
    // t ≈ -8mm to find the intersection.
    let t_min = t_min.max(-100.0);

    (t_min, t_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_ray_going_down() {
        // Ray at (0, 0, 10) going straight down
        let (t_min, t_max) = bracket_ray(0.0, 0.0, 10.0, 0.0, 0.0, -1.0, None, None);
        assert!(t_min < 10.0);
        assert!(t_max > 10.0);
    }

    #[test]
    fn bracket_with_limits() {
        // Ray at y=-100, going +y
        let (t_min, t_max) = bracket_ray(
            0.0,
            -100.0,
            10.0,
            0.0,
            1.0,
            -0.01,
            None,
            Some([-50.0, 50.0]),
        );
        // Should bracket based on y limits
        assert!(t_min < 50.0, "t_min = {t_min}");
        assert!(t_max > 50.0, "t_max = {t_max}");
    }
}
