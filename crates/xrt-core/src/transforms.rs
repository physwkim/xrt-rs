//! 3D coordinate rotation functions for ray tracing.
//!
//! Ported from `xrt/backends/raycing/__init__.py` (lines 263–415).
//! All rotations use pre-computed cos/sin angles for efficiency.

use ndarray::{Array1, Zip};

use crate::beam::Beam;
use crate::error::{Result, XrtError};

// ── Scalar rotations ────────────────────────────────────────────────────────

/// Rotate around the X axis (pitch). Returns (y', z').
#[inline]
pub fn rotate_x(y: f64, z: f64, cos_a: f64, sin_a: f64) -> (f64, f64) {
    (cos_a * y - sin_a * z, sin_a * y + cos_a * z)
}

/// Rotate around the Y axis (roll). Returns (x', z').
#[inline]
pub fn rotate_y(x: f64, z: f64, cos_a: f64, sin_a: f64) -> (f64, f64) {
    (cos_a * x + sin_a * z, -sin_a * x + cos_a * z)
}

/// Rotate around the Z axis (yaw). Returns (x', y').
#[inline]
pub fn rotate_z(x: f64, y: f64, cos_a: f64, sin_a: f64) -> (f64, f64) {
    (cos_a * x - sin_a * y, sin_a * x + cos_a * y)
}

// ── Array rotations ─────────────────────────────────────────────────────────

/// Rotate arrays around X axis (pitch) in-place.
pub fn rotate_x_arrays(
    y: &mut Array1<f64>,
    z: &mut Array1<f64>,
    cos_a: f64,
    sin_a: f64,
) {
    Zip::from(y.view_mut()).and(z.view_mut()).for_each(|yi, zi| {
        let (yn, zn) = rotate_x(*yi, *zi, cos_a, sin_a);
        *yi = yn;
        *zi = zn;
    });
}

/// Rotate arrays around Y axis (roll) in-place.
pub fn rotate_y_arrays(
    x: &mut Array1<f64>,
    z: &mut Array1<f64>,
    cos_a: f64,
    sin_a: f64,
) {
    Zip::from(x.view_mut()).and(z.view_mut()).for_each(|xi, zi| {
        let (xn, zn) = rotate_y(*xi, *zi, cos_a, sin_a);
        *xi = xn;
        *zi = zn;
    });
}

/// Rotate arrays around Z axis (yaw) in-place.
pub fn rotate_z_arrays(
    x: &mut Array1<f64>,
    y: &mut Array1<f64>,
    cos_a: f64,
    sin_a: f64,
) {
    Zip::from(x.view_mut()).and(y.view_mut()).for_each(|xi, yi| {
        let (xn, yn) = rotate_z(*xi, *yi, cos_a, sin_a);
        *xi = xn;
        *yi = yn;
    });
}

// ── Indexed array rotations ─────────────────────────────────────────────────

/// Rotate selected elements of arrays around X axis in-place.
pub fn rotate_x_indexed(
    y: &mut Array1<f64>,
    z: &mut Array1<f64>,
    indices: &[usize],
    cos_a: f64,
    sin_a: f64,
) {
    for &i in indices {
        let (yn, zn) = rotate_x(y[i], z[i], cos_a, sin_a);
        y[i] = yn;
        z[i] = zn;
    }
}

/// Rotate selected elements of arrays around Y axis in-place.
pub fn rotate_y_indexed(
    x: &mut Array1<f64>,
    z: &mut Array1<f64>,
    indices: &[usize],
    cos_a: f64,
    sin_a: f64,
) {
    for &i in indices {
        let (xn, zn) = rotate_y(x[i], z[i], cos_a, sin_a);
        x[i] = xn;
        z[i] = zn;
    }
}

/// Rotate selected elements of arrays around Z axis in-place.
pub fn rotate_z_indexed(
    x: &mut Array1<f64>,
    y: &mut Array1<f64>,
    indices: &[usize],
    cos_a: f64,
    sin_a: f64,
) {
    for &i in indices {
        let (xn, yn) = rotate_z(x[i], y[i], cos_a, sin_a);
        x[i] = xn;
        y[i] = yn;
    }
}

// ── Rotation sequence parsing ───────────────────────────────────────────────

/// Rotation axis identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Parse a rotation sequence string (e.g. "RzRyRx" or "-RzRyRx") into
/// an ordered list of axes.
///
/// Standard form: "RzRyRx" → [Z, Y, X]
/// Reversed form: "-RzRyRx" → [X, Y, Z]
pub fn parse_rotation_sequence(seq: &str) -> Result<[Axis; 3]> {
    let (reversed, body) = if let Some(rest) = seq.strip_prefix('-') {
        (true, rest)
    } else {
        (false, seq)
    };

    if body.len() != 6 {
        return Err(XrtError::InvalidRotationSequence(seq.to_string()));
    }

    let bytes = body.as_bytes();
    let mut axes = [Axis::X; 3];
    for i in 0..3 {
        let r = bytes[i * 2];
        let a = bytes[i * 2 + 1];
        if r != b'R' {
            return Err(XrtError::InvalidRotationSequence(seq.to_string()));
        }
        axes[i] = match a {
            b'x' => Axis::X,
            b'y' => Axis::Y,
            b'z' => Axis::Z,
            _ => return Err(XrtError::InvalidRotationSequence(seq.to_string())),
        };
    }

    if reversed {
        axes.reverse();
    }
    Ok(axes)
}

// ── High-level beam rotation ────────────────────────────────────────────────

/// Rotation parameters for `rotate_beam` and `rotate_xyz`.
#[derive(Debug, Clone, Copy)]
pub struct RotationParams {
    pub pitch: f64,
    pub roll: f64,
    pub yaw: f64,
    pub sequence: [Axis; 3],
}

impl RotationParams {
    /// Create rotation params from angles and a sequence string.
    pub fn new(pitch: f64, roll: f64, yaw: f64, sequence: &str) -> Result<Self> {
        Ok(Self {
            pitch,
            roll,
            yaw,
            sequence: parse_rotation_sequence(sequence)?,
        })
    }

    /// Default sequence "RzRyRx" with given angles.
    pub fn default_sequence(pitch: f64, roll: f64, yaw: f64) -> Self {
        Self {
            pitch,
            roll,
            yaw,
            sequence: [Axis::Z, Axis::Y, Axis::X],
        }
    }

    fn angle_for(&self, axis: Axis) -> f64 {
        match axis {
            Axis::X => self.pitch,
            Axis::Y => self.roll,
            Axis::Z => self.yaw,
        }
    }
}

/// Apply a rotation sequence to a beam's position and direction arrays.
///
/// Mirrors Python's `rotate_beam(beam, indarr, rotationSequence, pitch, roll, yaw)`.
///
/// If `indices` is `None`, all rays are rotated.
/// `skip_xyz` / `skip_abc` control which arrays are affected.
pub fn rotate_beam(
    beam: &mut Beam,
    indices: Option<&[usize]>,
    params: &RotationParams,
    skip_xyz: bool,
    skip_abc: bool,
) {
    for &axis in &params.sequence {
        let angle = params.angle_for(axis);
        if angle == 0.0 {
            continue;
        }
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        match indices {
            None => {
                if !skip_xyz {
                    apply_rotation_full_xyz(axis, &mut beam.x, &mut beam.y, &mut beam.z, cos_a, sin_a);
                }
                if !skip_abc {
                    apply_rotation_full_xyz(axis, &mut beam.a, &mut beam.b, &mut beam.c, cos_a, sin_a);
                }
            }
            Some(idx) => {
                if !skip_xyz {
                    apply_rotation_indexed_xyz(axis, &mut beam.x, &mut beam.y, &mut beam.z, idx, cos_a, sin_a);
                }
                if !skip_abc {
                    apply_rotation_indexed_xyz(axis, &mut beam.a, &mut beam.b, &mut beam.c, idx, cos_a, sin_a);
                }
            }
        }
    }
}

/// Rotate xyz arrays in-place by the given rotation parameters.
///
/// Mirrors Python's `rotate_xyz(x, y, z, ...)`.
pub fn rotate_xyz(
    x: &mut Array1<f64>,
    y: &mut Array1<f64>,
    z: &mut Array1<f64>,
    indices: Option<&[usize]>,
    params: &RotationParams,
) {
    for &axis in &params.sequence {
        let angle = params.angle_for(axis);
        if angle == 0.0 {
            continue;
        }
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        match indices {
            None => apply_rotation_full_xyz(axis, x, y, z, cos_a, sin_a),
            Some(idx) => apply_rotation_indexed_xyz(axis, x, y, z, idx, cos_a, sin_a),
        }
    }
}

/// Rotate a single 3D point by the given rotation parameters.
///
/// Mirrors Python's `rotate_point(point, ...)`.
pub fn rotate_point(point: [f64; 3], params: &RotationParams) -> [f64; 3] {
    let mut p = point;
    for &axis in &params.sequence {
        let angle = params.angle_for(axis);
        if angle == 0.0 {
            continue;
        }
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        match axis {
            Axis::X => {
                let (y, z) = rotate_x(p[1], p[2], cos_a, sin_a);
                p[1] = y;
                p[2] = z;
            }
            Axis::Y => {
                let (x, z) = rotate_y(p[0], p[2], cos_a, sin_a);
                p[0] = x;
                p[2] = z;
            }
            Axis::Z => {
                let (x, y) = rotate_z(p[0], p[1], cos_a, sin_a);
                p[0] = x;
                p[1] = y;
            }
        }
    }
    p
}

// ── 2D/3D distance utilities ────────────────────────────────────────────────

/// 2D distance between two points.
#[inline]
pub fn distance_xy(p1: &[f64], p2: &[f64]) -> f64 {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
}

/// 3D distance between two points.
#[inline]
pub fn distance_xyz(p1: &[f64], p2: &[f64]) -> f64 {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
}

// ── Global ↔ local coordinate transforms ────────────────────────────────────

/// Transform a beam from global to virgin (unrotated) local coordinates.
///
/// Mirrors Python's `global_to_virgin_local(bl, beam, lo, center, part)`.
/// Here we take the beamline azimuth (sin, cos) directly rather than
/// the full BeamLine object.
pub fn global_to_virgin_local(
    beam: &Beam,
    lo: &mut Beam,
    center: [f64; 3],
    sin_azimuth: f64,
    cos_azimuth: f64,
    part: Option<&[usize]>,
) {
    match part {
        None => {
            // All rays
            lo.x.assign(&(&beam.x - center[0]));
            lo.y.assign(&(&beam.y - center[1]));
            lo.z.assign(&(&beam.z - center[2]));

            if sin_azimuth == 0.0 {
                lo.a.assign(&beam.a);
                lo.b.assign(&beam.b);
            } else {
                rotate_z_arrays(&mut lo.x, &mut lo.y, cos_azimuth, sin_azimuth);
                // Direction vectors rotate from the original beam values
                lo.a.assign(&beam.a);
                lo.b.assign(&beam.b);
                rotate_z_arrays(&mut lo.a, &mut lo.b, cos_azimuth, sin_azimuth);
            }
            lo.c.assign(&beam.c);
        }
        Some(indices) => {
            for &i in indices {
                lo.x[i] = beam.x[i] - center[0];
                lo.y[i] = beam.y[i] - center[1];
                lo.z[i] = beam.z[i] - center[2];
            }

            if sin_azimuth == 0.0 {
                for &i in indices {
                    lo.a[i] = beam.a[i];
                    lo.b[i] = beam.b[i];
                }
            } else {
                rotate_z_indexed(&mut lo.x, &mut lo.y, indices, cos_azimuth, sin_azimuth);
                // Copy direction then rotate
                for &i in indices {
                    lo.a[i] = beam.a[i];
                    lo.b[i] = beam.b[i];
                }
                rotate_z_indexed(&mut lo.a, &mut lo.b, indices, cos_azimuth, sin_azimuth);
            }
            for &i in indices {
                lo.c[i] = beam.c[i];
            }
        }
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn apply_rotation_full_xyz(
    axis: Axis,
    x: &mut Array1<f64>,
    y: &mut Array1<f64>,
    z: &mut Array1<f64>,
    cos_a: f64,
    sin_a: f64,
) {
    match axis {
        Axis::X => rotate_x_arrays(y, z, cos_a, sin_a),
        Axis::Y => rotate_y_arrays(x, z, cos_a, sin_a),
        Axis::Z => rotate_z_arrays(x, y, cos_a, sin_a),
    }
}

fn apply_rotation_indexed_xyz(
    axis: Axis,
    x: &mut Array1<f64>,
    y: &mut Array1<f64>,
    z: &mut Array1<f64>,
    indices: &[usize],
    cos_a: f64,
    sin_a: f64,
) {
    match axis {
        Axis::X => rotate_x_indexed(y, z, indices, cos_a, sin_a),
        Axis::Y => rotate_y_indexed(x, z, indices, cos_a, sin_a),
        Axis::Z => rotate_z_indexed(x, y, indices, cos_a, sin_a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_rotate_x_90deg() {
        // Rotate (0, 1, 0) around X by 90°: expect (0, 0, 1)
        let (y, z) = rotate_x(1.0, 0.0, 0.0, 1.0); // cos(90°)=0, sin(90°)=1
        assert_abs_diff_eq!(y, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(z, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_rotate_y_90deg() {
        // Rotate (1, 0, 0) around Y by 90°: expect (0, 0, -1)
        let (x, z) = rotate_y(1.0, 0.0, 0.0, 1.0);
        assert_abs_diff_eq!(x, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(z, -1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_rotate_z_90deg() {
        // Rotate (1, 0, 0) around Z by 90°: expect (0, 1)
        let (x, y) = rotate_z(1.0, 0.0, 0.0, 1.0);
        assert_abs_diff_eq!(x, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(y, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_rotate_identity() {
        // Zero rotation → no change
        let (y, z) = rotate_x(3.0, 4.0, 1.0, 0.0);
        assert_abs_diff_eq!(y, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(z, 4.0, epsilon = 1e-15);
    }

    #[test]
    fn test_parse_rotation_sequence() {
        let axes = parse_rotation_sequence("RzRyRx").unwrap();
        assert_eq!(axes, [Axis::Z, Axis::Y, Axis::X]);

        let axes = parse_rotation_sequence("-RzRyRx").unwrap();
        assert_eq!(axes, [Axis::X, Axis::Y, Axis::Z]);

        let axes = parse_rotation_sequence("RxRzRy").unwrap();
        assert_eq!(axes, [Axis::X, Axis::Z, Axis::Y]);
    }

    #[test]
    fn test_parse_rotation_sequence_invalid() {
        assert!(parse_rotation_sequence("Rz").is_err());
        assert!(parse_rotation_sequence("RzRyRq").is_err());
        assert!(parse_rotation_sequence("AzRyRx").is_err());
    }

    #[test]
    fn test_rotate_beam_pitch() {
        let mut beam = Beam::new(3);
        // Place rays along +z: (0, 0, 1)
        beam.z.fill(1.0);
        beam.c.fill(1.0);
        beam.b.fill(0.0);

        let params = RotationParams::default_sequence(FRAC_PI_2, 0.0, 0.0);
        rotate_beam(&mut beam, None, &params, false, false);

        // After 90° pitch (around x): z→y, y→-z  actually (y,z) -> (cos*y-sin*z, sin*y+cos*z)
        // With cos=0, sin=1: y'=-z=-1... wait: rotate_x(y,z,cos,sin) = (cos*y-sin*z, sin*y+cos*z)
        // beam.y was 0, beam.z was 1 → y' = 0*0 - 1*1 = -1, z' = 1*0 + 0*1 = 0
        for i in 0..3 {
            assert_abs_diff_eq!(beam.y[i], -1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(beam.z[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rotate_point() {
        let params = RotationParams::default_sequence(0.0, 0.0, FRAC_PI_2);
        let p = rotate_point([1.0, 0.0, 0.0], &params);
        assert_abs_diff_eq!(p[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotate_xyz_arrays() {
        let mut x = Array1::from_vec(vec![1.0, 0.0]);
        let mut y = Array1::from_vec(vec![0.0, 1.0]);
        let mut z = Array1::zeros(2);

        let params = RotationParams::default_sequence(0.0, 0.0, FRAC_PI_2);
        rotate_xyz(&mut x, &mut y, &mut z, None, &params);

        assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_xy() {
        assert_abs_diff_eq!(distance_xy(&[0.0, 0.0], &[3.0, 4.0]), 5.0, epsilon = 1e-15);
    }

    #[test]
    fn test_distance_xyz() {
        assert_abs_diff_eq!(distance_xyz(&[0.0, 0.0, 0.0], &[1.0, 2.0, 2.0]), 3.0, epsilon = 1e-15);
    }

    #[test]
    fn test_global_to_virgin_local_no_azimuth() {
        let beam = Beam::new(3);
        let mut lo = Beam::new(3);
        global_to_virgin_local(&beam, &mut lo, [1.0, 2.0, 3.0], 0.0, 1.0, None);

        for i in 0..3 {
            assert_abs_diff_eq!(lo.x[i], -1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(lo.y[i], -2.0, epsilon = 1e-15);
            assert_abs_diff_eq!(lo.z[i], -3.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_rotate_indexed_subset() {
        let mut beam = Beam::new(4);
        beam.x.fill(1.0);
        beam.b.fill(0.0);

        let params = RotationParams::default_sequence(0.0, 0.0, FRAC_PI_2);
        rotate_beam(&mut beam, Some(&[0, 2]), &params, false, false);

        // Only indices 0 and 2 should be rotated
        assert_abs_diff_eq!(beam.x[0], 0.0, epsilon = 1e-10);
        assert_eq!(beam.x[1], 1.0); // untouched
        assert_abs_diff_eq!(beam.x[2], 0.0, epsilon = 1e-10);
        assert_eq!(beam.x[3], 1.0); // untouched
    }
}
