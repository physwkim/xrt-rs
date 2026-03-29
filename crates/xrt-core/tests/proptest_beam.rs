//! Property-based tests for Beam and transforms.

use proptest::prelude::*;
use xrt_core::beam::{Beam, RayState};
use xrt_core::transforms::{rotate_beam, rotate_xyz, RotationParams};
use ndarray::Array1;

proptest! {
    /// Rotation preserves vector magnitude.
    #[test]
    fn rotation_preserves_norm(
        x in -1000.0..1000.0_f64,
        y in -1000.0..1000.0_f64,
        z in -1000.0..1000.0_f64,
        pitch in -0.5..0.5_f64,
        roll in -0.5..0.5_f64,
        yaw in -0.5..0.5_f64,
    ) {
        let norm_before = (x * x + y * y + z * z).sqrt();
        let mut xa = Array1::from_vec(vec![x]);
        let mut ya = Array1::from_vec(vec![y]);
        let mut za = Array1::from_vec(vec![z]);
        let params = RotationParams::default_sequence(pitch, roll, yaw);
        rotate_xyz(&mut xa, &mut ya, &mut za, None, &params);
        let norm_after = (xa[0] * xa[0] + ya[0] * ya[0] + za[0] * za[0]).sqrt();
        prop_assert!((norm_after - norm_before).abs() < 1e-8 * norm_before.max(1.0),
            "norm changed: {} → {}", norm_before, norm_after);
    }

    /// Double rotation by θ then -θ returns to original.
    #[test]
    fn rotation_inverse(
        x in -100.0..100.0_f64,
        y in -100.0..100.0_f64,
        z in -100.0..100.0_f64,
        pitch in -0.3..0.3_f64,
    ) {
        let mut xa = Array1::from_vec(vec![x]);
        let mut ya = Array1::from_vec(vec![y]);
        let mut za = Array1::from_vec(vec![z]);

        let fwd = RotationParams::default_sequence(pitch, 0.0, 0.0);
        rotate_xyz(&mut xa, &mut ya, &mut za, None, &fwd);

        let inv = RotationParams::default_sequence(-pitch, 0.0, 0.0);
        rotate_xyz(&mut xa, &mut ya, &mut za, None, &inv);

        prop_assert!((xa[0] - x).abs() < 1e-8, "x: {} → {}", x, xa[0]);
        prop_assert!((ya[0] - y).abs() < 1e-8, "y: {} → {}", y, ya[0]);
        prop_assert!((za[0] - z).abs() < 1e-8, "z: {} → {}", z, za[0]);
    }

    /// Propagation preserves direction cosine normalization.
    #[test]
    fn propagate_preserves_direction(
        a in -0.1..0.1_f64,
        c in -0.1..0.1_f64,
        dist in 0.0..10000.0_f64,
    ) {
        let mut beam = Beam::new(1);
        beam.set_state(RayState::Good);
        beam.a[0] = a;
        let ac2 = a * a + c * c;
        beam.b[0] = if ac2 < 1.0 { (1.0 - ac2).sqrt() } else { 0.01 };
        beam.c[0] = c;

        let a_before = beam.a[0];
        let b_before = beam.b[0];
        let c_before = beam.c[0];

        beam.propagate(dist);

        // Direction should not change
        prop_assert!((beam.a[0] - a_before).abs() < 1e-15);
        prop_assert!((beam.b[0] - b_before).abs() < 1e-15);
        prop_assert!((beam.c[0] - c_before).abs() < 1e-15);

        // Path should increase by distance
        prop_assert!((beam.path[0] - dist).abs() < 1e-10);
    }

    /// Beam statistics are consistent.
    #[test]
    fn statistics_consistency(n in 10..500_usize) {
        let mut beam = Beam::new(n);
        beam.set_state(RayState::Good);
        for i in 0..n {
            beam.b[i] = 1.0;
            beam.e[i] = 10000.0;
        }

        let stats = beam.statistics().unwrap();
        prop_assert_eq!(stats.n_good, n);
        prop_assert_eq!(stats.n_total, n);
        prop_assert!(stats.sigma_x >= 0.0);
        prop_assert!(stats.sigma_z >= 0.0);
        prop_assert!(stats.e_min <= stats.mean_energy);
        prop_assert!(stats.e_max >= stats.mean_energy);
    }

    /// Filter-good preserves ray count correctly.
    #[test]
    fn filter_good_count(n in 1..200_usize, kill_frac in 0.0..1.0_f64) {
        let mut beam = Beam::new(n);
        beam.set_state(RayState::Good);
        let n_kill = (n as f64 * kill_frac) as usize;
        for i in 0..n_kill {
            beam.state[i] = RayState::Dead as i32;
        }

        let good = beam.filter_good();
        prop_assert_eq!(good.nrays(), n - n_kill);
    }
}
