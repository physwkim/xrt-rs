//! GeometricSource: a simple parametric X-ray source.
//!
//! Ported from sources_geoms.py:139-381.

use xrt_core::beam::{Beam, RayState};
use xrt_core::consts::DEFAULT_ENERGY;
use xrt_core::transforms::{rotate_beam, RotationParams};

use crate::distributions::{apply_distribution, make_energy, set_annulus, EnergyDist, SpatialDist};
use crate::polarization::{make_polarization, Polarization};

/// A geometric (parametric) X-ray source.
///
/// Generates rays with configurable position, angle, energy, and
/// polarization distributions.
#[derive(Debug, Clone)]
pub struct GeometricSource {
    /// Center position in global coordinates [mm]
    pub center: [f64; 3],
    /// Number of rays
    pub nrays: usize,

    /// Horizontal position distribution
    pub dist_x: SpatialDist,
    /// Longitudinal position distribution
    pub dist_y: SpatialDist,
    /// Vertical position distribution
    pub dist_z: SpatialDist,
    /// Horizontal divergence distribution
    pub dist_xprime: SpatialDist,
    /// Vertical divergence distribution
    pub dist_zprime: SpatialDist,
    /// Energy distribution
    pub dist_e: EnergyDist,

    /// Polarization
    pub polarization: Polarization,

    /// Pitch angle [rad]
    pub pitch: f64,
    /// Roll angle [rad]
    pub roll: f64,
    /// Yaw angle [rad]
    pub yaw: f64,
}

impl Default for GeometricSource {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            nrays: 10_000,
            dist_x: SpatialDist::Normal(0.32),
            dist_y: SpatialDist::None,
            dist_z: SpatialDist::Normal(0.018),
            dist_xprime: SpatialDist::Normal(1e-3),
            dist_zprime: SpatialDist::Normal(1e-4),
            dist_e: EnergyDist::Lines(vec![DEFAULT_ENERGY], None),
            polarization: Polarization::Horizontal,
            pitch: 0.0,
            roll: 0.0,
            yaw: 0.0,
        }
    }
}

impl GeometricSource {
    /// Generate a beam from this source.
    ///
    /// Returns a beam in local coordinates (use `to_global` to transform).
    pub fn shine(&self) -> Beam {
        let n = self.nrays;
        let mut beam = Beam::with_amplitudes(n);
        beam.set_state(RayState::Good);

        // Initialize polarization
        make_polarization(&self.polarization, &mut beam);

        // Position distributions
        apply_distribution(beam.y.as_slice_mut().expect("contiguous array"), &self.dist_y);

        // Handle annulus for x/z pair
        match (&self.dist_x, &self.dist_z) {
            (SpatialDist::Annulus(r_min, r_max), _) => {
                let (phi_min, phi_max) = match &self.dist_z {
                    SpatialDist::Flat(a, b) => (*a, *b),
                    _ => (0.0, std::f64::consts::TAU),
                };
                set_annulus(
                    beam.x.as_slice_mut().expect("contiguous array"),
                    beam.z.as_slice_mut().expect("contiguous array"),
                    *r_min,
                    *r_max,
                    phi_min,
                    phi_max,
                );
            }
            _ => {
                apply_distribution(beam.x.as_slice_mut().expect("contiguous array"), &self.dist_x);
                apply_distribution(beam.z.as_slice_mut().expect("contiguous array"), &self.dist_z);
            }
        }

        // Direction distributions
        match (&self.dist_xprime, &self.dist_zprime) {
            (SpatialDist::Annulus(r_min, r_max), _) => {
                let (phi_min, phi_max) = match &self.dist_zprime {
                    SpatialDist::Flat(a, b) => (*a, *b),
                    _ => (0.0, std::f64::consts::TAU),
                };
                set_annulus(
                    beam.a.as_slice_mut().expect("contiguous array"),
                    beam.c.as_slice_mut().expect("contiguous array"),
                    *r_min,
                    *r_max,
                    phi_min,
                    phi_max,
                );
            }
            _ => {
                apply_distribution(beam.a.as_slice_mut().expect("contiguous array"), &self.dist_xprime);
                apply_distribution(beam.c.as_slice_mut().expect("contiguous array"), &self.dist_zprime);
            }
        }

        // Normalize direction (a, b, c)
        for i in 0..n {
            let ac2 = beam.a[i] * beam.a[i] + beam.c[i] * beam.c[i];
            if ac2 > 1.0 {
                let norm = (ac2 + 1.0).sqrt();
                beam.a[i] /= norm;
                beam.c[i] /= norm;
                beam.b[i] = 1.0 / norm;
            } else {
                beam.b[i] = (1.0 - ac2).sqrt();
            }
        }

        // Energy
        let energies = make_energy(&self.dist_e, n);
        for (i, &e) in energies.iter().enumerate() {
            beam.e[i] = e;
        }

        // Optional rotation
        if self.pitch != 0.0 || self.roll != 0.0 || self.yaw != 0.0 {
            let params =
                RotationParams::default_sequence(self.pitch, self.roll, self.yaw);
            let indices: Vec<usize> = (0..n).collect();
            rotate_beam(&mut beam, Some(&indices), &params, false, false);
        }

        beam
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_source_shine() {
        let source = GeometricSource {
            nrays: 100,
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.nrays(), 100);

        // All rays should be good
        for i in 0..100 {
            assert_eq!(beam.state[i], RayState::Good as i32);
        }

        // Direction should be approximately forward (b ≈ 1)
        let avg_b: f64 = beam.b.iter().sum::<f64>() / 100.0;
        assert!(avg_b > 0.99, "avg b = {avg_b}, expected ~1");

        // Directions should be normalized
        for i in 0..100 {
            let norm =
                (beam.a[i] * beam.a[i] + beam.b[i] * beam.b[i] + beam.c[i] * beam.c[i]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "direction not normalized: {norm}"
            );
        }
    }

    #[test]
    fn flat_energy_distribution() {
        let source = GeometricSource {
            nrays: 1000,
            dist_e: EnergyDist::Flat(8000.0, 12000.0),
            ..Default::default()
        };
        let beam = source.shine();
        for i in 0..1000 {
            assert!(beam.e[i] >= 8000.0 && beam.e[i] <= 12000.0);
        }
    }

    #[test]
    fn point_source() {
        let source = GeometricSource {
            nrays: 100,
            dist_x: SpatialDist::None,
            dist_z: SpatialDist::None,
            dist_xprime: SpatialDist::Flat(-0.001, 0.001),
            dist_zprime: SpatialDist::Flat(-0.001, 0.001),
            ..Default::default()
        };
        let beam = source.shine();
        // All positions should be at origin
        for i in 0..100 {
            assert_eq!(beam.x[i], 0.0);
            assert_eq!(beam.z[i], 0.0);
        }
    }

    #[test]
    fn polarization_is_set() {
        let source = GeometricSource {
            nrays: 10,
            polarization: Polarization::Vertical,
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.jss[0], 0.0);
        assert_eq!(beam.jpp[0], 1.0);
    }

    #[test]
    fn annulus_distribution() {
        use crate::distributions::SpatialDist;
        let source = GeometricSource {
            nrays: 10000,
            dist_x: SpatialDist::Annulus(1.0, 5.0), // inner=1mm, outer=5mm
            dist_z: SpatialDist::Annulus(1.0, 5.0),
            dist_e: EnergyDist::Lines(vec![10000.0], None),
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.nrays(), 10000);
        // All points should be within the annulus
        for i in 0..beam.nrays() {
            let r = (beam.x[i].powi(2) + beam.z[i].powi(2)).sqrt();
            assert!(r >= 0.5, "ray[{i}] r={r:.4} < inner radius");
            assert!(r <= 6.0, "ray[{i}] r={r:.4} > outer radius");
        }
    }

    #[test]
    fn polarization_plus45() {
        let source = GeometricSource {
            nrays: 100,
            dist_e: EnergyDist::Lines(vec![10000.0], None),
            polarization: Polarization::Plus45,
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.nrays(), 100);
        // Plus45: Jss = Jpp = 0.5, Jsp = (0.5, 0)
        for i in 0..beam.nrays() {
            assert!((beam.jss[i] - 0.5).abs() < 1e-15);
            assert!((beam.jpp[i] - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn polarization_minus45() {
        let source = GeometricSource {
            nrays: 100,
            dist_e: EnergyDist::Lines(vec![10000.0], None),
            polarization: Polarization::Minus45,
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.nrays(), 100);
        // Minus45: Jss = Jpp = 0.5, Jsp = (-0.5, 0)
        for i in 0..beam.nrays() {
            assert!((beam.jss[i] - 0.5).abs() < 1e-15);
            assert!((beam.jpp[i] - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn polarization_left_circular() {
        let source = GeometricSource {
            nrays: 100,
            dist_e: EnergyDist::Lines(vec![10000.0], None),
            polarization: Polarization::Left,
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.nrays(), 100);
        // Left circular: Jss = Jpp = 0.5, Jsp = (0, -0.5)
        for i in 0..beam.nrays() {
            assert!((beam.jss[i] - 0.5).abs() < 1e-15);
            assert!((beam.jpp[i] - 0.5).abs() < 1e-15);
            assert!((beam.jsp[i].im - (-0.5)).abs() < 1e-15);
        }
    }

    #[test]
    fn weighted_energy_lines() {
        // Two energy lines: 8000 eV (weight 0.9) and 12000 eV (weight 0.1)
        let source = GeometricSource {
            nrays: 10000,
            dist_e: EnergyDist::Lines(
                vec![8000.0, 12000.0],
                Some(vec![0.9, 0.1]),
            ),
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.nrays(), 10000);

        // Count rays at each energy
        let n_8k = beam.e.iter().filter(|&&e| (e - 8000.0).abs() < 1.0).count();
        let n_12k = beam.e.iter().filter(|&&e| (e - 12000.0).abs() < 1.0).count();

        // Should roughly follow 9:1 ratio
        let ratio = n_8k as f64 / n_12k as f64;
        assert!(ratio > 5.0 && ratio < 15.0,
            "Expected ~9:1 ratio, got {n_8k}:{n_12k} = {ratio:.1}");
    }

    #[test]
    fn source_with_pitch_rotation() {
        let source = GeometricSource {
            nrays: 1000,
            dist_e: EnergyDist::Lines(vec![10000.0], None),
            pitch: 0.01, // small pitch rotation
            ..Default::default()
        };
        let beam = source.shine();
        assert_eq!(beam.nrays(), 1000);
        // After pitch rotation, mean b should still be ~1 (forward)
        let mean_b: f64 = beam.b.iter().sum::<f64>() / 1000.0;
        assert!(mean_b > 0.99, "mean(b) after small pitch should be ~1: {mean_b}");
    }
}
