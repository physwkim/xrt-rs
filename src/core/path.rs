//! Optical path length computation.
//!
//! Tracks the cumulative optical path for each ray through the beamline,
//! enabling coherence calculations and phase tracking.

/// Compute the optical path length for each ray given positions before and after.
///
/// Returns the Euclidean distance between (x0,y0,z0) and (x1,y1,z1) for each ray.
pub fn compute_path_lengths(
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    x1: &[f64],
    y1: &[f64],
    z1: &[f64],
) -> Vec<f64> {
    x0.iter()
        .zip(y0)
        .zip(z0)
        .zip(x1)
        .zip(y1)
        .zip(z1)
        .map(|(((((x0, y0), z0), x1), y1), z1)| {
            let dx = x1 - x0;
            let dy = y1 - y0;
            let dz = z1 - z0;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect()
}

/// Accumulated optical path tracker for a beamline.
#[derive(Debug, Clone)]
pub struct OpticalPathTracker {
    /// Cumulative path length for each ray [mm]
    pub path: Vec<f64>,
}

impl OpticalPathTracker {
    /// Create a new tracker for n_rays.
    pub fn new(n_rays: usize) -> Self {
        Self {
            path: vec![0.0; n_rays],
        }
    }

    /// Add a drift of `distance` mm to all rays.
    pub fn add_drift(&mut self, distance: f64) {
        for p in &mut self.path {
            *p += distance;
        }
    }

    /// Add per-ray path increments (e.g., from intersection calculation).
    pub fn add_increments(&mut self, increments: &[f64]) {
        for (p, inc) in self.path.iter_mut().zip(increments) {
            *p += inc;
        }
    }

    /// Get the optical path difference (OPD) relative to the mean.
    pub fn opd(&self) -> Vec<f64> {
        let mean: f64 = self.path.iter().sum::<f64>() / self.path.len() as f64;
        self.path.iter().map(|&p| p - mean).collect()
    }

    /// Get the phase for each ray at given energy [eV].
    pub fn phase(&self, energy: f64) -> Vec<f64> {
        use crate::core::consts::CHBAR;
        let k = energy / CHBAR * 1e7; // mm⁻¹
        self.path.iter().map(|&p| k * p).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_drift() {
        let mut t = OpticalPathTracker::new(3);
        t.add_drift(1000.0);
        assert_eq!(t.path, vec![1000.0; 3]);
        t.add_drift(500.0);
        assert_eq!(t.path, vec![1500.0; 3]);
    }

    #[test]
    fn tracker_opd() {
        let mut t = OpticalPathTracker::new(3);
        t.path = vec![1000.0, 1000.5, 999.5];
        let opd = t.opd();
        assert!((opd[0] - 0.0).abs() < 1e-12);
        assert!((opd[1] - 0.5).abs() < 1e-12);
        assert!((opd[2] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn tracker_phase() {
        let mut t = OpticalPathTracker::new(1);
        t.path = vec![1000.0];
        let phase = t.phase(10000.0);
        assert!(phase[0] > 0.0, "phase should be positive");
        assert!(phase[0].is_finite(), "phase should be finite");
    }

    #[test]
    fn path_lengths_simple() {
        let paths = compute_path_lengths(&[0.0], &[0.0], &[0.0], &[3.0], &[4.0], &[0.0]);
        assert!((paths[0] - 5.0).abs() < 1e-12);
    }
}
