//! Distribution sampling for source parameters.
//!
//! Ported from sources_geoms.py:16-290.

use rand_distr::{Distribution, Normal, Uniform, WeightedIndex};

use xrt_core::consts::DEFAULT_ENERGY;

/// Energy distribution type.
#[derive(Debug, Clone)]
pub enum EnergyDist {
    /// Normal (Gaussian) distribution: mean, sigma
    Normal(f64, f64),
    /// Flat (uniform) distribution: min, max
    Flat(f64, f64),
    /// Discrete lines with optional weights
    Lines(Vec<f64>, Option<Vec<f64>>),
}

/// Spatial/angular distribution type.
#[derive(Debug, Clone)]
pub enum SpatialDist {
    /// Gaussian: sigma
    Normal(f64),
    /// Uniform: half-width (or [min, max])
    Flat(f64, f64),
    /// Annulus: r_min, r_max (paired with another axis for phi)
    Annulus(f64, f64),
    /// No distribution (zero)
    None,
}

/// Generate energy values for `n` rays.
pub fn make_energy(dist: &EnergyDist, n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    match dist {
        EnergyDist::Normal(mean, sigma) => {
            if *sigma <= 0.0 {
                return vec![*mean; n];
            }
            match Normal::new(*mean, *sigma) {
                Ok(d) => (0..n).map(|_| d.sample(&mut rng)).collect(),
                Err(_) => vec![*mean; n],
            }
        }
        EnergyDist::Flat(min, max) => {
            if *max <= *min {
                return vec![*min; n];
            }
            let d = Uniform::new(*min, *max);
            (0..n).map(|_| d.sample(&mut rng)).collect()
        }
        EnergyDist::Lines(energies, weights) => {
            if energies.is_empty() {
                return vec![DEFAULT_ENERGY; n];
            }
            if let Some(w) = weights {
                match WeightedIndex::new(w) {
                    Ok(d) => (0..n).map(|_| energies[d.sample(&mut rng)]).collect(),
                    Err(_) => vec![energies[0]; n],
                }
            } else if energies.len() == 1 {
                vec![energies[0]; n]
            } else {
                let d = Uniform::new(0, energies.len());
                (0..n).map(|_| energies[d.sample(&mut rng)]).collect()
            }
        }
    }
}

/// Sample from a spatial distribution into a pre-allocated slice.
pub fn apply_distribution(axis: &mut [f64], dist: &SpatialDist) {
    let mut rng = rand::thread_rng();
    match dist {
        SpatialDist::Normal(sigma) => {
            if *sigma <= 0.0 {
                axis.iter_mut().for_each(|v| *v = 0.0);
                return;
            }
            match Normal::new(0.0, *sigma) {
                Ok(d) => axis.iter_mut().for_each(|v| *v = d.sample(&mut rng)),
                Err(_) => axis.iter_mut().for_each(|v| *v = 0.0),
            }
        }
        SpatialDist::Flat(min, max) => {
            if *max <= *min {
                axis.iter_mut().for_each(|v| *v = 0.0);
                return;
            }
            let d = Uniform::new(*min, *max);
            axis.iter_mut().for_each(|v| *v = d.sample(&mut rng));
        }
        SpatialDist::Annulus(_, _) => {
            // Annulus is handled by set_annulus which operates on two axes
            // This should not be called directly
        }
        SpatialDist::None => {
            axis.iter_mut().for_each(|v| *v = 0.0);
        }
    }
}

/// Sample from an annulus distribution (uniform in area).
///
/// Fills `axis1` (x or a) and `axis2` (z or c) with positions
/// uniformly distributed in the annulus [r_min, r_max] × [phi_min, phi_max].
pub fn set_annulus(
    axis1: &mut [f64],
    axis2: &mut [f64],
    r_min: f64,
    r_max: f64,
    phi_min: f64,
    phi_max: f64,
) {
    let mut rng = rand::thread_rng();
    let n = axis1.len();
    let phi_dist = Uniform::new(phi_min, phi_max);

    if r_max > r_min {
        // Area-uniform: r = sqrt(U * (r_max² - r_min²) + r_min²)
        let r_min2 = r_min * r_min;
        let r_diff2 = r_max * r_max - r_min2;
        let u_dist = Uniform::new(0.0_f64, 1.0);
        for i in 0..n {
            let r = (u_dist.sample(&mut rng) * r_diff2 + r_min2).sqrt();
            let phi = phi_dist.sample(&mut rng);
            axis1[i] = r * phi.cos();
            axis2[i] = r * phi.sin();
        }
    } else {
        let r = r_max;
        for i in 0..n {
            let phi = phi_dist.sample(&mut rng);
            axis1[i] = r * phi.cos();
            axis2[i] = r * phi.sin();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_normal_distribution() {
        let e = make_energy(&EnergyDist::Normal(10000.0, 1.0), 1000);
        assert_eq!(e.len(), 1000);
        let mean: f64 = e.iter().sum::<f64>() / e.len() as f64;
        assert!(
            (mean - 10000.0).abs() < 10.0,
            "mean = {mean}, expected ~10000"
        );
    }

    #[test]
    fn energy_flat_distribution() {
        let e = make_energy(&EnergyDist::Flat(8000.0, 12000.0), 1000);
        assert!(e.iter().all(|&v| (8000.0..=12000.0).contains(&v)));
    }

    #[test]
    fn energy_lines_distribution() {
        let e = make_energy(
            &EnergyDist::Lines(vec![8000.0, 10000.0, 12000.0], None),
            1000,
        );
        assert!(e
            .iter()
            .all(|&v| v == 8000.0 || v == 10000.0 || v == 12000.0));
    }

    #[test]
    fn spatial_normal() {
        let mut axis = vec![0.0; 1000];
        apply_distribution(&mut axis, &SpatialDist::Normal(1.0));
        let mean: f64 = axis.iter().sum::<f64>() / axis.len() as f64;
        assert!(mean.abs() < 0.2, "mean = {mean}");
    }

    #[test]
    fn annulus_distribution() {
        let n = 10000;
        let mut x = vec![0.0; n];
        let mut z = vec![0.0; n];
        set_annulus(&mut x, &mut z, 5.0, 10.0, 0.0, std::f64::consts::TAU);
        // All points should be within the annulus
        for i in 0..n {
            let r = (x[i] * x[i] + z[i] * z[i]).sqrt();
            assert!(
                (4.99..=10.01).contains(&r),
                "r = {r} outside annulus [5, 10]"
            );
        }
    }
}
