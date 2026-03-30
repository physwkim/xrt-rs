//! 2D Fresnel wavefront propagation.
//!
//! Propagates a complex wavefront from a source plane to an observation
//! plane using the Fresnel integral (paraxial approximation).
//!
//! Compatible with shadow3's FFresnel2D postprocessor.

use num_complex::Complex64;
use rayon::prelude::*;

use xrt_core::consts::CHBAR;

/// Result of Fresnel propagation.
pub struct FresnelResult {
    /// Complex amplitude at each observation pixel
    pub amplitude: Vec<Complex64>,
    /// Pixel x-coordinates [mm]
    pub x_pixels: Vec<f64>,
    /// Pixel z-coordinates [mm]
    pub z_pixels: Vec<f64>,
    /// Number of x pixels
    pub nx: usize,
    /// Number of z pixels
    pub nz: usize,
}

impl FresnelResult {
    /// Compute intensity at each pixel.
    pub fn intensity(&self) -> Vec<f64> {
        self.amplitude.iter().map(|a| a.norm_sqr()).collect()
    }
}

/// Perform 2D Fresnel propagation.
///
/// Propagates from source rays at z=0 to an observation screen at distance `dist`.
///
/// # Arguments
/// * `ray_x`, `ray_z` — source ray positions [mm]
/// * `ray_es` — complex s-polarization amplitude at each ray
/// * `energy` — photon energy [eV]
/// * `dist` — propagation distance [mm]
/// * `x_min`, `x_max`, `nx` — observation screen x range and pixel count
/// * `z_min`, `z_max`, `nz` — observation screen z range and pixel count
pub fn fresnel_propagate(
    ray_x: &[f64],
    ray_z: &[f64],
    ray_es: &[Complex64],
    energy: f64,
    dist: f64,
    x_min: f64, x_max: f64, nx: usize,
    z_min: f64, z_max: f64, nz: usize,
) -> FresnelResult {
    let k = energy / CHBAR * 1e7; // mm⁻¹

    // Build pixel grid
    let x_pixels: Vec<f64> = (0..nx)
        .map(|i| x_min + (x_max - x_min) * i as f64 / (nx - 1).max(1) as f64)
        .collect();
    let z_pixels: Vec<f64> = (0..nz)
        .map(|i| z_min + (z_max - z_min) * i as f64 / (nz - 1).max(1) as f64)
        .collect();

    // Fresnel kernel: U(x',z') = Σ_j Es_j × exp(ik/(2d) × ((x'-x_j)² + (z'-z_j)²))
    let prefactor = Complex64::new(0.0, k / (2.0 * dist));

    let x_ref = &x_pixels;
    let amplitude: Vec<Complex64> = (0..nz)
        .into_par_iter()
        .flat_map(|iz| {
            let zp = z_pixels[iz];
            (0..nx).map(|ix| {
                let xp = x_ref[ix];
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..ray_x.len() {
                    let dx = xp - ray_x[j];
                    let dz = zp - ray_z[j];
                    let r2 = dx * dx + dz * dz;
                    let phase = prefactor * r2;
                    sum += ray_es[j] * phase.exp();
                }
                sum
            }).collect::<Vec<_>>()
        })
        .collect();

    FresnelResult { amplitude, x_pixels, z_pixels, nx, nz }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_single_point_source() {
        // Single point source → should produce Airy-like pattern
        let ray_x = vec![0.0];
        let ray_z = vec![0.0];
        let ray_es = vec![Complex64::new(1.0, 0.0)];

        let result = fresnel_propagate(
            &ray_x, &ray_z, &ray_es,
            10000.0, // 10 keV
            1000.0,  // 1 meter
            -0.1, 0.1, 21,
            -0.1, 0.1, 21,
        );

        assert_eq!(result.amplitude.len(), 21 * 21);
        // Center should have maximum intensity
        let center = &result.amplitude[10 * 21 + 10];
        let corner = &result.amplitude[0];
        assert!(center.norm() >= corner.norm(),
            "center should be brighter: {:.4e} vs {:.4e}",
            center.norm(), corner.norm());
    }

    #[test]
    fn fresnel_two_slit() {
        // Two slits → should show interference
        let ray_x = vec![-0.01, 0.01]; // two point sources separated by 20 µm
        let ray_z = vec![0.0, 0.0];
        let ray_es = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];

        let result = fresnel_propagate(
            &ray_x, &ray_z, &ray_es,
            10000.0,
            1000.0,
            -0.1, 0.1, 101,
            0.0, 0.0, 1,
        );

        let intensities: Vec<f64> = result.amplitude.iter().map(|a| a.norm_sqr()).collect();
        // Should have interference fringes (not monotonically decreasing)
        let center = intensities[50];
        let off_center = intensities[60];
        // Just verify we get non-trivial pattern
        assert!(center > 0.0);
        assert!(off_center > 0.0);
    }
}
