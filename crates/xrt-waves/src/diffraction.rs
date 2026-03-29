//! CPU Kirchhoff diffraction integral: O(N_pixel × N_ray).
//!
//! Ported from the Python XRT diffraction integral (diffract.cl → CPU).
//!
//! For each pixel, sums over all good rays on the OE:
//!   U = (k·i)/(4π) × (nl + ns) × exp(i·k·path) / path
//!   Es_result += Es_j × U
//!   Ep_result += Ep_j × U

use num_complex::Complex64;
use rayon::prelude::*;

use xrt_core::consts::CHBAR;

/// A single ray on the OE surface, ready for diffraction computation.
#[derive(Debug, Clone, Copy)]
pub struct DiffractionRay {
    /// Position on the OE [mm]
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Surface normal at ray position
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
    /// Obliquity factor (local_n · beam_direction)
    pub nl: f64,
    /// Field amplitudes
    pub es: Complex64,
    pub ep: Complex64,
    /// Energy [eV]
    pub energy: f64,
}

/// A pixel in the observation plane.
#[derive(Debug, Clone, Copy)]
pub struct PixelPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Result of diffraction at a single pixel.
#[derive(Debug, Clone, Copy)]
pub struct DiffractionResult {
    pub es: Complex64,
    pub ep: Complex64,
}

/// Compute the Kirchhoff diffraction integral.
///
/// Outer loop (pixels): rayon parallel.
/// Inner loop (rays): sequential accumulation.
///
/// Returns one `DiffractionResult` per pixel.
pub fn diffraction_integral(
    rays: &[DiffractionRay],
    pixels: &[PixelPoint],
) -> Vec<DiffractionResult> {
    pixels
        .par_iter()
        .map(|pixel| diffraction_at_pixel(rays, pixel))
        .collect()
}

/// Compute diffraction at a single pixel by summing over all rays.
fn diffraction_at_pixel(rays: &[DiffractionRay], pixel: &PixelPoint) -> DiffractionResult {
    let mut es_sum = Complex64::new(0.0, 0.0);
    let mut ep_sum = Complex64::new(0.0, 0.0);

    // Kahan summation compensators
    let mut es_comp = Complex64::new(0.0, 0.0);
    let mut ep_comp = Complex64::new(0.0, 0.0);

    for ray in rays {
        let dx = pixel.x - ray.x;
        let dy = pixel.y - ray.y;
        let dz = pixel.z - ray.z;
        let path = (dx * dx + dy * dy + dz * dz).sqrt();

        if path < 1e-30 {
            continue;
        }

        // ns = dot(normal, (pixel - ray)) / path
        let ns = (ray.nx * dx + ray.ny * dy + ray.nz * dz) / path;

        // k = E / (c·ℏ) [Å⁻¹] → need to convert to mm⁻¹: k_mm = E / CHBAR * 1e7
        let k = ray.energy / CHBAR * 1e7; // mm⁻¹

        // U = k·i/(4π) × (nl + ns) × exp(i·k·path) / path
        let phase = k * path;
        let (sin_phase, cos_phase) = phase.sin_cos();
        let exp_ikr = Complex64::new(cos_phase, sin_phase);

        let obliquity = ray.nl + ns;
        let amplitude_factor = k / (4.0 * std::f64::consts::PI) * obliquity / path;
        let u = Complex64::i() * amplitude_factor * exp_ikr;

        // Kahan summation for Es
        let es_term = ray.es * u - es_comp;
        let es_new = es_sum + es_term;
        es_comp = (es_new - es_sum) - es_term;
        es_sum = es_new;

        // Kahan summation for Ep
        let ep_term = ray.ep * u - ep_comp;
        let ep_new = ep_sum + ep_term;
        ep_comp = (ep_new - ep_sum) - ep_term;
        ep_sum = ep_new;
    }

    DiffractionResult {
        es: es_sum,
        ep: ep_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_ray_single_pixel() {
        let ray = DiffractionRay {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            nl: 1.0,
            es: Complex64::new(1.0, 0.0),
            ep: Complex64::new(0.0, 0.0),
            energy: 10000.0,
        };
        let pixel = PixelPoint {
            x: 0.0,
            y: 1000.0, // 1 meter away
            z: 0.0,
        };

        let result = diffraction_integral(&[ray], &[pixel]);
        assert_eq!(result.len(), 1);
        // Should have non-zero Es
        assert!(result[0].es.norm() > 0.0, "Es = {}", result[0].es);
        // Ep should be zero since input Ep is zero
        assert!(result[0].ep.norm() < 1e-30);
    }

    #[test]
    fn parallel_pixels() {
        let n_rays = 100;
        let rays: Vec<DiffractionRay> = (0..n_rays)
            .map(|i| {
                let x = (i as f64 - 50.0) * 0.01;
                DiffractionRay {
                    x,
                    y: 0.0,
                    z: 0.0,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                    nl: 1.0,
                    es: Complex64::new(1.0, 0.0),
                    ep: Complex64::new(0.0, 0.0),
                    energy: 10000.0,
                }
            })
            .collect();

        let n_pixels = 50;
        let pixels: Vec<PixelPoint> = (0..n_pixels)
            .map(|i| PixelPoint {
                x: (i as f64 - 25.0) * 0.02,
                y: 1000.0,
                z: 0.0,
            })
            .collect();

        let results = diffraction_integral(&rays, &pixels);
        assert_eq!(results.len(), n_pixels);

        // Central pixel should have highest intensity
        let center_idx = n_pixels / 2;
        let center_intensity = results[center_idx].es.norm_sqr();
        let edge_intensity = results[0].es.norm_sqr();
        assert!(
            center_intensity > edge_intensity,
            "center {} should be > edge {}",
            center_intensity,
            edge_intensity
        );
    }
}
