//! Wave preparation utilities.
//!
//! Creates pixel grids for diffraction calculations and converts
//! beam data to diffraction-ready ray format.

use xrt_core::beam::Beam;

use crate::diffraction::{DiffractionRay, PixelPoint};

/// Parameters for wave/screen pixel generation.
#[derive(Debug, Clone)]
pub struct WaveParams {
    /// Screen center position [mm]
    pub center: [f64; 3],
    /// Screen extent in x [mm] (half-width)
    pub dx: f64,
    /// Screen extent in z [mm] (half-height)
    pub dz: f64,
    /// Number of pixels in x
    pub nx: usize,
    /// Number of pixels in z
    pub nz: usize,
}

impl WaveParams {
    pub fn new(center: [f64; 3], dx: f64, dz: f64, nx: usize, nz: usize) -> Self {
        Self {
            center,
            dx,
            dz,
            nx,
            nz,
        }
    }
}

/// Generate a 2D pixel grid for diffraction calculation.
///
/// Returns pixels in row-major order (z varies fastest).
pub fn prepare_wave(params: &WaveParams) -> Vec<PixelPoint> {
    let mut pixels = Vec::with_capacity(params.nx * params.nz);

    for ix in 0..params.nx {
        let frac_x = if params.nx > 1 {
            ix as f64 / (params.nx - 1) as f64 - 0.5
        } else {
            0.0
        };
        let px = params.center[0] + frac_x * 2.0 * params.dx;

        for iz in 0..params.nz {
            let frac_z = if params.nz > 1 {
                iz as f64 / (params.nz - 1) as f64 - 0.5
            } else {
                0.0
            };
            let pz = params.center[2] + frac_z * 2.0 * params.dz;

            pixels.push(PixelPoint {
                x: px,
                y: params.center[1],
                z: pz,
            });
        }
    }

    pixels
}

/// Convert beam data to diffraction rays.
///
/// Extracts good rays from the beam and pairs them with surface normals.
///
/// # Arguments
/// * `beam` - The beam after reflection/diffraction
/// * `normals` - Surface normals at each ray hit point [nx, ny, nz]
/// * `good` - Indices of good rays to include
pub fn beam_to_diffraction_rays(
    beam: &Beam,
    normals: &[[f64; 3]],
    good: &[usize],
) -> Vec<DiffractionRay> {
    good.iter()
        .map(|&i| {
            let n = normals[i];
            // Obliquity factor: dot(normal, beam_direction)
            let nl = n[0] * beam.a[i] + n[1] * beam.b[i] + n[2] * beam.c[i];

            let (es, ep) = match (&beam.es, &beam.ep) {
                (Some(es_arr), Some(ep_arr)) => (es_arr[i], ep_arr[i]),
                _ => (
                    num_complex::Complex64::new(1.0, 0.0),
                    num_complex::Complex64::new(0.0, 0.0),
                ),
            };

            DiffractionRay {
                x: beam.x[i],
                y: beam.y[i],
                z: beam.z[i],
                nx: n[0],
                ny: n[1],
                nz: n[2],
                nl,
                es,
                ep,
                energy: beam.e[i],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn prepare_wave_pixel_count() {
        let params = WaveParams::new([0.0, 1000.0, 0.0], 1.0, 1.0, 10, 10);
        let pixels = prepare_wave(&params);
        assert_eq!(pixels.len(), 100);
    }

    #[test]
    fn prepare_wave_center() {
        let params = WaveParams::new([5.0, 1000.0, 3.0], 1.0, 1.0, 1, 1);
        let pixels = prepare_wave(&params);
        assert_eq!(pixels.len(), 1);
        assert!((pixels[0].x - 5.0).abs() < 1e-10);
        assert!((pixels[0].y - 1000.0).abs() < 1e-10);
        assert!((pixels[0].z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn prepare_wave_grid_extent() {
        let params = WaveParams::new([0.0, 1000.0, 0.0], 5.0, 3.0, 11, 7);
        let pixels = prepare_wave(&params);
        // First pixel x should be -5.0, last should be +5.0
        assert!((pixels[0].x - (-5.0)).abs() < 1e-10);
        let last_row_start = (11 - 1) * 7;
        assert!((pixels[last_row_start].x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn beam_to_diffraction_rays_basic() {
        let mut beam = Beam::with_amplitudes(3);
        beam.x[0] = 1.0;
        beam.y[0] = 2.0;
        beam.z[0] = 3.0;
        beam.a[0] = 0.0;
        beam.b[0] = 1.0;
        beam.c[0] = 0.0;
        beam.e[0] = 10000.0;
        if let Some(ref mut es) = beam.es {
            es[0] = Complex64::new(1.0, 0.0);
        }

        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let good = vec![0];

        let rays = beam_to_diffraction_rays(&beam, &normals, &good);
        assert_eq!(rays.len(), 1);
        assert!((rays[0].x - 1.0).abs() < 1e-10);
        assert!((rays[0].energy - 10000.0).abs() < 1e-10);
    }
}
