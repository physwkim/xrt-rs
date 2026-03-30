//! Surface error map (figure error from external data).
//!
//! Wraps any Surface and adds a height error from a 2D grid.
//! Compatible with shadow3's surface error file format.

use crate::surface::Surface;

/// A surface with figure error from a 2D height map.
#[derive(Debug, Clone)]
pub struct ErrorMapSurface<S: Surface> {
    /// Underlying ideal surface
    pub base: S,
    /// Error map grid: x positions [mm]
    pub x_grid: Vec<f64>,
    /// Error map grid: y positions [mm]
    pub y_grid: Vec<f64>,
    /// Height errors [mm], stored row-major [ny × nx]
    pub errors: Vec<f64>,
    /// Scale factor for errors
    pub scale: f64,
}

impl<S: Surface> ErrorMapSurface<S> {
    pub fn new(
        base: S,
        x_grid: Vec<f64>,
        y_grid: Vec<f64>,
        errors: Vec<f64>,
        scale: f64,
    ) -> Self {
        assert_eq!(errors.len(), x_grid.len() * y_grid.len());
        Self {
            base,
            x_grid,
            y_grid,
            errors,
            scale,
        }
    }

    /// Bilinear interpolation of the error at (x, y).
    fn interp_error(&self, x: f64, y: f64) -> f64 {
        let nx = self.x_grid.len();
        let ny = self.y_grid.len();
        if nx < 2 || ny < 2 {
            return 0.0;
        }

        // Find x index
        let ix = self.x_grid.partition_point(|&v| v < x);
        let ix = ix.clamp(1, nx - 1);
        let ix0 = ix - 1;
        let tx = (x - self.x_grid[ix0]) / (self.x_grid[ix] - self.x_grid[ix0]);
        let tx = tx.clamp(0.0, 1.0);

        // Find y index
        let iy = self.y_grid.partition_point(|&v| v < y);
        let iy = iy.clamp(1, ny - 1);
        let iy0 = iy - 1;
        let ty = (y - self.y_grid[iy0]) / (self.y_grid[iy] - self.y_grid[iy0]);
        let ty = ty.clamp(0.0, 1.0);

        // Bilinear interpolation
        let e00 = self.errors[iy0 * nx + ix0];
        let e10 = self.errors[iy0 * nx + ix];
        let e01 = self.errors[iy * nx + ix0];
        let e11 = self.errors[iy * nx + ix];

        let e0 = e00 * (1.0 - tx) + e10 * tx;
        let e1 = e01 * (1.0 - tx) + e11 * tx;
        (e0 * (1.0 - ty) + e1 * ty) * self.scale
    }
}

impl<S: Surface> Surface for ErrorMapSurface<S> {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        self.base.local_z(x, y) + self.interp_error(x, y)
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        // Numerical normal including error contribution
        let eps = 1e-6;
        let dz_dx = (self.local_z(x + eps, y) - self.local_z(x - eps, y)) / (2.0 * eps);
        let dz_dy = (self.local_z(x, y + eps) - self.local_z(x, y - eps)) / (2.0 * eps);
        let nx = -dz_dx;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }

    fn local_n_bragg(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        self.base.local_n_bragg(x, y)
    }

    fn local_g(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        self.base.local_g(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::flat::FlatSurface;

    #[test]
    fn error_map_zero_error() {
        let s = ErrorMapSurface::new(
            FlatSurface,
            vec![-10.0, 10.0],
            vec![-10.0, 10.0],
            vec![0.0, 0.0, 0.0, 0.0],
            1.0,
        );
        assert!(s.local_z(0.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn error_map_adds_height() {
        let s = ErrorMapSurface::new(
            FlatSurface,
            vec![-10.0, 10.0],
            vec![-10.0, 10.0],
            vec![0.001, 0.001, 0.001, 0.001], // 1 µm uniform error
            1.0,
        );
        assert!((s.local_z(0.0, 0.0) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn error_map_interpolation() {
        let s = ErrorMapSurface::new(
            FlatSurface,
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![0.0, 0.01, 0.0, 0.01],
            1.0,
        );
        // At x=5, y=0: should interpolate to 0.005
        let z = s.local_z(5.0, 0.0);
        assert!((z - 0.005).abs() < 1e-10, "interpolated z={z}");
    }
}
