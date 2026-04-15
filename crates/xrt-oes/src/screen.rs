//! Screen / Detector: captures beam footprint and intensity distribution.
//!
//! A Screen is placed at a position in the beamline to record ray positions
//! and compute intensity distributions via 2D histogram binning.

use std::io::Write;

use xrt_core::beam::{Beam, RayState};

/// A screen (detector plane) that captures beam footprint.
#[derive(Debug, Clone)]
pub struct Screen {
    /// Center position [mm]
    pub center: [f64; 3],
    /// Half-width in x [mm]
    pub dx: f64,
    /// Half-width in z [mm]
    pub dz: f64,
    /// Number of bins in x
    pub nx: usize,
    /// Number of bins in z
    pub nz: usize,
}

/// Result of capturing a beam on a screen.
#[derive(Debug, Clone)]
pub struct ScreenCapture {
    /// 2D intensity histogram (row-major: [ix * nz + iz])
    pub intensity: Vec<f64>,
    /// Bin edges in x [mm]
    pub x_edges: Vec<f64>,
    /// Bin edges in z [mm]
    pub z_edges: Vec<f64>,
    /// Number of bins in x
    pub nx: usize,
    /// Number of bins in z
    pub nz: usize,
    /// Total number of captured rays
    pub n_captured: usize,
    /// Total number of missed rays (outside screen)
    pub n_missed: usize,
}

impl ScreenCapture {
    /// Get intensity at bin (ix, iz).
    pub fn get(&self, ix: usize, iz: usize) -> f64 {
        self.intensity[ix * self.nz + iz]
    }

    /// Total integrated intensity.
    pub fn total_intensity(&self) -> f64 {
        self.intensity.iter().sum()
    }

    /// Peak intensity.
    pub fn peak_intensity(&self) -> f64 {
        self.intensity.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Centroid position [x, z] in mm.
    pub fn centroid(&self) -> [f64; 2] {
        let total = self.total_intensity();
        if total <= 0.0 {
            return [0.0, 0.0];
        }

        let mut cx = 0.0;
        let mut cz = 0.0;
        for ix in 0..self.nx {
            let x = (self.x_edges[ix] + self.x_edges[ix + 1]) / 2.0;
            for iz in 0..self.nz {
                let z = (self.z_edges[iz] + self.z_edges[iz + 1]) / 2.0;
                let w = self.get(ix, iz);
                cx += x * w;
                cz += z * w;
            }
        }
        [cx / total, cz / total]
    }

    /// RMS beam size [σx, σz] in mm.
    pub fn rms_size(&self) -> [f64; 2] {
        let total = self.total_intensity();
        if total <= 0.0 {
            return [0.0, 0.0];
        }

        let [cx, cz] = self.centroid();
        let mut var_x = 0.0;
        let mut var_z = 0.0;
        for ix in 0..self.nx {
            let x = (self.x_edges[ix] + self.x_edges[ix + 1]) / 2.0;
            for iz in 0..self.nz {
                let z = (self.z_edges[iz] + self.z_edges[iz + 1]) / 2.0;
                let w = self.get(ix, iz);
                var_x += (x - cx) * (x - cx) * w;
                var_z += (z - cz) * (z - cz) * w;
            }
        }
        [(var_x / total).sqrt(), (var_z / total).sqrt()]
    }
}

impl ScreenCapture {
    /// 1D projection onto x-axis (sum over z bins).
    pub fn project_x(&self) -> Vec<f64> {
        (0..self.nx)
            .map(|ix| {
                (0..self.nz)
                    .map(|iz| self.intensity[ix * self.nz + iz])
                    .sum()
            })
            .collect()
    }

    /// 1D projection onto z-axis (sum over x bins).
    pub fn project_z(&self) -> Vec<f64> {
        let mut proj = vec![0.0; self.nz];
        for row in self.intensity.chunks(self.nz) {
            for (iz, val) in row.iter().enumerate() {
                proj[iz] += val;
            }
        }
        proj
    }

    /// Bin centers in x [mm].
    pub fn x_centers(&self) -> Vec<f64> {
        (0..self.nx)
            .map(|i| (self.x_edges[i] + self.x_edges[i + 1]) / 2.0)
            .collect()
    }

    /// Bin centers in z [mm].
    pub fn z_centers(&self) -> Vec<f64> {
        (0..self.nz)
            .map(|i| (self.z_edges[i] + self.z_edges[i + 1]) / 2.0)
            .collect()
    }

    /// FWHM of x-projection [mm]. Returns None if no intensity.
    pub fn fwhm_x(&self) -> Option<f64> {
        fwhm_of_projection(&self.project_x(), &self.x_centers())
    }

    /// FWHM of z-projection [mm]. Returns None if no intensity.
    pub fn fwhm_z(&self) -> Option<f64> {
        fwhm_of_projection(&self.project_z(), &self.z_centers())
    }

    /// Write the 2D intensity map as TSV (x, z, intensity).
    ///
    /// Format: three columns separated by tabs, one row per bin.
    /// Can be plotted with gnuplot `splot` or loaded into numpy.
    /// Write the 2D intensity map as TSV (x, z, intensity).
    ///
    /// Format: three columns separated by tabs, one row per bin.
    /// Blank lines between x-rows for gnuplot splot compatibility.
    pub fn write_tsv<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "# x_mm\tz_mm\tintensity")?;
        let xc = self.x_centers();
        let zc = self.z_centers();
        for (ix, &x) in xc.iter().enumerate() {
            for (iz, &z) in zc.iter().enumerate() {
                writeln!(w, "{:.6}\t{:.6}\t{:.6e}", x, z, self.get(ix, iz))?;
            }
            if self.nz > 0 {
                writeln!(w)?;
            }
        }
        Ok(())
    }

    /// Write 1D x-projection as TSV.
    pub fn write_projection_x<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "# x_mm\tintensity")?;
        for (x, v) in self.x_centers().iter().zip(self.project_x().iter()) {
            writeln!(w, "{:.6}\t{:.6e}", x, v)?;
        }
        Ok(())
    }

    /// Write 1D z-projection as TSV.
    pub fn write_projection_z<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "# z_mm\tintensity")?;
        for (z, v) in self.z_centers().iter().zip(self.project_z().iter()) {
            writeln!(w, "{:.6}\t{:.6e}", z, v)?;
        }
        Ok(())
    }
}

/// Compute FWHM from a 1D projection and its bin centers.
fn fwhm_of_projection(proj: &[f64], centers: &[f64]) -> Option<f64> {
    let peak = proj.iter().cloned().fold(0.0_f64, f64::max);
    if peak <= 0.0 {
        return None;
    }
    let half = peak * 0.5;
    let mut left = None;
    let mut right = None;
    for (i, &v) in proj.iter().enumerate() {
        if v >= half {
            if left.is_none() {
                left = Some(centers[i]);
            }
            right = Some(centers[i]);
        }
    }
    match (left, right) {
        (Some(l), Some(r)) => Some(r - l),
        _ => None,
    }
}

impl Screen {
    pub fn new(center: [f64; 3], dx: f64, dz: f64, nx: usize, nz: usize) -> Self {
        Self {
            center,
            dx,
            dz,
            nx,
            nz,
        }
    }

    /// Capture good rays from a beam onto this screen.
    ///
    /// Rays are projected onto the screen plane (at y = center[1]).
    /// Each good ray contributes weight = jss + jpp (total intensity).
    pub fn capture(&self, beam: &Beam) -> ScreenCapture {
        let x_min = self.center[0] - self.dx;
        let x_max = self.center[0] + self.dx;
        let z_min = self.center[2] - self.dz;
        let z_max = self.center[2] + self.dz;

        let bin_dx = (x_max - x_min) / self.nx as f64;
        let bin_dz = (z_max - z_min) / self.nz as f64;

        let x_edges: Vec<f64> = (0..=self.nx).map(|i| x_min + i as f64 * bin_dx).collect();
        let z_edges: Vec<f64> = (0..=self.nz).map(|i| z_min + i as f64 * bin_dz).collect();

        let mut intensity = vec![0.0; self.nx * self.nz];
        let mut n_captured = 0usize;
        let mut n_missed = 0usize;

        for i in 0..beam.nrays() {
            if beam.state[i] != RayState::Good as i32 {
                continue;
            }

            // Project ray to screen plane (propagate to y = center[1])
            let dy = self.center[1] - beam.y[i];
            let (rx, rz) = if beam.b[i].abs() > 1e-30 {
                let t = dy / beam.b[i];
                (beam.x[i] + beam.a[i] * t, beam.z[i] + beam.c[i] * t)
            } else {
                (beam.x[i], beam.z[i])
            };

            // Bin the ray
            let ix = ((rx - x_min) / bin_dx) as isize;
            let iz = ((rz - z_min) / bin_dz) as isize;

            if ix >= 0 && ix < self.nx as isize && iz >= 0 && iz < self.nz as isize {
                let weight = beam.jss[i] + beam.jpp[i];
                intensity[ix as usize * self.nz + iz as usize] += weight;
                n_captured += 1;
            } else {
                n_missed += 1;
            }
        }

        ScreenCapture {
            intensity,
            x_edges,
            z_edges,
            nx: self.nx,
            nz: self.nz,
            n_captured,
            n_missed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xrt_core::beam::Beam;

    #[test]
    fn screen_capture_centered_beam() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 5.0, 5.0, 10, 10);

        let mut beam = Beam::new(100);
        beam.set_state(RayState::Good);
        for i in 0..100 {
            beam.y[i] = 0.0;
            beam.b[i] = 1.0;
            beam.e[i] = 10000.0;
            // Rays at origin, going forward — should hit center of screen
        }

        let capture = screen.capture(&beam);
        assert_eq!(capture.n_captured, 100);
        assert_eq!(capture.n_missed, 0);
        assert!(capture.total_intensity() > 0.0);

        // All rays at origin → center bin should have all intensity
        let center = capture.get(5, 5);
        assert!(center > 0.0);
    }

    #[test]
    fn screen_capture_spread_beam() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 10.0, 10.0, 20, 20);

        let mut beam = Beam::new(1000);
        beam.set_state(RayState::Good);
        for i in 0..1000 {
            let frac = i as f64 / 999.0 - 0.5;
            beam.x[i] = frac * 8.0; // spread in x
            beam.z[i] = frac * 4.0; // spread in z
            beam.y[i] = 0.0;
            beam.b[i] = 1.0;
            beam.e[i] = 10000.0;
        }

        let capture = screen.capture(&beam);
        assert!(capture.n_captured > 900);
        assert!(capture.total_intensity() > 0.0);
    }

    #[test]
    fn screen_centroid() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 10.0, 10.0, 20, 20);

        let mut beam = Beam::new(100);
        beam.set_state(RayState::Good);
        for i in 0..100 {
            beam.x[i] = 2.0; // offset in x
            beam.z[i] = -1.0; // offset in z
            beam.y[i] = 0.0;
            beam.b[i] = 1.0;
        }

        let capture = screen.capture(&beam);
        let [cx, cz] = capture.centroid();
        assert!((cx - 2.0).abs() < 1.5, "centroid x = {cx}");
        assert!((cz - (-1.0)).abs() < 1.5, "centroid z = {cz}");
    }

    #[test]
    fn screen_rms_size() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 10.0, 10.0, 100, 100);

        let mut beam = Beam::new(10000);
        beam.set_state(RayState::Good);
        // Use a simple uniform distribution for predictable size
        for i in 0..10000 {
            let frac = i as f64 / 9999.0;
            beam.x[i] = (frac - 0.5) * 4.0; // ±2 mm → σ ≈ 1.15
            beam.z[i] = 0.0;
            beam.y[i] = 0.0;
            beam.b[i] = 1.0;
        }

        let capture = screen.capture(&beam);
        let [sx, _sz] = capture.rms_size();
        assert!(sx > 0.5 && sx < 2.0, "σx = {sx}");
    }

    #[test]
    fn screen_peak_intensity() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 5.0, 5.0, 10, 10);

        let mut beam = Beam::new(50);
        beam.set_state(RayState::Good);
        // All at same point
        for i in 0..50 {
            beam.b[i] = 1.0;
        }

        let capture = screen.capture(&beam);
        assert!((capture.peak_intensity() - capture.total_intensity()).abs() < 1e-10);
    }

    #[test]
    fn screen_projections() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 10.0, 10.0, 20, 20);

        let mut beam = Beam::new(100);
        beam.set_state(RayState::Good);
        for i in 0..100 {
            beam.b[i] = 1.0;
        }

        let capture = screen.capture(&beam);
        let proj_x = capture.project_x();
        let proj_z = capture.project_z();

        assert_eq!(proj_x.len(), 20);
        assert_eq!(proj_z.len(), 20);

        // Sum of projection should equal total
        let sum_x: f64 = proj_x.iter().sum();
        assert!((sum_x - capture.total_intensity()).abs() < 1e-10);
    }

    #[test]
    fn screen_fwhm() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 10.0, 10.0, 100, 100);

        let mut beam = Beam::new(1000);
        beam.set_state(RayState::Good);
        for i in 0..1000 {
            // Spread over ±2mm in x → FWHM should be ~4mm
            beam.x[i] = (i as f64 / 999.0 - 0.5) * 4.0;
            beam.b[i] = 1.0;
        }

        let capture = screen.capture(&beam);
        if let Some(fwhm_x) = capture.fwhm_x() {
            assert!(fwhm_x > 1.0 && fwhm_x < 6.0, "FWHM_x = {fwhm_x}");
        }
    }

    #[test]
    fn screen_write_tsv() {
        let screen = Screen::new([0.0, 1000.0, 0.0], 5.0, 5.0, 3, 3);
        let mut beam = Beam::new(10);
        beam.set_state(RayState::Good);
        for i in 0..10 {
            beam.b[i] = 1.0;
        }

        let capture = screen.capture(&beam);
        let mut buf = Vec::new();
        capture.write_tsv(&mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("# x_mm"));
        assert!(output.lines().count() > 3); // header + data + blank lines
    }

    #[test]
    fn screen_bin_centers() {
        let screen = Screen::new([5.0, 1000.0, -3.0], 2.0, 1.0, 4, 2);
        let capture = screen.capture(&Beam::new(0));

        let xc = capture.x_centers();
        assert_eq!(xc.len(), 4);
        // x range [3, 7], bin width=1, centers at 3.5, 4.5, 5.5, 6.5
        assert!((xc[0] - 3.5).abs() < 0.01, "xc[0]={}", xc[0]);

        let zc = capture.z_centers();
        assert_eq!(zc.len(), 2);
    }
}
