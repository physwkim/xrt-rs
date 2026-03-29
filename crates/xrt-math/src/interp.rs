//! Interpolation routines: cubic spline and linear interpolation.
//!
//! Replaces `scipy.interpolate.interp1d(kind='cubic')` and `np.interp`.

use ndarray::Array1;
use num_complex::Complex64;

/// Behavior when evaluating outside the data range.
#[derive(Debug, Clone, Copy)]
pub enum Extrapolation {
    /// Clamp to a constant value.
    Constant(f64),
    /// Extrapolate linearly from the boundary segment.
    Linear,
    /// Extrapolate using the cubic polynomial of the boundary segment.
    Cubic,
}

/// Natural cubic spline interpolation.
///
/// Uses the Thomas algorithm (O(n)) to solve the tridiagonal system for
/// natural spline boundary conditions (second derivative = 0 at endpoints).
#[derive(Debug, Clone)]
pub struct CubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Polynomial coefficients [a, b, c, d] per segment:
    /// S_i(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
    coeffs: Vec<[f64; 4]>,
    extrap: Extrapolation,
}

impl CubicSpline {
    /// Build a natural cubic spline from sorted (xs, ys) data.
    ///
    /// # Panics
    /// Panics if xs.len() != ys.len() or len < 2.
    pub fn new(xs: &[f64], ys: &[f64], extrap: Extrapolation) -> Self {
        let n = xs.len();
        assert!(n >= 2, "need at least 2 data points");
        assert_eq!(n, ys.len(), "xs and ys must have same length");

        if n == 2 {
            // Linear segment
            let dx = xs[1] - xs[0];
            let dy = ys[1] - ys[0];
            let slope = dy / dx;
            return Self {
                xs: xs.to_vec(),
                ys: ys.to_vec(),
                coeffs: vec![[ys[0], slope, 0.0, 0.0]],
                extrap,
            };
        }

        let m = n - 1; // number of segments

        // Step 1: compute h_i and alpha_i
        let mut h = vec![0.0; m];
        for i in 0..m {
            h[i] = xs[i + 1] - xs[i];
        }

        // Step 2: set up tridiagonal system for natural spline
        // A * c = rhs, where c[0] = c[n-1] = 0 (natural BC)
        let inner = n - 2; // number of interior points
        let mut diag = vec![0.0; inner]; // main diagonal
        let mut upper = vec![0.0; inner]; // upper diagonal
        let mut lower = vec![0.0; inner]; // lower diagonal
        let mut rhs = vec![0.0; inner];

        for i in 0..inner {
            let j = i + 1; // index into original arrays
            diag[i] = 2.0 * (h[j - 1] + h[j]);
            rhs[i] = 3.0 * ((ys[j + 1] - ys[j]) / h[j] - (ys[j] - ys[j - 1]) / h[j - 1]);
            if i > 0 {
                lower[i] = h[j - 1];
            }
            if i < inner - 1 {
                upper[i] = h[j];
            }
        }

        // Step 3: Thomas algorithm (forward sweep)
        for i in 1..inner {
            let w = lower[i] / diag[i - 1];
            diag[i] -= w * upper[i - 1];
            rhs[i] -= w * rhs[i - 1];
        }

        // Back substitution
        let mut c_inner = vec![0.0; inner];
        c_inner[inner - 1] = rhs[inner - 1] / diag[inner - 1];
        for i in (0..inner - 1).rev() {
            c_inner[i] = (rhs[i] - upper[i] * c_inner[i + 1]) / diag[i];
        }

        // Full c array with natural BC
        let mut c_arr = vec![0.0; n];
        c_arr[1..(inner + 1)].copy_from_slice(&c_inner[..inner]);

        // Step 4: compute polynomial coefficients per segment
        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let a = ys[i];
            let b_coeff =
                (ys[i + 1] - ys[i]) / h[i] - h[i] * (2.0 * c_arr[i] + c_arr[i + 1]) / 3.0;
            let c_coeff = c_arr[i];
            let d_coeff = (c_arr[i + 1] - c_arr[i]) / (3.0 * h[i]);
            coeffs.push([a, b_coeff, c_coeff, d_coeff]);
        }

        Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            coeffs,
            extrap,
        }
    }

    /// Find the segment index for a given x using binary search.
    #[inline]
    fn find_segment(&self, x: f64) -> usize {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return 0;
        }
        if x >= self.xs[n - 1] {
            return self.coeffs.len() - 1;
        }
        // Binary search: find i such that xs[i] <= x < xs[i+1]
        let mut lo = 0;
        let mut hi = n - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Evaluate the spline at a single point.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let m = self.coeffs.len();

        // Handle extrapolation
        if x < self.xs[0] {
            match self.extrap {
                Extrapolation::Constant(v) => return v,
                Extrapolation::Linear => {
                    let [_, b, _, _] = self.coeffs[0];
                    return self.ys[0] + b * (x - self.xs[0]);
                }
                Extrapolation::Cubic => {
                    // Fall through to use segment 0
                }
            }
        } else if x > self.xs[n - 1] {
            match self.extrap {
                Extrapolation::Constant(v) => return v,
                Extrapolation::Linear => {
                    // Use derivative at the last point
                    let [a, b, c, d] = self.coeffs[m - 1];
                    let dx_end = self.xs[n - 1] - self.xs[n - 2];
                    let deriv = b + 2.0 * c * dx_end + 3.0 * d * dx_end * dx_end;
                    let y_end = a + b * dx_end + c * dx_end * dx_end + d * dx_end * dx_end * dx_end;
                    return y_end + deriv * (x - self.xs[n - 1]);
                }
                Extrapolation::Cubic => {
                    // Fall through to use last segment
                }
            }
        }

        let i = self.find_segment(x);
        let dx = x - self.xs[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    /// Evaluate the spline for an array of points.
    pub fn eval_array(&self, xs: &Array1<f64>) -> Array1<f64> {
        xs.mapv(|x| self.eval(x))
    }
}

/// Complex cubic spline: separate splines for real and imaginary parts.
#[derive(Debug, Clone)]
pub struct CubicSplineComplex {
    real: CubicSpline,
    imag: CubicSpline,
}

impl CubicSplineComplex {
    pub fn new(xs: &[f64], ys: &[Complex64], extrap: Extrapolation) -> Self {
        let reals: Vec<f64> = ys.iter().map(|c| c.re).collect();
        let imags: Vec<f64> = ys.iter().map(|c| c.im).collect();
        Self {
            real: CubicSpline::new(xs, &reals, extrap),
            imag: CubicSpline::new(xs, &imags, extrap),
        }
    }

    pub fn eval(&self, x: f64) -> Complex64 {
        Complex64::new(self.real.eval(x), self.imag.eval(x))
    }

    pub fn eval_array(&self, xs: &Array1<f64>) -> Array1<Complex64> {
        xs.mapv(|x| self.eval(x))
    }
}

/// Linear interpolation equivalent to `np.interp`.
///
/// For each value in `xp`, interpolates in the sorted knot arrays (xk, yk).
/// Values outside the range are clamped to the boundary values.
pub fn interp_linear(xp: &Array1<f64>, xk: &[f64], yk: &[f64]) -> Array1<f64> {
    debug_assert_eq!(xk.len(), yk.len());
    let n = xk.len();
    xp.mapv(|x| {
        if x <= xk[0] {
            yk[0]
        } else if x >= xk[n - 1] {
            yk[n - 1]
        } else {
            // Binary search
            let mut lo = 0;
            let mut hi = n - 1;
            while lo < hi - 1 {
                let mid = (lo + hi) / 2;
                if xk[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let t = (x - xk[lo]) / (xk[hi] - xk[lo]);
            yk[lo] + t * (yk[hi] - yk[lo])
        }
    })
}

/// Scalar linear interpolation in sorted knot arrays.
/// Values outside the range are clamped to the boundary values.
#[inline]
pub fn interp_linear_scalar(x: f64, xk: &[f64], yk: &[f64]) -> f64 {
    let n = xk.len();
    if x <= xk[0] {
        return yk[0];
    }
    if x >= xk[n - 1] {
        return yk[n - 1];
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if xk[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xk[lo]) / (xk[hi] - xk[lo]);
    yk[lo] + t * (yk[hi] - yk[lo])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cubic_spline_accurate_on_smooth_function() {
        // sin(x) on [0, π] with 20 points — natural spline should be very accurate
        let n = 20;
        let xs: Vec<f64> = (0..=n)
            .map(|i| i as f64 * std::f64::consts::PI / n as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let spline = CubicSpline::new(&xs, &ys, Extrapolation::Cubic);

        // Test at midpoints — should be accurate to ~1e-6 with 20 pts
        for i in 1..n - 1 {
            let x = (xs[i] + xs[i + 1]) / 2.0;
            let expected = x.sin();
            let got = spline.eval(x);
            assert!(
                (got - expected).abs() < 1e-5,
                "at x={x}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn cubic_spline_interpolates_data_points() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let spline = CubicSpline::new(&xs, &ys, Extrapolation::Linear);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let got = spline.eval(x);
            assert!(
                (got - y).abs() < 1e-12,
                "at x={x}: got {got}, expected {y}"
            );
        }
    }

    #[test]
    fn linear_interp_matches_numpy() {
        let xk = vec![0.0, 1.0, 2.0, 3.0];
        let yk = vec![0.0, 2.0, 1.0, 3.0];
        let xp = array![0.5, 1.5, 2.5, -1.0, 5.0];
        let result = interp_linear(&xp, &xk, &yk);

        assert!((result[0] - 1.0).abs() < 1e-12); // midpoint 0-1
        assert!((result[1] - 1.5).abs() < 1e-12); // midpoint 1-2
        assert!((result[2] - 2.0).abs() < 1e-12); // midpoint 2-3
        assert!((result[3] - 0.0).abs() < 1e-12); // clamped left
        assert!((result[4] - 3.0).abs() < 1e-12); // clamped right
    }

    #[test]
    fn cubic_spline_two_points() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 2.0];
        let spline = CubicSpline::new(&xs, &ys, Extrapolation::Linear);
        assert!((spline.eval(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn complex_spline_roundtrip() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];
        let spline = CubicSplineComplex::new(&xs, &ys, Extrapolation::Linear);
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let got = spline.eval(x);
            assert!((got - y).norm() < 1e-12);
        }
    }

    #[test]
    fn extrapolation_constant() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 2.0, 3.0];
        let spline = CubicSpline::new(&xs, &ys, Extrapolation::Constant(0.0));
        assert_eq!(spline.eval(-1.0), 0.0);
        assert_eq!(spline.eval(5.0), 0.0);
    }
}
