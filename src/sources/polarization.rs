//! Polarization / coherency matrix initialization.
//!
//! Ported from sources_geoms.py:37-132.

use num_complex::Complex64;
use rand::Rng;

use crate::core::beam::Beam;

/// Polarization type.
#[derive(Debug, Clone)]
pub enum Polarization {
    /// Horizontal linear
    Horizontal,
    /// Vertical linear
    Vertical,
    /// +45° linear
    Plus45,
    /// -45° linear
    Minus45,
    /// Right circular
    Right,
    /// Left circular
    Left,
    /// Unpolarized (natural)
    Unpolarized,
    /// Custom coherency matrix: (Jss, Jpp, Re(Jsp), Im(Jsp))
    Custom(f64, f64, f64, f64),
}

/// Initialize the beam's coherency matrix and optional field amplitudes.
pub fn make_polarization(pol: &Polarization, beam: &mut Beam) {
    let sq2inv = std::f64::consts::FRAC_1_SQRT_2;

    match pol {
        Polarization::Horizontal => {
            fill_beam(beam, 1.0, 0.0, Complex64::new(0.0, 0.0), 1.0, 0.0);
        }
        Polarization::Vertical => {
            fill_beam(beam, 0.0, 1.0, Complex64::new(0.0, 0.0), 0.0, 1.0);
        }
        Polarization::Plus45 => {
            fill_beam(beam, 0.5, 0.5, Complex64::new(0.5, 0.0), sq2inv, sq2inv);
        }
        Polarization::Minus45 => {
            fill_beam(beam, 0.5, 0.5, Complex64::new(-0.5, 0.0), sq2inv, -sq2inv);
        }
        Polarization::Right => {
            fill_beam(
                beam,
                0.5,
                0.5,
                Complex64::new(0.0, 0.5),
                sq2inv,
                -sq2inv, // -i * 2^(-0.5) → real part for Ep
            );
        }
        Polarization::Left => {
            fill_beam(beam, 0.5, 0.5, Complex64::new(0.0, -0.5), sq2inv, sq2inv);
        }
        Polarization::Unpolarized => {
            fill_beam_unpolarized(beam, sq2inv);
        }
        Polarization::Custom(jss, jpp, re_jsp, im_jsp) => {
            beam.jss.fill(*jss);
            beam.jpp.fill(*jpp);
            beam.jsp.fill(Complex64::new(*re_jsp, *im_jsp));
        }
    }
}

fn fill_beam(beam: &mut Beam, jss: f64, jpp: f64, jsp: Complex64, es_val: f64, ep_val: f64) {
    beam.jss.fill(jss);
    beam.jpp.fill(jpp);
    beam.jsp.fill(jsp);
    if let Some(amps) = beam.amplitudes_mut() {
        amps.es.fill(Complex64::new(es_val, 0.0));
        amps.ep.fill(Complex64::new(ep_val, 0.0));
    }
}

fn fill_beam_unpolarized(beam: &mut Beam, sq2inv: f64) {
    beam.jss.fill(0.5);
    beam.jpp.fill(0.5);
    beam.jsp.fill(Complex64::new(0.0, 0.0));
    if let Some(amps) = beam.amplitudes_mut() {
        amps.es.fill(Complex64::new(sq2inv, 0.0));
        // Random phase for unpolarized light
        let mut rng = rand::thread_rng();
        for i in 0..amps.ep.len() {
            amps.ep[i] = Complex64::new(rng.r#gen::<f64>() * sq2inv, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_polarization() {
        let mut beam = Beam::new(10);
        make_polarization(&Polarization::Horizontal, &mut beam);
        assert_eq!(beam.jss[0], 1.0);
        assert_eq!(beam.jpp[0], 0.0);
        assert_eq!(beam.jsp[0], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn unpolarized() {
        let mut beam = Beam::new(10);
        make_polarization(&Polarization::Unpolarized, &mut beam);
        assert_eq!(beam.jss[0], 0.5);
        assert_eq!(beam.jpp[0], 0.5);
        assert_eq!(beam.jsp[0], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn right_circular() {
        let mut beam = Beam::new(10);
        make_polarization(&Polarization::Right, &mut beam);
        assert_eq!(beam.jss[0], 0.5);
        assert_eq!(beam.jpp[0], 0.5);
        assert!((beam.jsp[0].im - 0.5).abs() < 1e-15);
    }

    #[test]
    fn custom_coherency() {
        let mut beam = Beam::new(10);
        make_polarization(&Polarization::Custom(0.7, 0.3, 0.1, 0.2), &mut beam);
        assert_eq!(beam.jss[0], 0.7);
        assert_eq!(beam.jpp[0], 0.3);
        assert_eq!(beam.jsp[0], Complex64::new(0.1, 0.2));
    }
}
