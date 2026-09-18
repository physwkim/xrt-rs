//! xrt-waves: Kirchhoff diffraction integral.
//!
//! CPU implementation with rayon parallelization over pixels.
//! The inner loop over rays is sequential with Kahan summation.

pub mod diffraction;
pub mod fresnel;
pub mod prepare_wave;
