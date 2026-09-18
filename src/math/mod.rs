#![allow(clippy::too_many_arguments)]
//! xrt-math: numerical methods for X-ray tracing.
//!
//! - f0 Gaussian sum evaluation (atomic form factor)
//! - Cubic spline and linear interpolation
//! - Root finding (secant + Brent methods)

pub mod f0;
pub mod interp;
pub mod rootfind;
