//! xrt-pytte: Takagi-Taupin ODE solver for dynamical diffraction.
//!
//! Solves the Takagi-Taupin equations for bent/deformed crystals using
//! a Dormand-Prince RK45 adaptive integrator.

pub mod crystal;
pub mod deformation;
pub mod quantity;
pub mod scan;
pub mod solver;
