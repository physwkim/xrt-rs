#![allow(clippy::too_many_arguments)]
//! xrt-sources: X-ray source models.
//!
//! - GeometricSource: parametric source with configurable distributions
//! - BendingMagnet: synchrotron bending magnet with Monte Carlo sampling
//! - Distribution utilities and polarization initialization

pub mod bending_magnet;
pub mod distributions;
pub mod geometric;
pub mod polarization;
mod rejection;
pub mod undulator;
pub mod wiggler;
