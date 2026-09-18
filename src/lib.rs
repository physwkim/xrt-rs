//! xrt-rs: ray tracing and wave propagation in the x-ray regime.
//!
//! A Rust port of [xrt](https://github.com/kklmn/xrt), validated against it
//! numerically. The modules follow the physics rather than the pipeline, so a
//! beamline is assembled from a [`sources`] emitter, [`oes`] elements carrying
//! [`materials`] properties, and a [`waves`] or [`pytte`] calculation:
//!
//! - [`core`] — [`Beam`](core::Beam), physical constants, coordinate transforms
//! - [`math`] — interpolation, root finding, atomic form factors
//! - [`materials`] — elements, compounds, crystals, multilayers
//! - [`oes`] — surfaces, intersection, reflection, gratings, screens
//! - [`sources`] — geometric source, bending magnet, wiggler, undulator
//! - [`waves`] — Kirchhoff diffraction integral
//! - [`pytte`] — Takagi-Taupin solver for bent and deformed crystals
//! - [`gpu`] — wgpu compute kernels, behind the `gpu` feature
//!
//! The `gpu` feature is off by default: without it the auto-dispatching entry
//! points ([`sources::undulator::Undulator::build_i_map_auto`],
//! `waves`' Kirchhoff path) run their f64 CPU implementations, which are the
//! reference the shaders are checked against.

pub mod core;
pub mod materials;
pub mod math;
pub mod oes;
pub mod pytte;
pub mod sources;
pub mod waves;

#[cfg(feature = "gpu")]
pub mod gpu;
