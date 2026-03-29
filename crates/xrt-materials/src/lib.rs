#![allow(clippy::too_many_arguments, clippy::type_complexity)]
//! xrt-materials: material properties for X-ray tracing.
//!
//! - Element: atomic scattering factors (f0, f1, f2)
//! - Material: refractive index, absorption, Fresnel amplitudes
//! - Crystal: dynamical diffraction theory (Bragg/Laue)
//! - Data: file parsers for scattering factor tables

pub mod data;
pub mod elements;
pub mod material;
pub mod crystal;
pub mod crystal_variants;
pub mod multilayer;
