#![allow(clippy::too_many_arguments)]
//! xrt-oes: optical elements for X-ray tracing.
//!
//! # Surfaces
//!
//! - **Surface trait** (`z = f(x,y)`): Flat, Spherical, Toroid, ParaboloidLens, BlazedGrating, LaminarGrating, FZP
//! - **ParametricSurface trait** (`r = f(s,φ)`): Elliptical, Parabolical, Hyperbolic
//!
//! # Optical Elements
//!
//! - [`OpticalElement`](oe::OpticalElement) — geometric reflection only
//! - [`MaterialOpticalElement`](material_oe::MaterialOpticalElement) — mirror with Fresnel Rs/Rp from Material
//! - [`GratingOpticalElement`](grating_oe::GratingOpticalElement) — grating with diffraction order
//! - [`ParametricOpticalElement`](param_oe::ParametricOpticalElement) — parametric surface OE
//! - [`CrystalOpticalElement`](crystal_oe::CrystalOpticalElement) — crystal with dynamical diffraction
//!
//! # Pipeline
//!
//! Bracketing → intersection → aperture → deflection (reflect/grating/refract) → material amplitude
//!
//! # Beamline
//!
//! Chain OEs with [`Beamline`](beamline::Beamline):
//! `Beamline::new().add_material("M1", m1).drift(2000.0).add_grating("G1", g1).propagate(&mut beam)`

pub mod surface;
pub mod surfaces;
pub mod intersection;
pub mod aperture;
pub mod bracketing;
pub mod deflection;
pub mod reflect;
pub mod oe;
pub mod param_oe;
pub mod crystal_oe;
pub mod material_oe;
pub mod grating_oe;
pub mod screen;
pub mod beamline;
