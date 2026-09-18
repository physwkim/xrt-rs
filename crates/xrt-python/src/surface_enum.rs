//! SurfaceEnum and ParametricSurfaceEnum for Python-side type erasure.
//!
//! Python cannot use Rust generics, so we wrap all surface types into enums
//! that implement the Surface/ParametricSurface traits via match dispatch.

use xrt_rs::oes::surface::{ParametricSurface, Surface};
use xrt_rs::oes::surfaces::bent_flat::BentFlatSurface;
use xrt_rs::oes::surfaces::cylindrical::CylindricalSurface;
use xrt_rs::oes::surfaces::elliptical::EllipticalSurface;
use xrt_rs::oes::surfaces::flat::FlatSurface;
use xrt_rs::oes::surfaces::grating::{BlazedGrating, LaminarGrating};
use xrt_rs::oes::surfaces::lens::ParaboloidLensSurface;
use xrt_rs::oes::surfaces::parabolical::ParabolicalSurface;
use xrt_rs::oes::surfaces::spherical::SphericalSurface;
use xrt_rs::oes::surfaces::toroid::ToroidSurface;
use xrt_rs::oes::surfaces::vfm::VfmSurface;
use xrt_rs::oes::surfaces::vls_grating::VlsGrating;

/// Enum wrapping all Surface-trait types for Python dispatch.
///
/// Only includes surfaces that have corresponding Py* classes registered
/// in the Python module. Add new variants here when exposing new surface
/// types to Python.
#[derive(Clone)]
pub enum SurfaceEnum {
    Flat(FlatSurface),
    Toroid(ToroidSurface),
    Spherical(SphericalSurface),
    BentFlat(BentFlatSurface),
    Vfm(VfmSurface),
    Cylindrical(CylindricalSurface),
    ParaboloidLens(ParaboloidLensSurface),
    BlazedGrating(BlazedGrating),
    LaminarGrating(LaminarGrating),
    VlsGrating(VlsGrating),
}

macro_rules! dispatch_surface {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            SurfaceEnum::Flat(s) => s.$method($($arg),*),
            SurfaceEnum::Toroid(s) => s.$method($($arg),*),
            SurfaceEnum::Spherical(s) => s.$method($($arg),*),
            SurfaceEnum::BentFlat(s) => s.$method($($arg),*),
            SurfaceEnum::Vfm(s) => s.$method($($arg),*),
            SurfaceEnum::Cylindrical(s) => s.$method($($arg),*),
            SurfaceEnum::ParaboloidLens(s) => s.$method($($arg),*),
            SurfaceEnum::BlazedGrating(s) => s.$method($($arg),*),
            SurfaceEnum::LaminarGrating(s) => s.$method($($arg),*),
            SurfaceEnum::VlsGrating(s) => s.$method($($arg),*),
        }
    };
}

impl Surface for SurfaceEnum {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        dispatch_surface!(self, local_z, x, y)
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        dispatch_surface!(self, local_n, x, y)
    }

    fn local_z_distorted(&self, x: f64, y: f64) -> Option<f64> {
        dispatch_surface!(self, local_z_distorted, x, y)
    }

    fn local_g(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        dispatch_surface!(self, local_g, x, y)
    }

    fn local_n_bragg(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        dispatch_surface!(self, local_n_bragg, x, y)
    }
}

/// Enum wrapping all ParametricSurface-trait types for Python dispatch.
#[derive(Clone)]
pub enum ParametricSurfaceEnum {
    Elliptical(EllipticalSurface),
    Parabolical(ParabolicalSurface),
}

macro_rules! dispatch_parametric {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ParametricSurfaceEnum::Elliptical(s) => s.$method($($arg),*),
            ParametricSurfaceEnum::Parabolical(s) => s.$method($($arg),*),
        }
    };
}

impl ParametricSurface for ParametricSurfaceEnum {
    fn local_r(&self, s: f64, phi: f64) -> f64 {
        dispatch_parametric!(self, local_r, s, phi)
    }

    fn local_n(&self, s: f64, phi: f64) -> [f64; 3] {
        dispatch_parametric!(self, local_n, s, phi)
    }

    fn xyz_to_param(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        dispatch_parametric!(self, xyz_to_param, x, y, z)
    }

    fn param_to_xyz(&self, s: f64, phi: f64, r: f64) -> (f64, f64, f64) {
        dispatch_parametric!(self, param_to_xyz, s, phi, r)
    }

    fn local_r_distorted(&self, s: f64, phi: f64) -> Option<f64> {
        dispatch_parametric!(self, local_r_distorted, s, phi)
    }
}
