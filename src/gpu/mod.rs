#![allow(clippy::too_many_arguments)]
//! xrt-gpu: GPU compute shaders for X-ray tracing.
//!
//! Uses wgpu for cross-platform GPU compute (Metal/Vulkan/DX12).
//! Priority kernels:
//! 1. Kirchhoff diffraction integral (biggest GPU speedup, O(N_pixel × N_ray))
//! 2. Undulator radiation (electron trajectory integration)

pub mod context;
pub mod fallback;
pub mod kirchhoff;
pub mod undulator;
