//! GPU Kirchhoff diffraction integral dispatch.
//!
//! Sends ray and pixel data to the GPU, runs the WGSL compute shader,
//! and reads back the diffraction results.

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;

use xrt_core::consts::CHBAR;

use crate::context::GpuContext;

/// GPU-compatible ray data (f32, 48 bytes aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRay {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub nl: f32,
    pub energy: f32,
    pub es_re: f32,
    pub es_im: f32,
    pub ep_re: f32,
    pub ep_im: f32,
}

/// GPU-compatible pixel position (16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPixel {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

/// GPU-compatible result (16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPixelResult {
    pub es_re: f32,
    pub es_im: f32,
    pub ep_re: f32,
    pub ep_im: f32,
}

/// Uniform parameters for the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    n_rays: u32,
    n_pixels: u32,
    chbar_inv_1e7: f32,
    _pad: u32,
}

/// Run the Kirchhoff diffraction integral on the GPU.
///
/// Returns complex field amplitudes (Es, Ep) per pixel.
pub fn kirchhoff_gpu(
    ctx: &GpuContext,
    rays: &[GpuRay],
    pixels: &[GpuPixel],
) -> Vec<(Complex64, Complex64)> {
    pollster::block_on(kirchhoff_gpu_async(ctx, rays, pixels))
}

async fn kirchhoff_gpu_async(
    ctx: &GpuContext,
    rays: &[GpuRay],
    pixels: &[GpuPixel],
) -> Vec<(Complex64, Complex64)> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let n_rays = rays.len() as u32;
    let n_pixels = pixels.len() as u32;

    let params = GpuParams {
        n_rays,
        n_pixels,
        chbar_inv_1e7: (1.0 / CHBAR * 1e7) as f32,
        _pad: 0,
    };

    // Create buffers
    let ray_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rays"),
        contents: bytemuck::cast_slice(rays),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let pixel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pixels"),
        contents: bytemuck::cast_slice(pixels),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let result_size = (n_pixels as usize) * std::mem::size_of::<GpuPixelResult>();
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results"),
        size: result_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: result_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load shader
    let shader_source = include_str!("shaders/kirchhoff.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kirchhoff"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout and pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("kirchhoff_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("kirchhoff_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("kirchhoff_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kirchhoff_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ray_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pixel_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("kirchhoff_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kirchhoff_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_pixels.div_ceil(64), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_size as u64);
    queue.submit(Some(encoder.finish()));

    // Read back results
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .expect("GPU buffer map channel closed")
        .expect("GPU buffer map failed");

    let data = buffer_slice.get_mapped_range();
    let results: &[GpuPixelResult] = bytemuck::cast_slice(&data);

    results
        .iter()
        .map(|r| {
            (
                Complex64::new(r.es_re as f64, r.es_im as f64),
                Complex64::new(r.ep_re as f64, r.ep_im as f64),
            )
        })
        .collect()
}

// Re-export wgpu util for buffer creation
use wgpu::util::DeviceExt;
