//! GPU Kirchhoff diffraction integral dispatch.
//!
//! Sends ray and pixel data to the GPU, runs the WGSL compute shader,
//! and reads back the diffraction results.
//!
//! Phase reduction for f32 precision:
//! The shader computes `delta_phase = k * (path - ref_path)` where ref_path
//! is the distance from the ray centroid to the pixel. The path difference
//! is computed algebraically (via path² - ref²) to avoid f32 cancellation.
//! The base phase `exp(i*k*ref_path)` is restored on the CPU in f64.

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
    // Ray centroid for phase reduction
    cx: f32,
    cy: f32,
    cz: f32,
    _pad2: u32,
}

/// Run the Kirchhoff diffraction integral on the GPU.
///
/// Uses phase reduction to handle arbitrary ray-to-pixel distances
/// without f32 precision loss. Returns complex field amplitudes (Es, Ep)
/// per pixel.
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

    // Compute ray centroid (f64 for precision)
    let n = rays.len() as f64;
    let cx_f64: f64 = rays.iter().map(|r| r.x as f64).sum::<f64>() / n;
    let cy_f64: f64 = rays.iter().map(|r| r.y as f64).sum::<f64>() / n;
    let cz_f64: f64 = rays.iter().map(|r| r.z as f64).sum::<f64>() / n;

    // Mean energy for base phase correction
    let mean_energy: f64 = rays.iter().map(|r| r.energy as f64).sum::<f64>() / n;
    let k_f64 = mean_energy / CHBAR * 1e7;

    let params = GpuParams {
        n_rays,
        n_pixels,
        chbar_inv_1e7: (1.0 / CHBAR * 1e7) as f32,
        _pad: 0,
        cx: cx_f64 as f32,
        cy: cy_f64 as f32,
        cz: cz_f64 as f32,
        _pad2: 0,
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

    // Create bind group layout (4 bindings: rays, pixels, results, params)
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

    // Apply base phase correction per pixel: multiply by exp(i * k * ref_path)
    // where ref_path is computed in f64 for precision
    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let pix = &pixels[i];
            let dx = pix.x as f64 - cx_f64;
            let dy = pix.y as f64 - cy_f64;
            let dz = pix.z as f64 - cz_f64;
            let ref_path = (dx * dx + dy * dy + dz * dz).sqrt();
            let base_phase = k_f64 * ref_path;

            let (sin_bp, cos_bp) = base_phase.sin_cos();
            let phase_factor = Complex64::new(cos_bp, sin_bp);

            let es_reduced = Complex64::new(r.es_re as f64, r.es_im as f64);
            let ep_reduced = Complex64::new(r.ep_re as f64, r.ep_im as f64);

            (es_reduced * phase_factor, ep_reduced * phase_factor)
        })
        .collect()
}

// Re-export wgpu util for buffer creation
use wgpu::util::DeviceExt;
