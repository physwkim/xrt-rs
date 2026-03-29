//! GPU undulator radiation dispatch.
//!
//! Sends observation points to the GPU, runs the WGSL compute shader
//! for electron trajectory integration, and reads back radiation amplitudes.

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;
use wgpu::util::DeviceExt;

use xrt_core::consts::{FINE_STR, SIE0};

use crate::context::GpuContext;

/// GPU-compatible observation point (16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuObsPoint {
    pub energy: f32,
    pub theta: f32,
    pub psi: f32,
    pub _pad: f32,
}

/// GPU-compatible result per observation point (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuUndulatorResult {
    pub ax_re: f32,
    pub ax_im: f32,
    pub az_re: f32,
    pub az_im: f32,
    pub intensity: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Uniform parameters for the undulator shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuUndulatorParams {
    n_points: u32,
    n_steps: u32,
    n_periods: u32,
    _pad0: u32,
    kx: f32,
    ky: f32,
    gamma: f32,
    gamma2: f32,
    period_m: f32,
    phase_rad: f32,
    amp2flux: f32,
    _pad1: f32,
}

/// Result of GPU undulator computation.
pub struct UndulatorGpuResult {
    /// Horizontal radiation amplitude (σ-polarization)
    pub amp_s: Vec<Complex64>,
    /// Vertical radiation amplitude (π-polarization)
    pub amp_p: Vec<Complex64>,
    /// Total intensity
    pub intensity: Vec<f64>,
}

/// Run undulator radiation calculation on GPU.
pub fn undulator_gpu(
    ctx: &GpuContext,
    obs_points: &[GpuObsPoint],
    kx: f64,
    ky: f64,
    period_mm: f64,
    n_periods: usize,
    gamma: f64,
    beam_current: f64,
    n_integration_steps: u32,
    phase_deg: f64,
) -> UndulatorGpuResult {
    pollster::block_on(undulator_gpu_async(
        ctx,
        obs_points,
        kx,
        ky,
        period_mm,
        n_periods,
        gamma,
        beam_current,
        n_integration_steps,
        phase_deg,
    ))
}

async fn undulator_gpu_async(
    ctx: &GpuContext,
    obs_points: &[GpuObsPoint],
    kx: f64,
    ky: f64,
    period_mm: f64,
    n_periods: usize,
    gamma: f64,
    beam_current: f64,
    n_integration_steps: u32,
    phase_deg: f64,
) -> UndulatorGpuResult {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let n_points = obs_points.len() as u32;
    let amp2flux = FINE_STR * beam_current / SIE0 * n_periods as f64;

    let params = GpuUndulatorParams {
        n_points,
        n_steps: n_integration_steps,
        n_periods: n_periods as u32,
        _pad0: 0,
        kx: kx as f32,
        ky: ky as f32,
        gamma: gamma as f32,
        gamma2: (gamma * gamma) as f32,
        period_m: (period_mm * 1e-3) as f32,
        phase_rad: phase_deg.to_radians() as f32,
        amp2flux: amp2flux as f32,
        _pad1: 0.0,
    };

    let obs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("undulator_obs"),
        contents: bytemuck::cast_slice(obs_points),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let result_size = (n_points as usize) * std::mem::size_of::<GpuUndulatorResult>();
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("undulator_results"),
        size: result_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("undulator_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("undulator_staging"),
        size: result_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader_source = include_str!("shaders/undulator.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("undulator"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("undulator_layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
        label: Some("undulator_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("undulator_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("undulator_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: obs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("undulator_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("undulator_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_points.div_ceil(64), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_size as u64);
    queue.submit(Some(encoder.finish()));

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
    let results: &[GpuUndulatorResult] = bytemuck::cast_slice(&data);

    let mut amp_s = Vec::with_capacity(results.len());
    let mut amp_p = Vec::with_capacity(results.len());
    let mut intensity = Vec::with_capacity(results.len());

    for r in results {
        amp_s.push(Complex64::new(r.ax_re as f64, r.ax_im as f64));
        amp_p.push(Complex64::new(r.az_re as f64, r.az_im as f64));
        intensity.push(r.intensity as f64);
    }

    UndulatorGpuResult {
        amp_s,
        amp_p,
        intensity,
    }
}
