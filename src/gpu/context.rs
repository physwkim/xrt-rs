//! GPU context: wgpu device and queue initialization.

use std::sync::{Arc, OnceLock};

/// GPU context holding the wgpu device and queue.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize a GPU context, requesting the best available adapter.
    ///
    /// Returns None if no suitable GPU is found.
    ///
    /// Enumerating the adapters and creating the device costs 200-400 ms, so
    /// anything that asks per batch rather than per run spends all its time
    /// here: use [`GpuContext::shared`] unless a private device is wanted.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    /// The process-wide context, initialized at most once.
    ///
    /// Also answers whether a GPU exists at all, which is what the `*_auto`
    /// entry points need, without paying for a device to find out twice.
    pub fn shared() -> Option<&'static Self> {
        static SHARED: OnceLock<Option<GpuContext>> = OnceLock::new();
        SHARED.get_or_init(Self::new).as_ref()
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("xrt-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Get the adapter name.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.adapter_info.name)
            .field("backend", &self.adapter_info.backend)
            .finish()
    }
}
