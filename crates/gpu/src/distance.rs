//! Opt-in approximate Euclidean distance maps. Native interactive rendering only.
//! Seeds follow compose::effects; jump flooding approximates only the nearest-seed search.
use std::future::Future;

use photocraft_compose::effects::FieldKind;
use photocraft_geom::Rect;
use wgpu::util::DeviceExt;

const FRAME_BYTES: u64 = 96 * 1024 * 1024;
const FRAME_WORK: u64 = 250_000_000;

pub(crate) struct Engine {
    layout: wgpu::BindGroupLayout,
    init: wgpu::ComputePipeline,
    jump: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    frame: Option<u64>,
    bytes: u64,
    work: u64,
}

struct Size {
    width: u32,
    height: u32,
    stride: u32,
    seed_bytes: u64,
    output_bytes: u64,
    steps: Vec<u32>,
    work: u64,
}

fn size(device: &wgpu::Device, region: Rect, reach: i32) -> Option<Size> {
    let (width, height) = (region.width(), region.height());
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    if reach < 64 || width == 0 || height == 0 || width.max(height) > 4096 || !(65_536..=2_097_152).contains(&pixels) {
        return None;
    }
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage < 3
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 8
        || limits.max_compute_workgroup_size_y < 8
        || width.div_ceil(8) > limits.max_compute_workgroups_per_dimension
        || height.div_ceil(8) > limits.max_compute_workgroups_per_dimension
    {
        return None;
    }
    let stride = width.checked_add(63)? / 64 * 64;
    let seed_bytes = pixels.checked_mul(8)?;
    let output_bytes = u64::from(stride).checked_mul(u64::from(height))?.checked_mul(4)?;
    if seed_bytes.max(output_bytes) > u64::from(limits.max_storage_buffer_binding_size) || seed_bytes.max(output_bytes) > limits.max_buffer_size {
        return None;
    }
    let mut steps = Vec::new();
    let mut step = width.max(height).next_power_of_two() / 2;
    while step > 0 {
        steps.push(step);
        step /= 2;
    }
    // JFA+1 reduces, but does not eliminate, nearest-seed errors.
    steps.push(1);
    let work = pixels.checked_mul((steps.len() as u64).checked_mul(9)?.checked_add(2)?)?;
    if work > FRAME_WORK || seed_bytes.checked_mul(2)?.checked_add(output_bytes)? > FRAME_BYTES {
        return None;
    }
    Some(Size { width, height, stride, seed_bytes, output_bytes, steps, work })
}

fn supported_kind(kind: FieldKind) -> bool {
    matches!(kind, FieldKind::Outside | FieldKind::Inside | FieldKind::OutsideInverse)
}

/// Unlike a shared render encoder, a rejected compute encoder can be dropped safely.
/// Never block the render thread waiting for validation or a device readback.
fn guarded<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> Result<T, String> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out = f();
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut error = None;
    for pop in [internal.pop(), validation.pop(), oom.pop()] {
        match std::pin::pin!(pop).poll(&mut cx) {
            std::task::Poll::Ready(None) => {}
            std::task::Poll::Ready(Some(e)) => {
                error.get_or_insert_with(|| e.to_string());
            }
            std::task::Poll::Pending => {
                error.get_or_insert_with(|| "distance validation is asynchronous".to_owned());
            }
        }
    }
    match error {
        Some(error) => Err(error),
        None => Ok(out),
    }
}

impl Engine {
    pub(crate) fn opt_in(device: &wgpu::Device) -> Option<Self> {
        if std::env::var("PHOTOCRAFT_GPU_DISTANCE").ok().as_deref() != Some("1") {
            return None;
        }
        let result = guarded(device, || {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pc_distance"),
                source: wgpu::ShaderSource::Wgsl(include_str!("distance.wgsl").into()),
            });
            let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pc_distance"),
                entries: &[
                    storage(0, true),
                    storage(1, false),
                    storage(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pc_distance"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = |entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let init = pipeline("seed");
            let jump = pipeline("jump");
            let finish = pipeline("finish");
            Self { layout, init, jump, finish, frame: None, bytes: 0, work: 0 }
        });
        match result {
            Ok(engine) => Some(engine),
            Err(error) => {
                log::warn!("GPU distance unavailable; using exact CPU fields: {error}");
                None
            }
        }
    }

    pub(crate) fn supports(&self, device: &wgpu::Device, kind: FieldKind, reach: i32, region: Rect) -> bool {
        supported_kind(kind) && size(device, region, reach).is_some()
    }

    /// Entire region is rebuilt; partial CPU damage uploads must not overwrite this map later.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: u64,
        kind: FieldKind,
        reach: i32,
        region: Rect,
        shape: &super::Tex,
        target: &super::Tex,
    ) -> bool {
        if !supported_kind(kind) {
            return false;
        }
        let Some(s) = size(device, region, reach) else {
            return false;
        };
        if self.frame != Some(frame) {
            self.frame = Some(frame);
            self.bytes = 0;
            self.work = 0;
        }
        let bytes = s.seed_bytes * 2 + s.output_bytes;
        if bytes > FRAME_BYTES.saturating_sub(self.bytes) || s.work > FRAME_WORK.saturating_sub(self.work) {
            return false;
        }
        // Reserve even on failure: do not repeatedly allocate a failing workload this frame.
        self.bytes += bytes;
        self.work += s.work;
        let result = guarded(device, || {
            let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
            let a = buffer("pc_distance_a", s.seed_bytes, wgpu::BufferUsages::STORAGE);
            let b = buffer("pc_distance_b", s.seed_bytes, wgpu::BufferUsages::STORAGE);
            let output = buffer("pc_distance_output", s.output_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_distance") });
            let kind = match kind {
                FieldKind::Inside => 1,
                FieldKind::OutsideInverse => 2,
                _ => 0,
            };
            let dispatch = |encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, source: &wgpu::Buffer, dest: &wgpu::Buffer, step| {
                let params: Vec<u8> = [s.width, s.height, s.stride, step, kind, 0, 0, 0].into_iter().flat_map(u32::to_le_bytes).collect();
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pc_distance_params"),
                    contents: &params,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pc_distance"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: source.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: dest.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&shape.view) },
                        wgpu::BindGroupEntry { binding: 4, resource: uniform.as_entire_binding() },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pc_distance"), timestamp_writes: None });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(s.width.div_ceil(8), s.height.div_ceil(8), 1);
            };
            dispatch(&mut encoder, &self.init, &b, &a, 0);
            let (mut source, mut dest) = (&a, &b);
            for &step in &s.steps {
                dispatch(&mut encoder, &self.jump, source, dest, step);
                std::mem::swap(&mut source, &mut dest);
            }
            dispatch(&mut encoder, &self.finish, source, dest, 0);
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &output,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(s.stride * 4), rows_per_image: Some(s.height) },
                },
                wgpu::TexelCopyTextureInfo { texture: &target.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: s.width, height: s.height, depth_or_array_layers: 1 },
            );
            encoder.finish()
        });
        match result {
            Ok(commands) => {
                queue.submit([commands]);
                true
            }
            Err(error) => {
                log::warn!("GPU distance rejected; using exact CPU field: {error}");
                false
            }
        }
    }
}
