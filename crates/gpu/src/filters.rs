//! Native compute filters for bounded preview windows. Unsupported formats/devices fall back
//! to the CPU. Samples stay in their document space and original storage depth.
use crate::Unsupported;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

#[derive(Clone, Copy, Debug)]
pub enum Filter {
    Gaussian { radius: f32 },
    Box { radius: f32 },
    Motion { angle: f32, distance: f32 },
    Unsharp { amount: f32, radius: f32, threshold: f32 },
    HighPass { radius: f32 },
}

#[derive(Clone)]
pub struct FilterEngine {
    layout: wgpu::BindGroupLayout,
    convolution: wgpu::ComputePipeline,
    box_line: wgpu::ComputePipeline,
    motion: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

impl FilterEngine {
    pub fn new(device: &wgpu::Device) -> Result<Self, Unsupported> {
        let (engine, error) = crate::first_error(device, || {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("photocraft_filters"),
                source: wgpu::ShaderSource::Wgsl(include_str!("filters.wgsl").into()),
            });
            let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("photocraft_filters"),
                entries: &[
                    storage(0, true),
                    storage(1, false),
                    storage(2, true),
                    storage(3, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("photocraft_filters"),
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
            let convolution = pipeline("convolution");
            let box_line = pipeline("box_line");
            let motion = pipeline("motion");
            let finish = pipeline("finish");
            Self { layout, convolution, box_line, motion, finish }
        });
        if let Some(error) = error { Err(Unsupported(format!("GPU filters unavailable: {error}"))) } else { Ok(engine) }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn run_surface(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _source: &Surface,
        _filter: Filter,
        _area: Rect,
        _extent: Rect,
        _selection: Option<&Surface>,
    ) -> Result<Surface, Unsupported> {
        Err(Unsupported("GPU filter readback is native-only; use the CPU filter".into()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    pub fn run_surface(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Surface,
        filter: Filter,
        area: Rect,
        extent: Rect,
        selection: Option<&Surface>,
    ) -> Result<Surface, Unsupported> {
        use std::future::Future as _;
        let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let (result, error) = crate::first_error(device, || self.run_surface_inner(device, queue, source, filter, area, extent, selection));
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        if let std::task::Poll::Ready(Some(error)) = std::pin::pin!(scope.pop()).poll(&mut cx) {
            return Err(Unsupported(format!("GPU filter memory allocation failed: {error}")));
        }
        if let Some(error) = error {
            return Err(Unsupported(format!("GPU filter failed: {error}")));
        }
        result
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    fn run_surface_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Surface,
        filter: Filter,
        area: Rect,
        extent: Rect,
        selection: Option<&Surface>,
    ) -> Result<Surface, Unsupported> {
        use wgpu::util::DeviceExt;
        if std::env::var("PHOTOCRAFT_GPU_FILTERS").ok().as_deref() == Some("0") {
            return Err(Unsupported("GPU filters disabled".into()));
        }
        if matches!(filter, Filter::Motion { angle, distance } if angle.is_finite() && distance.is_finite() && distance.abs() < 0.5) {
            return Ok(source.clone());
        }
        let channels = source.channels();
        if channels == 0 || channels > 4 {
            return Err(Unsupported("GPU filters support up to four native channels".into()));
        }
        let area = area.intersect(&extent);
        if area.is_empty() {
            return Ok(source.clone());
        }
        let (radius, halo, kind, amount, threshold, delta, steps) = filter.config()?;
        let window = checked_inflate(area, halo)?;
        let inner = window.intersect(&extent);
        if inner.is_empty() {
            return Err(Unsupported("empty GPU filter source extent".into()));
        }
        let count = u64::from(window.width()).checked_mul(u64::from(window.height())).ok_or_else(|| Unsupported("GPU filter dimensions overflow".into()))?;
        let bytes = count.checked_mul(16).ok_or_else(|| Unsupported("GPU filter byte count overflow".into()))?;
        let samples_per_pixel = match filter {
            Filter::Gaussian { .. } if radius > 4.0 => 12,
            Filter::Motion { .. } => u64::from(steps).saturating_add(1).saturating_mul(4),
            Filter::Box { .. } => 2 * (2 * radius.round() as u64 + 1),
            _ => 2 * (2 * (3.0 * radius).ceil() as u64 + 1),
        };
        if count.saturating_mul(samples_per_pixel) > 256_000_000 {
            return Err(Unsupported("GPU filter dispatch exceeds the cooperative work budget".into()));
        }

        // Three working buffers plus one readback. Source expansion and readback decoding
        // also stay below this per-window cap; a wide halo uses the existing tiled CPU path.
        if bytes > 32 << 20 || bytes > u64::from(device.limits().max_storage_buffer_binding_size) || bytes > device.limits().max_buffer_size {
            return Err(Unsupported("GPU filter window exceeds the bounded buffer budget".into()));
        }
        let (width, height) = (window.width(), window.height());
        if width.div_ceil(8).max(height.div_ceil(8)) > device.limits().max_compute_workgroups_per_dimension {
            return Err(Unsupported("GPU filter dispatch exceeds device limits".into()));
        }
        let original_samples = source.read_region(inner);
        let alpha = source.format().alpha;
        let color_channels = channels - usize::from(alpha);
        let mut expanded = Vec::with_capacity(bytes as usize);
        let inner_width = inner.width() as usize;
        for y in window.y0..window.y1 {
            let sy = (y.clamp(inner.y0, inner.y1 - 1) - inner.y0) as usize;
            for x in window.x0..window.x1 {
                let sx = (x.clamp(inner.x0, inner.x1 - 1) - inner.x0) as usize;
                let offset = (sy * inner_width + sx) * channels;
                let pixel = original_samples.get(offset..offset + channels).ok_or_else(|| Unsupported("GPU filter source length mismatch".into()))?;
                let mut packed = [0.0f32; 4];
                for (out, value) in packed.iter_mut().zip(pixel.iter().take(color_channels)) {
                    *out = *value;
                }
                if alpha {
                    packed[3] = pixel.last().copied().unwrap_or(0.0);
                }
                for value in packed {
                    expanded.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let original = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pc_filter_original"),
            contents: &expanded,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffer = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_filter_work"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let work = [buffer(), buffer()];
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_filters") });
        let mut stage = 0usize;
        let mut run = |pipeline: &wgpu::ComputePipeline, axis: u32, radius: u32, weights: &[f32], lines: bool, final_pass: bool| {
            let input = if stage == 0 { &original } else { &work[(stage - 1) % 2] };
            let output = &work[stage % 2];
            let mut params = Vec::with_capacity(64);
            for value in [width, height, axis, radius] {
                params.extend_from_slice(&value.to_le_bytes());
            }
            for value in [amount, threshold, delta[0], delta[1]] {
                params.extend_from_slice(&value.to_le_bytes());
            }
            for value in [u32::from(alpha), u32::from(stage == 0), if final_pass { kind } else { 0 }, steps] {
                params.extend_from_slice(&value.to_le_bytes());
            }
            for value in [window.x0, window.y0, 0, 0] {
                params.extend_from_slice(&value.to_le_bytes());
            }
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pc_filter_params"),
                contents: &params,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let weights: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
            let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pc_filter_weights"),
                contents: &weights,
                usage: wgpu::BufferUsages::STORAGE,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pc_filters"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: original.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: weights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: uniform.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pc_filters"), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            if lines {
                pass.dispatch_workgroups(if axis == 0 { height } else { width }.div_ceil(64), 1, 1);
            } else {
                pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
            }
            stage += 1;
        };
        match filter {
            Filter::Motion { .. } => run(&self.motion, 0, 0, &[1.0], false, false),
            Filter::Gaussian { .. } if radius > 4.0 => {
                let widths = gaussian_boxes(radius);
                for axis in 0..2 {
                    for width in widths {
                        run(&self.box_line, axis, (width - 1) / 2, &[1.0], true, false);
                    }
                }
            }
            _ => {
                let weights = if matches!(filter, Filter::Box { .. }) {
                    let r = radius.max(0.0).round() as usize;
                    vec![1.0 / (2 * r + 1) as f32; 2 * r + 1]
                } else {
                    gaussian_kernel(radius)
                };
                for axis in 0..2 {
                    run(&self.convolution, axis, (weights.len() / 2) as u32, &weights, false, false);
                }
            }
        }
        run(&self.finish, 0, 0, &[1.0], false, true);
        drop(run);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_filter_readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&work[(stage - 1) % 2], 0, &readback, 0, bytes);
        let submission = queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(std::time::Duration::from_secs(30)) })
            .map_err(|error| Unsupported(format!("GPU filter wait failed: {error}")))?;
        receive
            .recv_timeout(std::time::Duration::from_secs(1))
            .map_err(|error| Unsupported(format!("GPU filter readback callback failed: {error}")))?
            .map_err(|error| Unsupported(format!("GPU filter map failed: {error}")))?;
        let data = readback.slice(..).get_mapped_range().map_err(|error| Unsupported(format!("GPU filter readback failed: {error}")))?;
        let mut result = Vec::with_capacity(area.width() as usize * area.height() as usize * channels);
        for y in area.y0..area.y1 {
            for x in area.x0..area.x1 {
                let offset = ((y - window.y0) as usize * width as usize + (x - window.x0) as usize) * 16;
                let pixel = data.get(offset..offset + 16).ok_or_else(|| Unsupported("GPU filter readback length mismatch".into()))?;
                let mask = selection.map(|s| s.pixel(x, y).first().copied().unwrap_or(0.0).clamp(0.0, 1.0)).unwrap_or(1.0);
                let original_offset = ((y - inner.y0) as usize * inner_width + (x - inner.x0) as usize) * channels;
                for channel in 0..channels {
                    let lane = if alpha && channel + 1 == channels { 3 } else { channel };
                    let bytes = pixel
                        .get(lane * 4..lane * 4 + 4)
                        .and_then(|b| <[u8; 4]>::try_from(b).ok())
                        .ok_or_else(|| Unsupported("GPU filter channel readback mismatch".into()))?;
                    let filtered = f32::from_le_bytes(bytes);
                    let old =
                        original_samples.get(original_offset + channel).copied().ok_or_else(|| Unsupported("GPU filter original channel mismatch".into()))?;
                    result.push(if mask >= 1.0 { filtered } else { old + (filtered - old) * mask });
                }
            }
        }
        drop(data);
        readback.unmap();
        let mut output = source.clone();
        output.write_region(area, &result);
        Ok(output)
    }
}

impl Filter {
    // (radius, halo, final kind, amount factor, threshold, motion vector, samples)
    fn config(self) -> Result<(f32, i32, u32, f32, f32, [f32; 2], u32), Unsupported> {
        let (radius, kind, amount, threshold) = match self {
            Self::Gaussian { radius } | Self::Box { radius } => (radius, 0, 0.0, 0.0),
            Self::HighPass { radius } => (radius, 2, 0.0, 0.0),
            Self::Unsharp { radius, amount, threshold } => (radius, 1, amount / 100.0, threshold / 255.0),
            Self::Motion { angle, distance } => {
                if !angle.is_finite() || !distance.is_finite() || distance.abs() > 2000.0 {
                    return Err(Unsupported("invalid GPU motion filter parameters".into()));
                }
                let (sin, cos) = angle.to_radians().sin_cos();
                let distance = distance.abs();
                return Ok((
                    0.0,
                    (distance / 2.0).ceil() as i32 + 2,
                    0,
                    0.0,
                    0.0,
                    [cos * distance, -sin * distance],
                    if distance < 0.5 { 0 } else { distance.ceil() as u32 },
                ));
            }
        };
        if !radius.is_finite() || !(0.0..=1000.0).contains(&radius) || !amount.is_finite() || !threshold.is_finite() {
            return Err(Unsupported("invalid GPU filter parameters".into()));
        }
        let halo = if matches!(self, Self::Box { .. }) { radius.ceil() as i32 + 1 } else { (3.0 * radius).ceil() as i32 + 1 };
        Ok((radius, halo, kind, amount, threshold, [0.0; 2], 0))
    }
}

fn checked_inflate(rect: Rect, halo: i32) -> Result<Rect, Unsupported> {
    let invalid = || Unsupported("GPU filter coordinate overflow".into());
    Ok(Rect::new(
        rect.x0.checked_sub(halo).ok_or_else(invalid)?,
        rect.y0.checked_sub(halo).ok_or_else(invalid)?,
        rect.x1.checked_add(halo).ok_or_else(invalid)?,
        rect.y1.checked_add(halo).ok_or_else(invalid)?,
    ))
}

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    if sigma < 0.05 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma).ceil() as i32;
    let mut weights: Vec<f32> = (-radius..=radius).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

fn gaussian_boxes(sigma: f32) -> [u32; 3] {
    let ideal = (4.0 * sigma * sigma + 1.0).sqrt();
    let mut low = ideal.floor() as i32;
    if low % 2 == 0 {
        low -= 1;
    }
    low = low.max(1);
    let count = ((12.0 * sigma * sigma - 3.0 * (low * low) as f32 - 12.0 * low as f32 - 9.0) / (-4.0 * low as f32 - 4.0)).round().clamp(0.0, 3.0) as usize;
    std::array::from_fn(|i| if i < count { low as u32 } else { low as u32 + 2 })
}
