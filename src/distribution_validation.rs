//! Regression tests that exercise the production distribution WGSL on a real GPU.
use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ProjectedPoint {
    position: [f32; 4],
    clip: [f32; 4],
}

struct ProjectionGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroupLayout,
}

impl ProjectionGpu {
    fn new() -> Self {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("distribution GPU tests require an available wgpu adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        // Append only a readback entry point: the coordinates and projection are
        // evaluated by the same functions used by the application's vertex shader.
        let source = format!(
            "{}\n{}",
            include_str!("../shaders/color_distribution.wgsl"),
            r#"
struct ProjectedPoint {
    position : vec4<f32>,
    clip : vec4<f32>,
}
@group(0) @binding(2) var<storage, read_write> projected_points : array<ProjectedPoint>;

@compute @workgroup_size(64)
fn validate_projection(@builtin(global_invocation_id) invocation : vec3<u32>) {
    let index = invocation.x;
    if (index >= parameters.image_size.x * parameters.image_size.y) {
        return;
    }
    let pixel = vec2<u32>(index % parameters.image_size.x, index / parameters.image_size.x);
    let position = distribution_position(textureLoad(output_image, pixel, 0).rgb);
    projected_points[index].position = vec4<f32>(position, 1.0);
    projected_points[index].clip = project(position);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HDR distribution projection validation shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let bindings = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HDR distribution validation bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            DistributionParameters,
                        >() as u64),
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
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HDR distribution validation layout"),
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HDR distribution projection validation"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("validate_projection"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            pipeline,
            bindings,
        }
    }

    fn project(
        &self,
        encoded: &[[f32; 4]],
        space: ColorSpace,
        headroom: f32,
        rotation: [f32; 2],
        canvas: egui::Vec2,
    ) -> Vec<ProjectedPoint> {
        let count = encoded.len() as u32;
        let extent = wgpu::Extent3d {
            width: count,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR distribution test colors"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(encoded),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(count * 16),
                rows_per_image: Some(1),
            },
            extent,
        );
        let view = texture.create_view(&Default::default());
        let parameters =
            DistributionParameters::new(rotation, [count, 1], space, false, headroom, canvas);
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("HDR distribution validation parameters"),
                contents: bytemuck::bytes_of(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output_size = encoded.len() as u64 * std::mem::size_of::<ProjectedPoint>() as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HDR distribution projected points"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HDR distribution projection readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HDR distribution validation bindings"),
            layout: &self.bindings,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        bytemuck::cast_slice::<u8, ProjectedPoint>(&bytes).to_vec()
    }
}

fn encode_channel(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn rgb_cube(headroom: f32) -> Vec<[f32; 4]> {
    let mut colors = Vec::with_capacity(11 * 11 * 11);
    for r in 0..=10 {
        for g in 0..=10 {
            for b in 0..=10 {
                colors.push([
                    encode_channel(r as f32 / 10.0 * headroom),
                    encode_channel(g as f32 / 10.0 * headroom),
                    encode_channel(b as f32 / 10.0 * headroom),
                    1.0,
                ]);
            }
        }
    }
    colors
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn hdr_distribution_stays_inside_canvas_at_every_rotation_and_aspect() {
    let gpu = ProjectionGpu::new();
    let rotations = [
        [DEFAULT_YAW, DEFAULT_PITCH],
        [0.0, 0.0],
        [std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4],
        [2.9, -1.45],
        [-4.9, 1.45],
    ];
    let canvases = [
        egui::vec2(900.0, 600.0),
        egui::vec2(200.0, 800.0),
        egui::vec2(1.0, 1200.0),
        egui::vec2(1200.0, 1.0),
    ];
    for headroom in [1.0, 4.0, 64.0] {
        let colors = rgb_cube(headroom);
        for space in [ColorSpace::Srgb, ColorSpace::Oklab] {
            for rotation in rotations {
                for canvas in canvases {
                    for (index, point) in gpu
                        .project(&colors, space, headroom, rotation, canvas)
                        .iter()
                        .enumerate()
                    {
                        let [x, y, z, w] = point.clip;
                        assert!(
                            point
                                .position
                                .iter()
                                .chain(&point.clip)
                                .all(|x| x.is_finite())
                                && w == 1.0
                                && (-1.0..=1.0).contains(&x)
                                && (-1.0..=1.0).contains(&y)
                                && (0.0..=1.0).contains(&z),
                            "{space:?}, headroom {headroom}, rotation {rotation:?}, canvas \
                             {canvas:?}, color {:?}: {point:?}",
                            colors[index],
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn hdr_distribution_preserves_highlights_and_oklab_shape() {
    let gpu = ProjectionGpu::new();
    let canvas = egui::vec2(600.0, 600.0);
    let rotation = [DEFAULT_YAW, DEFAULT_PITCH];
    let sdr = gpu.project(&rgb_cube(1.0), ColorSpace::Oklab, 1.0, rotation, canvas);
    for headroom in [4.0, 64.0] {
        let hdr = gpu.project(
            &rgb_cube(headroom),
            ColorSpace::Oklab,
            headroom,
            rotation,
            canvas,
        );
        for (sdr, hdr) in sdr.iter().zip(&hdr) {
            for (sdr, hdr) in sdr.position.iter().zip(hdr.position) {
                assert!(
                    (sdr - hdr).abs() < 0.000_02,
                    "proportional HDR Oklab colors changed normalized shape: {sdr} != {hdr}"
                );
            }
        }
        let peak = encode_channel(headroom);
        let colors = [
            [1.0, 1.0, 1.0, 1.0],
            [peak, peak, peak, 1.0],
            [encode_channel(headroom * 0.5); 4],
        ];
        for space in [ColorSpace::Srgb, ColorSpace::Oklab] {
            let points = gpu.project(&colors, space, headroom, rotation, canvas);
            let distance = |a: &ProjectedPoint, b: &ProjectedPoint| {
                a.position
                    .iter()
                    .zip(b.position)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            };
            assert!(distance(&points[0], &points[1]) > 0.5);
            assert!(distance(&points[2], &points[1]) > 0.1);
            if space == ColorSpace::Srgb {
                for coordinate in &points[1].position[..3] {
                    assert!((coordinate - 1.0).abs() < 0.000_002);
                }
            } else {
                assert!((points[1].position[1] - 1.0).abs() < 0.000_002);
            }
        }
    }
}
