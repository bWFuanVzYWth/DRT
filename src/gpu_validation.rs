//! Opt-in tests that execute the DRT pipelines on a real wgpu adapter.
use super::*;

struct TestGpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    bindings: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
}

impl TestGpu {
    fn new() -> Self {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("GPU tests require an available wgpu adapter");
        eprintln!("GPU validation: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let bindings = create_bind_group_layout(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DRT validation layout"),
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        Self {
            instance,
            adapter,
            device,
            queue,
            bindings,
            layout,
        }
    }

    fn render(
        &self,
        pipeline: &wgpu::ComputePipeline,
        parameters: Parameters,
        input: &[f32],
    ) -> Vec<f32> {
        let extent = wgpu::Extent3d {
            width: parameters.width,
            height: parameters.height,
            depth_or_array_layers: 1,
        };
        assert_eq!(input.len(), (extent.width * extent.height * 4) as usize);
        let texture = |usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("DRT validation texture"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage,
                view_formats: &[],
            })
        };
        let source = texture(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let output = texture(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let source_view = source.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let half_input: Vec<_> = input.iter().copied().map(half::f16::from_f32).collect();
        self.queue.write_texture(
            source.as_image_copy(),
            bytemuck::cast_slice(&half_input),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(extent.width * 8),
                rows_per_image: Some(extent.height),
            },
            extent,
        );
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DRT validation parameters"),
                contents: bytemuck::bytes_of(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DRT validation bindings"),
            layout: &self.bindings,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let row_bytes = (extent.width * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DRT validation readback"),
            size: u64::from(row_bytes * extent.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(extent.height),
                },
            },
            extent,
        );
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        bytes
            .chunks_exact(row_bytes as usize)
            .flat_map(|row| {
                row[..(extent.width * 8) as usize]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| half::f16::from_le_bytes([pair[0], pair[1]]).to_f32())
            })
            .collect()
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn drt_gpu_outputs() {
    let gpu = TestGpu::new();
    let image = crate::image_io::test_pattern(63, 9);
    let shaders = [
        BUILT_NONE_SHADER,
        BUILT_OKLAB_SHADER,
        BUILT_AGX_S2O3_SHADER,
        BUILT_AGX_HSV_SHADER,
        BUILT_REINHARD_GAMUT_SHADER,
    ];
    for (drt, shader) in DrtKind::ALL.into_iter().zip(shaders) {
        let pipeline = create_pipeline(&gpu.device, &gpu.layout, shader, drt.label()).unwrap();
        for variant in 0..6 {
            let mut parameters = Parameters::new(image.width, image.height);
            let headroom = [1.0, 4.0, 64.0][variant % 3];
            let mut reinhard = ReinhardParameters::default();
            let mut oklab = ReinhardParameters::oklab_default();
            if variant >= 3 {
                reinhard.compression_start = 0.3;
                reinhard.hue_retention = 0.25;
                reinhard.gamut_expansion = 0.2;
                reinhard.input_scale = 2.0;
                reinhard.highlight_reach_ev = 8.0;
                oklab.compression_start = 0.65;
                oklab.input_scale = 1.5;
                oklab.highlight_reach_ev = 12.0;
                parameters.exposure_multiplier = 4.0;
            }
            parameters.set_reinhard_for_drt(drt, oklab, reinhard, headroom);
            parameters.set_agx_for_headroom(
                if drt == DrtKind::AgxHsv {
                    AgxParameters::hsv_default()
                } else {
                    AgxParameters::default()
                },
                if drt == DrtKind::AgxHsv {
                    headroom
                } else {
                    1.0
                },
            );
            let pixels = gpu.render(&pipeline, parameters, &image.rgba);
            assert!(pixels.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!(pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 1.0));
            let peak = match drt {
                DrtKind::None | DrtKind::ReinhardGamut => extended_srgb_oetf(headroom),
                DrtKind::AgxHsv => parameters.agx_output_peak,
                _ => 1.0,
            };
            assert!(
                pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| p[..3].iter().all(|v| *v <= peak + 0.005))
            );
            // The first row is a neutral ramp. It must stay neutral and monotonic.
            let ramp = &pixels[..image.width as usize * 4];
            for pixel in ramp.as_chunks::<4>().0 {
                assert!((pixel[0] - pixel[1]).abs() < 0.003 * peak);
                assert!((pixel[1] - pixel[2]).abs() < 0.003 * peak);
            }
            if drt == DrtKind::Oklab {
                // Oklab always applies the independent piecewise curve in L^3.
                // Allow fp16 storage and the AP0/Rec.709 neutral-axis rounding.
                let curve = oklab.curve_for_headroom(1.0);
                for (pixel, input) in ramp
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(image.rgba.as_chunks::<4>().0)
                {
                    let value =
                        half::f16::from_f32(input[0]).to_f32() * parameters.exposure_multiplier;
                    let expected =
                        extended_srgb_oetf((curve.map_linear(value) * 0.99999).clamp(0.0, 1.0));
                    assert!(
                        (pixel[0] - expected).abs() < 0.003,
                        "Oklab curve: expected {expected}, got {}",
                        pixel[0]
                    );
                }
            }
            assert!(
                ramp.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| p[0])
                    .collect::<Vec<_>>()
                    .windows(2)
                    .all(|pair| pair[1] + 0.0001 >= pair[0])
            );
            if headroom > 1.0
                && matches!(
                    drt,
                    DrtKind::None | DrtKind::AgxHsv | DrtKind::ReinhardGamut
                )
            {
                assert!(
                    ramp.iter().any(|v| *v > 1.0),
                    "{} lost HDR headroom",
                    drt.label()
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn drt_gpu_diagnostics() {
    let gpu = TestGpu::new();
    let shaders = [
        BUILT_NONE_SHADER,
        BUILT_OKLAB_SHADER,
        BUILT_AGX_S2O3_SHADER,
        BUILT_AGX_HSV_SHADER,
        BUILT_REINHARD_GAMUT_SHADER,
    ];
    for (drt, shader) in DrtKind::ALL.into_iter().zip(shaders) {
        let pipeline = create_pipeline(&gpu.device, &gpu.layout, shader, drt.label()).unwrap();
        let mut pixels = Vec::new();
        for show_anomalies in [0, 1] {
            for rgb in [
                [0.0; 3],
                [-0.1; 3],
                [1000.0; 3],
                [f32::NAN, 0.0, 0.0],
                [f32::INFINITY, 0.0, 0.0],
                [f32::NEG_INFINITY, 0.0, 0.0],
                [f32::INFINITY, f32::NEG_INFINITY, 0.0],
            ] {
                let mut parameters = Parameters::new(1, 1);
                parameters.show_anomalies = show_anomalies;
                pixels.extend(gpu.render(&pipeline, parameters, &[rgb[0], rgb[1], rgb[2], 1.0]));
            }
        }
        let expected_special = [
            [[1.0, 0.0, 1.0], [1.0; 3], [0.0; 3], [1.0, 0.0, 1.0]],
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.35, 0.0],
            ],
        ];
        for (mode, expected) in expected_special.into_iter().enumerate() {
            for (index, color) in expected.into_iter().enumerate() {
                let offset = (mode * 7 + index + 3) * 4;
                assert_color(&pixels[offset..offset + 4], color);
            }
        }
        assert_color(&pixels[..4], [0.0; 3]);
        if drt == DrtKind::None {
            assert_color(&pixels[8 * 4..9 * 4], [0.0, 0.25, 1.0]);
            assert_color(&pixels[9 * 4..10 * 4], [1.0, 0.05, 0.0]);
        }
    }
}

fn assert_color(actual: &[f32], expected: [f32; 3]) {
    for (value, expected) in actual.iter().zip(expected) {
        assert!(
            (value - expected).abs() < 0.001,
            "expected {expected}, got {value}"
        );
    }
    assert_eq!(actual[3], 1.0);
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn drt_gpu_hot_reload_recovers() {
    let gpu = TestGpu::new();
    let target_format = wgpu::TextureFormat::Rgba16Float;
    let renderer = egui_wgpu::Renderer::new(&gpu.device, target_format, Default::default());
    let render_state = egui_wgpu::RenderState {
        adapter: gpu.adapter.clone(),
        available_adapters: vec![gpu.adapter.clone()],
        instance: gpu.instance.clone(),
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
        target_format,
        renderer: std::sync::Arc::new(egui::epaint::mutex::RwLock::new(renderer)),
        surface_config: egui_wgpu::SurfaceConfig::HIGH_THROUGHPUT,
    };
    let mut drt = DrtGpu::new(&render_state, crate::image_io::test_pattern(63, 9)).unwrap();
    // Exercise image + neutral-axis dispatch and shader-file lookup for every DRT.
    for kind in DrtKind::ALL {
        drt.set_drt(kind);
        drt.reload_shader(
            kind,
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join(kind.shader_file()),
        )
        .unwrap();
    }
    let input = [0.18, 0.18, 0.18, 1.0];
    let parameters = Parameters::new(1, 1);
    let original = gpu.render(&drt.pipelines.reinhard_gamut, parameters, &input);
    let file = std::env::temp_dir().join(format!("drt-reload-test-{}.wgsl", std::process::id()));
    std::fs::write(&file, "invalid WGSL").unwrap();
    let error = drt
        .reload_shader(DrtKind::ReinhardGamut, &file)
        .unwrap_err();
    assert!(format!("{error:#}").contains("parsing error"));
    assert_eq!(
        original,
        gpu.render(&drt.pipelines.reinhard_gamut, parameters, &input)
    );
    // A valid edit must actually replace the pipeline after a failed edit.
    std::fs::write(
        &file,
        r#"
        @group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            if (all(id.xy < textureDimensions(output))) {
                textureStore(output, vec2i(id.xy), vec4f(0.25, 0.5, 0.75, 1.0));
            }
        }
    "#,
    )
    .unwrap();
    drt.reload_shader(DrtKind::ReinhardGamut, &file).unwrap();
    assert_color(
        &gpu.render(&drt.pipelines.reinhard_gamut, parameters, &input),
        [0.25, 0.5, 0.75],
    );
    std::fs::remove_file(&file).unwrap();
    assert!(drt.reload_shader(DrtKind::ReinhardGamut, &file).is_err());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}
