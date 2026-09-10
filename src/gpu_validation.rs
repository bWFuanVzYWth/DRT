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
    let shaders: [&str; DrtKind::ALL.len()] = [
        BUILT_NONE_SHADER,
        BUILT_OKLAB_REINHARD_SHADER,
        BUILT_OKLAB_LOG_SHOULDER_SHADER,
        BUILT_AGX_S2O3_SHADER,
        BUILT_RGB_LOG_SIGMOID_SHADER,
        BUILT_RGB_REINHARD_SHADER,
        BUILT_RGB_LOG_SHOULDER_SHADER,
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
            parameters.set_log_sigmoid_for_headroom(
                if drt == DrtKind::RgbLogSigmoid {
                    RgbLogSigmoidParameters::default().tone_scale()
                } else {
                    LogSigmoidParameters::default()
                },
                if drt == DrtKind::RgbLogSigmoid {
                    headroom
                } else {
                    1.0
                },
            );
            let log_shoulder = if variant >= 3 {
                LogShoulderParameters {
                    base: if drt.is_oklab() { oklab } else { reinhard },
                    shoulder_power: 1.5,
                }
            } else if drt.is_oklab() {
                LogShoulderParameters::oklab_default()
            } else {
                LogShoulderParameters::default()
            };
            if drt.uses_log_shoulder() {
                parameters.set_log_shoulder_for_headroom(
                    log_shoulder,
                    direct_output_headroom(drt, headroom),
                );
            }
            let pixels = gpu.render(&pipeline, parameters, &image.rgba);
            assert!(pixels.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!(pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 1.0));
            let peak = match drt {
                DrtKind::None | DrtKind::RgbReinhard | DrtKind::RgbLogShoulder => {
                    extended_srgb_oetf(headroom)
                }
                DrtKind::RgbLogSigmoid => parameters.log_sigmoid_output_peak,
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
            if drt.is_oklab() {
                // Both Oklab transforms apply their independent curve in L^3.
                // Allow fp16 storage and the AP0/Rec.709 neutral-axis rounding.
                let curve = oklab.curve_for_headroom(1.0);
                let log_curve = log_shoulder.curve_for_headroom(1.0);
                for (pixel, input) in ramp
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(image.rgba.as_chunks::<4>().0)
                {
                    let value =
                        half::f16::from_f32(input[0]).to_f32() * parameters.exposure_multiplier;
                    let mapped = if drt.uses_log_shoulder() {
                        log_curve.map_linear(value)
                    } else {
                        curve.map_linear(value)
                    };
                    let expected = extended_srgb_oetf((mapped * 0.99999).clamp(0.0, 1.0));
                    assert!(
                        (pixel[0] - expected).abs() < 0.003,
                        "{} curve: expected {expected}, got {}",
                        drt.label(),
                        pixel[0]
                    );
                }
            }
            if drt == DrtKind::RgbLogShoulder {
                let curve = log_shoulder.curve_for_headroom(headroom);
                for (pixel, input) in ramp
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(image.rgba.as_chunks::<4>().0)
                {
                    let value =
                        half::f16::from_f32(input[0]).to_f32() * parameters.exposure_multiplier;
                    let expected = extended_srgb_oetf(curve.map_linear(value).clamp(0.0, headroom));
                    assert!(
                        (pixel[0] - expected).abs() < 0.003 * peak,
                        "RGB Log Shoulder: expected {expected}, got {}",
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
                    DrtKind::None
                        | DrtKind::RgbLogSigmoid
                        | DrtKind::RgbReinhard
                        | DrtKind::RgbLogShoulder
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
    let shaders: [&str; DrtKind::ALL.len()] = [
        BUILT_NONE_SHADER,
        BUILT_OKLAB_REINHARD_SHADER,
        BUILT_OKLAB_LOG_SHOULDER_SHADER,
        BUILT_AGX_S2O3_SHADER,
        BUILT_RGB_LOG_SIGMOID_SHADER,
        BUILT_RGB_REINHARD_SHADER,
        BUILT_RGB_LOG_SHOULDER_SHADER,
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
                if drt.uses_log_shoulder() {
                    parameters.set_log_shoulder_for_headroom(
                        if drt.is_oklab() {
                            LogShoulderParameters::oklab_default()
                        } else {
                            LogShoulderParameters::default()
                        },
                        1.0,
                    );
                }
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

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn rgb_log_sigmoid_analytic_shadows_and_original_shoulder() {
    let gpu = TestGpu::new();
    let pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        BUILT_RGB_LOG_SIGMOID_SHADER,
        "analytic log curve",
    )
    .unwrap();
    // Probe the production scalar curve separately from the color transforms.
    let scalar_shader = BUILT_RGB_LOG_SIGMOID_SHADER.replace(
        "let mapped: vec3f = adjustHsv(originalLinear, rgbLogSigmoid(originalLinear));",
        "let mapped: vec3f = logSigmoidCurve(ap0);",
    );
    assert_ne!(scalar_shader, BUILT_RGB_LOG_SIGMOID_SHADER);
    let scalar_pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &scalar_shader,
        "scalar log curve probe",
    )
    .unwrap();
    let mut ramp = vec![0.0];
    ramp.extend((0..=256).map(|i| 0.18 * 2.0_f32.powf(-20.0 + i as f32 * 20.0 / 256.0)));
    // Include both sides of the join and a dense highlight ramp.
    ramp.extend([0.1798, 0.18, 0.1802]);
    ramp.extend((1..=256).map(|i| 0.18 * 2.0_f32.powf(i as f32 * 18.0 / 256.0)));
    ramp.sort_by(f32::total_cmp);
    let input: Vec<_> = ramp.iter().flat_map(|&x| [x, x, x, 1.0]).collect();
    for linear_slope in [0.1, 1.0, 4.0] {
        for headroom in [1.0, 4.0, 64.0] {
            let source = RgbLogSigmoidParameters {
                linear_slope,
                ..RgbLogSigmoidParameters::default()
            };
            let mut parameters = Parameters::new(ramp.len() as u32, 1);
            parameters.set_log_sigmoid_for_headroom(source.tone_scale(), headroom);
            let scalar = gpu.render(&scalar_pipeline, parameters, &input);
            let output = gpu.render(&pipeline, parameters, &input);
            let mut original = parameters;
            original.set_log_sigmoid_for_headroom(
                LogSigmoidParameters::original_rgb_reference(),
                headroom,
            );
            let mut previous = 0.0;
            for ((&x, pixel), raw) in ramp
                .iter()
                .zip(output.as_chunks::<4>().0)
                .zip(scalar.as_chunks::<4>().0)
            {
                let x = half::f16::from_f32(x).to_f32();
                assert!(pixel.iter().all(|v| v.is_finite()));
                assert!(
                    pixel[0] >= previous,
                    "non-monotonic at {x}: {previous} -> {}",
                    pixel[0]
                );
                previous = pixel[0];
                if x <= 0.18 {
                    let expected = extended_srgb_oetf(linear_slope * x);
                    // Input/output storage is fp16; the AP0 matrix is approximately neutral.
                    for value in [&pixel[..3], &raw[..3]].into_iter().flatten() {
                        assert!(
                            (value - expected).abs() <= expected * 0.002 + 1.0e-7,
                            "linear gain {linear_slope}, headroom {headroom}, x {x}: expected {expected}, got {value}"
                        );
                    }
                    if x == 0.0 {
                        assert_color(pixel, [0.0; 3]);
                    } else {
                        assert!(pixel[0] > 0.0, "finite black floor at {x}");
                    }
                } else if linear_slope == 1.0 {
                    // Compare the new shader shoulder with the previous default tone scale.
                    let distance = ((x.log2() - original.log_sigmoid_minimum_log2)
                        * original.log_sigmoid_inverse_dynamic_range)
                        .min(original.log_sigmoid_maximum_log_coordinate)
                        - original.log_sigmoid_input_pivot;
                    let expected = original.log_sigmoid_output_pivot
                        + original.log_sigmoid_pivot_slope
                            * distance
                            * (1.0
                                + original.sigmoid_shoulder_coefficient
                                    * distance.powf(original.sigmoid_shoulder_power))
                            .powf(-1.0 / original.sigmoid_shoulder_power);
                    assert!(
                        (raw[0] - expected).abs() <= expected * 0.001 + 1.0e-6,
                        "original shoulder, headroom {headroom}, x {x}: expected {expected}, got {}",
                        raw[0]
                    );
                }
            }
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
    // Other DRT controls must not overwrite the log_shoulder's curve or gamut/hue settings.
    let log_shoulder = LogShoulderParameters {
        base: ReinhardParameters {
            compression_start: 0.3,
            hue_retention: 0.2,
            ..ReinhardParameters::default()
        },
        shoulder_power: 2.5,
    };
    drt.set_rgb_log_shoulder_parameters(log_shoulder);
    drt.set_drt(DrtKind::RgbLogShoulder);
    let log_shoulder_uniform = drt.parameters;
    drt.set_rgb_log_sigmoid_parameters(RgbLogSigmoidParameters::default());
    drt.set_rgb_reinhard_parameters(ReinhardParameters {
        compression_start: 0.7,
        ..ReinhardParameters::default()
    });
    drt.set_oklab_reinhard_parameters(ReinhardParameters::oklab_default());
    let oklab_log_shoulder = LogShoulderParameters {
        base: ReinhardParameters {
            compression_start: 0.4,
            input_scale: 1.5,
            highlight_reach_ev: 9.0,
            ..ReinhardParameters::oklab_default()
        },
        shoulder_power: 3.0,
    };
    drt.set_oklab_log_shoulder_parameters(oklab_log_shoulder);
    assert_eq!(
        bytemuck::bytes_of(&log_shoulder_uniform),
        bytemuck::bytes_of(&drt.parameters)
    );
    drt.set_drt(DrtKind::RgbLogSigmoid);
    drt.set_hdr_headroom(4.0);
    drt.set_drt(DrtKind::RgbLogShoulder);
    let expected = log_shoulder.curve_for_headroom(4.0);
    assert_eq!(
        drt.parameters.sigmoid_shoulder_coefficient,
        expected.shoulder_coefficient
    );
    assert_eq!(
        drt.parameters.sigmoid_shoulder_power,
        log_shoulder.shoulder_power
    );
    assert_eq!(
        drt.parameters.linear_compression_start,
        log_shoulder.base.compression_start
    );
    assert_eq!(
        drt.parameters.rgb_hue_retention,
        log_shoulder.base.hue_retention
    );

    // Oklab Log Shoulder keeps its own controls and SDR boundary, even when
    // other transforms or the display's HDR target change.
    drt.set_drt(DrtKind::OklabLogShoulder);
    let oklab_uniform = drt.parameters;
    let image = crate::image_io::test_pattern(drt.width(), drt.height());
    let oklab_output = gpu.render(
        &drt.pipelines.oklab_log_shoulder,
        oklab_uniform,
        &image.rgba,
    );
    drt.set_rgb_log_shoulder_parameters(LogShoulderParameters::default());
    drt.set_oklab_reinhard_parameters(ReinhardParameters {
        compression_start: 0.7,
        ..ReinhardParameters::oklab_default()
    });
    drt.set_agx_s2o3_parameters(LogSigmoidParameters::s2o3_reference());
    assert_eq!(
        bytemuck::bytes_of(&oklab_uniform),
        bytemuck::bytes_of(&drt.parameters)
    );
    for headroom in [1.0, 4.0, 64.0] {
        for kind in DrtKind::ALL {
            drt.set_drt(kind);
            drt.set_hdr_headroom(headroom);
        }
        drt.set_drt(DrtKind::OklabLogShoulder);
        assert_eq!(drt.parameters.linear_output_peak, 1.0);
        assert_eq!(drt.oklab_log_shoulder_parameters, oklab_log_shoulder);
        assert_eq!(
            gpu.render(
                &drt.pipelines.oklab_log_shoulder,
                drt.parameters,
                &image.rgba
            ),
            oklab_output
        );
    }
    drt.set_hdr_headroom(1.0);
    drt.set_drt(DrtKind::RgbReinhard);
    let input = [0.18, 0.18, 0.18, 1.0];
    let parameters = Parameters::new(1, 1);
    let original = gpu.render(&drt.pipelines.rgb_reinhard, parameters, &input);
    let file = std::env::temp_dir().join(format!("drt-reload-test-{}.wgsl", std::process::id()));
    std::fs::write(&file, "invalid WGSL").unwrap();
    let error = drt.reload_shader(DrtKind::RgbReinhard, &file).unwrap_err();
    assert!(format!("{error:#}").contains("parsing error"));
    assert_eq!(
        original,
        gpu.render(&drt.pipelines.rgb_reinhard, parameters, &input)
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
    drt.reload_shader(DrtKind::RgbReinhard, &file).unwrap();
    assert_color(
        &gpu.render(&drt.pipelines.rgb_reinhard, parameters, &input),
        [0.25, 0.5, 0.75],
    );
    std::fs::remove_file(&file).unwrap();
    assert!(drt.reload_shader(DrtKind::RgbReinhard, &file).is_err());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}
