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
        BUILT_AGX_S2O3_SHADER,
        BUILT_OKLAB_LOG_SIGMOID_SHADER,
        BUILT_OKLAB_REINHARD_SHADER,
        BUILT_RGB_LOG_SIGMOID_SHADER,
        BUILT_RGB_REINHARD_SHADER,
    ];
    for (drt, shader) in DrtKind::ALL.into_iter().zip(shaders) {
        let pipeline = create_pipeline(&gpu.device, &gpu.layout, shader, drt.label()).unwrap();
        for variant in 0..6 {
            let mut parameters = Parameters::new(image.width, image.height);
            let headroom = [1.0, 4.0, 64.0][variant % 3];
            let mut reinhard = ReinhardParameters::default();
            let mut oklab = ReinhardParameters::oklab_default();
            let mut sigmoid = LinearLogSigmoidParameters::default();
            if variant >= 3 {
                reinhard.hue_retention = 0.25;
                reinhard.linear_slope = 2.0;
                reinhard.compression_start = 0.3;
                reinhard.gamut_compression = 0.2;
                reinhard.highlight_reach_ev = 8.0;
                oklab.linear_slope = 1.5;
                oklab.compression_start = 0.5;
                oklab.highlight_reach_ev = 12.0;
                sigmoid.compression_start = 0.5;
                sigmoid.gamut_compression = 0.2;
                parameters.exposure_multiplier = 4.0;
            }
            parameters.set_reinhard_for_drt(drt, oklab, reinhard, headroom);
            if drt.uses_linear_log_sigmoid() {
                parameters.set_linear_log_sigmoid_for_headroom(
                    sigmoid,
                    if drt == DrtKind::RgbLogSigmoid {
                        headroom
                    } else {
                        1.0
                    },
                );
            } else {
                parameters.set_log_sigmoid(LogSigmoidParameters::default());
            }
            let pixels = gpu.render(&pipeline, parameters, &image.rgba);
            assert!(pixels.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!(pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 1.0));
            let peak = match drt {
                DrtKind::None | DrtKind::RgbReinhard => extended_srgb_oetf(headroom),
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
            if drt == DrtKind::OklabReinhard {
                // Oklab Reinhard applies its independent curve in L^3.
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
                    let mapped = curve.map_linear(value);
                    let expected = extended_srgb_oetf((mapped * 0.99999).clamp(0.0, 1.0));
                    assert!(
                        (pixel[0] - expected).abs() < 0.003,
                        "{} curve: expected {expected}, got {}",
                        drt.label(),
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
                    DrtKind::None | DrtKind::RgbLogSigmoid | DrtKind::RgbReinhard
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
        BUILT_AGX_S2O3_SHADER,
        BUILT_OKLAB_LOG_SIGMOID_SHADER,
        BUILT_OKLAB_REINHARD_SHADER,
        BUILT_RGB_LOG_SIGMOID_SHADER,
        BUILT_RGB_REINHARD_SHADER,
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
                if drt.uses_linear_log_sigmoid() {
                    parameters.set_linear_log_sigmoid_for_headroom(
                        LinearLogSigmoidParameters::default(),
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
            let source = LinearLogSigmoidParameters {
                linear_slope,
                ..LinearLogSigmoidParameters::default()
            };
            let mut parameters = Parameters::new(ramp.len() as u32, 1);
            parameters.set_linear_log_sigmoid_for_headroom(source, headroom);
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

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_log_sigmoid_matches_the_rgb_neutral_curve() {
    let gpu = TestGpu::new();
    let oklab = create_pipeline(
        &gpu.device,
        &gpu.layout,
        BUILT_OKLAB_LOG_SIGMOID_SHADER,
        "Oklab log sigmoid",
    )
    .unwrap();
    let rgb = create_pipeline(
        &gpu.device,
        &gpu.layout,
        BUILT_RGB_LOG_SIGMOID_SHADER,
        "RGB log sigmoid",
    )
    .unwrap();
    let mut ramp = vec![0.0];
    ramp.extend((0..=512).map(|i| 0.18 * 2.0_f32.powf(-20.0 + i as f32 * 38.0 / 512.0)));
    ramp.extend([0.1798, 0.18, 0.1802]);
    ramp.sort_by(f32::total_cmp);
    let input: Vec<_> = ramp.iter().flat_map(|&x| [x, x, x, 1.0]).collect();
    for linear_slope in [0.1, 1.0, 4.0] {
        for compression_start in [0.0, 0.18, 10.0] {
            for (highlight_ev, shoulder_power) in [(1.0, 1.0), (6.5, 5.2), (20.0, 8.0)] {
                let mut source = LinearLogSigmoidParameters {
                    linear_slope,
                    compression_start,
                    highlight_ev,
                    shoulder_power,
                    ..LinearLogSigmoidParameters::default()
                };
                source.constrain();
                let mut parameters = Parameters::new(ramp.len() as u32, 1);
                parameters.set_linear_log_sigmoid_for_headroom(source, 1.0);
                let output = gpu.render(&oklab, parameters, &input);
                let reference = gpu.render(&rgb, parameters, &input);
                let mut previous = 0.0;
                for ((&x, pixel), expected) in ramp
                    .iter()
                    .zip(output.as_chunks::<4>().0)
                    .zip(reference.as_chunks::<4>().0)
                {
                    assert!(
                        pixel
                            .iter()
                            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
                    );
                    assert!(
                        pixel[0] >= previous,
                        "non-monotonic at {x}: {previous} -> {}",
                        pixel[0]
                    );
                    previous = pixel[0];
                    for channel in 0..3 {
                        // Account for fp16 storage and the existing 0.99999 gamut margin.
                        assert!(
                            (pixel[channel] - expected[channel]).abs()
                                < expected[channel] * 0.003 + 2.0e-7,
                            "{source:?}, x {x}: Oklab {pixel:?}, RGB {expected:?}"
                        );
                    }
                    if x == 0.0 {
                        assert_eq!(&pixel[..3], &[0.0; 3]);
                    } else if x < source.compression_start {
                        assert!(pixel[0] > 0.0, "finite black floor at {x}");
                        let linear =
                            extended_srgb_oetf(linear_slope * half::f16::from_f32(x).to_f32());
                        assert!(
                            (pixel[0] - linear).abs() < linear * 0.003 + 2.0e-7,
                            "{source:?}, x {x}: expected linear {linear}, got {pixel:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn rgb_gamut_controls_change_colors_and_preserve_neutrals() {
    let gpu = TestGpu::new();
    let image = crate::image_io::test_pattern(63, 9);
    for (name, shader) in [
        ("RGB Reinhard", BUILT_RGB_REINHARD_SHADER),
        ("RGB Log Sigmoid", BUILT_RGB_LOG_SIGMOID_SHADER),
    ] {
        let pipeline = create_pipeline(&gpu.device, &gpu.layout, shader, name).unwrap();
        for headroom in [1.0, 64.0] {
            let outputs: Vec<_> = [0.0, 0.04, 0.8]
                .into_iter()
                .map(|gamut_compression| {
                    let mut parameters = Parameters::new(image.width, image.height);
                    parameters.set_reinhard_for_headroom(
                        ReinhardParameters {
                            compression_start: 0.3,
                            gamut_compression,
                            ..ReinhardParameters::default()
                        },
                        headroom,
                    );
                    parameters.set_linear_log_sigmoid_for_headroom(
                        LinearLogSigmoidParameters {
                            compression_start: 0.3,
                            gamut_compression,
                            ..LinearLogSigmoidParameters::default()
                        },
                        headroom,
                    );
                    gpu.render(&pipeline, parameters, &image.rgba)
                })
                .collect();
            for pair in outputs.windows(2) {
                assert!(
                    pair[1]
                        .iter()
                        .all(|value| value.is_finite() && *value >= 0.0)
                );
                let gray_end = image.width as usize * 4;
                for (before, after) in pair[0][..gray_end].iter().zip(&pair[1][..gray_end]) {
                    assert!(
                        (before - after).abs() < before * 0.003 + 2.0e-7,
                        "{name}, headroom {headroom}: neutral changed from {before} to {after}"
                    );
                }
                let color_difference = pair[0][gray_end..]
                    .iter()
                    .zip(&pair[1][gray_end..])
                    .map(|(before, after)| (before - after).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    color_difference > 0.01,
                    "{name}: gamut slider has no visible effect"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn rgb_log_sigmoid_hue_retention_is_uniform_across_brightness() {
    let gpu = TestGpu::new();
    // Exercise the production hue repair on a known wrap-around hue pair.
    // H=0.98 -> H=0.02 has midpoint red; S=0.6 and V must stay unchanged.
    let shader = BUILT_RGB_LOG_SIGMOID_SHADER.replace(
        "let mapped: vec3f = adjustHsv(originalLinear, rgbLogSigmoid(originalLinear));",
        "let mapped: vec3f = adjustHsv(\n\
            pow(hsvToRgb(vec3f(0.02, 0.6, 0.7)), vec3f(2.2)),\n\
            hsvToRgb(vec3f(0.98, 0.6, ap0.x)));",
    );
    assert_ne!(shader, BUILT_RGB_LOG_SIGMOID_SHADER);
    let pipeline =
        create_pipeline(&gpu.device, &gpu.layout, &shader, "uniform hue repair").unwrap();
    for headroom in [1.0, 64.0] {
        let values = if headroom == 1.0 {
            vec![0.0, 0.01, 0.1, 0.5, 1.0]
        } else {
            vec![0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0]
        };
        let input: Vec<_> = values.iter().flat_map(|&v| [v, v, v, 1.0]).collect();
        let mut parameters = Parameters::new(values.len() as u32, 1);
        parameters.set_log_sigmoid_for_headroom(
            LinearLogSigmoidParameters::default().tone_scale(),
            headroom,
        );
        for (retention, rgb_ratios) in [
            (0.0, [1.0, 0.4, 0.472]),
            (0.5, [1.0, 0.4, 0.4]),
            (1.0, [1.0, 0.472, 0.4]),
        ] {
            parameters.rgb_log_sigmoid_hue_retention = retention;
            let output = gpu.render(&pipeline, parameters, &input);
            for (&value, pixel) in values.iter().zip(output.as_chunks::<4>().0) {
                let value = half::f16::from_f32(value).to_f32();
                for channel in 0..3 {
                    let expected = value * rgb_ratios[channel];
                    assert!(
                        (pixel[channel] - expected).abs() < value * 0.001 + 1.0e-7,
                        "retention {retention}, V {value}, headroom {headroom}: expected {expected}, got {}",
                        pixel[channel]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_chroma_controls_preserve_defaults_and_neutral_axis() {
    let gpu = TestGpu::new();
    let image = crate::image_io::test_pattern(127, 17);
    for (kind, shader) in [
        (DrtKind::OklabReinhard, BUILT_OKLAB_REINHARD_SHADER),
        (DrtKind::OklabLogSigmoid, BUILT_OKLAB_LOG_SIGMOID_SHADER),
    ] {
        // Inspect pre-clamp linear output so a diagnostic color or display clamp
        // cannot conceal a broken gamut boundary at the new slider limits.
        let raw_shader = shader.replace(
            "vec4f(prepareOutput(ap0, mapped), 1.0)",
            "vec4f(mapped, 1.0)",
        );
        assert_ne!(raw_shader, shader);
        let pipeline =
            create_pipeline(&gpu.device, &gpu.layout, &raw_shader, kind.label()).unwrap();
        let mut parameters = Parameters::new(image.width, image.height);
        parameters.set_reinhard_for_headroom(ReinhardParameters::oklab_default(), 1.0);
        parameters.set_linear_log_sigmoid_for_headroom(LinearLogSigmoidParameters::default(), 1.0);
        let original = gpu.render(&pipeline, parameters, &image.rgba);

        // Original hard-coded Oklab expressions, before exposing the controls.
        let legacy_shader = raw_shader
            .replace("return 1.0 - pow(clamp(lightness, 0.0, 1.0), parameters.oklabHighlightChromaPower);",
                "let l2 = lightness * lightness; let l4 = l2 * l2; let l8 = l4 * l4; return 1.0 - l8 * l4;")
            .replace("return mix(parameters.oklabEndpointCompressionPower, parameters.oklabMidtoneCompressionPower, midtoneWeight);",
                "return 32.0 - 256.0 * endpointDistance * endpointDistance;")
            .replace("softMin(blackChroma, whiteChroma, parameters.oklabGamutRoundingPower)",
                "legacySoftMin4(blackChroma, whiteChroma)");
        let legacy_shader = format!(
            "{legacy_shader}\n{}",
            r#"
            fn legacySoftMin4(value: f32, limit: f32) -> f32 {
                if (value <= 0.0 || limit <= 0.0) { return 0.0; }
                let lower = min(value, limit);
                let ratio = lower / max(value, limit);
                let ratio2 = ratio * ratio;
                return lower * inverseSqrt(sqrt(1.0 + ratio2 * ratio2));
            }
        "#
        );
        let legacy = create_pipeline(
            &gpu.device,
            &gpu.layout,
            &legacy_shader,
            "legacy Oklab chroma",
        )
        .unwrap();
        let reference = gpu.render(&legacy, parameters, &image.rgba);
        assert!(
            original
                .iter()
                .zip(reference)
                .all(|(a, b)| (a - b).abs() < 0.002)
        );

        for control in 0..4 {
            for upper in [false, true] {
                let mut source = OklabChromaParameters::default();
                match control {
                    0 => source.highlight_chroma_power = if upper { 32.0 } else { 1.0 },
                    1 => source.gamut_rounding_power = if upper { 8.0 } else { 1.0 },
                    2 => source.endpoint_compression_power = if upper { 64.0 } else { 1.0 },
                    _ => source.midtone_compression_power = if upper { 64.0 } else { 1.0 },
                }
                parameters.set_oklab_chroma(source);
                let output = gpu.render(&pipeline, parameters, &image.rgba);
                // Reinhard intentionally permits neutral overexposure up to its
                // asymptote before the final display clamp.
                let upper_bound = if kind == DrtKind::OklabReinhard {
                    parameters.linear_curve_peak + 0.001
                } else {
                    1.001
                };
                assert!(
                    output
                        .iter()
                        .all(|v| v.is_finite() && *v >= -0.001 && *v <= upper_bound),
                    "{} invalid gamut output with {source:?}: min {}, max {}",
                    kind.label(),
                    output.iter().copied().fold(f32::INFINITY, f32::min),
                    output.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                );
                let neutral_end = image.width as usize * 4;
                assert!(
                    output[..neutral_end]
                        .iter()
                        .zip(&original[..neutral_end])
                        .all(|(a, b)| (a - b).abs() < 0.001)
                );
                assert!(
                    output[neutral_end..]
                        .iter()
                        .zip(&original[neutral_end..])
                        .any(|(a, b)| (a - b).abs() > 0.001),
                    "{} control {control} had no visible effect",
                    kind.label()
                );
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
    let image = crate::image_io::test_pattern(drt.width(), drt.height());
    // The new Oklab curve has its own parameters and keeps the SDR boundary.
    let oklab_sigmoid = LinearLogSigmoidParameters {
        linear_slope: 1.5,
        compression_start: 0.3,
        highlight_ev: 8.0,
        shoulder_power: 2.25,
        ..LinearLogSigmoidParameters::default()
    };
    let oklab_chroma = OklabChromaParameters {
        highlight_chroma_power: 8.0,
        gamut_rounding_power: 2.0,
        endpoint_compression_power: 24.0,
        midtone_compression_power: 12.0,
    };
    drt.set_oklab_chroma_parameters(DrtKind::OklabLogSigmoid, oklab_chroma);
    drt.set_oklab_log_sigmoid_parameters(oklab_sigmoid);
    drt.set_drt(DrtKind::OklabLogSigmoid);
    let oklab_sigmoid_output = gpu.render(
        &drt.pipelines.oklab_log_sigmoid,
        drt.parameters,
        &image.rgba,
    );
    drt.set_oklab_chroma_parameters(
        DrtKind::OklabReinhard,
        OklabChromaParameters {
            highlight_chroma_power: 20.0,
            ..OklabChromaParameters::default()
        },
    );
    drt.set_rgb_log_sigmoid_parameters(LinearLogSigmoidParameters {
        linear_slope: 0.4,
        compression_start: 0.5,
        gamut_compression: 0.25,
        highlight_ev: 12.0,
        shoulder_power: 7.0,
    });
    for headroom in [1.0, 4.0, 64.0] {
        for kind in DrtKind::ALL {
            drt.set_drt(kind);
            drt.set_hdr_headroom(headroom);
        }
        drt.set_drt(DrtKind::OklabLogSigmoid);
        assert_eq!(drt.oklab_log_sigmoid_parameters, oklab_sigmoid);
        assert_eq!(drt.oklab_log_sigmoid_chroma, oklab_chroma);
        assert_eq!(
            drt.parameters.oklab_highlight_chroma_power,
            oklab_chroma.highlight_chroma_power
        );
        assert_eq!(drt.parameters.log_sigmoid_output_peak, 1.0);
        assert_eq!(
            gpu.render(
                &drt.pipelines.oklab_log_sigmoid,
                drt.parameters,
                &image.rgba
            ),
            oklab_sigmoid_output
        );
    }
    drt.set_drt(DrtKind::RgbLogSigmoid);
    assert_eq!(
        drt.parameters.log_sigmoid_output_pivot,
        extended_srgb_oetf(0.5 * 0.4)
    );
    assert_eq!(drt.parameters.log_sigmoid_gamut_compression, 0.25);
    assert!(drt.parameters.log_sigmoid_output_peak > 1.0);
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
