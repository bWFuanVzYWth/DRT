// GPU properties for the inexpensive max-driven, root-LMS neutral experiment.

fn oklab_neutral_validation_shader(expression: &str) -> String {
    oklab_aces_kernel_shader_from_source(&builtin_shader(DrtKind::OklabNeutral), expression)
}

fn oklab_neutral_validation_pipeline(gpu: &TestGpu, expression: &str) -> wgpu::ComputePipeline {
    create_pipeline(
        &gpu.device,
        &gpu.layout,
        &oklab_neutral_validation_shader(expression),
        "Oklab Neutral validation",
    )
    .unwrap()
}

fn oklab_neutral_validation_oracle(
    rgb: [f64; 3],
    parameters: OklabNeutralParameters,
    headroom: f64,
) -> [f64; 3] {
    // Independent f64 evaluation of the proposed transform. This deliberately
    // does not call its Rust scalar evaluator or any WGSL helper function.
    let maximum = rgb.into_iter().fold(0.0_f64, f64::max);
    let gain = f64::from(parameters.linear_slope);
    let join = f64::from(parameters.compression_start);
    if maximum <= join {
        return rgb.map(|channel| gain * channel);
    }
    let extent = headroom - gain * join;
    let q = gain * (maximum - join) / extent;
    let power = f64::from(parameters.shoulder_power);
    let tail = (1.0 + q / power).powf(-power);
    let output_maximum = headroom - extent * tail;
    let keep = tail * (1.0 + tail * (3.0 + tail * (-5.0 + 2.0 * tail)));
    let normalized = rgb.map(|channel| channel / maximum);
    let matrix = [
        [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
        [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
        [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
    ];
    let mixed_roots = matrix.map(|row| {
        let response = row
            .into_iter()
            .zip(normalized)
            .map(|(m, x)| m * x)
            .sum::<f64>();
        1.0 + keep * (response.cbrt() - 1.0)
    });
    let cones = mixed_roots.map(|root| root.powi(3));
    let inverse = [
        [4.076_741_662_1, -3.307_711_591_3, 0.230_969_929_2],
        [-1.268_438_004_6, 2.609_757_401_1, -0.341_319_396_5],
        [-0.004_196_086_3, -0.703_418_614_7, 1.707_614_701_0],
    ];
    let color = inverse.map(|row| row.into_iter().zip(cones).map(|(m, x)| m * x).sum::<f64>());
    let lift = -color.into_iter().fold(0.0_f64, f64::min);
    let lifted = color.map(|channel| channel + lift);
    let denominator = lifted.into_iter().fold(0.0_f64, f64::max);
    lifted.map(|channel| output_maximum * channel / denominator)
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_neutral_shadows_and_display_peak_follow_the_scalar_curve() {
    let gpu = TestGpu::new();
    let pipeline = oklab_neutral_validation_pipeline(&gpu, "mapLinearRgb(source)");
    // Compare against a simple GPU gain reference so both paths use the same
    // texture-store rounding mode instead of assuming CPU fp16 conversion.
    let linear_pipeline =
        oklab_neutral_validation_pipeline(&gpu, "source*parameters.oklabNeutralLinearSlope");
    let colors = [
        [0.0, 0.0, 0.0, 1.0],
        [0.02, 0.01, 0.005, 1.0],
        [0.005, 0.02, 0.01, 1.0],
        [0.01, 0.005, 0.02, 1.0],
        [0.02, 0.0, 0.02, 1.0],
        [0.0, 0.0, 0.02, 1.0],
    ]
    .concat();
    for headroom in [1.0_f32, 4.0, 64.0] {
        for gain in [0.1, 1.0, 4.0] {
            let mut source = OklabNeutralParameters {
                linear_slope: gain,
                ..Default::default()
            };
            source.constrain();
            let mut parameters = Parameters::new((colors.len() / 4) as u32, 1);
            parameters.set_oklab_neutral_for_headroom(source, headroom);
            let actual = gpu.render(&pipeline, parameters, &colors);
            let expected = gpu.render(&linear_pipeline, parameters, &colors);
            assert_eq!(
                actual, expected,
                "nonlinear colored shadow: gain {gain}, peak {headroom}"
            );
        }
        let source = OklabNeutralParameters::default();
        let curve = source.curve_for_headroom(headroom);
        let reach = curve.highlight_reach_ev();
        if headroom == 1.0 {
            assert!((reach - 10.0).abs() < 0.001);
            assert!((source.compression_start - 0.6).abs() < f32::EPSILON);
        }
        let values = [
            0.0,
            0.02,
            0.599,
            0.6,
            0.601,
            1.0,
            4.0,
            16.0,
            0.18 * 2.0_f32.powf(reach),
            0.18 * 2.0_f32.powf(reach + 2.0),
        ];
        // Upload 8 EV lower so a large HDR scene value stays finite in fp16.
        let input: Vec<_> = values
            .into_iter()
            .flat_map(|value| [value / 256.0, value / 256.0, value / 256.0, 1.0])
            .collect();
        let mut parameters = Parameters::new(values.len() as u32, 1);
        parameters.set_oklab_neutral_for_headroom(source, headroom);
        parameters.exposure_multiplier = 256.0;
        let actual = gpu.render(&pipeline, parameters, &input);
        for (pixel, input) in actual
            .as_chunks::<4>()
            .0
            .iter()
            .zip(input.as_chunks::<4>().0)
        {
            let scene = half::f16::from_f32(input[0]).to_f32() * 256.0;
            let expected = curve.map_linear(scene);
            for channel in &pixel[..3] {
                assert!(channel.is_finite() && *channel >= 0.0 && *channel <= 1.002 * headroom);
                assert!((channel - expected).abs() <= 0.002 * headroom + 0.000_01);
            }
        }
        let at_reach = actual.as_chunks::<4>().0[8][0];
        assert!((at_reach / headroom - 0.98).abs() < 0.002);
        assert!(actual.as_chunks::<4>().0[9][0] > at_reach);
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_neutral_color_trajectories_stay_bounded_and_fade_without_rebrightening() {
    let gpu = TestGpu::new();
    let pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &builtin_shader(DrtKind::OklabNeutral),
        "Oklab Neutral 61 colors",
    )
    .unwrap();
    let raw_pipeline = oklab_neutral_validation_pipeline(
        &gpu,
        "mapLinearRgb(max(acesAp0ToRec709(source),vec3f(0.0)))",
    );
    let image = crate::image_io::color_trajectory_pattern(513, 61);
    let originals: Vec<_> = image
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            let rgb = oklab_aces_validation_ap0_to_rgb([pixel[0], pixel[1], pixel[2]])
                .map(|channel| channel.max(0.0));
            oklab_aces_validation_lab(rgb)
        })
        .collect();
    for headroom in [1.0_f32, 4.0, 64.0] {
        let mut parameters = Parameters::new(image.width, image.height);
        parameters.set_oklab_neutral_for_headroom(Default::default(), headroom);
        let encoded = gpu.render(&pipeline, parameters, &image.rgba);
        let raw = gpu.render(&raw_pipeline, parameters, &image.rgba);
        let labs = oklab_aces_validation_display_labs(&encoded);
        let scale = f64::from(headroom).cbrt();
        let peak_signal = extended_srgb_oetf(headroom);
        for (band, ray) in labs.chunks_exact(image.width as usize).enumerate() {
            let offset = band * image.width as usize;
            let mut chroma = Vec::new();
            let mut previous_l = 0.0;
            let mut hue_count = 0;
            for (sample, lab) in ray.iter().enumerate() {
                let index = offset + sample;
                let linear = &raw[index * 4..index * 4 + 3];
                let output = &encoded[index * 4..index * 4 + 3];
                assert!(
                    linear
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.002 * headroom)
                );
                assert!(
                    output
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0 && *v <= peak_signal + 0.005)
                );
                let normalized_l = lab[0] / scale;
                assert!(
                    normalized_l + 0.001 >= previous_l,
                    "lightness reversed, band {band}, sample {sample}, peak {headroom}"
                );
                previous_l = normalized_l;
                let normalized_c = lab[1].hypot(lab[2]) / scale;
                chroma.push(normalized_c);
                if normalized_c >= 0.01 && normalized_l >= 0.03 && band != 60 {
                    let error = oklab_aces_validation_hue_error(originals[index], *lab);
                    assert!(
                        error < 3.0,
                        "root-white hue drift {error} degrees, band {band}, peak {headroom}"
                    );
                    hue_count += 1;
                }
            }
            if band != 60 {
                assert!(
                    hue_count >= 16,
                    "prematurely white color band {band}, peak {headroom}"
                );
            }
            let maximum = chroma
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let mut lowest = chroma[maximum];
            for (sample, c) in chroma.iter().enumerate().skip(maximum) {
                assert!(
                    *c <= lowest + 0.001,
                    "chroma brightened again, band {band}, sample {sample}, peak {headroom}"
                );
                lowest = lowest.min(*c);
            }
        }
        if headroom == 1.0
            && let Some(directory) = std::env::var_os("DRT_TRAJECTORY_REPORT_DIR")
        {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            let document = serde_json::json!({
                "width": image.width, "height": image.height, "headroom": headroom,
                "inputs": image.rgba, "encoded_rgb": encoded,
            });
            let file =
                std::fs::File::create(directory.join("oklab_neutral_trajectory.json")).unwrap();
            serde_json::to_writer(file, &document).unwrap();
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_neutral_matches_f64_reference_and_refreshes_independent_parameters() {
    let gpu = TestGpu::new();
    let pipeline = oklab_neutral_validation_pipeline(&gpu, "mapLinearRgb(source)");
    let input = [
        [0.0, 0.0, 0.0, 1.0],
        [0.4, 0.1, 0.2, 1.0],
        [0.6, 0.2, 0.4, 1.0],
        [0.61, 0.0, 0.61, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [4.0, 3.0, 3.0, 1.0],
        [16.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 16.0, 1.0],
        [64.0, 48.0, 64.0, 1.0],
        [48.0, 64.0, 64.0, 1.0],
        [256.0, 256.0, 256.0, 1.0],
    ]
    .concat();
    for headroom in [1.0_f32, 4.0, 64.0] {
        for mut source in [
            OklabNeutralParameters::default(),
            OklabNeutralParameters {
                linear_slope: 0.5,
                compression_start: 0.75,
                shoulder_power: 0.8,
            },
        ] {
            source.constrain();
            let mut parameters = Parameters::new((input.len() / 4) as u32, 1);
            parameters.set_oklab_neutral_for_headroom(source, headroom);
            let actual = gpu.render(&pipeline, parameters, &input);
            for (pixel, input) in actual
                .as_chunks::<4>()
                .0
                .iter()
                .zip(input.as_chunks::<4>().0)
            {
                let source_rgb = [input[0], input[1], input[2]]
                    .map(|v| f64::from(half::f16::from_f32(v).to_f32()));
                let expected =
                    oklab_neutral_validation_oracle(source_rgb, source, f64::from(headroom));
                for channel in 0..3 {
                    assert!(
                        (f64::from(pixel[channel]) - expected[channel]).abs()
                            < 0.002 * f64::from(headroom) + 0.000_01,
                        "GPU/f64 mismatch: input {source_rgb:?}, peak {headroom}, parameters {source:?}, expected {expected:?}, actual {pixel:?}"
                    );
                }
            }
        }
    }
    let mut drt = DrtGpu::new(&gpu.render_state(), crate::image_io::test_pattern(63, 9)).unwrap();
    drt.set_drt(DrtKind::OklabAces);
    drt.set_comparison_drt(Some(DrtKind::OklabNeutral));
    let defaults = OklabNeutralParameters::default();
    let baseline = gpu.read_texture(&drt.comparison.as_ref().unwrap()._output);
    let left_output = gpu.read_texture(&drt.image._output);
    let mut edits = [defaults; 3];
    edits[0].linear_slope = 1.5;
    edits[1].compression_start = 0.8;
    edits[2].set_highlight_reach_ev(14.0, 1.0);
    for edited in edits {
        let left_parameters = bytemuck::bytes_of(&drt.parameters).to_vec();
        drt.set_oklab_neutral_parameters(edited);
        let comparison = gpu.read_texture(&drt.comparison.as_ref().unwrap()._output);
        assert_ne!(
            comparison, baseline,
            "parameter has no visible effect: {edited:?}"
        );
        assert_eq!(bytemuck::bytes_of(&drt.parameters), left_parameters);
        assert_eq!(gpu.read_texture(&drt.image._output), left_output);
        drt.set_drt(DrtKind::OklabNeutral);
        assert_eq!(gpu.read_texture(&drt.image._output), comparison);
        drt.set_drt(DrtKind::OklabAces);
        drt.set_oklab_neutral_parameters(defaults);
        assert_eq!(
            gpu.read_texture(&drt.comparison.as_ref().unwrap()._output),
            baseline
        );
    }
}
