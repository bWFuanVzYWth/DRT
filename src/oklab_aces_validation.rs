// GPU checks for the original Oklab experiment, including final display output.

fn oklab_aces_validation_ap0_to_rgb(ap0: [f32; 3]) -> [f64; 3] {
    let source = ap0.map(|value| f64::from(half::f16::from_f32(value).to_f32()));
    [
        [2.521_400_888_6, -1.133_995_749_4, -0.387_561_856_8],
        [-0.276_214_061_6, 1.372_595_566_3, -0.096_282_355_7],
        [-0.015_320_200_1, -0.152_992_561_8, 1.168_387_199_6],
    ]
    .map(|row| row.into_iter().zip(source).map(|(m, x)| m * x).sum())
}

// Match the workbench's published input coefficients verbatim.
#[allow(clippy::excessive_precision)]
fn oklab_aces_validation_rgb_to_ap0(rgb: [f32; 3]) -> [f32; 3] {
    [
        [0.439_643_004_0, 0.383_005_471_4, 0.177_399_308_9],
        [0.089_715_731_9, 0.813_475_053_8, 0.096_782_252_4],
        [0.017_512_720_5, 0.111_551_438_5, 0.870_882_793_0],
    ]
    .map(|row| row.into_iter().zip(rgb).map(|(m, x)| m * x).sum())
}

fn oklab_aces_validation_lab(rgb: [f64; 3]) -> [f64; 3] {
    let roots = [
        [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
        [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
        [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
    ]
    .map(|row| {
        row.into_iter()
            .zip(rgb)
            .map(|(m, x)| m * x)
            .sum::<f64>()
            .cbrt()
    });
    [
        [0.210_454_255_3, 0.793_617_785_0, -0.004_072_046_8],
        [1.977_998_495_1, -2.428_592_205_0, 0.450_593_709_9],
        [0.025_904_037_1, 0.782_771_766_2, -0.808_675_766_0],
    ]
    .map(|row| row.into_iter().zip(roots).map(|(m, x)| m * x).sum())
}

fn oklab_aces_validation_decode(value: f32) -> f64 {
    let value = f64::from(value);
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn oklab_aces_validation_hue_error(before: [f64; 3], after: [f64; 3]) -> f64 {
    let cross = before[1] * after[2] - before[2] * after[1];
    let dot = before[1] * after[1] + before[2] * after[2];
    cross.atan2(dot).abs().to_degrees()
}

fn oklab_aces_kernel_shader(expression: &str) -> String {
    let source = builtin_shader(DrtKind::OklabAces);
    oklab_aces_kernel_shader_from_source(&source, expression)
}

fn oklab_aces_kernel_shader_from_source(source: &str, expression: &str) -> String {
    let helpers = source.split("@compute").next().unwrap();
    format!(
        r#"{helpers}
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) id: vec3u) {{
            if (id.x >= parameters.width || id.y >= parameters.height) {{ return; }}
            let source = textureLoad(inputTexture, vec2i(id.xy), 0).rgb
                * parameters.exposureMultiplier;
            let result = {expression};
            textureStore(outputTexture, vec2i(id.xy), vec4f(result, 1.0));
        }}
        "#
    )
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_colored_shadows_keep_linear_tone_with_gamut_fitting() {
    let gpu = TestGpu::new();
    let pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &builtin_shader(DrtKind::OklabAces),
        "Oklab ACES-inspired linear shadows",
    )
    .unwrap();
    let none = create_pipeline(
        &gpu.device,
        &gpu.layout,
        BUILT_NONE_SHADER,
        "linear reference",
    )
    .unwrap();
    // The dark scalar curve remains linear; gamut fitting can now make small
    // chroma adjustments even below the join. Interior Rec.709 colors should
    // still match the linear display reference within fp16/margin precision.
    let input = [
        [0.0, 0.0, 0.0],
        [0.000_02, 0.000_04, 0.000_08],
        [0.005, 0.002, 0.001],
        [0.002, 0.005, 0.001],
        [0.001, 0.002, 0.005],
        [0.005, 0.001, 0.005],
        [0.001, 0.005, 0.005],
        [0.005, 0.005, 0.001],
        [0.005, 0.005, 0.005],
    ]
    .into_iter()
    .flat_map(|rgb| {
        let ap0 = oklab_aces_validation_rgb_to_ap0(rgb);
        [ap0[0], ap0[1], ap0[2], 1.0]
    })
    .collect::<Vec<_>>();
    for headroom in [1.0, 4.0, 64.0] {
        for gain in [0.1, 1.0, 4.0] {
            for exposure in [0.25, 1.0, 4.0] {
                let mut parameters = Parameters::new((input.len() / 4) as u32, 1);
                parameters.set_oklab_aces_for_headroom(
                    crate::oklab_aces::OklabAcesParameters {
                        linear_slope: gain,
                        ..Default::default()
                    },
                    headroom,
                );
                parameters.exposure_multiplier = exposure;
                let output = gpu.render(&pipeline, parameters, &input);
                parameters.exposure_multiplier = exposure * gain;
                let expected = gpu.render(&none, parameters, &input);
                for (actual, reference) in output.iter().zip(&expected) {
                    assert!(
                        (actual - reference).abs() < 0.001,
                        "gain {gain}, exposure {exposure}, peak {headroom}: {actual} versus {reference}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_highlights_keep_extended_detail_in_sdr_and_hdr() {
    let gpu = TestGpu::new();
    let pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &builtin_shader(DrtKind::OklabAces),
        "Oklab ACES-inspired extended highlights",
    )
    .unwrap();
    let old = create_pipeline(
        &gpu.device,
        &gpu.layout,
        BUILT_OKLAB_LOG_SIGMOID_SHADER,
        "original Oklab highlight reference",
    )
    .unwrap();
    let evs = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0];
    // The highest scene values exceed fp16's finite source range. Upload four
    // EV lower and apply exposure in f32 within the real GPU pipeline.
    let input: Vec<_> = evs
        .iter()
        .flat_map(|ev| {
            let value = 0.18 * 2.0_f32.powf(ev - 4.0);
            [value, value, value, 1.0]
        })
        .collect();
    for headroom in [1.0, 4.0, 64.0] {
        // Explicitly select the original long tail: the default now reaches
        // 98% peak at 10 EV and can legitimately quantize late samples alike.
        let source = crate::oklab_aces::OklabAcesParameters {
            shoulder_power: 0.4,
            ..Default::default()
        };
        let curve = source.curve_for_headroom(headroom);
        let mut parameters = Parameters::new(evs.len() as u32, 1);
        parameters.set_oklab_aces_for_headroom(source, headroom);
        parameters.exposure_multiplier = 16.0;
        let output = gpu.render(&pipeline, parameters, &input);
        let ramp = output.as_chunks::<4>().0;
        assert!(ramp.windows(2).all(|pair| pair[1][0] > pair[0][0]));
        for (pixel, source_pixel) in ramp.iter().zip(input.as_chunks::<4>().0) {
            let value = half::f16::from_f32(source_pixel[0]).to_f32() * 16.0;
            let expected = extended_srgb_oetf(curve.map_linear(value));
            assert!((pixel[0] - expected).abs() < 0.003 * headroom.cbrt());
            assert!(pixel[..3].iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!(pixel[0] < extended_srgb_oetf(headroom));
            assert!((pixel[0] - pixel[1]).abs() < 0.003);
            assert!((pixel[1] - pixel[2]).abs() < 0.003);
        }
        if headroom == 1.0 {
            parameters
                .set_linear_log_sigmoid_for_headroom(LinearLogSigmoidParameters::default(), 1.0);
            let old_output = gpu.render(&old, parameters, &input);
            let old_ramp = old_output.as_chunks::<4>().0;
            // The old finite input reach has already reached its display
            // plateau while the experiment still distinguishes +14/+16/+20.
            assert!((old_ramp[4][0] - old_ramp[5][0]).abs() <= 0.001);
            assert!(ramp[5][0] - ramp[4][0] > 0.002);
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_highlight_reach_controls_brightness_and_current_display_peak() {
    let gpu = TestGpu::new();
    let pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &builtin_shader(DrtKind::OklabAces),
        "Oklab ACES-inspired highlight reach",
    )
    .unwrap();
    let defaults = crate::oklab_aces::OklabAcesParameters::default();
    assert!((defaults.curve_for_headroom(1.0).highlight_reach_ev() - 10.0).abs() < 0.001);
    let values = [0.005, 0.18, 0.36, 1.0, 4.0, 16.0, 64.0, 512.0];
    let input: Vec<_> = values
        .iter()
        .flat_map(|value| [*value, *value, *value, 1.0])
        .collect();
    let mut previous: Option<Vec<f32>> = None;
    for reach in [6.0, 8.0, 14.0] {
        let mut source = crate::oklab_aces::OklabAcesParameters::default();
        source.set_highlight_reach_ev(reach, 1.0);
        let curve = source.curve_for_headroom(1.0);
        assert!((curve.highlight_reach_ev() - reach).abs() < 0.001);
        let mut parameters = Parameters::new(values.len() as u32, 1);
        parameters.set_oklab_aces_for_headroom(source, 1.0);
        let output = gpu.render(&pipeline, parameters, &input);
        assert!(
            output
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        if let Some(shorter_reach) = previous {
            // The actual UI setter changes only the shoulder. Both shadow
            // samples stay bit-identical; a shorter reach brightens highlights.
            assert_eq!(&shorter_reach[..8], &output[..8]);
            for (shorter, longer) in shorter_reach.as_chunks::<4>().0[2..]
                .iter()
                .zip(&output.as_chunks::<4>().0[2..])
            {
                assert!(shorter[0] > longer[0]);
            }
        }
        previous = Some(output);
    }

    for headroom in [1.0, 4.0, 64.0] {
        for requested_reach in [6.0_f32, 8.0, 10.0, 14.0] {
            let mut source = crate::oklab_aces::OklabAcesParameters::default();
            let [minimum, maximum] = source.highlight_reach_range(headroom);
            let reach = requested_reach.clamp(minimum, maximum);
            source.set_highlight_reach_ev(requested_reach, headroom);
            let curve = source.curve_for_headroom(headroom);
            assert!((curve.highlight_reach_ev() - reach).abs() < 0.001);
            // High HDR scene values can exceed the source texture's fp16
            // range. Exposure raises this finite upload to the selected EV.
            let upload = 0.18 * 2.0_f32.powf(reach - 8.0);
            let input = [upload, upload, upload, 1.0];
            let mut parameters = Parameters::new(1, 1);
            parameters.set_oklab_aces_for_headroom(source, headroom);
            parameters.exposure_multiplier = 256.0;
            let output = gpu.render(&pipeline, parameters, &input);
            let actual_input = half::f16::from_f32(upload).to_f32() * 256.0;
            let expected = extended_srgb_oetf(curve.map_linear(actual_input));
            let target = extended_srgb_oetf(0.98 * headroom);
            let tolerance = 0.003 * headroom.cbrt();
            let peak = extended_srgb_oetf(headroom);
            for value in &output[..3] {
                assert!(value.is_finite() && (0.0..=peak).contains(value));
                assert!((value - expected).abs() < tolerance);
                assert!((value - target).abs() < tolerance);
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_61_color_trajectories_preserve_final_display_hue() {
    let gpu = TestGpu::new();
    let pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &builtin_shader(DrtKind::OklabAces),
        "61-color final display hue",
    )
    .unwrap();
    let raw_pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &oklab_aces_kernel_shader("mapLinearRgb(acesAp0ToRec709(source))"),
        "61-color output before presentation clip",
    )
    .unwrap();
    let image = crate::image_io::color_trajectory_pattern(513, 61);
    let input_lab: Vec<_> = image
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            oklab_aces_validation_lab(oklab_aces_validation_ap0_to_rgb([
                pixel[0], pixel[1], pixel[2],
            ]))
        })
        .collect();
    // The palette enumerates four 9-color faces, then the 25 colors with R=1.
    // Magenta is the last B sample in the first R=1, G=0 row; white is last.
    const MAGENTA_BAND: usize = 40;
    const WHITE_BAND: usize = 60;
    for headroom in [1.0, 4.0, 64.0] {
        let source = crate::oklab_aces::OklabAcesParameters::default();
        let curve = source.curve_for_headroom(headroom);
        let mut parameters = Parameters::new(image.width, image.height);
        parameters.set_oklab_aces_for_headroom(source, headroom);
        let output = gpu.render(&pipeline, parameters, &image.rgba);
        let raw = gpu.render(&raw_pipeline, parameters, &image.rgba);
        let peak_signal = extended_srgb_oetf(headroom);
        let lab_peak = f64::from(headroom).cbrt();
        let mut covered = [0_usize; 61];
        let mut magenta_covered = 0;
        for (index, ((pixel, linear), original)) in output
            .as_chunks::<4>()
            .0
            .iter()
            .zip(raw.as_chunks::<4>().0)
            .zip(&input_lab)
            .enumerate()
        {
            let band = index / image.width as usize;
            assert!(
                pixel[..3]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0 && *v <= peak_signal + 0.005)
            );
            // Inspect the actual transform before the display clamp. fp16
            // readback can round a margin onto the endpoint, but large outliers
            // would prove that hard channel clipping still changes its hue.
            assert!(
                linear[..3]
                    .iter()
                    .all(|v| v.is_finite() && *v >= -0.002 * headroom && *v <= 1.002 * headroom),
                "unfitted RGB at band {band}, sample {index}, peak {headroom}: {linear:?}"
            );
            let decoded = pixel[..3]
                .try_into()
                .map(|channels: [f32; 3]| channels.map(oklab_aces_validation_decode))
                .unwrap();
            let displayed = oklab_aces_validation_lab(decoded);
            let chroma = displayed[1].hypot(displayed[2]);
            if band == WHITE_BAND {
                let value = half::f16::from_f32(image.rgba[index * 4]).to_f32();
                let expected = extended_srgb_oetf(curve.map_linear(value));
                assert!((pixel[0] - expected).abs() < 0.003 * headroom.cbrt());
                assert!(chroma / lab_peak < 0.001);
                continue;
            }
            // Near black and near white, fp16's channel quantization makes a
            // hue angle poorly defined. Every colored band still needs many
            // measurable points, so premature whitening cannot bypass this.
            if chroma / lab_peak >= 0.01 && displayed[0] / lab_peak >= 0.03 {
                let error = oklab_aces_validation_hue_error(*original, displayed);
                assert!(
                    error < 2.0,
                    "display hue shifted {error:.3} degrees at band {band}, sample {index}, peak {headroom}"
                );
                covered[band] += 1;
                if band == MAGENTA_BAND && chroma / lab_peak >= 0.02 {
                    assert!(
                        error < 1.0,
                        "magenta bent {error:.3} degrees at peak {headroom}"
                    );
                    magenta_covered += 1;
                }
            }
        }
        assert!(
            covered[..WHITE_BAND].iter().all(|count| *count >= 16),
            "insufficient hue coverage at peak {headroom}: {covered:?}"
        );
        assert!(magenta_covered >= 16);
        if headroom == 1.0 {
            assert!((curve.highlight_reach_ev() - 10.0).abs() < 0.001);
            assert!(curve.map_linear(0.18 * 2.0_f32.powf(6.5)) < 0.98);
            if let Some(directory) = std::env::var_os("DRT_TRAJECTORY_REPORT_DIR") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                let old_pipeline = create_pipeline(
                    &gpu.device,
                    &gpu.layout,
                    BUILT_OKLAB_REINHARD_SHADER,
                    "trajectory report old Oklab",
                )
                .unwrap();
                let mut old_parameters = Parameters::new(image.width, image.height);
                old_parameters.set_reinhard_for_headroom(ReinhardParameters::oklab_default(), 1.0);
                let old_output = gpu.render(&old_pipeline, old_parameters, &image.rgba);
                let document = serde_json::json!({
                    "width": image.width, "height": image.height, "headroom": headroom,
                    "input_space": "scene-linear AP0", "output_space": "extended sRGB",
                    "inputs": image.rgba, "old_output": old_output, "new_output": output,
                });
                let file =
                    std::fs::File::create(directory.join("oklab_aces_trajectory.json")).unwrap();
                serde_json::to_writer(file, &document).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_core_preserves_hue_with_late_color_gamut_fitting_before_clip() {
    let gpu = TestGpu::new();
    let mapped = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &oklab_aces_kernel_shader("rgbToOklab(mapLinearRgb(source))"),
        "Oklab ACES-inspired unclipped Lab",
    )
    .unwrap();
    let source_lab = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &oklab_aces_kernel_shader("rgbToOklab(source)"),
        "independent source Lab",
    )
    .unwrap();
    // Instrumented entry receives working Rec.709 directly, so it can inspect
    // the original Lab direction without AP0 rounding or final display clip.
    let input = [
        [64.0, 0.0, 0.0, 1.0],
        [0.0, 64.0, 0.0, 1.0],
        [0.0, 0.0, 64.0, 1.0],
        [64.0, 56.0, 56.0, 1.0],
        [0.18, 0.18, 0.18, 1.0],
        [64.0, 64.0, 64.0, 1.0],
        [2.0, -0.3, 0.1, 1.0],
        [-1.0, 0.3, 2.0, 1.0],
        [-1.0, -1.0, -1.0, 1.0],
        [4.0, 3.0, 3.0, 1.0],
        [48.0, 64.0, 64.0, 1.0],
    ]
    .concat();
    let mut parameters = Parameters::new((input.len() / 4) as u32, 1);
    parameters.set_oklab_aces_for_headroom(Default::default(), 1.0);
    let before = gpu.render(&source_lab, parameters, &input);
    for headroom in [1.0, 4.0, 64.0] {
        let curve = crate::oklab_aces::OklabAcesParameters::default().curve_for_headroom(headroom);
        parameters.set_oklab_aces_for_headroom(Default::default(), headroom);
        let output = gpu.render(&mapped, parameters, &input);
        assert!(output.iter().all(|v| v.is_finite()));
        for (original, kept) in before
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks::<4>().0)
        {
            let chroma = original[1].hypot(original[2]);
            let mapped_chroma = kept[1].hypot(kept[2]);
            if chroma > 0.01 && original[0].abs() > 0.001 {
                let sine =
                    (original[1] * kept[2] - original[2] * kept[1]) / (chroma * mapped_chroma);
                let dot = original[1] * kept[1] + original[2] * kept[2];
                assert!(
                    sine.abs() < 0.004 && dot > 0.0,
                    "hue changed: {original:?} -> {kept:?}"
                );
                if original[0] > 0.0 {
                    let brightness = original[0] * original[0] * original[0];
                    let expected_lightness = curve.map_linear(brightness).cbrt();
                    assert!(
                        (kept[0] - expected_lightness).abs() < 0.005 * headroom.cbrt(),
                        "gamut fitting changed the new tone curve: {original:?} -> {kept:?}"
                    );
                }
            }
        }
        // The middle gray, bright white and signed negative neutral retain
        // negligible a/b even though their lightness follows different paths.
        for index in [4, 5, 8] {
            let kept = output.as_chunks::<4>().0[index];
            assert!(kept[1].abs() < 0.001 && kept[2].abs() < 0.001);
        }
        // A pale highlight whose desired chroma already fits must retain it.
        // The old 1-w^2 fade discards about half of this sample's C/L in SDR.
        let original = before.as_chunks::<4>().0[9];
        let kept = output.as_chunks::<4>().0[9];
        let ratio =
            (kept[1].hypot(kept[2]) / kept[0]) / (original[1].hypot(original[2]) / original[0]);
        assert!(
            ratio > 0.9,
            "pale highlight faded early at peak {headroom}: {ratio}"
        );
    }
}

fn oklab_aces_validation_rgb(lab: [f64; 3]) -> [f64; 3] {
    let [l, a, b] = lab;
    let cones = [
        l + 0.396_337_777_4 * a + 0.215_803_757_3 * b,
        l - 0.105_561_345_8 * a - 0.063_854_172_8 * b,
        l - 0.089_484_177_5 * a - 1.291_485_548_0 * b,
    ]
    .map(|root| root.powi(3));
    [
        [4.076_741_662_1, -3.307_711_591_3, 0.230_969_929_2],
        [-1.268_438_004_6, 2.609_757_401_1, -0.341_319_396_5],
        [-0.004_196_086_3, -0.703_418_614_7, 1.707_614_701_0],
    ]
    .map(|row| row.into_iter().zip(cones).map(|(m, x)| m * x).sum())
}

fn oklab_aces_validation_source_saturation(hue: [f64; 2]) -> f64 {
    // Independently locate the first black boundary on this Oklab ray.
    // This does not call the shader's analytic saturation/cusp/cap helpers.
    let valid = |s: f64| {
        oklab_aces_validation_rgb([1.0, s * hue[0], s * hue[1]])
            .into_iter()
            .all(|channel| channel >= 0.0)
    };
    let mut lower = 0.0;
    let mut upper = 0.01;
    while valid(upper) {
        lower = upper;
        upper += 0.01;
        assert!(upper < 10.0, "failed to locate a connected black boundary");
    }
    for _ in 0..48 {
        let midpoint = 0.5 * (lower + upper);
        if valid(midpoint) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    lower
}

fn oklab_aces_validation_input_lightness(
    curve: crate::oklab_aces::OklabAcesCurve,
    normalized: f64,
) -> f32 {
    let peak = f64::from(curve.output_peak);
    let join = f64::from(curve.compression_start);
    let gain = f64::from(curve.linear_slope);
    let output = peak * normalized.powi(3);
    let output_join = gain * join;
    let brightness = if output <= output_join {
        output / gain
    } else {
        let extent = peak - output_join;
        let remaining = (peak - output) / extent;
        let power = f64::from(curve.shoulder_power);
        let q = power * (remaining.powf(-power.recip()) - 1.0);
        join + extent * q / gain
    };
    brightness.cbrt() as f32
}

fn oklab_aces_validation_same_h_expression(hue: [f64; 2], operation: &str) -> String {
    // Store L and S in fp16, then construct the working RGB inside the GPU.
    // This keeps source L identical between levels and avoids uploading HDR
    // scene RGB above fp16's finite range; the complete mapper still runs.
    let input = format!(
        "oklabToRgb(vec3f(source.x, source.x*source.y*{:.9}, source.x*source.y*{:.9}))",
        hue[0], hue[1]
    );
    let mapped = format!("mapLinearRgb({input})");
    match operation {
        "lab" => format!("rgbToOklab({mapped})"),
        "encoded" => format!("prepareOutput(source, {mapped})"),
        "rgb" => mapped,
        _ => panic!("unknown same-h readback operation"),
    }
}

fn oklab_aces_validation_display_labs(encoded: &[f32]) -> Vec<[f64; 3]> {
    encoded
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            oklab_aces_validation_lab(
                [pixel[0], pixel[1], pixel[2]].map(oklab_aces_validation_decode),
            )
        })
        .collect()
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_same_h_highlights_preserve_saturation_rank() {
    let gpu = TestGpu::new();
    let source = crate::oklab_aces::OklabAcesParameters::default();
    for degrees in [30.0_f64, 90.0, 194.0, 264.0, 328.0] {
        let angle = degrees.to_radians();
        let hue = [angle.cos(), angle.sin()];
        let reference = oklab_aces_validation_source_saturation(hue);
        let make_pipeline = |operation| {
            create_pipeline(
                &gpu.device,
                &gpu.layout,
                &oklab_aces_kernel_shader(&oklab_aces_validation_same_h_expression(hue, operation)),
                "same-h saturation rank",
            )
            .unwrap()
        };
        let lab_pipeline = make_pipeline("lab");
        let rgb_pipeline = make_pipeline("rgb");
        let encoded_pipeline = make_pipeline("encoded");
        for headroom in [1.0, 4.0, 64.0] {
            let curve = source.curve_for_headroom(headroom);
            for normalized in [0.95, 0.98, 0.995] {
                let input_l = oklab_aces_validation_input_lightness(curve, normalized);
                let input: Vec<_> = [0.2, 0.4, 0.6, 0.8]
                    .into_iter()
                    .flat_map(|rank| [input_l, (reference * rank) as f32, 0.0, 1.0])
                    .collect();
                let mut parameters = Parameters::new(4, 1);
                parameters.set_oklab_aces_for_headroom(source, headroom);
                let raw_lab = gpu.render(&lab_pipeline, parameters, &input);
                let raw_rgb = gpu.render(&rgb_pipeline, parameters, &input);
                let encoded = gpu.render(&encoded_pipeline, parameters, &input);
                let actual = raw_lab.as_chunks::<4>().0;
                let saturation: Vec<_> = actual
                    .iter()
                    .map(|lab| lab[1].hypot(lab[2]) / lab[0])
                    .collect();
                let span = saturation[3] - saturation[0];
                let context =
                    format!("h {degrees}, Ln {normalized}, peak {headroom}: {saturation:?}");
                assert!(span > 0.0, "collapsed same-h highlights: {context}");
                assert!(
                    saturation[3] / saturation[0] >= 1.8,
                    "saturation levels converged on their cap: {context}"
                );
                for pair in saturation.windows(2) {
                    assert!(
                        pair[1] - pair[0] >= 0.05 * span,
                        "indistinguishable adjacent ranks: {context}"
                    );
                }
                for (lab, rgb) in actual.iter().zip(raw_rgb.as_chunks::<4>().0) {
                    assert!(lab[..3].iter().all(|channel| channel.is_finite()));
                    assert!((lab[0] / headroom.cbrt() - normalized as f32).abs() < 0.002);
                    assert!(rgb[..3].iter().all(|channel| channel.is_finite()
                        && *channel >= -0.002 * headroom
                        && *channel <= 1.002 * headroom));
                    let error = oklab_aces_validation_hue_error(
                        [1.0, hue[0], hue[1]],
                        [f64::from(lab[0]), f64::from(lab[1]), f64::from(lab[2])],
                    );
                    assert!(
                        error < 0.5,
                        "raw same-h angle changed {error} degrees: {context}"
                    );
                }
                if normalized <= 0.98 {
                    let display_lab = oklab_aces_validation_display_labs(&encoded);
                    let display_saturation: Vec<_> = display_lab
                        .iter()
                        .map(|lab| lab[1].hypot(lab[2]) / lab[0])
                        .collect();
                    for pair in display_saturation.windows(2) {
                        assert!(
                            pair[1] > pair[0],
                            "encoded same-h ranks merged: {context}, {display_saturation:?}"
                        );
                    }
                    for pair in encoded.as_chunks::<4>().0.windows(2) {
                        assert_ne!(
                            &pair[0][..3],
                            &pair[1][..3],
                            "fp16 output merged visible ranks: {context}"
                        );
                    }
                }
            }
        }
    }
    if let Some(directory) = std::env::var_os("DRT_TRAJECTORY_REPORT_DIR") {
        oklab_aces_validation_same_h_report(&gpu, std::path::Path::new(&directory));
    }
}

fn oklab_aces_validation_same_h_report(gpu: &TestGpu, directory: &std::path::Path) {
    let new_shader = builtin_shader(DrtKind::OklabAces);
    let baseline = std::env::var_os("DRT_SATURATION_BASELINE_SHADER_PATH")
        .map(|path| std::fs::read_to_string(path).expect("read requested saturation baseline"));
    let source = crate::oklab_aces::OklabAcesParameters::default();
    let curve = source.curve_for_headroom(1.0);
    let lightness: Vec<_> = (0..64)
        .map(|step| 0.90 + 0.099 * f64::from(step) / 63.0)
        .collect();
    let mut slices = Vec::new();
    for degrees in [194.0_f64, 328.0] {
        let angle = degrees.to_radians();
        let hue = [angle.cos(), angle.sin()];
        let reference = oklab_aces_validation_source_saturation(hue);
        let input: Vec<_> = lightness
            .iter()
            .flat_map(|normalized| {
                let l = oklab_aces_validation_input_lightness(curve, *normalized);
                [0.2, 0.4, 0.6, 0.8]
                    .into_iter()
                    .flat_map(move |rank| [l, (reference * rank) as f32, 0.0, 1.0])
            })
            .collect();
        let mut parameters = Parameters::new(4, 64);
        parameters.set_oklab_aces_for_headroom(source, 1.0);
        let render = |shader: &str, operation| {
            let pipeline = create_pipeline(
                &gpu.device,
                &gpu.layout,
                &oklab_aces_kernel_shader_from_source(
                    shader,
                    &oklab_aces_validation_same_h_expression(hue, operation),
                ),
                "same-h audit readback",
            )
            .unwrap();
            gpu.render(&pipeline, parameters, &input)
        };
        let raw_lab = render(&new_shader, "lab");
        let encoded = render(&new_shader, "encoded");
        let displayed = oklab_aces_validation_display_labs(&encoded);
        let old = baseline.as_ref().map(|shader| {
            let encoded = render(shader,"encoded");
            serde_json::json!({"raw_lab": render(shader,"lab"),"encoded_lab": oklab_aces_validation_display_labs(&encoded),"encoded_rgb": encoded})
        });
        slices.push(serde_json::json!({
            "hue_degrees":degrees,"source_reference_saturation":reference,
            "normalized_output_lightness":lightness,"source_saturation_ranks":[0.2,0.4,0.6,0.8],
            "inputs_L_S":input,"new_raw_lab":raw_lab,"new_encoded_lab":displayed,"new_encoded_rgb":encoded,"baseline":old,
        }));
    }
    std::fs::create_dir_all(directory).unwrap();
    let file = std::fs::File::create(directory.join("oklab_aces_same_h_rank.json")).unwrap();
    serde_json::to_writer(
        file,
        &serde_json::json!({"width":4,"height":64,"headroom":1,"slices":slices}),
    )
    .unwrap();
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_magenta_is_continuous_across_the_tone_join() {
    let gpu = TestGpu::new();
    let pipeline = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &oklab_aces_kernel_shader("rgbToOklab(mapLinearRgb(source))"),
        "magenta tone-join continuity",
    )
    .unwrap();
    let source = crate::oklab_aces::OklabAcesParameters::default();
    let unit_lab = oklab_aces_validation_lab([1.0, 0.0, 1.0]);
    let unit_brightness = unit_lab[0].powi(3);
    let input: Vec<_> = (-4..=4)
        .flat_map(|step| {
            let brightness = f64::from(source.compression_start) * (1.0 + f64::from(step) * 0.002);
            let value = (brightness / unit_brightness) as f32;
            [value, 0.0, value, 1.0]
        })
        .collect();
    for headroom in [1.0, 4.0, 64.0] {
        let mut parameters = Parameters::new(9, 1);
        parameters.set_oklab_aces_for_headroom(source, headroom);
        let output = gpu.render(&pipeline, parameters, &input);
        for pair in output.as_chunks::<4>().0.windows(2) {
            let distance = pair[0][..3]
                .iter()
                .zip(&pair[1][..3])
                .map(|(left, right)| (right - left).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                distance < 0.002,
                "magenta jumps at the join, peak {headroom}: {pair:?}"
            );
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_aces_parameters_refresh_comparison_without_changing_the_left() {
    let gpu = TestGpu::new();
    let mut drt = DrtGpu::new(&gpu.render_state(), crate::image_io::test_pattern(63, 9)).unwrap();
    drt.set_drt(DrtKind::RgbReinhard);
    drt.set_comparison_drt(Some(DrtKind::OklabAces));
    let defaults = crate::oklab_aces::OklabAcesParameters::default();
    let baseline = gpu.read_texture(&drt.comparison.as_ref().unwrap()._output);
    let left_output = gpu.read_texture(&drt.image._output);
    let mut edits = [defaults; 3];
    edits[0].linear_slope = 1.5;
    edits[1].compression_start = 0.4;
    edits[2].set_highlight_reach_ev(14.0, 1.0);
    for edited in edits {
        // Switching back from the experiment can retain its unused uniform
        // fields. Compare the active left state immediately before this edit.
        let left_parameters = bytemuck::bytes_of(&drt.parameters).to_vec();
        drt.set_oklab_aces_parameters(edited);
        let comparison = gpu.read_texture(&drt.comparison.as_ref().unwrap()._output);
        assert_ne!(
            comparison, baseline,
            "parameter edit did not affect image: {edited:?}"
        );
        assert_eq!(bytemuck::bytes_of(&drt.parameters), left_parameters);
        assert_eq!(gpu.read_texture(&drt.image._output), left_output);
        drt.set_drt(DrtKind::OklabAces);
        assert_eq!(comparison, gpu.read_texture(&drt.image._output));
        drt.set_drt(DrtKind::RgbReinhard);
        drt.set_oklab_aces_parameters(defaults);
        assert_eq!(
            gpu.read_texture(&drt.comparison.as_ref().unwrap()._output),
            baseline
        );
    }
}
