// GPU checks for the original Oklab experiment, independent of display clipping.

fn oklab_aces_kernel_shader(expression: &str) -> String {
    let source = builtin_shader(DrtKind::OklabAces);
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
fn oklab_aces_colored_shadows_are_exactly_linear() {
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
    // AP0 samples include colored shadows and colors outside Rec.709. Both
    // paths apply the same final presentation clip, so equality checks the
    // complete colored shadow result rather than only its neutral axis.
    let input = [
        [0.0, 0.0, 0.0, 1.0],
        [0.000_001, 0.000_002, 0.000_004, 1.0],
        [0.005, 0.002, 0.001, 1.0],
        [0.002, 0.005, 0.001, 1.0],
        [0.001, 0.002, 0.005, 1.0],
        [0.005, 0.0, 0.0, 1.0],
        [0.0, 0.005, 0.0, 1.0],
        [0.0, 0.0, 0.005, 1.0],
        [-0.001, 0.001, 0.002, 1.0],
    ]
    .concat();
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
                assert_eq!(
                    output, expected,
                    "gain {gain}, exposure {exposure}, peak {headroom}"
                );
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
fn oklab_aces_core_preserves_hue_with_fixed_highlight_fade_before_clip() {
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
                // This is the former desaturation=1, protection=0 behavior:
                // independently evaluate the CPU tone curve, then verify the
                // fixed 1-w^2 retention in addition to the L'/L rescaling.
                // Source Lab is read back through fp16, so its inferred L^3
                // and the GPU's unquantized input Lab can differ slightly.
                let chroma_scale = mapped_chroma / chroma;
                let lightness_scale = kept[0] / original[0];
                assert!(lightness_scale > 0.0);
                let brightness = original[0] * original[0] * original[0];
                let output_join = curve.linear_slope * curve.compression_start;
                let progress = if brightness > curve.compression_start {
                    (curve.map_linear(brightness) - output_join) / (curve.output_peak - output_join)
                } else {
                    0.0
                };
                let expected_retention = (1.0 - progress * progress).max(0.0);
                assert!(
                    (chroma_scale / lightness_scale - expected_retention).abs() < 0.01,
                    "fixed chroma fade differs at peak {headroom}: expected {expected_retention}, {original:?} -> {kept:?}"
                );
            }
        }
        // The middle gray, bright white and signed negative neutral retain
        // negligible a/b even though their lightness follows different paths.
        for index in [4, 5, 8] {
            let kept = output.as_chunks::<4>().0[index];
            assert!(kept[1].abs() < 0.001 && kept[2].abs() < 0.001);
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
