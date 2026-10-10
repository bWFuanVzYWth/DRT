// Verify that the ACES curve experiment changes only its scalar tone stage.

const ACES_CURVE_TONE_LINE: &str = "    let Y = a20_curve(max(0.0,linear))*100.0;";
const ACES_ORIGINAL_TONE_LINES: &str = "    let f = reference_data[89u]*pow(max(0.0,linear)/(linear+reference_data[87u]),reference_data[84u]);\n    let Y = max(0.0,f*f/(f+reference_data[85u]))*100.0;";

fn aces_curve_restore_original(fragment: &str) -> String {
    let fragment = fragment.replace("\r\n", "\n");
    assert_eq!(fragment.matches(ACES_CURVE_TONE_LINE).count(), 1);
    let helper_start = fragment.find("// Scalar scene Y -> display-linear Y").unwrap();
    let helper_end = fragment.find("fn a20_tonemap_compress(").unwrap();
    assert!(helper_start < helper_end);
    format!("{}{}", &fragment[..helper_start], &fragment[helper_end..])
        .replace(ACES_CURVE_TONE_LINE, ACES_ORIGINAL_TONE_LINES)
}

#[test]
fn aces_curve_preserves_the_pinned_color_pipeline_and_reference_data() {
    let original = crate::reference::fragment(DrtKind::Aces20);
    let experiment = crate::reference::fragment(DrtKind::Aces20Curve);
    let restored = aces_curve_restore_original(experiment);
    // Discard only the derivative's additional provenance header. The complete
    // constants/functions body must then be identical to the pinned ACES port.
    assert_eq!(
        original.replace("\r\n", "\n").split_once("const A20_AP0_AP1").unwrap().1,
        restored.split_once("const A20_AP0_AP1").unwrap().1,
        "the experiment changed an ACES color stage in addition to its tone curve"
    );
    let entry = include_str!("../references/entries/aces_20.md");
    assert!(experiment.contains("SPDX-License-Identifier: Apache-2.0"));
    assert!(experiment.contains("069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80"));
    assert!(entry.contains("069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80"));
    assert!(!DrtKind::Aces20Curve.is_reference());
    assert!(DrtKind::Aces20Curve.supports_hdr());
    for headroom in [1.0, 4.0, 64.0] {
        let expected = crate::aces2_data::generate(headroom);
        assert_eq!(crate::reference::data(DrtKind::Aces20, headroom), expected);
        assert_eq!(crate::reference::data(DrtKind::Aces20Curve, headroom), expected);
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn aces_curve_restoring_original_tone_is_bit_identical_to_aces20() {
    let gpu = TestGpu::new();
    let original = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &builtin_shader(DrtKind::Aces20),
        "ACES 2.0 unchanged color reference",
    )
    .unwrap();
    let restored_source = crate::reference::compose(&aces_curve_restore_original(
        crate::reference::fragment(DrtKind::Aces20Curve),
    ));
    let restored = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &restored_source,
        "ACES curve restored official tone",
    )
    .unwrap();
    let input = [
        [0.0, 0.0, 0.0, 1.0],
        [0.001, 0.001, 0.001, 1.0],
        [0.18, 0.18, 0.18, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 1.0],
        [256.0, 256.0, 256.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [16.0, 1.0, 8.0, 1.0],
        [1.0, 16.0, 8.0, 1.0],
        [8.0, 1.0, 16.0, 1.0],
        [-1.0, -1.0, -1.0, 1.0],
        [-0.5, 1.0, 0.2, 1.0],
        [1.0, -0.5, 0.2, 1.0],
        [0.2, 1.0, -0.5, 1.0],
    ]
    .concat();
    for headroom in [1.0, 4.0, 64.0] {
        for controls in [
            OklabAcesParameters::default(),
            OklabAcesParameters {
                linear_slope: 0.5,
                compression_start: 0.75,
                shoulder_power: 0.8,
            },
        ] {
            let mut parameters = Parameters::new((input.len() / 4) as u32, 1);
            parameters.set_oklab_aces_for_headroom(controls, headroom);
            let expected = gpu.render_kind(
                &original,
                parameters,
                &input,
                DrtKind::Aces20,
                headroom,
            );
            let actual = gpu.render_kind(
                &restored,
                parameters,
                &input,
                DrtKind::Aces20Curve,
                headroom,
            );
            assert_eq!(actual, expected, "color pipeline changed at peak {headroom}");
        }
    }
}

fn aces_curve_f64(input: f64, controls: OklabAcesParameters, peak: f64) -> f64 {
    let gain = f64::from(controls.linear_slope);
    let join = f64::from(controls.compression_start);
    if input <= join {
        return gain * input;
    }
    let extent = peak - gain * join;
    let power = f64::from(controls.shoulder_power);
    let q = gain * (input - join) / extent;
    peak - extent * (1.0 + q / power).powf(-power)
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn aces_curve_scalar_matches_independent_f64_and_display_peak_reach() {
    let gpu = TestGpu::new();
    let scalar_source = oklab_aces_kernel_shader_from_source(
        &builtin_shader(DrtKind::Aces20Curve),
        "vec3f(a20_curve(max(source.r,0.0)))",
    );
    let scalar = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &scalar_source,
        "ACES curve independent scalar oracle",
    )
    .unwrap();
    let neutral_source = oklab_aces_kernel_shader_from_source(
        &builtin_shader(DrtKind::Aces20Curve),
        "reference_transform(source, parameters.linearOutputPeak)",
    );
    let neutral = create_pipeline(
        &gpu.device,
        &gpu.layout,
        &neutral_source,
        "ACES curve full neutral pipeline",
    )
    .unwrap();
    for headroom in [1.0_f32, 4.0, 64.0] {
        for controls in [
            OklabAcesParameters::default(),
            OklabAcesParameters {
                linear_slope: 0.5,
                compression_start: 0.75,
                shoulder_power: 0.8,
            },
        ] {
            let curve = controls.curve_for_headroom(headroom);
            let reach_input = 0.18 * 2.0_f32.powf(curve.highlight_reach_ev());
            // Divide the uploaded values and apply exposure in WGSL, avoiding
            // fp16 overflow for the long HDR shoulder's scene-white reach.
            let samples = [
                0.0,
                0.001,
                controls.compression_start * 0.5,
                controls.compression_start,
                controls.compression_start + 0.0001,
                0.18,
                1.0,
                16.0,
                reach_input,
                reach_input * 2.0,
            ];
            let upload_gain = 256.0;
            let input: Vec<f32> = samples
                .into_iter()
                .flat_map(|value| [value / upload_gain; 3].into_iter().chain([1.0]))
                .collect();
            let mut parameters = Parameters::new(samples.len() as u32, 1);
            parameters.exposure_multiplier = upload_gain;
            parameters.set_oklab_aces_for_headroom(controls, headroom);
            let actual = gpu.render_kind(
                &scalar,
                parameters,
                &input,
                DrtKind::Aces20Curve,
                headroom,
            );
            for (index, pixel) in actual.as_chunks::<4>().0.iter().enumerate() {
                let uploaded = half::f16::from_f32(input[index * 4]).to_f32();
                let expected = aces_curve_f64(
                    f64::from(uploaded * upload_gain),
                    controls,
                    f64::from(headroom),
                );
                for channel in &pixel[..3] {
                    assert!(
                        (f64::from(*channel) - expected).abs()
                            < f64::from(headroom) * 0.002 + 0.000_01,
                        "scalar mismatch at sample {index}, peak {headroom}, controls {controls:?}"
                    );
                }
            }
            assert!((actual[8 * 4] / headroom - 0.98).abs() < 0.002);
            // The official AP1 input limiter is deliberately retained; verify
            // final neutral response only before that limiter takes effect.
            let data = crate::reference::data(DrtKind::Aces20Curve, headroom);
            let neutral_input: Vec<f32> = samples
                .into_iter()
                .filter(|value| *value < 0.5 * data[90])
                .flat_map(|value| [value / upload_gain; 3].into_iter().chain([1.0]))
                .collect();
            parameters.width = (neutral_input.len() / 4) as u32;
            let output = gpu.render_kind(
                &neutral,
                parameters,
                &neutral_input,
                DrtKind::Aces20Curve,
                headroom,
            );
            for (pixel, input) in output
                .as_chunks::<4>()
                .0
                .iter()
                .zip(neutral_input.as_chunks::<4>().0)
            {
                let uploaded = half::f16::from_f32(input[0]).to_f32() * upload_gain;
                let expected = aces_curve_f64(
                    f64::from(uploaded),
                    controls,
                    f64::from(headroom),
                );
                for channel in &pixel[..3] {
                    assert!(
                        (f64::from(*channel) - expected).abs()
                            < f64::from(headroom) * 0.006 + 0.000_02,
                        "full neutral mismatch: input {uploaded}, expected {expected}, actual {pixel:?}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn aces_curve_controls_compare_hdr_and_reload_are_independent() {
    let gpu = TestGpu::new();
    let image = crate::image_io::color_trajectory_pattern(129, 61);
    let mut drt = DrtGpu::new(&gpu.render_state(), image.clone()).unwrap();
    drt.set_drt(DrtKind::OklabAces);
    drt.set_comparison_drt(Some(DrtKind::Aces20Curve));
    let defaults = OklabAcesParameters::default();
    let baseline = gpu.read_texture(&drt.comparison.as_ref().unwrap()._output);
    let left_output = gpu.read_texture(&drt.image._output);
    let mut edits = [defaults; 3];
    edits[0].linear_slope = 1.5;
    edits[1].compression_start = 0.6;
    edits[2].set_highlight_reach_ev(6.0, 1.0);
    for edited in edits {
        let left_parameters = bytemuck::bytes_of(&drt.parameters).to_vec();
        drt.set_aces20_curve_parameters(edited);
        let comparison = gpu.read_texture(&drt.comparison.as_ref().unwrap()._output);
        assert_ne!(comparison, baseline, "control has no effect: {edited:?}");
        assert_eq!(bytemuck::bytes_of(&drt.parameters), left_parameters);
        assert_eq!(gpu.read_texture(&drt.image._output), left_output);
        assert_eq!(drt.oklab_aces_parameters, defaults);
        drt.set_drt(DrtKind::Aces20Curve);
        assert_eq!(gpu.read_texture(&drt.image._output), comparison);
        assert_eq!(drt.aces20_curve_parameters, edited);
        drt.set_drt(DrtKind::OklabAces);
        drt.set_aces20_curve_parameters(defaults);
        assert_eq!(
            gpu.read_texture(&drt.comparison.as_ref().unwrap()._output),
            baseline
        );
    }
    drt.set_drt(DrtKind::Aces20Curve);
    drt.set_comparison_drt(Some(DrtKind::Aces20));
    for headroom in [1.0, 4.0, 64.0] {
        drt.set_hdr_headroom(headroom);
        assert_eq!(drt.active_output_headroom(), headroom);
        let reference_before = gpu.read_texture(&drt.comparison.as_ref().unwrap()._output);
        let output = gpu.read_texture(&drt.image._output);
        let expected = gpu.render_kind(
            drt.pipelines.get(DrtKind::Aces20Curve),
            drt.parameters,
            &image.rgba,
            DrtKind::Aces20Curve,
            headroom,
        );
        assert_eq!(output, expected, "HDR reference data did not refresh");
        let encoded_peak = extended_srgb_oetf(headroom);
        assert!(output.as_chunks::<4>().0.iter().all(|pixel| {
            pixel[3] == 1.0
                && pixel[..3]
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0 && *value <= encoded_peak + 0.005)
        }));
        let file = std::env::temp_dir().join(format!(
            "drt-aces-curve-reload-{}-{headroom}.wgsl",
            std::process::id()
        ));
        std::fs::write(&file, "invalid WGSL").unwrap();
        assert!(drt.reload_shader(DrtKind::Aces20Curve, &file).is_err());
        assert_eq!(gpu.read_texture(&drt.image._output), output);
        std::fs::write(
            &file,
            crate::reference::fragment(DrtKind::Aces20Curve)
                .replace(ACES_CURVE_TONE_LINE, "    let Y = a20_curve(max(0.0,linear))*50.0;"),
        )
        .unwrap();
        drt.reload_shader(DrtKind::Aces20Curve, &file).unwrap();
        assert_ne!(gpu.read_texture(&drt.image._output), output);
        assert_eq!(
            gpu.read_texture(&drt.comparison.as_ref().unwrap()._output),
            reference_before,
            "experiment reload changed the official ACES reference"
        );
        std::fs::write(&file, crate::reference::fragment(DrtKind::Aces20Curve)).unwrap();
        drt.reload_shader(DrtKind::Aces20Curve, &file).unwrap();
        assert_eq!(gpu.read_texture(&drt.image._output), output);
        std::fs::remove_file(file).unwrap();
    }
}
