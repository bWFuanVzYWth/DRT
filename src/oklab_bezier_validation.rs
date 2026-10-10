// Numerical audit of Bezier replacements for highlight-color retention.
// This compiles helper variants at test runtime; no application DRT is changed.

struct OklabBezierProbe {
    name: &'static str,
    expression: &'static str,
    retention_controls: &'static [f64],
}

fn oklab_bezier_value(controls: &[f64], coordinate: f64) -> f64 {
    // de Casteljau/Bernstein evaluation, independent of the WGSL Horner form.
    let mut work = controls.to_vec();
    for count in (1..controls.len()).rev() {
        for index in 0..count {
            work[index] = (1.0 - coordinate) * work[index] + coordinate * work[index + 1];
        }
    }
    work[0]
}

fn oklab_bezier_oracle(
    rgb: [f64; 3],
    parameters: OklabNeutralParameters,
    peak: f64,
    controls: &[f64],
) -> [f64; 3] {
    // Independent f64 scalar shoulder, Bernstein retention, and color transform.
    let maximum = rgb.into_iter().fold(0.0_f64, f64::max);
    let gain = f64::from(parameters.linear_slope);
    let join = f64::from(parameters.compression_start);
    if maximum <= join {
        return rgb.map(|channel| gain * channel);
    }
    let extent = peak - gain * join;
    let q = gain * (maximum - join) / extent;
    let power = f64::from(parameters.shoulder_power);
    let tail = (1.0 + q / power).powf(-power);
    let mapped_peak = peak - extent * tail;
    if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
        return [mapped_peak; 3];
    }
    let retention = oklab_bezier_value(controls, tail);
    let normalized = rgb.map(|channel| channel / maximum);
    let roots = [
        [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
        [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
        [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
    ]
    .map(|row| {
        let response = row
            .into_iter()
            .zip(normalized)
            .map(|(m, x)| m * x)
            .sum::<f64>();
        1.0 + retention * (response.cbrt() - 1.0)
    });
    let cones = roots.map(|root| root.powi(3));
    let raw = [
        [4.076_741_662_1, -3.307_711_591_3, 0.230_969_929_2],
        [-1.268_438_004_6, 2.609_757_401_1, -0.341_319_396_5],
        [-0.004_196_086_3, -0.703_418_614_7, 1.707_614_701_0],
    ]
    .map(|row| row.into_iter().zip(cones).map(|(m, x)| m * x).sum::<f64>());
    let lift = -raw.into_iter().fold(0.0_f64, f64::min);
    let denominator = raw.into_iter().fold(0.0_f64, f64::max) + lift;
    raw.map(|channel| mapped_peak * (channel + lift) / denominator)
}

fn oklab_bezier_pipeline(gpu: &TestGpu, probe: &OklabBezierProbe) -> wgpu::ComputePipeline {
    const CURRENT: &str = "tail * (1.0 + tail * (3.0 + tail * (-5.0 + 2.0 * tail)))";
    let original = builtin_shader(DrtKind::OklabNeutral);
    assert_eq!(
        original.matches(CURRENT).count(),
        1,
        "retention patch anchor changed"
    );
    let patched = original.replacen(CURRENT, probe.expression, 1);
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/research/oklab_neutral_bezier.wgsl"),
        patched,
    );
    create_pipeline(
        &gpu.device,
        &gpu.layout,
        &oklab_aces_kernel_shader_from_source(&source, "mapLinearRgb(source)"),
        probe.name,
    )
    .unwrap()
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn oklab_bezier_retention_gpu_audit() {
    let probes = [
        OklabBezierProbe {
            name: "current_quartic_k1",
            expression: "bezierHighlightRetentionQuartic(tail, 1.0)",
            retention_controls: &[0.0, 0.25, 1.0, 1.0, 1.0],
        },
        OklabBezierProbe {
            name: "quartic_k2_25",
            expression: "bezierHighlightRetentionQuartic(tail, 2.25)",
            retention_controls: &[0.0, 0.5625, 1.0, 1.0, 1.0],
        },
        OklabBezierProbe {
            name: "cubic_equivalent_quartic_k3",
            expression: "bezierHighlightRetentionCubic(tail)",
            retention_controls: &[0.0, 1.0, 1.0, 1.0],
        },
        OklabBezierProbe {
            name: "quintic_fade_a055_b055",
            expression: "bezierHighlightRetentionQuintic(tail, 0.55, 0.55)",
            retention_controls: &[0.0, 0.45, 0.45, 1.0, 1.0, 1.0],
        },
        OklabBezierProbe {
            name: "quintic_fade_a04_b04",
            expression: "bezierHighlightRetentionQuintic(tail, 0.4, 0.4)",
            retention_controls: &[0.0, 0.6, 0.6, 1.0, 1.0, 1.0],
        },
    ];
    // Verify equivalent degree representations and the join derivatives before
    // comparing their GPU outputs. All candidate control polygons are monotone.
    for step in 0..=1_000 {
        let tail = f64::from(step) / 1_000.0;
        assert!(
            (oklab_bezier_value(&[0.0, 0.25, 1.0, 1.0, 1.0], tail)
                - oklab_bezier_value(&[0.0, 0.2, 0.7, 1.0, 1.0, 1.0], tail))
            .abs()
                < 1e-14
        );
        assert!(
            (oklab_bezier_value(&[0.0, 1.0, 1.0, 1.0], tail)
                - oklab_bezier_value(&[0.0, 0.75, 1.0, 1.0, 1.0], tail))
            .abs()
                < 1e-14
        );
    }
    for probe in &probes {
        let controls = probe.retention_controls;
        assert!(controls.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(controls[0], 0.0);
        assert_eq!(controls[controls.len() - 3..], [1.0, 1.0, 1.0]);
    }
    let gpu = TestGpu::new();
    let pipelines: Vec<_> = probes
        .iter()
        .map(|probe| oklab_bezier_pipeline(&gpu, probe))
        .collect();
    let baseline = oklab_neutral_validation_pipeline(&gpu, "mapLinearRgb(source)");
    let anchors = [
        [0.0, 0.0, 0.0],
        [0.01, 0.02, 0.005],
        [0.05, 0.02, 0.0],
        [0.0, 0.01, 0.05],
        [0.05, 0.0, 0.05],
        [0.1, 0.1, 0.1],
        [0.6, 0.6, 0.6],
        [1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0],
        [256.0, 256.0, 256.0],
        [0.61, 0.0, 0.61],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [4.0, 3.0, 3.0],
        [64.0, 48.0, 64.0],
    ];
    const MIX_SAMPLES: usize = 2_049;
    let mixes = [
        ([0.0, 32.0, 0.0], [4.0; 3]),
        ([0.0, 2_048.0, 2_048.0], [1.0; 3]),
    ];
    let mut colors = anchors.to_vec();
    for (start, end) in mixes {
        for sample in 0..MIX_SAMPLES {
            let t = sample as f32 / (MIX_SAMPLES - 1) as f32;
            colors.push(std::array::from_fn(|channel| {
                (1.0 - t) * start[channel] + t * end[channel]
            }));
        }
    }
    let input: Vec<_> = colors
        .iter()
        .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 1.0])
        .collect();
    let quantized: Vec<_> = colors
        .iter()
        .map(|rgb| rgb.map(|value| f64::from(half::f16::from_f32(value).to_f32())))
        .collect();
    let mut low_join = OklabNeutralParameters {
        compression_start: 0.1,
        ..Default::default()
    };
    low_join.set_highlight_reach_ev(10.390, 1.0);
    let mut report = Vec::new();
    for headroom in [1.0_f32, 4.0, 64.0] {
        for (configuration, tone) in [
            ("default_join", OklabNeutralParameters::default()),
            ("low_join", low_join),
        ] {
            let mut parameters = Parameters::new(colors.len() as u32, 1);
            parameters.set_oklab_neutral_for_headroom(tone, headroom);
            let reference = gpu.render(&baseline, parameters, &input);
            for (probe, pipeline) in probes.iter().zip(&pipelines) {
                let actual = gpu.render(pipeline, parameters, &input);
                let expected: Vec<_> = quantized
                    .iter()
                    .map(|rgb| {
                        oklab_bezier_oracle(
                            *rgb,
                            tone,
                            f64::from(headroom),
                            probe.retention_controls,
                        )
                    })
                    .collect();
                let mut maximum_error = 0.0_f64;
                for (index, pixel) in actual.as_chunks::<4>().0.iter().enumerate() {
                    for channel in 0..3 {
                        assert!(
                            pixel[channel].is_finite()
                                && pixel[channel] >= 0.0
                                && pixel[channel] <= 1.002 * headroom,
                            "unbounded {} / {configuration} / {headroom}: {pixel:?}",
                            probe.name
                        );
                        let error = (f64::from(pixel[channel]) - expected[index][channel]).abs();
                        maximum_error = maximum_error.max(error);
                        assert!(
                            error <= 0.002 * f64::from(headroom) + 0.000_01,
                            "GPU/Bernstein mismatch {} / {configuration} / {headroom} / {index}: {pixel:?} vs {:?}",
                            probe.name,
                            expected[index]
                        );
                    }
                }
                // Colored linear shadows and the complete neutral curve must
                // retain bit-identical GPU output; only colored highlights vary.
                assert_eq!(actual[..10 * 4], reference[..10 * 4]);
                if probe.name == "current_quartic_k1" {
                    assert_eq!(
                        actual, reference,
                        "degree representation changed the baseline"
                    );
                }
                let mut mix_report = Vec::new();
                for (mix_index, (start, end)) in mixes.iter().enumerate() {
                    let offset = anchors.len() + mix_index * MIX_SAMPLES;
                    let mut samples = Vec::new();
                    let mut maximum_chroma = 0.0_f64;
                    let mut maximum_at = 0;
                    let mut initial_chroma = 0.0_f64;
                    let mut oracle_maximum_chroma = 0.0_f64;
                    let mut oracle_initial_chroma = 0.0_f64;
                    for sample in 0..MIX_SAMPLES {
                        let index = offset + sample;
                        let pixel = &actual[index * 4..index * 4 + 3];
                        let lab = oklab_aces_validation_lab([
                            f64::from(pixel[0]),
                            f64::from(pixel[1]),
                            f64::from(pixel[2]),
                        ]);
                        let oracle_lab = oklab_aces_validation_lab(expected[index]);
                        let input_lab = oklab_aces_validation_lab(quantized[index]);
                        let c = lab[1].hypot(lab[2]);
                        let oracle_c = oracle_lab[1].hypot(oracle_lab[2]);
                        if sample == 0 {
                            initial_chroma = c;
                            oracle_initial_chroma = oracle_c;
                        }
                        if c > maximum_chroma {
                            maximum_chroma = c;
                            maximum_at = sample;
                        }
                        oracle_maximum_chroma = oracle_maximum_chroma.max(oracle_c);
                        samples.push(serde_json::json!({
                            "t": sample as f64 / (MIX_SAMPLES - 1) as f64,
                            "input_c": input_lab[1].hypot(input_lab[2]),
                            "gpu_l": lab[0], "gpu_c": c,
                            "oracle_l": oracle_lab[0], "oracle_c": oracle_c,
                        }));
                    }
                    let rebound_ratio = maximum_chroma / initial_chroma;
                    let oracle_rebound_ratio = oracle_maximum_chroma / oracle_initial_chroma;
                    eprintln!(
                        "{} / {configuration} / peak {headroom} / mix {mix_index}: GPU C peak/start {rebound_ratio:.5}, f64 {oracle_rebound_ratio:.5}",
                        probe.name
                    );
                    mix_report.push(serde_json::json!({
                        "start": start, "end": end,
                        "gpu_c_initial": initial_chroma,
                        "gpu_c_maximum": maximum_chroma,
                        "gpu_c_maximum_t": maximum_at as f64 / (MIX_SAMPLES - 1) as f64,
                        "gpu_rebound_ratio": rebound_ratio,
                        "oracle_rebound_ratio": oracle_rebound_ratio,
                        "samples": samples,
                    }));
                }
                report.push(serde_json::json!({
                    "curve": probe.name, "retention_controls": probe.retention_controls,
                    "configuration": configuration, "headroom": headroom,
                    "compression_start": tone.compression_start,
                    "shoulder_power": tone.shoulder_power,
                    "highlight_reach_ev": tone.curve_for_headroom(headroom).highlight_reach_ev(),
                    "maximum_gpu_oracle_rgb_error": maximum_error,
                    "mixes": mix_report,
                }));
            }
        }
    }
    if let Some(directory) = std::env::var_os("DRT_BEZIER_REPORT_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        let file = std::fs::File::create(directory.join("gpu_report.json")).unwrap();
        serde_json::to_writer_pretty(
            file,
            &serde_json::json!({
                "adapter": gpu.adapter.get_info().name,
                "input_domain": "nonnegative linear Rec.709, bypassing AP0 adapter",
                "input_texture": "rgba16float",
                "output_texture": "rgba16float, linear display RGB",
                "samples_per_mix": MIX_SAMPLES,
                "cases": report,
            }),
        )
        .unwrap();
    }
}
