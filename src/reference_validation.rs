//! Real GPU comparisons against independent, pinned upstream reference values.
//! Run with: cargo +stable test reference_validation -- --ignored --nocapture
use super::*;
use serde_json::Value;

struct ReferenceGpu {
    state: egui_wgpu::RenderState,
}

impl ReferenceGpu {
    fn new() -> Self {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("reference GPU tests require an available wgpu adapter");
        eprintln!("Reference validation GPU: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let target_format = wgpu::TextureFormat::Rgba16Float;
        let renderer = egui_wgpu::Renderer::new(&device, target_format, Default::default());
        Self {
            state: egui_wgpu::RenderState {
                adapter: adapter.clone(),
                available_adapters: vec![adapter],
                instance,
                device,
                queue,
                target_format,
                renderer: std::sync::Arc::new(egui::epaint::mutex::RwLock::new(renderer)),
                surface_config: egui_wgpu::SurfaceConfig::HIGH_THROUGHPUT,
            },
        }
    }

    fn read_output(&self, drt: &DrtGpu) -> Vec<[f32; 4]> {
        let texture = &drt.image._output;
        let extent = texture.size();
        let row_bytes = (extent.width * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reference golden readback"),
            size: u64::from(row_bytes * extent.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .state
            .device
            .create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(extent.height),
                },
            },
            extent,
        );
        self.state.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.state
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        bytes
            .chunks_exact(row_bytes as usize)
            .flat_map(|row| {
                row[..(extent.width * 8) as usize]
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|pixel| {
                        std::array::from_fn(|channel| {
                            half::f16::from_le_bytes([pixel[channel * 2], pixel[channel * 2 + 1]])
                                .to_f32()
                        })
                    })
            })
            .collect()
    }
}

fn kind_for_algorithm(algorithm: &str) -> DrtKind {
    match algorithm {
        "pbr_neutral" => DrtKind::PbrNeutral,
        "hable" => DrtKind::Hable,
        "lottes" => DrtKind::Lottes,
        "uchimura" => DrtKind::Uchimura,
        "aces_fitted" => DrtKind::AcesFitted,
        "opendrt" => DrtKind::OpenDrt,
        "fidelityfx_lpm" => DrtKind::FidelityFxLpm,
        "aces_13" => DrtKind::Aces13,
        "aces_20" => DrtKind::Aces20,
        "blender_agx" => DrtKind::BlenderAgx,
        "filmic" => DrtKind::Filmic,
        other => panic!("unregistered reference fixture algorithm: {other}"),
    }
}

fn triplet(value: &Value) -> [f32; 3] {
    let values = value.as_array().expect("fixture RGB must be an array");
    assert_eq!(values.len(), 3, "fixture RGB must have three components");
    std::array::from_fn(|channel| {
        let number = values[channel]
            .as_f64()
            .expect("fixture RGB must be numeric") as f32;
        assert!(
            number.is_finite(),
            "oracle fixture contains a nonfinite value"
        );
        number
    })
}

fn rec709_to_ap0(rgb: [f32; 3]) -> [f32; 3] {
    // Invert the existing workbench adapter in f64 before f16 upload. The
    // oracle evaluates the tonemapper independently; this is only its input
    // conversion for fixtures originally specified in Rec.709 working RGB.
    let matrix = [
        [2.521_400_888_6, -1.133_995_749_4, -0.387_561_856_8],
        [-0.276_214_061_6, 1.372_595_566_3, -0.096_282_355_7],
        [-0.015_320_200_1, -0.152_992_561_8, 1.168_387_199_6],
    ];
    let [x, y, z] = matrix;
    let determinant = x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
        + x[2] * (y[0] * z[1] - y[1] * z[0]);
    let inverse = [
        [
            y[1] * z[2] - y[2] * z[1],
            x[2] * z[1] - x[1] * z[2],
            x[1] * y[2] - x[2] * y[1],
        ],
        [
            y[2] * z[0] - y[0] * z[2],
            x[0] * z[2] - x[2] * z[0],
            x[2] * y[0] - x[0] * y[2],
        ],
        [
            y[0] * z[1] - y[1] * z[0],
            x[1] * z[0] - x[0] * z[1],
            x[0] * y[1] - x[1] * y[0],
        ],
    ];
    inverse.map(|row| {
        ((row[0] * f64::from(rgb[0]) + row[1] * f64::from(rgb[1]) + row[2] * f64::from(rgb[2]))
            / determinant) as f32
    })
}

fn fixture(path: &str) -> Value {
    let absolute = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let text = std::fs::read_to_string(&absolute)
        .unwrap_or_else(|error| panic!("missing upstream fixture {}: {error}", absolute.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("invalid upstream fixture {}: {error}", absolute.display()))
}

fn check_fixture(path: &str, required_algorithms: &[DrtKind]) {
    let document = fixture(path);
    assert_eq!(document["format_version"], 1);
    let records = document["records"].as_array().expect("fixture records");
    let gpu = ReferenceGpu::new();
    let mut drt = DrtGpu::new(
        &gpu.state,
        LinearImage {
            width: 1,
            height: 1,
            rgba: vec![0.0, 0.0, 0.0, 1.0],
        },
    )
    .expect("all registered reference and research shaders must compile");
    let mut covered = Vec::new();
    let mut checked = 0;
    for record in records {
        let algorithm = record["algorithm"].as_str().expect("fixture algorithm");
        let kind = kind_for_algorithm(algorithm);
        let headroom = record["headroom"].as_f64().expect("fixture headroom") as f32;
        assert!(
            (1.0..=64.0).contains(&headroom),
            "fixture must use reachable display headroom"
        );
        let cases = record["vectors"].as_array().expect("fixture vectors");
        assert!(
            cases.len() >= 8,
            "oracle must cover shadows, colors and highlights"
        );
        let rgba = cases
            .iter()
            .flat_map(|case| {
                let ap0 = if case["input_ap0"].is_array() {
                    triplet(&case["input_ap0"])
                } else {
                    rec709_to_ap0(triplet(&case["input_rec709"]))
                };
                [ap0[0], ap0[1], ap0[2], 1.0]
            })
            .collect();
        drt.set_image(LinearImage {
            width: cases.len() as u32,
            height: 1,
            rgba,
        })
        .unwrap();
        drt.set_drt(kind);
        drt.set_hdr_headroom(headroom);
        drt.set_show_anomalies(false);
        let actual = gpu.read_output(&drt);
        assert_eq!(
            actual.len(),
            cases.len(),
            "every upstream vector must be rendered"
        );
        let active_peak = drt.active_output_headroom();
        assert_eq!(
            active_peak,
            if kind.supports_hdr() { headroom } else { 1.0 }
        );
        let mut minimum_expected = f32::INFINITY;
        let mut maximum_expected = 0.0_f32;
        for (index, (pixel, case)) in actual.iter().zip(cases).enumerate() {
            let linear = triplet(&case["expected_linear_rec709"]);
            assert!(
                pixel.iter().all(|value| value.is_finite()),
                "{algorithm} case {index} produced a nonfinite pixel"
            );
            assert_eq!(pixel[3], 1.0, "reference output must remain opaque");
            for channel in 0..3 {
                let expected = extended_srgb_oetf(linear[channel].clamp(0.0, active_peak));
                minimum_expected = minimum_expected.min(expected);
                maximum_expected = maximum_expected.max(expected);
                // The runtime uses f16 textures for both scene-linear input
                // and display output. Quantized upstream fixtures remove
                // input error; residual tolerance covers f32 evaluation and
                // half-precision output, including HDR values above 1.
                let tolerance = 0.0015 + 0.001 * expected.abs();
                assert!(
                    (pixel[channel] - expected).abs() <= tolerance,
                    "{algorithm} headroom {headroom}, case {index} {}, channel {channel}: GPU {} vs upstream {expected} (tolerance {tolerance}, linear {})",
                    case["name"].as_str().unwrap_or(""),
                    pixel[channel],
                    linear[channel]
                );
            }
        }
        assert!(
            maximum_expected - minimum_expected > 0.1,
            "oracle must distinguish dark and bright output"
        );
        if headroom > 1.0 && kind.supports_hdr() {
            assert!(
                actual
                    .iter()
                    .any(|pixel| pixel[..3].iter().any(|&value| value > 1.0)),
                "{algorithm} failed to render HDR highlights"
            );
        }
        if !kind.supports_hdr() {
            // An SDR reference remains its original preset when the user
            // enables HDR presentation; it must not become a scaled SDR look.
            drt.set_hdr_headroom(4.0);
            assert_eq!(drt.active_output_headroom(), 1.0);
            assert_eq!(
                gpu.read_output(&drt),
                actual,
                "{algorithm} SDR reference changed when presentation headroom changed"
            );
        }
        if !covered.contains(&kind) {
            covered.push(kind);
        }
        checked += cases.len();
    }
    for kind in required_algorithms {
        assert!(
            covered.contains(kind),
            "missing upstream oracle for {}",
            kind.label()
        );
    }
    assert_eq!(
        covered.len(),
        required_algorithms.len(),
        "unexpected oracle algorithms"
    );
    eprintln!("Validated {checked} upstream GPU vectors from {path}");
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn reference_simple_ports_match_upstream_vectors() {
    check_fixture(
        "references/tests/simple_reference_vectors.json",
        &[
            DrtKind::PbrNeutral,
            DrtKind::Hable,
            DrtKind::Lottes,
            DrtKind::Uchimura,
            DrtKind::AcesFitted,
        ],
    );
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn reference_hdr_ports_match_upstream_vectors() {
    check_fixture(
        "references/validation/hdr_reference_vectors.json",
        &[DrtKind::OpenDrt, DrtKind::FidelityFxLpm],
    );
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn reference_aces_ports_match_official_ocio_vectors() {
    check_fixture(
        "references/validation/aces_reference_vectors.json",
        &[DrtKind::Aces13, DrtKind::Aces20],
    );
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn reference_lut_ports_match_official_ocio_vectors() {
    check_fixture(
        "references/validation/lut_reference_vectors.json",
        &[DrtKind::BlenderAgx, DrtKind::Filmic],
    );
}

#[test]
fn reference_catalogue_has_separate_shaders_and_pinned_remote_records() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(DrtKind::REFERENCES.len(), 12);
    assert_eq!(DrtKind::RESEARCH.len(), 6);
    for kind in DrtKind::REFERENCES {
        assert!(kind.shader_file().starts_with("reference/"));
        assert!(!DrtKind::RESEARCH.contains(&kind));
        assert!(kind.source_url().unwrap().starts_with("https://"));
        let filename = Path::new(kind.shader_file())
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        let entry = root.join(format!("references/entries/{filename}.md"));
        let text = std::fs::read_to_string(&entry).unwrap_or_else(|error| {
            panic!(
                "missing traceable source record {}: {error}",
                entry.display()
            )
        });
        assert!(
            text.contains("https://"),
            "source record must link a remote reference"
        );
        assert!(
            text.split(|c: char| !c.is_ascii_hexdigit())
                .any(|token| token.len() == 40 || token.len() == 64),
            "source record must pin an immutable commit or source snapshot hash: {}",
            entry.display()
        );
        assert!(root.join("shaders").join(kind.shader_file()).is_file());
        assert!(
            root.join("THIRD_PARTY_LICENSES/references")
                .join(filename)
                .is_dir(),
            "missing license/attribution directory for {}",
            kind.label()
        );
    }
    for kind in DrtKind::RESEARCH {
        assert!(kind.shader_file().starts_with("research/"));
    }
    assert!(
        std::fs::read_to_string(root.join(".gitignore"))
            .unwrap()
            .lines()
            .any(|line| line.trim() == "/third_party/")
    );
}
