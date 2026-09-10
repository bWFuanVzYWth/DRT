use std::path::Path;

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu};
use wgpu::util::DeviceExt;

use crate::{
    distribution::DistributionRenderer,
    image_io::LinearImage,
    tone_curve::{INPUT_MAX_EV, INPUT_MIN_EV, SAMPLE_COUNT, ToneCurveRenderer},
};

const BUILT_OKLAB_REINHARD_SHADER: &str = include_str!("../shaders/oklab_reinhard.wgsl");
const BUILT_AGX_S2O3_SHADER: &str = include_str!("../shaders/agx_s2o3.wgsl");
const BUILT_RGB_LOG_SIGMOID_SHADER: &str = include_str!("../shaders/rgb_log_sigmoid.wgsl");
const BUILT_RGB_REINHARD_SHADER: &str = include_str!("../shaders/rgb_reinhard.wgsl");
const BUILT_RGB_LOG_SHOULDER_SHADER: &str = include_str!("../shaders/rgb_log_shoulder.wgsl");
const BUILT_NONE_SHADER: &str = include_str!("../shaders/none_drt.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrtKind {
    None,
    OklabReinhard,
    AgxS2O3,
    RgbLogSigmoid,
    RgbReinhard,
    RgbLogShoulder,
}

impl DrtKind {
    pub const ALL: [Self; 6] = [
        Self::None,
        Self::OklabReinhard,
        Self::AgxS2O3,
        Self::RgbLogSigmoid,
        Self::RgbReinhard,
        Self::RgbLogShoulder,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::OklabReinhard => "Oklab Reinhard",
            Self::AgxS2O3 => "AgX-S2O3",
            Self::RgbLogSigmoid => "RGB Log Sigmoid",
            Self::RgbReinhard => "RGB Reinhard",
            Self::RgbLogShoulder => "RGB Log Shoulder",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::None => "Display conversion and clipping without tone compression",
            Self::OklabReinhard => {
                "Linear segment and Reinhard shoulder in Oklab L^3, with soft chroma compression along a fixed Oklab hue direction"
            }
            Self::AgxS2O3 => {
                "Reference port by linlin preserving the original AgX-S2O3 structure; not the original author's Python implementation"
            }
            Self::RgbLogSigmoid => {
                "Virtual RGB coordinates with a per-channel log2 toe/shoulder sigmoid and optional HSV hue repair"
            }
            Self::RgbReinhard => {
                "Virtual RGB coordinates with a per-channel linear segment and Reinhard shoulder, followed by optional HSV hue repair"
            }
            Self::RgbLogShoulder => {
                "Virtual RGB coordinates with a per-channel linear segment and log1p sigmoid shoulder, followed by optional HSV hue repair"
            }
        }
    }

    pub fn shader_file(self) -> &'static str {
        match self {
            Self::None => "none_drt.wgsl",
            Self::OklabReinhard => "oklab_reinhard.wgsl",
            Self::AgxS2O3 => "agx_s2o3.wgsl",
            Self::RgbLogSigmoid => "rgb_log_sigmoid.wgsl",
            Self::RgbReinhard => "rgb_reinhard.wgsl",
            Self::RgbLogShoulder => "rgb_log_shoulder.wgsl",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::None => 0,
            Self::OklabReinhard => 1,
            Self::AgxS2O3 => 2,
            Self::RgbLogSigmoid => 3,
            Self::RgbReinhard => 4,
            Self::RgbLogShoulder => 5,
        }
    }

    pub fn uses_log_sigmoid(self) -> bool {
        matches!(self, Self::AgxS2O3 | Self::RgbLogSigmoid)
    }
}

struct DrtPipelines {
    none: wgpu::ComputePipeline,
    oklab_reinhard: wgpu::ComputePipeline,
    agx_s2o3: wgpu::ComputePipeline,
    rgb_log_sigmoid: wgpu::ComputePipeline,
    rgb_reinhard: wgpu::ComputePipeline,
    rgb_log_shoulder: wgpu::ComputePipeline,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReinhardParameters {
    pub gamut_expansion: f32,
    pub input_scale: f32,
    pub compression_start: f32,
    pub highlight_reach_ev: f32,
    pub hue_retention: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReinhardCurveParameters {
    pub compression_start: f32,
    pub compression_start_output: f32,
    pub linear_slope: f32,
    pub shoulder_scale: f32,
    pub curve_peak: f32,
    pub highlight_reach_ev: f32,
}

impl ReinhardCurveParameters {
    pub fn map_linear(self, value: f32) -> f32 {
        if value <= self.compression_start {
            self.linear_slope * value
        } else {
            let distance = value - self.compression_start;
            self.compression_start_output
                + self.linear_slope * distance / (1.0 + self.shoulder_scale * distance)
        }
    }
}

impl Default for ReinhardParameters {
    fn default() -> Self {
        Self {
            gamut_expansion: 0.04,
            // scale*x/(1+scale*x) maps scene-linear 18% gray back to 18%.
            input_scale: 1.0 / (1.0 - 0.18),
            compression_start: 0.5,
            highlight_reach_ev: 10.0,
            hue_retention: 0.75,
        }
    }
}

impl ReinhardParameters {
    pub const fn oklab_default() -> Self {
        Self {
            gamut_expansion: 0.03,
            input_scale: 1.0 / (1.0 - 0.18),
            compression_start: 0.18,
            highlight_reach_ev: 6.5,
            hue_retention: 0.5,
        }
    }

    pub fn constrain(&mut self) {
        self.gamut_expansion = self.gamut_expansion.clamp(0.0, 0.8);
        self.input_scale = self.input_scale.clamp(0.1, 8.0);
        self.compression_start = self
            .compression_start
            .clamp(0.0, self.maximum_compression_start());
        self.highlight_reach_ev = self
            .highlight_reach_ev
            .clamp(self.minimum_highlight_reach_ev(), 20.0);
        self.hue_retention = self.hue_retention.clamp(0.0, 1.0);
    }

    pub fn linear_middle_gray(self) -> f32 {
        let scaled = 0.18 * self.input_scale;
        scaled / (1.0 + scaled)
    }

    pub fn minimum_highlight_reach_ev(self) -> f32 {
        (1.0 / self.linear_middle_gray()).log2() + 0.1
    }

    pub fn maximum_compression_start(self) -> f32 {
        // Leave room for a positive shoulder below SDR white at every input scale.
        0.99 * 0.18 / self.linear_middle_gray()
    }

    pub fn curve_for_headroom(mut self, headroom: f32) -> ReinhardCurveParameters {
        self.constrain();
        let headroom = headroom.clamp(1.0, 64.0);
        let effective_reach = self.highlight_reach_ev + headroom.log2();
        let reach_input = 0.18 * 2.0_f32.powf(effective_reach);
        let linear_slope = self.linear_middle_gray() / 0.18;
        let compression_start_output = linear_slope * self.compression_start;
        let tangent_distance = linear_slope * (reach_input - self.compression_start);
        let output_distance = headroom - compression_start_output;
        let shoulder_extent =
            output_distance * tangent_distance / (tangent_distance - output_distance);
        let curve_peak = compression_start_output + shoulder_extent;
        let shoulder_scale = linear_slope / shoulder_extent;
        ReinhardCurveParameters {
            compression_start: self.compression_start,
            compression_start_output,
            linear_slope,
            shoulder_scale,
            curve_peak,
            highlight_reach_ev: effective_reach,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbLogShoulderParameters {
    // Reuse the Reinhard control conventions, replacing its shoulder curve below.
    pub base: ReinhardParameters,
    pub shoulder_power: f32,
}

impl Default for RgbLogShoulderParameters {
    fn default() -> Self {
        Self {
            base: ReinhardParameters {
                gamut_expansion: 0.04,
                compression_start: 0.18,
                highlight_reach_ev: 8.0,
                hue_retention: 0.5,
                ..ReinhardParameters::default()
            },
            shoulder_power: 5.0,
        }
    }
}

impl RgbLogShoulderParameters {
    pub fn minimum_highlight_reach_ev(self) -> f32 {
        // ln(1 + distance) must exceed one at the requested peak, so the
        // sigmoid coefficient stays positive even at a zero join and HDR headroom.
        ((std::f32::consts::E - 1.0) / self.base.linear_middle_gray()).log2() + 0.1
    }

    pub fn constrain(&mut self) {
        self.base.constrain();
        self.base.highlight_reach_ev = self
            .base
            .highlight_reach_ev
            .clamp(self.minimum_highlight_reach_ev(), 20.0);
        self.shoulder_power = self.shoulder_power.clamp(1.0, 8.0);
    }

    pub fn curve_for_headroom(mut self, headroom: f32) -> RgbLogShoulderCurveParameters {
        self.constrain();
        let output_peak = headroom.clamp(1.0, 64.0);
        let highlight_reach_ev = self.base.highlight_reach_ev + output_peak.log2();
        let linear_slope = self.base.linear_middle_gray() / 0.18;
        let compression_start = self.base.compression_start;
        let join = linear_slope * compression_start;
        let extent = f64::from(output_peak - join);
        let reach = 0.18_f64 * 2.0_f64.powf(f64::from(highlight_reach_ev));
        let distance = f64::from(linear_slope) * (reach - f64::from(compression_start)) / extent;
        // Normalized sigmoid shoulder, using the AgX form: z / (1 + a*z^p)^(1/p).
        // z = ln(1 + m*(x-P)/(H-mP)) gives a unit tangent at the join,
        // supports P=0, and scales the log coordinate with HDR output room.
        // Solve f(reach)=H analytically, preserving Reinhard's reach semantics.
        let shoulder_coefficient =
            (1.0 - distance.ln_1p().powf(-f64::from(self.shoulder_power))) as f32;
        RgbLogShoulderCurveParameters {
            compression_start,
            linear_slope,
            output_peak,
            shoulder_power: self.shoulder_power,
            shoulder_coefficient,
            curve_peak: join
                + (extent
                    * f64::from(shoulder_coefficient).powf(-1.0 / f64::from(self.shoulder_power)))
                    as f32,
            highlight_reach_ev,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RgbLogShoulderCurveParameters {
    pub compression_start: f32,
    pub linear_slope: f32,
    pub output_peak: f32,
    pub shoulder_power: f32,
    pub shoulder_coefficient: f32,
    pub curve_peak: f32,
    pub highlight_reach_ev: f32,
}

impl RgbLogShoulderCurveParameters {
    pub fn map_linear(self, value: f32) -> f32 {
        if value <= self.compression_start {
            return self.linear_slope * value;
        }
        let join = self.linear_slope * self.compression_start;
        let extent = self.output_peak - join;
        let distance = (self.linear_slope * (value - self.compression_start) / extent).ln_1p();
        join + extent
            * distance
            * (1.0 + self.shoulder_coefficient * distance.powf(self.shoulder_power))
                .powf(-1.0 / self.shoulder_power)
    }
}

/// Log-domain toe/shoulder controls shared by the AgX-S2O3 reference port
/// and RGB Log Sigmoid, with separate defaults for each transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogSigmoidParameters {
    pub shadow_ev: f32,
    pub highlight_ev: f32,
    pub output_pivot: f32,
    pub pivot_slope: f32,
    pub toe_power: f32,
    pub shoulder_power: f32,
    pub gamut_compression: f32,
}

impl Default for LogSigmoidParameters {
    fn default() -> Self {
        Self::s2o3_reference()
    }
}

impl LogSigmoidParameters {
    pub const fn s2o3_reference() -> Self {
        Self {
            shadow_ev: -10.0,
            highlight_ev: 6.5,
            output_pivot: 0.5,
            pivot_slope: 2.0,
            toe_power: 3.0,
            shoulder_power: 3.25,
            gamut_compression: 0.2,
        }
    }

    pub const fn rgb_default() -> Self {
        Self {
            shadow_ev: -10.0,
            highlight_ev: 6.5,
            // IEC sRGB OETF(0.18): preserves scene-linear 18% gray on display.
            output_pivot: 0.461_356_13,
            // Matches the local None slope for the default 16.5-stop allocation.
            pivot_slope: 2.460_636_6,
            // Least-squares fit to the None neutral ramp over the visible shadows.
            toe_power: 1.55,
            // TODO: Known issue: the per-channel shoulder can create perceptual
            // banding across high-to-low-saturation highlight transitions. 5.2 is
            // the accepted artistic compromise until the color trajectory is
            // redesigned independently from the tone curve.
            shoulder_power: 5.2,
            gamut_compression: 0.05,
        }
    }

    pub fn input_pivot(self) -> f32 {
        -self.shadow_ev / (self.highlight_ev - self.shadow_ev)
    }

    pub fn minimum_pivot_slope(self) -> f32 {
        let input_pivot = self.input_pivot();
        (self.output_pivot / input_pivot).max((1.0 - self.output_pivot) / (1.0 - input_pivot))
    }

    pub fn output_highlight_ev(self, output_peak: f32) -> f32 {
        self.highlight_ev * (output_peak - self.output_pivot) / (1.0 - self.output_pivot)
    }

    pub fn constrain(&mut self) {
        self.shadow_ev = self.shadow_ev.clamp(-20.0, -1.0);
        self.highlight_ev = self.highlight_ev.clamp(1.0, 20.0);
        self.output_pivot = self.output_pivot.clamp(0.1, 0.9);
        self.toe_power = self.toe_power.clamp(1.0, 8.0);
        self.shoulder_power = self.shoulder_power.clamp(1.0, 8.0);
        self.gamut_compression = self.gamut_compression.clamp(0.0, 0.8);
        self.pivot_slope = self
            .pivot_slope
            .max(self.minimum_pivot_slope() + 1.0e-3)
            .min(32.0);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Parameters {
    exposure_multiplier: f32,
    width: u32,
    height: u32,
    show_anomalies: u32,
    log_sigmoid_minimum_log2: f32,
    log_sigmoid_inverse_dynamic_range: f32,
    log_sigmoid_input_pivot: f32,
    log_sigmoid_output_pivot: f32,
    log_sigmoid_pivot_slope: f32,
    log_sigmoid_toe_power: f32,
    sigmoid_shoulder_power: f32,
    log_sigmoid_gamut_compression: f32,
    log_sigmoid_toe_coefficient: f32,
    sigmoid_shoulder_coefficient: f32,
    rgb_log_sigmoid_black_hue_retention: f32,
    rgb_log_sigmoid_white_hue_retention: f32,
    linear_compression_start: f32,
    log_sigmoid_maximum_log_coordinate: f32,
    log_sigmoid_output_peak: f32,
    rgb_gamut_expansion: f32,
    linear_slope: f32,
    linear_output_peak: f32,
    rgb_hue_retention: f32,
    linear_curve_peak: f32,
}

impl Parameters {
    fn new(width: u32, height: u32) -> Self {
        let mut parameters = Self {
            exposure_multiplier: 1.0,
            width,
            height,
            show_anomalies: 0,
            log_sigmoid_minimum_log2: 0.0,
            log_sigmoid_inverse_dynamic_range: 0.0,
            log_sigmoid_input_pivot: 0.0,
            log_sigmoid_output_pivot: 0.0,
            log_sigmoid_pivot_slope: 0.0,
            log_sigmoid_toe_power: 0.0,
            sigmoid_shoulder_power: 0.0,
            log_sigmoid_gamut_compression: 0.0,
            log_sigmoid_toe_coefficient: 0.0,
            sigmoid_shoulder_coefficient: 0.0,
            rgb_log_sigmoid_black_hue_retention: 1.0,
            rgb_log_sigmoid_white_hue_retention: 0.5,
            linear_compression_start: ReinhardParameters::default().compression_start,
            log_sigmoid_maximum_log_coordinate: 1.0,
            log_sigmoid_output_peak: 1.0,
            rgb_gamut_expansion: ReinhardParameters::default().gamut_expansion,
            linear_slope: 1.0,
            linear_output_peak: 1.0,
            rgb_hue_retention: ReinhardParameters::default().hue_retention,
            linear_curve_peak: 1.0,
        };
        parameters.set_log_sigmoid(LogSigmoidParameters::default());
        parameters.set_reinhard_for_headroom(ReinhardParameters::default(), 1.0);
        parameters
    }

    fn set_log_sigmoid(&mut self, source: LogSigmoidParameters) {
        self.set_log_sigmoid_for_headroom(source, 1.0);
    }

    fn set_log_sigmoid_for_headroom(&mut self, mut source: LogSigmoidParameters, headroom: f32) {
        source.constrain();
        let dynamic_range = source.highlight_ev - source.shadow_ev;
        let input_pivot = source.input_pivot();
        let headroom = headroom.clamp(1.0, 64.0);
        let output_peak = if headroom == 1.0 {
            1.0
        } else {
            extended_srgb_oetf(headroom)
        };
        let shoulder_scale = (output_peak - source.output_pivot) / (1.0 - source.output_pivot);
        let shoulder_extent = (1.0 - input_pivot) * shoulder_scale;

        self.log_sigmoid_minimum_log2 = 0.18_f32.log2() + source.shadow_ev;
        self.log_sigmoid_inverse_dynamic_range = dynamic_range.recip();
        self.log_sigmoid_input_pivot = input_pivot;
        self.log_sigmoid_output_pivot = source.output_pivot;
        self.log_sigmoid_pivot_slope = source.pivot_slope;
        self.log_sigmoid_toe_power = source.toe_power;
        self.sigmoid_shoulder_power = source.shoulder_power;
        self.log_sigmoid_gamut_compression = source.gamut_compression;
        self.log_sigmoid_toe_coefficient = curve_coefficient(
            input_pivot,
            source.output_pivot,
            source.pivot_slope,
            source.toe_power,
        );
        self.sigmoid_shoulder_coefficient = curve_coefficient(
            shoulder_extent,
            output_peak - source.output_pivot,
            source.pivot_slope,
            source.shoulder_power,
        );
        self.log_sigmoid_maximum_log_coordinate = if headroom == 1.0 {
            1.0
        } else {
            input_pivot + shoulder_extent
        };
        self.log_sigmoid_output_peak = output_peak;
    }

    fn set_reinhard_for_headroom(&mut self, mut source: ReinhardParameters, headroom: f32) {
        source.constrain();
        let headroom = headroom.clamp(1.0, 64.0);
        let curve = source.curve_for_headroom(headroom);
        self.rgb_gamut_expansion = source.gamut_expansion;
        self.linear_compression_start = curve.compression_start;
        self.linear_slope = curve.linear_slope;
        self.linear_output_peak = headroom;
        self.rgb_hue_retention = source.hue_retention;
        self.linear_curve_peak = curve.curve_peak;
    }

    fn set_reinhard_for_drt(
        &mut self,
        drt: DrtKind,
        oklab: ReinhardParameters,
        reinhard: ReinhardParameters,
        headroom: f32,
    ) {
        let source = if drt == DrtKind::OklabReinhard {
            oklab
        } else {
            reinhard
        };
        self.set_reinhard_for_headroom(source, direct_output_headroom(drt, headroom));
    }

    fn set_rgb_log_shoulder_for_headroom(
        &mut self,
        mut source: RgbLogShoulderParameters,
        headroom: f32,
    ) {
        source.constrain();
        let curve = source.curve_for_headroom(headroom);
        self.rgb_gamut_expansion = source.base.gamut_expansion;
        self.linear_compression_start = curve.compression_start;
        self.linear_slope = curve.linear_slope;
        self.linear_output_peak = curve.output_peak;
        self.rgb_hue_retention = source.base.hue_retention;
        self.linear_curve_peak = curve.curve_peak;
        self.sigmoid_shoulder_power = curve.shoulder_power;
        self.sigmoid_shoulder_coefficient = curve.shoulder_coefficient;
    }
}

fn extended_srgb_oetf(linear: f32) -> f32 {
    if linear == 1.0 {
        return 1.0;
    }
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn direct_output_headroom(drt: DrtKind, headroom: f32) -> f32 {
    if matches!(
        drt,
        DrtKind::None | DrtKind::RgbReinhard | DrtKind::RgbLogShoulder
    ) {
        headroom
    } else {
        1.0
    }
}

fn curve_coefficient(x_extent: f32, y_extent: f32, slope: f32, power: f32) -> f32 {
    let x_extent = f64::from(x_extent);
    let y_extent = f64::from(y_extent);
    let slope = f64::from(slope);
    let power = f64::from(power);
    (((slope * x_extent / y_extent).powf(power) - 1.0) / x_extent.powf(power)) as f32
}

struct ImageResources {
    _input: wgpu::Texture,
    _output: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    analysis_view: wgpu::TextureView,
    texture_id: egui::TextureId,
    width: u32,
    height: u32,
}

struct CurveResources {
    _input: wgpu::Texture,
    _output: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    output_view: wgpu::TextureView,
}

pub struct DrtGpu {
    render_state: egui_wgpu::RenderState,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: DrtPipelines,
    active_drt: DrtKind,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    curve_uniform: wgpu::Buffer,
    image: ImageResources,
    curve: CurveResources,
    parameters: Parameters,
    log_sigmoid_parameters: LogSigmoidParameters,
    oklab_reinhard_parameters: ReinhardParameters,
    rgb_reinhard_parameters: ReinhardParameters,
    rgb_log_shoulder_parameters: RgbLogShoulderParameters,
    hdr_headroom: f32,
    adapter_name: String,
    backend: wgpu::Backend,
}

impl DrtGpu {
    pub fn new(render_state: &egui_wgpu::RenderState, image: LinearImage) -> Result<Self> {
        let render_state = render_state.clone();
        let device = &render_state.device;
        let info = render_state.adapter.get_info();
        let bind_group_layout = create_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DRT pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipelines = DrtPipelines {
            none: create_pipeline(device, &pipeline_layout, BUILT_NONE_SHADER, "No DRT")?,
            oklab_reinhard: create_pipeline(
                device,
                &pipeline_layout,
                BUILT_OKLAB_REINHARD_SHADER,
                "Oklab Reinhard DRT",
            )?,
            agx_s2o3: create_pipeline(
                device,
                &pipeline_layout,
                BUILT_AGX_S2O3_SHADER,
                "AgX-S2O3 DRT",
            )?,
            rgb_log_sigmoid: create_pipeline(
                device,
                &pipeline_layout,
                BUILT_RGB_LOG_SIGMOID_SHADER,
                "RGB Log Sigmoid DRT",
            )?,
            rgb_reinhard: create_pipeline(
                device,
                &pipeline_layout,
                BUILT_RGB_REINHARD_SHADER,
                "RGB Reinhard DRT",
            )?,
            rgb_log_shoulder: create_pipeline(
                device,
                &pipeline_layout,
                BUILT_RGB_LOG_SHOULDER_SHADER,
                "RGB Log Shoulder DRT",
            )?,
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DRT input sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut parameters = Parameters::new(image.width, image.height);
        parameters.set_reinhard_for_headroom(ReinhardParameters::oklab_default(), 1.0);
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DRT parameters"),
            contents: bytemuck::bytes_of(&parameters),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let curve_parameters = curve_parameters(parameters);
        let curve_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("neutral-axis DRT parameters"),
            contents: bytemuck::bytes_of(&curve_parameters),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let image_resources = create_image_resources(
            &render_state,
            &bind_group_layout,
            &sampler,
            &uniform,
            image,
            None,
        )?;
        let curve_resources =
            create_curve_resources(&render_state, &bind_group_layout, &sampler, &curve_uniform);
        DistributionRenderer::install(
            &render_state,
            &image_resources.analysis_view,
            image_resources.width,
            image_resources.height,
        );
        ToneCurveRenderer::install(&render_state, &curve_resources.output_view);

        let mut gpu = Self {
            render_state,
            bind_group_layout,
            pipeline_layout,
            pipelines,
            active_drt: DrtKind::OklabReinhard,
            sampler,
            uniform,
            curve_uniform,
            image: image_resources,
            curve: curve_resources,
            parameters,
            log_sigmoid_parameters: LogSigmoidParameters::default(),
            oklab_reinhard_parameters: ReinhardParameters::oklab_default(),
            rgb_reinhard_parameters: ReinhardParameters::default(),
            rgb_log_shoulder_parameters: RgbLogShoulderParameters::default(),
            hdr_headroom: 1.0,
            adapter_name: info.name,
            backend: info.backend,
        };
        gpu.dispatch();
        Ok(gpu)
    }

    pub fn set_image(&mut self, image: LinearImage) -> Result<()> {
        let previous_id = self.image.texture_id;
        self.parameters.width = image.width;
        self.parameters.height = image.height;
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.image = create_image_resources(
            &self.render_state,
            &self.bind_group_layout,
            &self.sampler,
            &self.uniform,
            image,
            Some(previous_id),
        )?;
        DistributionRenderer::install(
            &self.render_state,
            &self.image.analysis_view,
            self.image.width,
            self.image.height,
        );
        self.dispatch();
        Ok(())
    }

    pub fn set_exposure(&mut self, exposure_ev: f32) {
        self.parameters.exposure_multiplier = 2.0_f32.powf(exposure_ev);
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_log_sigmoid_parameters(&mut self, parameters: LogSigmoidParameters) {
        self.log_sigmoid_parameters = parameters;
        self.apply_log_sigmoid_parameters();
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_hdr_headroom(&mut self, headroom: f32) {
        let headroom = headroom.clamp(1.0, 64.0);
        if (self.hdr_headroom - headroom).abs() < 1.0e-4 {
            return;
        }
        self.hdr_headroom = headroom;
        self.apply_log_sigmoid_parameters();
        self.apply_linear_shoulder_parameters();
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_rgb_log_sigmoid_hue_retention(&mut self, black: f32, white: f32) {
        self.parameters.rgb_log_sigmoid_black_hue_retention = black.clamp(0.0, 1.0);
        self.parameters.rgb_log_sigmoid_white_hue_retention = white.clamp(0.0, 1.0);
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_rgb_reinhard_parameters(&mut self, mut source: ReinhardParameters) {
        source.constrain();
        self.rgb_reinhard_parameters = source;
        self.apply_linear_shoulder_parameters();
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_oklab_reinhard_parameters(&mut self, mut source: ReinhardParameters) {
        source.constrain();
        self.oklab_reinhard_parameters = source;
        self.apply_linear_shoulder_parameters();
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_rgb_log_shoulder_parameters(&mut self, mut source: RgbLogShoulderParameters) {
        source.constrain();
        self.rgb_log_shoulder_parameters = source;
        self.apply_linear_shoulder_parameters();
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_show_anomalies(&mut self, show: bool) {
        self.parameters.show_anomalies = u32::from(show);
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_drt(&mut self, drt: DrtKind) {
        if self.active_drt != drt {
            self.active_drt = drt;
            self.apply_log_sigmoid_parameters();
            self.apply_linear_shoulder_parameters();
            self.render_state.queue.write_buffer(
                &self.uniform,
                0,
                bytemuck::bytes_of(&self.parameters),
            );
            self.dispatch();
        }
    }

    pub fn active_drt(&self) -> DrtKind {
        self.active_drt
    }

    pub fn reload_shader(&mut self, drt: DrtKind, source: &Path) -> Result<()> {
        let shader = std::fs::read_to_string(source)
            .with_context(|| format!("cannot read {}", source.display()))?;

        let next = create_pipeline(
            &self.render_state.device,
            &self.pipeline_layout,
            &shader,
            drt.label(),
        )
        .with_context(|| format!("cannot reload {}", source.display()))?;
        match drt {
            DrtKind::None => self.pipelines.none = next,
            DrtKind::OklabReinhard => self.pipelines.oklab_reinhard = next,
            DrtKind::AgxS2O3 => self.pipelines.agx_s2o3 = next,
            DrtKind::RgbLogSigmoid => self.pipelines.rgb_log_sigmoid = next,
            DrtKind::RgbReinhard => self.pipelines.rgb_reinhard = next,
            DrtKind::RgbLogShoulder => self.pipelines.rgb_log_shoulder = next,
        }
        if self.active_drt == drt {
            self.dispatch();
        }
        Ok(())
    }

    pub fn width(&self) -> u32 {
        self.image.width
    }
    pub fn height(&self) -> u32 {
        self.image.height
    }
    pub fn texture_id(&self) -> egui::TextureId {
        self.image.texture_id
    }
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }
    pub fn backend(&self) -> wgpu::Backend {
        self.backend
    }

    pub fn output_peak(&self) -> f32 {
        extended_srgb_oetf(self.hdr_headroom)
    }

    pub fn output_headroom(&self) -> f32 {
        self.hdr_headroom
    }

    fn apply_log_sigmoid_parameters(&mut self) {
        if self.active_drt == DrtKind::RgbLogShoulder {
            self.parameters.set_rgb_log_shoulder_for_headroom(
                self.rgb_log_shoulder_parameters,
                self.hdr_headroom,
            );
            return;
        }
        let headroom = if self.active_drt == DrtKind::RgbLogSigmoid {
            self.hdr_headroom
        } else {
            1.0
        };
        self.parameters
            .set_log_sigmoid_for_headroom(self.log_sigmoid_parameters, headroom);
    }

    fn apply_linear_shoulder_parameters(&mut self) {
        if self.active_drt == DrtKind::RgbLogShoulder {
            self.parameters.set_rgb_log_shoulder_for_headroom(
                self.rgb_log_shoulder_parameters,
                self.hdr_headroom,
            );
            return;
        }
        self.parameters.set_reinhard_for_drt(
            self.active_drt,
            self.oklab_reinhard_parameters,
            self.rgb_reinhard_parameters,
            self.hdr_headroom,
        );
    }

    fn dispatch(&mut self) {
        let curve_parameters = curve_parameters(self.parameters);
        self.render_state.queue.write_buffer(
            &self.curve_uniform,
            0,
            bytemuck::bytes_of(&curve_parameters),
        );
        let mut encoder =
            self.render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("DRT encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DRT pass"),
                timestamp_writes: None,
            });
            let pipeline = match self.active_drt {
                DrtKind::None => &self.pipelines.none,
                DrtKind::OklabReinhard => &self.pipelines.oklab_reinhard,
                DrtKind::AgxS2O3 => &self.pipelines.agx_s2o3,
                DrtKind::RgbLogSigmoid => &self.pipelines.rgb_log_sigmoid,
                DrtKind::RgbReinhard => &self.pipelines.rgb_reinhard,
                DrtKind::RgbLogShoulder => &self.pipelines.rgb_log_shoulder,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.image.bind_group, &[]);
            pass.dispatch_workgroups(
                self.image.width.div_ceil(8),
                self.image.height.div_ceil(8),
                1,
            );
            pass.set_bind_group(0, &self.curve.bind_group, &[]);
            pass.dispatch_workgroups(SAMPLE_COUNT.div_ceil(8), 1, 1);
        }
        self.render_state.queue.submit([encoder.finish()]);
    }
}

fn curve_parameters(mut source: Parameters) -> Parameters {
    source.exposure_multiplier = 1.0;
    source.width = SAMPLE_COUNT;
    source.height = 1;
    source.show_anomalies = 0;
    source
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("DRT bindings"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<Parameters>() as u64
                    ),
                },
                count: None,
            },
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    source: &str,
    label: &str,
) -> Result<wgpu::ComputePipeline> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = pollster::block_on(scope.pop()) {
        bail!("wgpu rejected the WGSL pipeline: {error}");
    }
    Ok(pipeline)
}

fn create_image_resources(
    render_state: &egui_wgpu::RenderState,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    image: LinearImage,
    reuse_id: Option<egui::TextureId>,
) -> Result<ImageResources> {
    if image.width == 0 || image.height == 0 {
        bail!("image dimensions cannot be zero");
    }
    let device = &render_state.device;
    let extent = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let input = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DRT AP0 input"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let input_f16: Vec<half::f16> = image
        .rgba
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    render_state.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&input_f16),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(image.width * 8),
            rows_per_image: Some(image.height),
        },
        extent,
    );
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DRT extended-sRGB output"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let input_view = input.create_view(&Default::default());
    let storage_view = output.create_view(&Default::default());
    let analysis_view = output.create_view(&wgpu::TextureViewDescriptor {
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        ..Default::default()
    });
    // The shader writes display-encoded extended-sRGB values. The custom
    // presenter decodes the final fp16 egui target to linear scRGB for HDR.
    let display_view = output.create_view(&wgpu::TextureViewDescriptor {
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DRT bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&storage_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let texture_id = {
        let mut renderer = render_state.renderer.write();
        if let Some(id) = reuse_id {
            renderer.update_egui_texture_from_wgpu_texture(
                device,
                &display_view,
                wgpu::FilterMode::Linear,
                id,
            );
            id
        } else {
            renderer.register_native_texture(device, &display_view, wgpu::FilterMode::Linear)
        }
    };
    Ok(ImageResources {
        _input: input,
        _output: output,
        bind_group,
        analysis_view,
        texture_id,
        width: image.width,
        height: image.height,
    })
}

fn create_curve_resources(
    render_state: &egui_wgpu::RenderState,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> CurveResources {
    let device = &render_state.device;
    let extent = wgpu::Extent3d {
        width: SAMPLE_COUNT,
        height: 1,
        depth_or_array_layers: 1,
    };
    let input = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("neutral-axis AP0 input"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut input_f16 = Vec::with_capacity(SAMPLE_COUNT as usize * 4);
    for index in 0..SAMPLE_COUNT {
        let t = index as f32 / (SAMPLE_COUNT - 1) as f32;
        let ev = INPUT_MIN_EV + (INPUT_MAX_EV - INPUT_MIN_EV) * t;
        let value = 0.18 * 2.0_f32.powf(ev);
        input_f16.extend([
            half::f16::from_f32(value),
            half::f16::from_f32(value),
            half::f16::from_f32(value),
            half::f16::ONE,
        ]);
    }
    render_state.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&input_f16),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SAMPLE_COUNT * 8),
            rows_per_image: Some(1),
        },
        extent,
    );
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("neutral-axis DRT output"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let input_view = input.create_view(&Default::default());
    let storage_view = output.create_view(&Default::default());
    let output_view = output.create_view(&wgpu::TextureViewDescriptor {
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("neutral-axis DRT bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&storage_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    CurveResources {
        _input: input,
        _output: output,
        bind_group,
        output_view,
    }
}

#[cfg(test)]
#[path = "gpu_validation.rs"]
mod validation;

#[cfg(test)]
mod tests {
    use super::{
        DrtKind, LogSigmoidParameters, Parameters, ReinhardParameters, RgbLogShoulderParameters,
        curve_coefficient, curve_parameters, direct_output_headroom, extended_srgb_oetf,
    };
    use crate::tone_curve::SAMPLE_COUNT;

    fn to_expanded_gamut(color: [f32; 3], expansion: f32) -> [f32; 3] {
        let neutral = color[0] * 0.212_005_35 + color[1] * 0.392_182_5 + color[2] * 0.395_812_12;
        color.map(|component| component + expansion * (neutral - component))
    }

    fn from_expanded_gamut(color: [f32; 3], expansion: f32) -> [f32; 3] {
        let neutral = color[0] * 0.212_005_35 + color[1] * 0.392_182_5 + color[2] * 0.395_812_12;
        color.map(|component| (component - expansion * neutral) / (1.0 - expansion))
    }

    fn curve_value(
        value: f32,
        input_pivot: f32,
        output_pivot: f32,
        slope: f32,
        toe_power: f32,
        shoulder_power: f32,
    ) -> f32 {
        let (x_extent, y_extent, power) = if value <= input_pivot {
            (input_pivot, output_pivot, toe_power)
        } else {
            (1.0 - input_pivot, 1.0 - output_pivot, shoulder_power)
        };
        let coefficient = curve_coefficient(x_extent, y_extent, slope, power);
        let distance = value - input_pivot;
        output_pivot
            + slope * distance * (1.0 + coefficient * distance.abs().powf(power)).powf(-1.0 / power)
    }

    #[test]
    fn s2o3_reference_parameters_reproduce_the_original_constants() {
        let source = LogSigmoidParameters::s2o3_reference();
        let parameters = Parameters::new(1280, 720);

        assert!((source.input_pivot() - 0.606_060_6).abs() < 1.0e-7);
        assert_eq!(source.output_pivot, 0.5);
        assert_eq!(source.pivot_slope, 2.0);
        assert_eq!(source.toe_power, 3.0);
        assert_eq!(source.shoulder_power, 3.25);
        assert_eq!(source.gamut_compression, 0.2);
        assert!((parameters.log_sigmoid_minimum_log2 - -12.473_931).abs() < 2.0e-6);
        assert!((parameters.log_sigmoid_inverse_dynamic_range - 1.0 / 16.5).abs() < 1.0e-7);
        assert!((parameters.log_sigmoid_toe_coefficient - 59.507_874).abs() < 1.0e-4);
        assert!((parameters.sigmoid_shoulder_coefficient - 69.862_79).abs() < 1.0e-3);
        assert_eq!(parameters.rgb_log_sigmoid_black_hue_retention, 1.0);
        assert_eq!(parameters.rgb_log_sigmoid_white_hue_retention, 0.5);
    }

    #[test]
    fn rgb_log_sigmoid_keeps_the_reference_curve_defaults_separate() {
        let source = LogSigmoidParameters::rgb_default();
        let mut parameters = Parameters::new(1280, 720);
        parameters.set_log_sigmoid(source);

        assert!((source.output_pivot - 0.461_356_13).abs() < 1.0e-7);
        assert!((source.pivot_slope - 2.460_636_6).abs() < 1.0e-6);
        assert_eq!(source.toe_power, 1.55);
        assert_eq!(source.shoulder_power, 5.2);
        assert_eq!(source.gamut_compression, 0.05);
        assert!((parameters.log_sigmoid_toe_coefficient - 11.219_474).abs() < 1.0e-4);
        assert!((parameters.sigmoid_shoulder_coefficient - 2_568.749_8).abs() < 1.0e-2);
        assert_eq!(parameters.log_sigmoid_maximum_log_coordinate, 1.0);
        assert_eq!(parameters.log_sigmoid_output_peak, 1.0);
        assert_ne!(source, LogSigmoidParameters::s2o3_reference());
    }

    #[test]
    fn rgb_log_sigmoid_extends_only_the_shoulder_for_hdr() {
        let source = LogSigmoidParameters::rgb_default();
        let mut sdr = Parameters::new(1, 1);
        sdr.set_log_sigmoid(source);
        let mut hdr = Parameters::new(1, 1);
        hdr.set_log_sigmoid_for_headroom(source, 4.0);

        assert_eq!(hdr.log_sigmoid_minimum_log2, sdr.log_sigmoid_minimum_log2);
        assert_eq!(
            hdr.log_sigmoid_inverse_dynamic_range,
            sdr.log_sigmoid_inverse_dynamic_range
        );
        assert_eq!(hdr.log_sigmoid_input_pivot, sdr.log_sigmoid_input_pivot);
        assert_eq!(hdr.log_sigmoid_output_pivot, sdr.log_sigmoid_output_pivot);
        assert_eq!(hdr.log_sigmoid_pivot_slope, sdr.log_sigmoid_pivot_slope);
        assert_eq!(
            hdr.log_sigmoid_toe_coefficient,
            sdr.log_sigmoid_toe_coefficient
        );
        assert!(hdr.log_sigmoid_output_peak > 1.0);
        assert!(hdr.log_sigmoid_maximum_log_coordinate > 1.0);

        let distance = hdr.log_sigmoid_maximum_log_coordinate - hdr.log_sigmoid_input_pivot;
        let mapped_peak = hdr.log_sigmoid_output_pivot
            + hdr.log_sigmoid_pivot_slope
                * distance
                * (1.0
                    + hdr.sigmoid_shoulder_coefficient * distance.powf(hdr.sigmoid_shoulder_power))
                .powf(-1.0 / hdr.sigmoid_shoulder_power);
        assert!((mapped_peak - hdr.log_sigmoid_output_peak).abs() < 2.0e-5);
        assert!(
            (source.output_highlight_ev(hdr.log_sigmoid_output_peak)
                - (hdr.log_sigmoid_maximum_log_coordinate - hdr.log_sigmoid_input_pivot)
                    / hdr.log_sigmoid_inverse_dynamic_range)
                .abs()
                < 2.0e-5
        );
    }

    #[test]
    fn neutral_axis_curve_is_independent_from_the_loaded_image_and_exposure() {
        let mut image_parameters = Parameters::new(3840, 2160);
        image_parameters.exposure_multiplier = 32.0;
        image_parameters.show_anomalies = 1;
        image_parameters.set_log_sigmoid_for_headroom(LogSigmoidParameters::rgb_default(), 4.0);
        image_parameters.set_reinhard_for_headroom(
            ReinhardParameters {
                compression_start: 0.42,
                ..ReinhardParameters::default()
            },
            4.0,
        );

        let curve = curve_parameters(image_parameters);
        assert_eq!(curve.width, SAMPLE_COUNT);
        assert_eq!(curve.height, 1);
        assert_eq!(curve.exposure_multiplier, 1.0);
        assert_eq!(curve.show_anomalies, 0);
        assert_eq!(
            curve.log_sigmoid_output_peak,
            image_parameters.log_sigmoid_output_peak
        );
        assert_eq!(
            curve.sigmoid_shoulder_coefficient,
            image_parameters.sigmoid_shoulder_coefficient
        );
        assert_eq!(
            curve.rgb_gamut_expansion,
            image_parameters.rgb_gamut_expansion
        );
        assert_eq!(curve.linear_slope, image_parameters.linear_slope);
        assert_eq!(
            curve.linear_compression_start,
            image_parameters.linear_compression_start
        );
        assert_eq!(
            curve.linear_output_peak,
            image_parameters.linear_output_peak
        );
        assert_eq!(curve.rgb_hue_retention, image_parameters.rgb_hue_retention);
        assert_eq!(curve.linear_curve_peak, image_parameters.linear_curve_peak);
    }

    #[test]
    fn reinhard_defaults_preserve_scene_linear_middle_gray() {
        let source = ReinhardParameters::default();
        let parameters = Parameters::new(1280, 720);

        assert_eq!(source.gamut_expansion, 0.04);
        assert!((source.input_scale - 1.219_512_2).abs() < 1.0e-7);
        assert_eq!(source.highlight_reach_ev, 10.0);
        assert_eq!(source.hue_retention, 0.75);
        assert_eq!(source.compression_start, 0.5);
        assert!((source.linear_middle_gray() - 0.18).abs() < 1.0e-7);
        let curve = source.curve_for_headroom(1.0);
        assert_eq!(parameters.rgb_gamut_expansion, source.gamut_expansion);
        assert_eq!(parameters.linear_slope, curve.linear_slope);
        assert_eq!(parameters.linear_compression_start, 0.5);
        assert!((curve.map_linear(0.18) - 0.18).abs() < 1.0e-7);
        assert_eq!(parameters.linear_output_peak, 1.0);
        assert_eq!(parameters.rgb_hue_retention, 0.75);
        assert_eq!(parameters.linear_curve_peak, curve.curve_peak);
        assert!(curve.curve_peak > 1.001 && curve.curve_peak < 1.002);
        assert_eq!(std::mem::size_of::<Parameters>(), 96);
    }

    #[test]
    fn oklab_and_rgb_reinhard_keep_independent_curves_across_drt_and_hdr_changes() {
        let mut oklab = ReinhardParameters::oklab_default();
        let mut reinhard = ReinhardParameters::default();
        assert_eq!(oklab.compression_start, 0.18);
        assert_eq!(oklab.input_scale, reinhard.input_scale);
        assert_eq!(oklab.highlight_reach_ev, 6.5);
        assert_eq!(reinhard.highlight_reach_ev, 10.0);
        let original_oklab = oklab.curve_for_headroom(1.0);
        assert!(original_oklab.curve_peak > 1.04 && original_oklab.curve_peak < 1.05);

        let mut parameters = Parameters::new(1, 1);
        for drt in [
            DrtKind::OklabReinhard,
            DrtKind::RgbReinhard,
            DrtKind::OklabReinhard,
        ] {
            parameters.set_reinhard_for_drt(drt, oklab, reinhard, 4.0);
            let (source, headroom) = if drt == DrtKind::OklabReinhard {
                (oklab, 1.0)
            } else {
                (reinhard, 4.0)
            };
            let expected = source.curve_for_headroom(headroom);
            assert_eq!(
                parameters.linear_compression_start,
                expected.compression_start
            );
            assert_eq!(parameters.linear_slope, expected.linear_slope);
            assert_eq!(parameters.linear_curve_peak, expected.curve_peak);
            assert_eq!(parameters.linear_output_peak, headroom);

            // Edits to the other DRT must leave the active curve intact.
            if drt == DrtKind::OklabReinhard {
                reinhard.input_scale = 2.0;
                reinhard.compression_start = 0.4;
                reinhard.highlight_reach_ev = 8.0;
            } else {
                oklab.input_scale = 1.5;
                oklab.compression_start = 0.3;
                oklab.highlight_reach_ev = 7.5;
            }
            parameters.set_reinhard_for_drt(drt, oklab, reinhard, 4.0);
            assert_eq!(
                parameters.linear_compression_start,
                expected.compression_start
            );
            assert_eq!(parameters.linear_slope, expected.linear_slope);
            assert_eq!(parameters.linear_curve_peak, expected.curve_peak);
        }
    }

    #[test]
    fn none_uses_the_requested_hdr_headroom_without_tone_mapping() {
        assert_eq!(direct_output_headroom(DrtKind::None, 4.0), 4.0);
        assert_eq!(direct_output_headroom(DrtKind::RgbReinhard, 4.0), 4.0);
        assert_eq!(direct_output_headroom(DrtKind::OklabReinhard, 4.0), 1.0);
        assert_eq!(direct_output_headroom(DrtKind::AgxS2O3, 4.0), 1.0);
        assert_eq!(direct_output_headroom(DrtKind::RgbLogSigmoid, 4.0), 1.0);

        let encoded_peak = extended_srgb_oetf(direct_output_headroom(DrtKind::None, 4.0));
        assert!(encoded_peak > 1.0);
    }

    #[test]
    fn reinhard_reach_hits_the_display_peak_and_preserves_middle_gray() {
        for source in [
            ReinhardParameters::default(),
            ReinhardParameters {
                gamut_expansion: 0.35,
                input_scale: 2.0,
                compression_start: 0.18,
                highlight_reach_ev: 7.0,
                hue_retention: 0.75,
            },
        ] {
            let target_middle_gray = source.linear_middle_gray();
            for headroom in [1.0, 2.0, 4.0, 16.0] {
                let curve = source.curve_for_headroom(headroom);
                let mapped_gray = curve.map_linear(0.18);
                let reach_input = 0.18 * 2.0_f32.powf(curve.highlight_reach_ev);
                let mapped_reach = curve.map_linear(reach_input);
                assert!((mapped_gray - target_middle_gray).abs() < 1.0e-6);
                assert!((mapped_reach - headroom).abs() < 2.0e-5);
                assert!(curve.curve_peak > headroom);
                assert_eq!(curve.map_linear(0.0), 0.0);
                assert!((curve.map_linear(0.09) - 0.5 * target_middle_gray).abs() < 1.0e-6);
                assert!(
                    (curve.shoulder_scale * (curve.curve_peak - curve.compression_start_output)
                        - curve.linear_slope)
                        .abs()
                        < 1.0e-6
                );
                assert!(
                    (curve.highlight_reach_ev - (source.highlight_reach_ev + headroom.log2()))
                        .abs()
                        < 1.0e-6
                );

                let mut parameters = Parameters::new(1, 1);
                parameters.set_reinhard_for_headroom(source, headroom);
                assert_eq!(parameters.rgb_gamut_expansion, source.gamut_expansion);
                assert_eq!(parameters.linear_slope, curve.linear_slope);
                assert_eq!(parameters.linear_compression_start, curve.compression_start);
                assert_eq!(parameters.linear_output_peak, headroom);
                assert_eq!(parameters.rgb_hue_retention, source.hue_retention);
                assert_eq!(parameters.linear_curve_peak, curve.curve_peak);
            }
        }
    }

    #[test]
    fn rgb_log_shoulder_preserves_the_line_tangent_and_reach() {
        for input_scale in [0.1, ReinhardParameters::default().input_scale, 8.0] {
            for compression_start in [0.0, 0.01, 0.18, 0.5, 100.0] {
                for highlight_reach_ev in [0.0, 10.0, 20.0] {
                    for shoulder_power in [1.0, 5.2, 8.0] {
                        let mut source = RgbLogShoulderParameters {
                            base: ReinhardParameters {
                                input_scale,
                                compression_start,
                                highlight_reach_ev,
                                ..ReinhardParameters::default()
                            },
                            shoulder_power,
                        };
                        source.constrain();
                        for headroom in [1.0, 4.0, 64.0] {
                            let curve = source.curve_for_headroom(headroom);
                            let start = curve.compression_start;
                            let join = curve.linear_slope * start;
                            assert_eq!(curve.map_linear(0.0), 0.0);
                            assert_eq!(curve.map_linear(start * 0.5), join * 0.5);
                            assert_eq!(curve.map_linear(start), join);
                            assert!(curve.shoulder_coefficient > 0.0);
                            assert!(curve.curve_peak.is_finite() && curve.curve_peak >= headroom);
                            let step = 1.0e-5 / curve.linear_slope;
                            let right_slope = (curve.map_linear(start + step) - join) / step;
                            assert!((right_slope / curve.linear_slope - 1.0).abs() < 0.02);
                            let reach = 0.18 * 2.0_f32.powf(curve.highlight_reach_ev);
                            assert!((curve.map_linear(reach) / headroom - 1.0).abs() < 2.0e-5);
                            let mut previous = join;
                            for index in 1..=64 {
                                let mapped =
                                    curve.map_linear(start + (reach - start) * index as f32 / 32.0);
                                assert!(
                                    mapped.is_finite() && mapped + 1.0e-5 * headroom >= previous
                                );
                                assert!(mapped <= curve.curve_peak + 1.0e-5 * headroom);
                                previous = mapped;
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rgb_log_shoulder_power_changes_highlights_without_changing_shadows() {
        let source = RgbLogShoulderParameters::default();
        assert_eq!(source.base.gamut_expansion, 0.04);
        assert_eq!(source.base.compression_start, 0.18);
        assert_eq!(source.base.highlight_reach_ev, 8.0);
        assert_eq!(source.base.hue_retention, 0.5);
        assert_eq!(source.shoulder_power, 5.0);
        let soft = RgbLogShoulderParameters {
            shoulder_power: 1.0,
            ..source
        }
        .curve_for_headroom(1.0);
        let hard = RgbLogShoulderParameters {
            shoulder_power: 8.0,
            ..source
        }
        .curve_for_headroom(1.0);
        for input in [0.0, 0.045, 0.09, 0.18] {
            assert_eq!(soft.map_linear(input), hard.map_linear(input));
            assert!((soft.map_linear(input) - input).abs() < 1.0e-6);
        }
        assert!(hard.map_linear(1.0) > soft.map_linear(1.0) + 0.05);
        let hdr = source.curve_for_headroom(4.0);
        assert_eq!(hdr.highlight_reach_ev, source.base.highlight_reach_ev + 2.0);
        assert!((hdr.map_linear(0.18) - source.base.linear_middle_gray()).abs() < 1.0e-6);
    }

    #[test]
    fn adjustable_reinhard_join_preserves_the_line_tangent_and_peak() {
        for input_scale in [0.1, ReinhardParameters::default().input_scale, 8.0] {
            for compression_start in [0.0, 0.01, 0.18, 0.5, 100.0] {
                for highlight_reach_ev in [0.0, 6.5, 20.0] {
                    let mut source = ReinhardParameters {
                        input_scale,
                        compression_start,
                        highlight_reach_ev,
                        ..ReinhardParameters::default()
                    };
                    source.constrain();
                    for headroom in [1.0, 4.0, 64.0] {
                        let curve = source.curve_for_headroom(headroom);
                        let start = curve.compression_start;
                        let join = curve.map_linear(start);
                        assert_eq!(join, curve.compression_start_output);
                        assert!(join < 1.0);
                        assert_eq!(curve.map_linear(0.0), 0.0);
                        assert_eq!(curve.map_linear(start * 0.5), join * 0.5);
                        assert_eq!(curve.linear_slope, source.linear_middle_gray() / 0.18);
                        assert!(curve.curve_peak.is_finite() && curve.curve_peak >= headroom);
                        assert!(curve.shoulder_scale.is_finite() && curve.shoulder_scale > 0.0);

                        // Check the right tangent numerically; the left segment has this exact slope.
                        let step = 1.0e-5 / curve.linear_slope;
                        let right_slope = (curve.map_linear(start + step) - join) / step;
                        assert!((right_slope / curve.linear_slope - 1.0).abs() < 0.02);

                        let reach = 0.18 * 2.0_f32.powf(curve.highlight_reach_ev);
                        assert!(reach > start);
                        assert!((curve.map_linear(reach) / headroom - 1.0).abs() < 2.0e-5);
                        let mut previous = 0.0;
                        for index in 1..=100 {
                            let mapped = curve.map_linear(reach * index as f32 / 100.0);
                            assert!(
                                mapped.is_finite()
                                    && mapped + 2.0 * f32::EPSILON * headroom >= previous,
                                "{source:?}, headroom={headroom}, sample={index}: {previous} -> {mapped}"
                            );
                            previous = mapped;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn virtual_gamut_coordinates_have_an_exact_inverse() {
        let color = [1.0, 0.25, 0.03];
        for expansion in [0.0, 0.2, 0.5, 0.8] {
            let expanded = to_expanded_gamut(color, expansion);
            let restored = from_expanded_gamut(expanded, expansion);
            for channel in 0..3 {
                assert!((restored[channel] - color[channel]).abs() < 2.0e-6);
            }

            let original_span = color[0] - color[2];
            let expanded_span = expanded[0] - expanded[2];
            assert!((expanded_span - original_span * (1.0 - expansion)).abs() < 2.0e-6);

            let gray = [0.18; 3];
            assert_eq!(to_expanded_gamut(gray, expansion), gray);
            assert_eq!(from_expanded_gamut(gray, expansion), gray);
        }
    }

    #[test]
    fn reinhard_parameter_constraints_keep_the_inverse_well_conditioned() {
        let mut source = ReinhardParameters {
            gamut_expansion: 1.0,
            input_scale: -1.0,
            compression_start: -1.0,
            highlight_reach_ev: -10.0,
            hue_retention: 2.0,
        };
        source.constrain();

        assert_eq!(source.gamut_expansion, 0.8);
        assert_eq!(source.input_scale, 0.1);
        assert_eq!(source.compression_start, 0.0);
        assert_eq!(
            source.highlight_reach_ev,
            source.minimum_highlight_reach_ev()
        );
        assert_eq!(source.hue_retention, 1.0);
    }

    #[test]
    fn log_sigmoid_constraints_keep_the_curve_in_its_real_domain() {
        let mut source = LogSigmoidParameters {
            shadow_ev: -0.1,
            highlight_ev: 0.1,
            output_pivot: 1.0,
            pivot_slope: 0.0,
            toe_power: 0.0,
            shoulder_power: 20.0,
            gamut_compression: 1.0,
        };
        source.constrain();

        assert_eq!(source.shadow_ev, -1.0);
        assert_eq!(source.highlight_ev, 1.0);
        assert_eq!(source.output_pivot, 0.9);
        assert_eq!(source.toe_power, 1.0);
        assert_eq!(source.shoulder_power, 8.0);
        assert_eq!(source.gamut_compression, 0.8);
        assert!(source.pivot_slope > source.minimum_pivot_slope());

        let mut parameters = Parameters::new(1, 1);
        parameters.set_log_sigmoid(source);
        assert!(
            parameters.log_sigmoid_toe_coefficient.is_finite()
                && parameters.log_sigmoid_toe_coefficient > 0.0
        );
        assert!(
            parameters.sigmoid_shoulder_coefficient.is_finite()
                && parameters.sigmoid_shoulder_coefficient > 0.0
        );
    }

    #[test]
    fn generated_curve_passes_through_both_endpoints_and_the_pivot() {
        for mut source in [
            LogSigmoidParameters::default(),
            LogSigmoidParameters {
                shadow_ev: -20.0,
                highlight_ev: 1.0,
                output_pivot: 0.9,
                pivot_slope: 20.0,
                toe_power: 1.0,
                shoulder_power: 8.0,
                gamut_compression: 0.8,
            },
        ] {
            source.constrain();
            let input_pivot = source.input_pivot();
            let evaluate = |value| {
                curve_value(
                    value,
                    input_pivot,
                    source.output_pivot,
                    source.pivot_slope,
                    source.toe_power,
                    source.shoulder_power,
                )
            };

            assert!(evaluate(0.0).abs() < 2.0e-5);
            assert!((evaluate(input_pivot) - source.output_pivot).abs() < 1.0e-6);
            assert!((evaluate(1.0) - 1.0).abs() < 2.0e-5);
        }
    }

    #[test]
    fn hsv_hue_repair_uses_the_shortest_wrapped_path() {
        let repair = |mapped: f32, original: f32, amount: f32| {
            let mut offset = original - mapped;
            offset -= (offset + 0.5).floor();
            (mapped + amount * offset).rem_euclid(1.0)
        };

        assert!((repair(0.98, 0.02, 0.5) - 0.0).abs() < 1.0e-6);
        assert!((repair(0.02, 0.98, 0.5) - 0.0).abs() < 1.0e-6);
        assert!((repair(0.25, 0.75, 0.5) - 0.0).abs() < 1.0e-6);
        assert_eq!(repair(0.37, 0.82, 0.0), 0.37);
    }

    #[test]
    fn hsv_hue_retention_interpolates_from_black_to_white_by_value() {
        let retention =
            |black: f32, white: f32, value: f32| black + (white - black) * value.clamp(0.0, 1.0);

        assert_eq!(retention(0.2, 0.8, -0.5), 0.2);
        assert_eq!(retention(0.2, 0.8, 0.0), 0.2);
        assert!((retention(0.2, 0.8, 0.5) - 0.5).abs() < 1.0e-6);
        assert_eq!(retention(0.2, 0.8, 1.0), 0.8);
        assert_eq!(retention(0.2, 0.8, 1.5), 0.8);
        assert_eq!(retention(0.5, 0.5, 0.37), 0.5);
    }
}
