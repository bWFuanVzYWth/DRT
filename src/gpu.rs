use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu};
use wgpu::util::DeviceExt;

use crate::{distribution::DistributionRenderer, image_io::LinearImage};

const BUILT_OKLAB_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oklab_drt.spv"));
const BUILT_AGX_S2O3_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/agx_s2o3.spv"));
const BUILT_AGX_HSV_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/agx_hsv.spv"));
const BUILT_NONE_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/none_drt.spv"));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrtKind {
    None,
    Oklab,
    AgxS2O3,
    AgxHsv,
}

impl DrtKind {
    pub const ALL: [Self; 4] = [Self::None, Self::Oklab, Self::AgxS2O3, Self::AgxHsv];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Oklab => "Oklab",
            Self::AgxS2O3 => "AgX-S2O3",
            Self::AgxHsv => "AgX-HSV",
        }
    }

    pub fn shader_file(self) -> &'static str {
        match self {
            Self::None => "none_drt.slang",
            Self::Oklab => "oklab_drt.slang",
            Self::AgxS2O3 => "agx_s2o3.slang",
            Self::AgxHsv => "agx_hsv.slang",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::None => 0,
            Self::Oklab => 1,
            Self::AgxS2O3 => 2,
            Self::AgxHsv => 3,
        }
    }

    pub fn uses_agx(self) -> bool {
        matches!(self, Self::AgxS2O3 | Self::AgxHsv)
    }
}

struct DrtPipelines {
    none: wgpu::ComputePipeline,
    oklab: wgpu::ComputePipeline,
    agx_s2o3: wgpu::ComputePipeline,
    agx_hsv: wgpu::ComputePipeline,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgxParameters {
    pub shadow_ev: f32,
    pub highlight_ev: f32,
    pub output_pivot: f32,
    pub pivot_slope: f32,
    pub toe_power: f32,
    pub shoulder_power: f32,
    pub gamut_compression: f32,
}

impl Default for AgxParameters {
    fn default() -> Self {
        Self {
            shadow_ev: -10.0,
            highlight_ev: 6.5,
            output_pivot: 0.455,
            pivot_slope: 3.0,
            toe_power: 1.5,
            shoulder_power: 3.0,
            gamut_compression: 0.05,
        }
    }
}

impl AgxParameters {
    pub fn input_pivot(self) -> f32 {
        -self.shadow_ev / (self.highlight_ev - self.shadow_ev)
    }

    pub fn minimum_pivot_slope(self) -> f32 {
        let input_pivot = self.input_pivot();
        (self.output_pivot / input_pivot).max((1.0 - self.output_pivot) / (1.0 - input_pivot))
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
    overexposure: f32,
    width: u32,
    height: u32,
    show_anomalies: u32,
    _padding: [u32; 3],
    agx_minimum_log2: f32,
    agx_inverse_dynamic_range: f32,
    agx_input_pivot: f32,
    agx_output_pivot: f32,
    agx_pivot_slope: f32,
    agx_toe_power: f32,
    agx_shoulder_power: f32,
    agx_gamut_compression: f32,
    agx_toe_a: f32,
    agx_shoulder_a: f32,
    agx_black_hue_retention: f32,
    agx_white_hue_retention: f32,
    agx_black_gamut_onset: f32,
    _agx_padding: [u32; 3],
}

impl Parameters {
    fn new(width: u32, height: u32) -> Self {
        let mut parameters = Self {
            exposure_multiplier: 1.0,
            overexposure: 1.1,
            width,
            height,
            show_anomalies: 0,
            _padding: [0; 3],
            agx_minimum_log2: 0.0,
            agx_inverse_dynamic_range: 0.0,
            agx_input_pivot: 0.0,
            agx_output_pivot: 0.0,
            agx_pivot_slope: 0.0,
            agx_toe_power: 0.0,
            agx_shoulder_power: 0.0,
            agx_gamut_compression: 0.0,
            agx_toe_a: 0.0,
            agx_shoulder_a: 0.0,
            agx_black_hue_retention: 1.0,
            agx_white_hue_retention: 0.5,
            agx_black_gamut_onset: 0.95,
            _agx_padding: [0; 3],
        };
        parameters.set_agx(AgxParameters::default());
        parameters
    }

    fn set_agx(&mut self, mut source: AgxParameters) {
        source.constrain();
        let dynamic_range = source.highlight_ev - source.shadow_ev;
        let input_pivot = source.input_pivot();

        self.agx_minimum_log2 = 0.18_f32.log2() + source.shadow_ev;
        self.agx_inverse_dynamic_range = dynamic_range.recip();
        self.agx_input_pivot = input_pivot;
        self.agx_output_pivot = source.output_pivot;
        self.agx_pivot_slope = source.pivot_slope;
        self.agx_toe_power = source.toe_power;
        self.agx_shoulder_power = source.shoulder_power;
        self.agx_gamut_compression = source.gamut_compression;
        self.agx_toe_a = curve_coefficient(
            input_pivot,
            source.output_pivot,
            source.pivot_slope,
            source.toe_power,
        );
        self.agx_shoulder_a = curve_coefficient(
            1.0 - input_pivot,
            1.0 - source.output_pivot,
            source.pivot_slope,
            source.shoulder_power,
        );
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

pub struct DrtGpu {
    render_state: egui_wgpu::RenderState,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: DrtPipelines,
    active_drt: DrtKind,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    image: ImageResources,
    parameters: Parameters,
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
            oklab: create_pipeline(device, &pipeline_layout, BUILT_OKLAB_SHADER, "Oklab DRT")?,
            agx_s2o3: create_pipeline(
                device,
                &pipeline_layout,
                BUILT_AGX_S2O3_SHADER,
                "AgX-S2O3 DRT",
            )?,
            agx_hsv: create_pipeline(
                device,
                &pipeline_layout,
                BUILT_AGX_HSV_SHADER,
                "AgX-HSV DRT",
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
        let parameters = Parameters::new(image.width, image.height);
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DRT parameters"),
            contents: bytemuck::bytes_of(&parameters),
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
        DistributionRenderer::install(
            &render_state,
            &image_resources.analysis_view,
            image_resources.width,
            image_resources.height,
        );

        let mut gpu = Self {
            render_state,
            bind_group_layout,
            pipeline_layout,
            pipelines,
            active_drt: DrtKind::Oklab,
            sampler,
            uniform,
            image: image_resources,
            parameters,
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

    pub fn set_parameters(&mut self, exposure_ev: f32, overexposure: f32) {
        self.parameters.exposure_multiplier = 2.0_f32.powf(exposure_ev);
        self.parameters.overexposure = overexposure;
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_agx_parameters(&mut self, parameters: AgxParameters) {
        self.parameters.set_agx(parameters);
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_agx_hue_retention(&mut self, black: f32, white: f32) {
        self.parameters.agx_black_hue_retention = black.clamp(0.0, 1.0);
        self.parameters.agx_white_hue_retention = white.clamp(0.0, 1.0);
        self.render_state.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.parameters),
        );
        self.dispatch();
    }

    pub fn set_agx_black_gamut_onset(&mut self, onset: f32) {
        self.parameters.agx_black_gamut_onset = onset.clamp(0.0, 1.0);
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
            self.dispatch();
        }
    }

    pub fn active_drt(&self) -> DrtKind {
        self.active_drt
    }

    pub fn reload_shader(&mut self, drt: DrtKind, source: &Path) -> Result<()> {
        let temporary = std::env::temp_dir().join(format!(
            "drt-{}-{}.spv",
            drt.shader_file().trim_end_matches(".slang"),
            std::process::id()
        ));
        compile_slang(source, &temporary)?;
        let bytes = std::fs::read(&temporary)
            .with_context(|| format!("cannot read {}", temporary.display()))?;
        let _ = std::fs::remove_file(&temporary);

        let next = create_pipeline(
            &self.render_state.device,
            &self.pipeline_layout,
            &bytes,
            drt.label(),
        )?;
        match drt {
            DrtKind::None => self.pipelines.none = next,
            DrtKind::Oklab => self.pipelines.oklab = next,
            DrtKind::AgxS2O3 => self.pipelines.agx_s2o3 = next,
            DrtKind::AgxHsv => self.pipelines.agx_hsv = next,
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

    fn dispatch(&mut self) {
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
                DrtKind::Oklab => &self.pipelines.oklab,
                DrtKind::AgxS2O3 => &self.pipelines.agx_s2o3,
                DrtKind::AgxHsv => &self.pipelines.agx_hsv,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.image.bind_group, &[]);
            pass.dispatch_workgroups(
                self.image.width.div_ceil(8),
                self.image.height.div_ceil(8),
                1,
            );
        }
        self.render_state.queue.submit([encoder.finish()]);
    }
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
                    format: wgpu::TextureFormat::Rgba8Unorm,
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
    spirv: &[u8],
    label: &str,
) -> Result<wgpu::ComputePipeline> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::util::make_spirv(spirv),
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
        bail!("wgpu rejected the SPIR-V pipeline: {error}");
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
        label: Some("DRT sRGB output"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let input_view = input.create_view(&Default::default());
    let storage_view = output.create_view(&Default::default());
    let analysis_view = output.create_view(&wgpu::TextureViewDescriptor {
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        ..Default::default()
    });
    // The shader writes display-encoded sRGB values. egui-wgpu expects ordinary
    // registered textures to return gamma-encoded samples, so this view must stay
    // UNORM. egui handles the target framebuffer's transfer behavior itself.
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

fn compile_slang(source: &Path, output: &Path) -> Result<()> {
    let slangc = find_slangc();
    let result = Command::new(&slangc)
        .args([
            source.as_os_str(),
            "-entry".as_ref(),
            "main".as_ref(),
            "-stage".as_ref(),
            "compute".as_ref(),
            "-target".as_ref(),
            "spirv".as_ref(),
            "-profile".as_ref(),
            "glsl_460".as_ref(),
            "-capability".as_ref(),
            "SPIRV_1_3".as_ref(),
            "-matrix-layout-row-major".as_ref(),
            "-O2".as_ref(),
            "-o".as_ref(),
            output.as_os_str(),
        ])
        .output()
        .with_context(|| format!("cannot launch {:?}", slangc))?;
    if !result.status.success() {
        bail!(
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    Ok(())
}

fn find_slangc() -> PathBuf {
    if let Some(path) = env::var_os("SLANGC") {
        return PathBuf::from(path);
    }

    let executable = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };
    if let Some(path) = env::var_os("PATH").and_then(|path| {
        env::split_paths(&path).find_map(|directory| {
            let candidate = directory.join(executable);
            candidate.is_file().then_some(candidate)
        })
    }) {
        return path;
    }

    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        let bin = if cfg!(windows) { "Bin" } else { "bin" };
        let candidate = PathBuf::from(sdk).join(bin).join(executable);
        if candidate.is_file() {
            return candidate;
        }
    }

    PathBuf::from(executable)
}

#[cfg(test)]
mod tests {
    use super::{AgxParameters, Parameters, curve_coefficient};

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
    fn default_agx_parameters_reproduce_the_tuned_constants() {
        let source = AgxParameters::default();
        let parameters = Parameters::new(1280, 720);

        assert!((source.input_pivot() - 0.606_060_6).abs() < 1.0e-7);
        assert!((parameters.agx_minimum_log2 - -12.473_931).abs() < 2.0e-6);
        assert!((parameters.agx_inverse_dynamic_range - 1.0 / 16.5).abs() < 1.0e-7);
        assert!((parameters.agx_toe_a - 14.810_842).abs() < 1.0e-4);
        assert!((parameters.agx_shoulder_a - 150.434_33).abs() < 1.0e-3);
        assert_eq!(parameters.agx_black_hue_retention, 1.0);
        assert_eq!(parameters.agx_white_hue_retention, 0.5);
        assert_eq!(parameters.agx_black_gamut_onset, 0.95);
    }

    #[test]
    fn agx_constraints_keep_the_curve_in_its_real_domain() {
        let mut source = AgxParameters {
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
        parameters.set_agx(source);
        assert!(parameters.agx_toe_a.is_finite() && parameters.agx_toe_a > 0.0);
        assert!(parameters.agx_shoulder_a.is_finite() && parameters.agx_shoulder_a > 0.0);
    }

    #[test]
    fn generated_curve_passes_through_both_endpoints_and_the_pivot() {
        for mut source in [
            AgxParameters::default(),
            AgxParameters {
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

    #[test]
    fn hsv_gamut_soft_shoulder_is_continuous_and_bounded() {
        let compress = |saturation: f32, onset: f32| {
            if saturation <= onset {
                saturation
            } else {
                let headroom = 1.0 - onset;
                let excess = saturation - onset;
                onset + headroom * excess / (headroom + excess)
            }
        };

        assert_eq!(compress(0.8, 0.9), 0.8);
        assert_eq!(compress(0.9, 0.9), 0.9);
        assert!((compress(0.900_001, 0.9) - 0.900_001).abs() < 1.0e-6);
        assert!(compress(1.0, 0.9) < 1.0);
        assert!(compress(100.0, 0.9) < 1.0);
        assert_eq!(compress(1.0, 1.0), 1.0);
        assert_eq!(compress(2.0, 1.0), 1.0);

        for onset in [0.0, 0.5, 0.95, 1.0] {
            for saturation in [0.0, onset, 1.0, 2.0, 100.0] {
                let compressed = compress(saturation, onset);
                assert!((0.0..=1.0).contains(&compressed));
            }
        }
    }
}
