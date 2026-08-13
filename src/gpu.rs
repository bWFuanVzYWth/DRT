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

const BUILT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oklab_drt.spv"));

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Parameters {
    exposure_multiplier: f32,
    overexposure: f32,
    width: u32,
    height: u32,
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
    pipeline: wgpu::ComputePipeline,
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
            label: Some("Oklab DRT pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = create_pipeline(device, &pipeline_layout, BUILT_SHADER)?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Oklab DRT input sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let parameters = Parameters {
            exposure_multiplier: 1.0,
            overexposure: 1.1,
            width: image.width,
            height: image.height,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Oklab DRT parameters"),
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
            pipeline,
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

    pub fn reload_shader(&mut self, source: &Path) -> Result<()> {
        let temporary = std::env::temp_dir().join(format!("drt-oklab-{}.spv", std::process::id()));
        compile_slang(source, &temporary)?;
        let bytes = std::fs::read(&temporary)
            .with_context(|| format!("cannot read {}", temporary.display()))?;
        let _ = std::fs::remove_file(&temporary);

        let next = create_pipeline(&self.render_state.device, &self.pipeline_layout, &bytes)?;
        self.pipeline = next;
        self.dispatch();
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
                    label: Some("Oklab DRT encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Oklab DRT pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
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
        label: Some("Oklab DRT bindings"),
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
) -> Result<wgpu::ComputePipeline> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Oklab DRT SPIR-V"),
        source: wgpu::util::make_spirv(spirv),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Oklab DRT pipeline"),
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
        label: Some("Oklab DRT AP0 input"),
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
        label: Some("Oklab DRT sRGB output"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });
    let input_view = input.create_view(&Default::default());
    let storage_view = output.create_view(&Default::default());
    let analysis_view = output.create_view(&wgpu::TextureViewDescriptor {
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        ..Default::default()
    });
    // The shader writes display-encoded values. Sampling the same bytes through
    // an sRGB view decodes them before egui renders into its sRGB surface.
    let display_view = output.create_view(&wgpu::TextureViewDescriptor {
        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Oklab DRT bind group"),
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
