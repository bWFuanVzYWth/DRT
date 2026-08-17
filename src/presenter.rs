use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use crate::{DrtApp, WorkspaceView};

const EGUI_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Debug)]
pub struct StartupOptions {
    pub initial_image: Option<std::path::PathBuf>,
    pub initial_folder: Option<std::path::PathBuf>,
    pub initial_view: WorkspaceView,
    pub initial_show_anomalies: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayOutput {
    pub hdr_surface: bool,
    pub detected_headroom: Option<f32>,
    pub max_nits: Option<f32>,
    pub sdr_white_nits: Option<f32>,
}

pub fn run(startup: StartupOptions) -> Result<()> {
    let event_loop = EventLoop::new().context("cannot create the native event loop")?;
    let mut native_app = NativeApplication {
        startup: Some(startup),
        graphics: None,
        initialization_error: None,
    };
    event_loop
        .run_app(&mut native_app)
        .context("native event loop failed")?;
    if let Some(error) = native_app.initialization_error {
        return Err(error);
    }
    Ok(())
}

struct NativeApplication {
    startup: Option<StartupOptions>,
    graphics: Option<Graphics>,
    initialization_error: Option<anyhow::Error>,
}

impl ApplicationHandler for NativeApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.graphics.is_some() || self.initialization_error.is_some() {
            return;
        }
        let startup = self.startup.take().expect("startup options are available");
        let attributes = Window::default_attributes()
            .with_title("DRT Bench")
            .with_inner_size(LogicalSize::new(1280.0, 760.0))
            .with_min_inner_size(LogicalSize::new(800.0, 520.0));
        let result = event_loop
            .create_window(attributes)
            .context("cannot create the DRT Bench window")
            .and_then(|window| Graphics::new(Arc::new(window), startup));
        match result {
            Ok(graphics) => {
                graphics.window.request_redraw();
                self.graphics = Some(graphics);
            }
            Err(error) => {
                self.initialization_error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(graphics) = self.graphics.as_mut() else {
            return;
        };
        if window_id != graphics.window.id() {
            return;
        }

        let response = graphics
            .egui_state
            .on_window_event(&graphics.window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Err(error) = graphics.resize(size) {
                    self.initialization_error = Some(error);
                    event_loop.exit();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } | WindowEvent::Moved(_) => {
                if let Err(error) = graphics.refresh_display_output() {
                    self.initialization_error = Some(error);
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => match graphics.render(event_loop) {
                Ok(RenderResult::Continue) => graphics.window.request_redraw(),
                Ok(RenderResult::Exit) => event_loop.exit(),
                Err(error) => {
                    self.initialization_error = Some(error);
                    event_loop.exit();
                }
            },
            _ => {
                if response.repaint {
                    graphics.window.request_redraw();
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(graphics) = &self.graphics {
            graphics.window.request_redraw();
        }
    }
}

enum RenderResult {
    Continue,
    Exit,
}

struct Graphics {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    render_state: egui_wgpu::RenderState,
    egui_context: egui::Context,
    egui_state: egui_winit::State,
    app: DrtApp,
    compositor: Compositor,
    display_output: DisplayOutput,
    last_display_poll: std::time::Instant,
}

impl Graphics {
    fn new(window: Arc<Window>, startup: StartupOptions) -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .context("cannot create the wgpu presentation surface")?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .context("no compatible GPU adapter was found")?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("DRT Bench device"),
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
            .context("cannot create the wgpu device")?;

        let size = nonzero_size(window.inner_size());
        let (config, hdr_surface) = make_surface_config(&surface, &adapter, size)?;
        surface.configure(&device, &config);
        let display_output = query_display_output(&surface, &adapter, hdr_surface);

        let egui_context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_context.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(
            &device,
            EGUI_TARGET_FORMAT,
            egui_wgpu::RendererOptions::default(),
        );
        let renderer = Arc::new(egui::epaint::mutex::RwLock::new(renderer));
        let available_adapters =
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let render_state = egui_wgpu::RenderState {
            adapter: adapter.clone(),
            available_adapters,
            instance: instance.clone(),
            device: device.clone(),
            queue: queue.clone(),
            target_format: EGUI_TARGET_FORMAT,
            renderer,
            surface_config: egui_wgpu::SurfaceConfig::HIGH_THROUGHPUT,
        };
        let app = DrtApp::new(
            &render_state,
            &egui_context,
            startup.initial_image.as_deref(),
            startup.initial_folder.as_deref(),
            startup.initial_view,
            startup.initial_show_anomalies,
            display_output,
        )?;
        let compositor = Compositor::new(&device, size, config.format, hdr_surface);

        Ok(Self {
            window,
            surface,
            config,
            render_state,
            egui_context,
            egui_state,
            app,
            compositor,
            display_output,
            last_display_poll: std::time::Instant::now(),
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.refresh_surface(size)
    }

    fn refresh_display_output(&mut self) -> Result<()> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.refresh_surface(size)
    }

    fn refresh_surface(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let (next_config, hdr_surface) =
            make_surface_config(&self.surface, &self.render_state.adapter, size)?;
        let format_changed = self.config.format != next_config.format;
        let hdr_changed = self.display_output.hdr_surface != hdr_surface;
        self.config = next_config;
        self.surface
            .configure(&self.render_state.device, &self.config);
        if format_changed || hdr_changed {
            self.compositor = Compositor::new(
                &self.render_state.device,
                size,
                self.config.format,
                hdr_surface,
            );
        } else {
            self.compositor.resize(&self.render_state.device, size);
        }
        let output = query_display_output(&self.surface, &self.render_state.adapter, hdr_surface);
        self.display_output = output;
        self.app.set_display_output(output);
        self.last_display_poll = std::time::Instant::now();
        Ok(())
    }

    fn poll_display_output(&mut self) -> Result<()> {
        if self.last_display_poll.elapsed() < std::time::Duration::from_secs(1) {
            return Ok(());
        }
        self.last_display_poll = std::time::Instant::now();
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let (next_config, hdr_surface) =
            make_surface_config(&self.surface, &self.render_state.adapter, size)?;
        if self.config.format != next_config.format
            || self.config.color_space != next_config.color_space
        {
            self.config = next_config;
            self.surface
                .configure(&self.render_state.device, &self.config);
            self.compositor = Compositor::new(
                &self.render_state.device,
                size,
                self.config.format,
                hdr_surface,
            );
        }
        let output = query_display_output(&self.surface, &self.render_state.adapter, hdr_surface);
        if output != self.display_output {
            self.display_output = output;
            self.app.set_display_output(output);
        }
        Ok(())
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) -> Result<RenderResult> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(RenderResult::Continue);
        }
        self.poll_display_output()?;

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let full_output = self.egui_context.run_ui(raw_input, |ui| self.app.ui(ui));
        let egui::FullOutput {
            platform_output,
            mut textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = full_output;
        let wants_close = viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|output| {
                output
                    .commands
                    .iter()
                    .any(|command| matches!(command, egui::ViewportCommand::Close))
            });
        self.egui_state.handle_platform_output_with_event_loop(
            &self.window,
            event_loop,
            platform_output,
        );

        let paint_jobs = self.egui_context.tessellate(shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point,
        };
        let mut encoder =
            self.render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("DRT Bench frame encoder"),
                });
        let user_command_buffers = {
            let mut renderer = self.render_state.renderer.write();
            for (id, deltas) in textures_delta.set.drain() {
                for delta in deltas {
                    renderer.update_texture(
                        &self.render_state.device,
                        &self.render_state.queue,
                        id,
                        &delta,
                    );
                }
            }
            renderer.update_buffers(
                &self.render_state.device,
                &self.render_state.queue,
                &mut encoder,
                &paint_jobs,
                &screen,
            )
        };

        {
            let renderer = self.render_state.renderer.read();
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui fp16 pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.compositor.offscreen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
        }

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(RenderResult::Continue);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.refresh_surface(size)?;
                return Ok(RenderResult::Continue);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                bail!("the presentation surface was lost");
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                bail!("wgpu rejected the presentation surface");
            }
        };
        let surface_view = surface_texture.texture.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HDR presentation pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.compositor.pipeline);
            pass.set_bind_group(0, &self.compositor.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.render_state.queue.submit(
            user_command_buffers
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        {
            let mut renderer = self.render_state.renderer.write();
            for id in textures_delta.free.drain() {
                renderer.free_texture(&id);
            }
        }
        self.window.pre_present_notify();
        self.render_state.queue.present(surface_texture);

        Ok(if wants_close {
            RenderResult::Exit
        } else {
            RenderResult::Continue
        })
    }
}

fn make_surface_config(
    surface: &wgpu::Surface<'_>,
    adapter: &wgpu::Adapter,
    size: PhysicalSize<u32>,
) -> Result<(wgpu::SurfaceConfiguration, bool)> {
    let capabilities = surface.get_capabilities(adapter);
    let hdr_supported = capabilities
        .color_spaces(wgpu::TextureFormat::Rgba16Float)
        .contains(wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR);
    let (format, color_space, hdr_surface) = if hdr_supported {
        (
            wgpu::TextureFormat::Rgba16Float,
            wgpu::SurfaceColorSpace::ExtendedSrgbLinear,
            true,
        )
    } else {
        let format = capabilities
            .formats
            .first()
            .copied()
            .context("the window surface exposes no SDR format")?;
        let color_space = if capabilities
            .color_spaces(format)
            .contains(wgpu::SurfaceColorSpaces::SRGB)
        {
            wgpu::SurfaceColorSpace::Srgb
        } else {
            wgpu::SurfaceColorSpace::Auto
        };
        (format, color_space, false)
    };
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        color_space,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: 2,
        alpha_mode: capabilities
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto),
        view_formats: vec![],
    };
    Ok((config, hdr_surface))
}

fn query_display_output(
    surface: &wgpu::Surface<'_>,
    adapter: &wgpu::Adapter,
    hdr_surface: bool,
) -> DisplayOutput {
    let hdr = surface.display_hdr_info(adapter);
    let luminance = hdr.luminance;
    DisplayOutput {
        hdr_surface,
        detected_headroom: hdr.tone_map_headroom().filter(|value| *value >= 1.0),
        max_nits: luminance.and_then(|value| value.max_nits),
        sdr_white_nits: luminance.and_then(|value| value.sdr_white_nits),
    }
}

fn nonzero_size(size: PhysicalSize<u32>) -> PhysicalSize<u32> {
    PhysicalSize::new(size.width.max(1), size.height.max(1))
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositorParameters {
    decode_to_linear: u32,
    _padding: [u32; 3],
}

struct Compositor {
    _offscreen: wgpu::Texture,
    offscreen_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Compositor {
    fn new(
        device: &wgpu::Device,
        size: PhysicalSize<u32>,
        target_format: wgpu::TextureFormat,
        hdr_surface: bool,
    ) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HDR compositor bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HDR compositor sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let decode_to_linear = hdr_surface || target_format.is_srgb();
        let parameters = CompositorParameters {
            decode_to_linear: u32::from(decode_to_linear),
            _padding: [0; 3],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HDR compositor parameters"),
            contents: bytemuck::bytes_of(&parameters),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HDR compositor shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("presenter.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HDR compositor pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HDR compositor pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let (offscreen, offscreen_view) = create_offscreen(device, size);
        let bind_group = create_compositor_bind_group(
            device,
            &bind_group_layout,
            &offscreen_view,
            &sampler,
            &uniform,
        );
        Self {
            _offscreen: offscreen,
            offscreen_view,
            pipeline,
            bind_group_layout,
            sampler,
            uniform,
            bind_group,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, size: PhysicalSize<u32>) {
        let (offscreen, offscreen_view) = create_offscreen(device, size);
        self.bind_group = create_compositor_bind_group(
            device,
            &self.bind_group_layout,
            &offscreen_view,
            &self.sampler,
            &self.uniform,
        );
        self._offscreen = offscreen;
        self.offscreen_view = offscreen_view;
    }
}

fn create_offscreen(
    device: &wgpu::Device,
    size: PhysicalSize<u32>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("egui extended-sRGB fp16 target"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: EGUI_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_compositor_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HDR compositor bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}
