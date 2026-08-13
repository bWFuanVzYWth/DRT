use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
    Oklab,
}

impl ColorSpace {
    pub fn label(self) -> &'static str {
        match self {
            Self::Srgb => "sRGB",
            Self::Oklab => "Oklab",
        }
    }

    fn shader_value(self) -> u32 {
        match self {
            Self::Srgb => 0,
            Self::Oklab => 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DistributionParameters {
    rotation: [f32; 4],
    image_size: [u32; 2],
    space: u32,
    _padding: f32,
}

pub struct DistributionRenderer;

impl DistributionRenderer {
    pub fn install(
        render_state: &egui_wgpu::RenderState,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let mut renderer = render_state.renderer.write();
        if renderer
            .callback_resources
            .get::<DistributionResources>()
            .is_none()
        {
            renderer
                .callback_resources
                .insert(DistributionResources::new(
                    &render_state.device,
                    render_state.target_format,
                    output_view,
                    width,
                    height,
                ));
        } else {
            renderer
                .callback_resources
                .get_mut::<DistributionResources>()
                .expect("distribution resources exist")
                .set_image(&render_state.device, output_view, width, height);
        }
    }

    pub fn paint(
        ui: &mut egui::Ui,
        rect: egui::Rect,
        yaw: &mut f32,
        pitch: &mut f32,
        color_space: ColorSpace,
        point_count: u32,
    ) -> egui::Response {
        let response = ui.interact(
            rect,
            ui.id().with("color_distribution_canvas"),
            egui::Sense::drag(),
        );
        if response.dragged() {
            let motion = response.drag_motion();
            *yaw += motion.x * 0.008;
            *pitch = (*pitch - motion.y * 0.008).clamp(-1.45, 1.45);
        }

        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_filled(rect, 10.0, egui::Color32::from_rgb(12, 15, 20));
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            DistributionCallback {
                yaw: *yaw,
                pitch: *pitch,
                color_space,
                point_count,
            },
        ));
        response
    }
}

struct DistributionCallback {
    yaw: f32,
    pitch: f32,
    color_space: ColorSpace,
    point_count: u32,
}

impl egui_wgpu::CallbackTrait for DistributionCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &DistributionResources = resources
            .get()
            .expect("distribution renderer was installed");
        let parameters = DistributionParameters {
            rotation: [self.yaw, self.pitch, 0.0, 0.0],
            image_size: [resources.width, resources.height],
            space: self.color_space.shader_value(),
            _padding: 0.0,
        };
        queue.write_buffer(&resources.uniform, 0, bytemuck::bytes_of(&parameters));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &DistributionResources = resources
            .get()
            .expect("distribution renderer was installed");
        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.bind_group, &[]);
        render_pass.draw(0..self.point_count, 0..1);
    }
}

struct DistributionResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl DistributionResources {
    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("color distribution shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/color_distribution.wgsl").into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("color distribution bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            DistributionParameters,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("color distribution pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("color distribution pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let parameters = DistributionParameters {
            rotation: [0.75, -0.35, 0.0, 0.0],
            image_size: [width, height],
            space: 0,
            _padding: 0.0,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color distribution parameters"),
            contents: bytemuck::bytes_of(&parameters),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = create_bind_group(device, &bind_group_layout, output_view, &uniform);
        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniform,
            width,
            height,
        }
    }

    fn set_image(
        &mut self,
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        self.width = width;
        self.height = height;
        self.bind_group =
            create_bind_group(device, &self.bind_group_layout, output_view, &self.uniform);
    }
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    output_view: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("color distribution bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}
