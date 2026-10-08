use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu};
use wgpu::util::DeviceExt;

pub const DEFAULT_YAW: f32 = 0.75;
pub const DEFAULT_PITCH: f32 = -0.35;

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
    target_is_srgb: u32,
    // Encoded RGB peak, Oklab L peak, then equal-pixel projection scales.
    range_and_fit: [f32; 4],
}

impl DistributionParameters {
    fn new(
        rotation: [f32; 2],
        image_size: [u32; 2],
        space: ColorSpace,
        target_is_srgb: bool,
        headroom: f32,
        canvas_size: egui::Vec2,
    ) -> Self {
        let headroom = headroom.clamp(1.0, 64.0);
        let encoded_peak = if headroom == 1.0 {
            1.0
        } else {
            1.055 * headroom.powf(1.0 / 2.4) - 0.055
        };
        let width = canvas_size.x.max(1.0);
        let height = canvas_size.y.max(1.0);
        let shortest = width.min(height);
        Self {
            rotation: [rotation[0], rotation[1], 0.0, 0.0],
            image_size,
            space: space.shader_value(),
            target_is_srgb: u32::from(target_is_srgb),
            range_and_fit: [
                encoded_peak,
                headroom.cbrt(),
                shortest / width,
                shortest / height,
            ],
        }
    }
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
        output_headroom: f32,
        point_count: u32,
    ) -> egui::Response {
        let rect = rect.intersect(ui.clip_rect());
        if !rect.is_positive() {
            return ui.interact(
                egui::Rect::from_min_size(rect.min, egui::Vec2::ZERO),
                ui.id().with("color_distribution_canvas"),
                egui::Sense::hover(),
            );
        }
        let response = ui.interact(
            rect,
            ui.id().with("color_distribution_canvas"),
            egui::Sense::click_and_drag(),
        );
        if response.dragged_by(egui::PointerButton::Primary) {
            let motion = response.drag_motion();
            *yaw += motion.x * 0.008;
            *pitch = (*pitch - motion.y * 0.008).clamp(-1.45, 1.45);
        }
        if response.clicked_by(egui::PointerButton::Secondary)
            || (response.hovered()
                && ui.input(|input| input.pointer.button_clicked(egui::PointerButton::Secondary)))
        {
            *yaw = DEFAULT_YAW;
            *pitch = DEFAULT_PITCH;
        }

        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_filled(rect, 10.0, egui::Color32::from_rgb(12, 15, 20));
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            DistributionCallback {
                yaw: *yaw,
                pitch: *pitch,
                color_space,
                output_headroom,
                canvas_size: rect.size(),
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
    output_headroom: f32,
    canvas_size: egui::Vec2,
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
        let parameters = DistributionParameters::new(
            [self.yaw, self.pitch],
            [resources.width, resources.height],
            self.color_space,
            resources.target_is_srgb,
            self.output_headroom,
            self.canvas_size,
        );
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
        if self.color_space == ColorSpace::Srgb {
            render_pass.set_pipeline(&resources.guide_pipeline);
            let guide_vertices = if self.output_headroom > 1.0 { 54 } else { 30 };
            render_pass.draw(0..guide_vertices, 0..1);
        }
    }
}

struct DistributionResources {
    pipeline: wgpu::RenderPipeline,
    guide_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    width: u32,
    height: u32,
    target_is_srgb: bool,
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
        let guide_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sRGB distribution reference pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_guide"),
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
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let parameters = DistributionParameters::new(
            [DEFAULT_YAW, DEFAULT_PITCH],
            [width, height],
            ColorSpace::Srgb,
            target_format.is_srgb(),
            1.0,
            egui::Vec2::splat(1.0),
        );
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color distribution parameters"),
            contents: bytemuck::bytes_of(&parameters),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = create_bind_group(device, &bind_group_layout, output_view, &uniform);
        Self {
            pipeline,
            guide_pipeline,
            bind_group_layout,
            bind_group,
            uniform,
            width,
            height,
            target_is_srgb: target_format.is_srgb(),
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

#[cfg(test)]
#[path = "distribution_validation.rs"]
mod validation;

#[cfg(test)]
#[path = "distribution_layout_validation.rs"]
mod layout_validation;

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
