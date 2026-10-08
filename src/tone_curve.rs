use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu};
use wgpu::util::DeviceExt;

pub const SAMPLE_COUNT: u32 = 512;
pub const INPUT_MIN_EV: f32 = -16.0;
pub const INPUT_MAX_EV: f32 = 18.0;
const OUTPUT_MIN_EV: f32 = -12.0;
const OUTPUT_MAX_EV: f32 = 10.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlotParameters {
    sample_count: u32,
    target_is_srgb: u32,
    _padding: [u32; 2],
}

pub struct ToneCurveRenderer;

impl ToneCurveRenderer {
    pub fn install(render_state: &egui_wgpu::RenderState, curve_view: &wgpu::TextureView) {
        let mut renderer = render_state.renderer.write();
        renderer.callback_resources.insert(ToneCurveResources::new(
            &render_state.device,
            render_state.target_format,
            curve_view,
        ));
    }

    pub fn paint(ui: &mut egui::Ui, drt_label: &str) -> egui::Response {
        ui.label(egui::RichText::new("Neutral-axis tone curve · log₂ EV").strong());
        ui.horizontal(|ui| {
            legend(ui, egui::Color32::from_gray(215), drt_label);
            legend(ui, egui::Color32::from_rgb(83, 139, 155), "Linear");
        });

        let size = egui::vec2(ui.available_width(), 174.0);
        let (outer, response) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(outer);
        painter.rect_filled(outer, 4.0, egui::Color32::from_rgb(13, 16, 20));
        painter.rect_stroke(
            outer,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(55)),
            egui::StrokeKind::Inside,
        );

        let plot = egui::Rect::from_min_max(
            outer.min + egui::vec2(28.0, 19.0),
            outer.max - egui::vec2(7.0, 23.0),
        );
        let grid = egui::Color32::from_gray(42);
        let axis = egui::Color32::from_gray(75);
        let text = egui::Color32::from_gray(125);
        let font = egui::FontId::monospace(8.5);

        for value in [-16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0] {
            let x = map_x(plot, value);
            painter.line_segment(
                [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
                egui::Stroke::new(1.0, if value == 0.0 { axis } else { grid }),
            );
            painter.text(
                egui::pos2(x, plot.bottom() + 4.0),
                egui::Align2::CENTER_TOP,
                format_ev(value),
                font.clone(),
                text,
            );
        }
        for value in [-12.0, -6.0, 0.0, 6.0, 10.0] {
            let y = map_y(plot, value);
            painter.line_segment(
                [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
                egui::Stroke::new(1.0, if value == 0.0 { axis } else { grid }),
            );
            painter.text(
                egui::pos2(plot.left() - 4.0, y),
                egui::Align2::RIGHT_CENTER,
                format_ev(value),
                font.clone(),
                text,
            );
        }

        // Untonemapped display-linear light is y=x in the same log2/EV axes.
        let identity_start = INPUT_MIN_EV.max(OUTPUT_MIN_EV);
        let identity_end = INPUT_MAX_EV.min(OUTPUT_MAX_EV);
        painter.line_segment(
            [
                egui::pos2(map_x(plot, identity_start), map_y(plot, identity_start)),
                egui::pos2(map_x(plot, identity_end), map_y(plot, identity_end)),
            ],
            egui::Stroke::new(1.25, egui::Color32::from_rgb(83, 139, 155)),
        );
        painter.add(egui_wgpu::Callback::new_paint_callback(
            plot,
            ToneCurveCallback,
        ));
        painter.text(
            egui::pos2(plot.center().x, outer.bottom() - 2.0),
            egui::Align2::CENTER_BOTTOM,
            "input EV rel. 18%",
            font.clone(),
            text,
        );
        painter.text(
            egui::pos2(plot.left(), outer.top() + 3.0),
            egui::Align2::LEFT_TOP,
            "output EV rel. 18%",
            font,
            text,
        );
        response.on_hover_text(
            "Both axes are log2 stops relative to 18% gray. The DRT curve uses a synthetic neutral AP0 axis and is independent of the loaded image.",
        )
    }
}

fn legend(ui: &mut egui::Ui, color: egui::Color32, label: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 8.0), egui::Sense::hover());
    ui.painter().line_segment(
        [rect.left_center(), rect.right_center()],
        egui::Stroke::new(1.5, color),
    );
    ui.label(egui::RichText::new(label).small().weak());
}

fn map_x(rect: egui::Rect, ev: f32) -> f32 {
    egui::remap(ev, INPUT_MIN_EV..=INPUT_MAX_EV, rect.x_range())
}

fn map_y(rect: egui::Rect, ev: f32) -> f32 {
    let t = (ev - OUTPUT_MIN_EV) / (OUTPUT_MAX_EV - OUTPUT_MIN_EV);
    egui::lerp(rect.bottom()..=rect.top(), t)
}

fn format_ev(value: f32) -> String {
    if value > 0.0 {
        format!("+{value:.0}")
    } else {
        format!("{value:.0}")
    }
}

struct ToneCurveCallback;

impl egui_wgpu::CallbackTrait for ToneCurveCallback {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &ToneCurveResources =
            resources.get().expect("tone-curve renderer was installed");
        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.bind_group, &[]);
        render_pass.draw(0..SAMPLE_COUNT, 0..1);
    }
}

struct ToneCurveResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl ToneCurveResources {
    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        curve_view: &wgpu::TextureView,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neutral-axis tone-curve shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/analysis/tone_curve.wgsl").into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("neutral-axis tone-curve bindings"),
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
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<PlotParameters>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let parameters = PlotParameters {
            sample_count: SAMPLE_COUNT,
            target_is_srgb: u32::from(target_format.is_srgb()),
            _padding: [0; 2],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("neutral-axis tone-curve plot parameters"),
            contents: bytemuck::bytes_of(&parameters),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("neutral-axis tone-curve bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(curve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("neutral-axis tone-curve pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("neutral-axis tone-curve pipeline"),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            bind_group,
        }
    }
}
