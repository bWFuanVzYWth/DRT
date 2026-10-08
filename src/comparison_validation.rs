//! Opt-in coverage of the Compare control and full application paint routing.
use super::*;

fn render_state() -> eframe::egui_wgpu::RenderState {
    use eframe::egui_wgpu;
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("comparison integration requires an available wgpu adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let target_format = wgpu::TextureFormat::Rgba16Float;
    let renderer = egui_wgpu::Renderer::new(&device, target_format, Default::default());
    egui_wgpu::RenderState {
        available_adapters: vec![adapter.clone()],
        adapter,
        instance,
        device,
        queue,
        target_format,
        renderer: std::sync::Arc::new(egui::epaint::mutex::RwLock::new(renderer)),
        surface_config: egui_wgpu::SurfaceConfig::HIGH_THROUGHPUT,
    }
}

fn frame(
    context: &egui::Context,
    app: &mut DrtApp,
    time: f64,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut output = context.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            time: Some(time),
            events,
            ..Default::default()
        },
        |ui| app.ui(ui),
    );
    // This headless check inspects shapes without applying font-atlas texture changes.
    output.textures_delta.clear();
    output
}

fn click(
    context: &egui::Context,
    app: &mut DrtApp,
    time: f64,
    position: egui::Pos2,
) -> egui::FullOutput {
    frame(
        context,
        app,
        time,
        vec![egui::Event::PointerMoved(position)],
    );
    let button = |pressed| egui::Event::PointerButton {
        pos: position,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    frame(context, app, time + 0.01, vec![button(true)]);
    frame(context, app, time + 0.02, vec![button(false)])
}

fn text_center(output: &egui::FullOutput, label: &str) -> egui::Pos2 {
    output
        .shapes
        .iter()
        .find_map(|shape| match &shape.shape {
            egui::Shape::Text(text) if text.galley.text() == label => {
                Some(text.visual_bounding_rect().center())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing visible control text: {label}"))
}

fn image_mesh(
    output: &egui::FullOutput,
    texture: egui::TextureId,
) -> Option<(egui::Rect, &egui::epaint::Mesh)> {
    output.shapes.iter().find_map(|shape| match &shape.shape {
        egui::Shape::Mesh(mesh) if mesh.texture_id == texture => {
            Some((shape.clip_rect, mesh.as_ref()))
        }
        _ => None,
    })
}

fn has_texture(output: &egui::FullOutput, texture: egui::TextureId) -> bool {
    output
        .shapes
        .iter()
        .any(|shape| shape.shape.texture_id() == texture)
}

fn assert_aligned_sides(
    output: &egui::FullOutput,
    left_texture: egui::TextureId,
    right_texture: egui::TextureId,
    split: f32,
) {
    let (left_clip, left) = image_mesh(output, left_texture).expect("left render is painted");
    let (right_clip, right) = image_mesh(output, right_texture).expect("right render is painted");
    assert_eq!(left.vertices, right.vertices);
    assert_eq!(left.vertices.len(), 4);
    let image = left.calc_bounds();
    let expected_divider = image.left() + image.width() * split;
    assert!((left_clip.right() - expected_divider).abs() < 0.01);
    assert!((right_clip.left() - expected_divider).abs() < 0.01);
    assert_eq!(left_clip.left(), image.left());
    assert_eq!(right_clip.right(), image.right());
    assert_eq!(left_clip.top(), right_clip.top());
    assert_eq!(left_clip.bottom(), right_clip.bottom());
    let mut uv = egui::Rect::NOTHING;
    for vertex in &left.vertices {
        uv.extend_with(vertex.uv);
    }
    assert_eq!(
        uv,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0))
    );
}

#[test]
#[ignore = "requires a GPU; run with --ignored --nocapture"]
fn compare_control_routes_both_workspace_views_and_turns_off() {
    let render_state = render_state();
    let context = egui::Context::default();
    let mut app = DrtApp::new(
        &render_state,
        &context,
        None,
        None,
        WorkspaceView::Image,
        false,
        DisplayOutput {
            hdr_surface: false,
            detected_headroom: Some(1.0),
            max_nits: None,
            sdr_white_nits: None,
        },
    )
    .unwrap();
    app.gpu.set_image(image_io::test_pattern(64, 36)).unwrap();

    // Simulate enabling through the visible checkbox, including its distinct-DRT fallback.
    app.comparison.right_drt = app.gpu.active_drt();
    let initial = frame(&context, &mut app, 0.0, vec![]);
    let compare = text_center(&initial, "Compare");
    let enabled = click(&context, &mut app, 0.1, compare);
    assert!(app.comparison.enabled);
    assert_ne!(app.gpu.active_drt(), app.comparison.right_drt);
    let left_texture = app.gpu.texture_id();
    let right_texture = app
        .gpu
        .comparison_texture_id()
        .expect("enabling Compare registers the right render");
    assert_ne!(left_texture, right_texture);
    assert_aligned_sides(&enabled, left_texture, right_texture, 0.5);

    // Analysis shares the comparison renderer while keeping its separate distribution canvas.
    app.workspace_view = WorkspaceView::Analysis;
    app.comparison.split = 0.3;
    let analysis = frame(&context, &mut app, 0.2, vec![]);
    assert_aligned_sides(&analysis, left_texture, right_texture, 0.3);
    let compare = text_center(&analysis, "Compare");
    let disabled = click(&context, &mut app, 0.3, compare);
    assert!(!app.comparison.enabled);
    assert!(app.gpu.comparison_texture_id().is_none());
    assert!(has_texture(&disabled, left_texture));
    assert!(!has_texture(&disabled, right_texture));

    app.workspace_view = WorkspaceView::Image;
    let single_image = frame(&context, &mut app, 0.4, vec![]);
    assert!(has_texture(&single_image, left_texture));
    assert!(!has_texture(&single_image, right_texture));
    render_state
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}
