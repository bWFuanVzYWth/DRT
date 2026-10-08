use super::*;

fn frame(canvas: egui::Rect, parent_clip: egui::Rect, space: ColorSpace) -> egui::FullOutput {
    let context = egui::Context::default();
    let mut output = context.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(640.0, 360.0),
            )),
            ..Default::default()
        },
        |ui| {
            ui.set_clip_rect(parent_clip);
            let mut yaw = DEFAULT_YAW;
            let mut pitch = DEFAULT_PITCH;
            DistributionRenderer::paint(ui, canvas, &mut yaw, &mut pitch, space, 16.0, 1024);
        },
    );
    // Headless frames inspect paint shapes without a GPU texture uploader.
    output.textures_delta.clear();
    output
}

#[test]
fn color_cloud_viewport_fits_the_visible_canvas() {
    let parent_clip = egui::Rect::from_min_max(egui::pos2(80.0, 60.0), egui::pos2(580.0, 300.0));
    let canvas = egui::Rect::from_min_max(egui::pos2(120.0, 90.0), egui::pos2(800.0, 500.0));
    let visible_canvas = canvas.intersect(parent_clip);

    for space in [ColorSpace::Srgb, ColorSpace::Oklab] {
        let output = frame(canvas, parent_clip, space);
        let callbacks: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                egui::Shape::Callback(callback) => Some((callback.rect, shape.clip_rect)),
                _ => None,
            })
            .collect();
        assert_eq!(callbacks.len(), 1, "{space:?} should paint one cloud");
        let (viewport, clip) = callbacks[0];
        assert!(viewport.is_positive());
        assert!(parent_clip.contains_rect(viewport));
        assert!(parent_clip.contains_rect(clip));
        assert_eq!(viewport, visible_canvas);
        assert_eq!(clip, visible_canvas);
    }
}

#[test]
fn collapsed_or_hidden_analysis_canvas_skips_gpu_callback() {
    let parent_clip = egui::Rect::from_min_max(egui::pos2(80.0, 60.0), egui::pos2(580.0, 300.0));
    let canvases = [
        // No horizontal space remains beside the analysis controls.
        egui::Rect::from_min_max(egui::pos2(120.0, 90.0), egui::pos2(120.0, 250.0)),
        // A short window leaves the analysis header below the available bottom.
        egui::Rect::from_min_max(egui::pos2(120.0, 240.0), egui::pos2(500.0, 200.0)),
        // The canvas is entirely outside the parent viewport.
        egui::Rect::from_min_max(egui::pos2(120.0, 320.0), egui::pos2(500.0, 500.0)),
    ];

    for space in [ColorSpace::Srgb, ColorSpace::Oklab] {
        for canvas in canvases {
            let output = frame(canvas, parent_clip, space);
            assert!(
                output
                    .shapes
                    .iter()
                    .all(|shape| !matches!(shape.shape, egui::Shape::Callback(_))),
                "{space:?} must skip a nonvisible canvas: {canvas:?}"
            );
        }
    }
}
