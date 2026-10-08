use eframe::egui::{self, Color32, Rect, Stroke, TextureId, Vec2};

use crate::gpu::DrtKind;

const DIVIDER_HIT_WIDTH: f32 = 16.0;
const HANDLE_SIZE: f32 = 30.0;

pub struct ComparisonView {
    pub enabled: bool,
    pub right_drt: DrtKind,
    /// Horizontal position relative to the displayed image, preserved across resizes.
    pub split: f32,
}

impl Default for ComparisonView {
    fn default() -> Self {
        Self {
            enabled: false,
            right_drt: DrtKind::RgbReinhard,
            split: 0.5,
        }
    }
}

impl ComparisonView {
    /// Paint two aligned renders of the same image. The caller controls whether comparison is enabled.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        ui: &mut egui::Ui,
        bounds: Rect,
        left_texture: TextureId,
        right_texture: TextureId,
        image_size: Vec2,
        left_label: &str,
        right_label: &str,
    ) -> egui::Response {
        let id = ui.id().with("drt_comparison_divider");
        let Some(image_rect) = fitted_image_rect(bounds, image_size) else {
            return ui.interact(bounds, id, egui::Sense::hover());
        };
        self.split = self.split.clamp(0.0, 1.0);

        let x = image_rect.left() + image_rect.width() * self.split;
        let handle_center = egui::pos2(x, image_rect.center().y);
        let divider_hit = Rect::from_center_size(
            handle_center,
            egui::vec2(DIVIDER_HIT_WIDTH, image_rect.height()),
        )
        .intersect(image_rect);
        let handle_hit =
            Rect::from_center_size(handle_center, Vec2::splat(HANDLE_SIZE)).intersect(image_rect);
        let mut response = ui.interact(divider_hit, id, egui::Sense::click_and_drag())
            | ui.interact(handle_hit, id.with("handle"), egui::Sense::click_and_drag());

        let previous_split = self.split;
        if response.double_clicked() {
            self.split = 0.5;
        } else if response.dragged_by(egui::PointerButton::Primary)
            || response.drag_stopped_by(egui::PointerButton::Primary)
            || (response.is_pointer_button_down_on()
                && ui.input(|input| input.pointer.primary_down()))
        {
            // Use the absolute pointer position, so capture continues beyond the hit band and image.
            if let Some(pointer) = response.interact_pointer_pos() {
                self.split = ((pointer.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
            }
        }
        if self.split != previous_split {
            response.mark_changed();
        }
        if response.hovered() || response.is_pointer_button_down_on() || response.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }

        let x = image_rect.left() + image_rect.width() * self.split;
        let left_rect = Rect::from_min_max(image_rect.min, egui::pos2(x, image_rect.bottom()));
        let right_rect = Rect::from_min_max(egui::pos2(x, image_rect.top()), image_rect.max);
        let uv = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));
        let painter = ui.painter().with_clip_rect(image_rect);

        // Both meshes cover the full image with the same UVs; clipping reveals each side without
        // scaling either half or changing which source pixel is visible at the divider.
        painter
            .with_clip_rect(left_rect)
            .image(left_texture, image_rect, uv, Color32::WHITE);
        painter
            .with_clip_rect(right_rect)
            .image(right_texture, image_rect, uv, Color32::WHITE);
        paint_badge(&painter, left_rect, left_label, false);
        paint_badge(&painter, right_rect, right_label, true);

        let divider = [
            egui::pos2(x, image_rect.top()),
            egui::pos2(x, image_rect.bottom()),
        ];
        painter.line_segment(divider, Stroke::new(4.0, Color32::from_black_alpha(220)));
        painter.line_segment(divider, Stroke::new(1.5, Color32::WHITE));
        let center = egui::pos2(x, image_rect.center().y);
        painter.circle_filled(center, HANDLE_SIZE * 0.5, Color32::from_black_alpha(220));
        painter.circle_stroke(
            center,
            HANDLE_SIZE * 0.5 - 1.0,
            Stroke::new(1.5, Color32::WHITE),
        );
        for direction in [-1.0, 1.0] {
            let tip = center + egui::vec2(direction * 9.0, 0.0);
            for y in [-4.0, 4.0] {
                painter.line_segment(
                    [tip, center + egui::vec2(direction * 5.0, y)],
                    Stroke::new(1.5, Color32::WHITE),
                );
            }
        }
        response.on_hover_text("Drag to compare DRTs · Double-click to center")
    }
}

fn fitted_image_rect(bounds: Rect, image_size: Vec2) -> Option<Rect> {
    if !bounds.is_finite()
        || !image_size.is_finite()
        || bounds.width() <= 0.0
        || bounds.height() <= 0.0
        || image_size.x <= 0.0
        || image_size.y <= 0.0
    {
        return None;
    }
    let scale = (bounds.width() / image_size.x).min(bounds.height() / image_size.y);
    Some(Rect::from_center_size(bounds.center(), image_size * scale))
}

fn paint_badge(painter: &egui::Painter, side: Rect, label: &str, align_right: bool) {
    if side.width() <= 0.0 || side.height() <= 0.0 {
        return;
    }
    let painter = painter.with_clip_rect(side);
    let galley = painter.layout_no_wrap(
        label.to_owned(),
        egui::FontId::proportional(12.0),
        Color32::WHITE,
    );
    let size = galley.size() + egui::vec2(16.0, 10.0);
    let x = if align_right {
        side.right() - 8.0 - size.x
    } else {
        side.left() + 8.0
    };
    let badge = Rect::from_min_size(egui::pos2(x, side.top() + 8.0), size);
    painter.rect_filled(badge, 4.0, Color32::from_black_alpha(210));
    painter.rect_stroke(
        badge,
        4.0,
        Stroke::new(1.0, Color32::from_white_alpha(90)),
        egui::StrokeKind::Inside,
    );
    painter.galley(badge.min + egui::vec2(8.0, 5.0), galley, Color32::WHITE);
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEFT_TEXTURE: TextureId = TextureId::User(7);
    const RIGHT_TEXTURE: TextureId = TextureId::User(8);

    fn frame(
        context: &egui::Context,
        comparison: &mut ComparisonView,
        bounds: Rect,
        time: f64,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        let mut output = context.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000.0, 800.0),
                )),
                time: Some(time),
                events,
                ..Default::default()
            },
            |ui| {
                comparison.paint(
                    ui,
                    bounds,
                    LEFT_TEXTURE,
                    RIGHT_TEXTURE,
                    egui::vec2(1600.0, 900.0),
                    "Left DRT",
                    "Right DRT",
                );
            },
        );
        // A headless frame intentionally has no GPU texture uploader.
        output.textures_delta.clear();
        output
    }

    fn pointer_button(position: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos: position,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }
    }

    #[test]
    fn drag_keeps_capture_outside_image_and_releases_it() {
        let context = egui::Context::default();
        let mut comparison = ComparisonView::default();
        let bounds = Rect::from_min_max(egui::pos2(100.0, 100.0), egui::pos2(900.0, 700.0));
        // The fitted image spans x=100..900 and y=175..625. Start on the line away from its handle.
        let start = egui::pos2(500.0, 250.0);
        frame(&context, &mut comparison, bounds, 0.0, vec![]);
        frame(
            &context,
            &mut comparison,
            bounds,
            0.1,
            vec![
                egui::Event::PointerMoved(start),
                pointer_button(start, true),
            ],
        );
        let outside_right = egui::pos2(980.0, 40.0);
        let output = frame(
            &context,
            &mut comparison,
            bounds,
            0.2,
            vec![egui::Event::PointerMoved(outside_right)],
        );
        assert_eq!(comparison.split, 1.0);
        assert_eq!(
            output.platform_output.cursor_icon,
            egui::CursorIcon::ResizeHorizontal
        );
        let outside_left = egui::pos2(20.0, 40.0);
        frame(
            &context,
            &mut comparison,
            bounds,
            0.3,
            vec![egui::Event::PointerMoved(outside_left)],
        );
        assert_eq!(comparison.split, 0.0);
        frame(
            &context,
            &mut comparison,
            bounds,
            0.4,
            vec![pointer_button(outside_left, false)],
        );
        frame(
            &context,
            &mut comparison,
            bounds,
            0.5,
            vec![egui::Event::PointerMoved(start)],
        );
        assert_eq!(comparison.split, 0.0);
    }

    #[test]
    fn image_clicks_away_from_divider_do_not_move_it() {
        let context = egui::Context::default();
        let mut comparison = ComparisonView::default();
        let bounds = Rect::from_min_max(egui::pos2(100.0, 100.0), egui::pos2(900.0, 700.0));
        frame(&context, &mut comparison, bounds, 0.0, vec![]);
        let start = egui::pos2(300.0, 250.0);
        frame(
            &context,
            &mut comparison,
            bounds,
            0.1,
            vec![
                egui::Event::PointerMoved(start),
                pointer_button(start, true),
            ],
        );
        frame(
            &context,
            &mut comparison,
            bounds,
            0.2,
            vec![egui::Event::PointerMoved(egui::pos2(700.0, 250.0))],
        );
        assert_eq!(comparison.split, 0.5);
    }

    #[test]
    fn handle_outer_edge_can_drag_and_double_click_centers() {
        let context = egui::Context::default();
        let mut comparison = ComparisonView {
            split: 0.25,
            ..Default::default()
        };
        let bounds = Rect::from_min_max(egui::pos2(100.0, 100.0), egui::pos2(900.0, 700.0));
        frame(&context, &mut comparison, bounds, 0.0, vec![]);
        // This is outside the 16-point line hit band, but inside the larger center handle.
        let handle_edge = egui::pos2(313.0, 400.0);
        frame(
            &context,
            &mut comparison,
            bounds,
            0.1,
            vec![
                egui::Event::PointerMoved(handle_edge),
                pointer_button(handle_edge, true),
            ],
        );
        let destination = egui::pos2(700.0, 400.0);
        frame(
            &context,
            &mut comparison,
            bounds,
            0.2,
            vec![egui::Event::PointerMoved(destination)],
        );
        frame(
            &context,
            &mut comparison,
            bounds,
            0.3,
            vec![pointer_button(destination, false)],
        );
        assert_eq!(comparison.split, 0.75);

        for (time, pressed) in [(1.0, true), (1.1, false), (1.2, true), (1.3, false)] {
            frame(
                &context,
                &mut comparison,
                bounds,
                time,
                vec![pointer_button(destination, pressed)],
            );
        }
        assert_eq!(comparison.split, 0.5);
    }

    #[test]
    fn both_renders_keep_full_matching_uvs_and_aspect_after_resize() {
        let context = egui::Context::default();
        let mut comparison = ComparisonView {
            split: 0.25,
            ..Default::default()
        };
        let cases = [
            (
                Rect::from_min_max(egui::pos2(100.0, 100.0), egui::pos2(900.0, 700.0)),
                Rect::from_min_max(egui::pos2(100.0, 175.0), egui::pos2(900.0, 625.0)),
            ),
            (
                Rect::from_min_max(egui::pos2(100.0, 100.0), egui::pos2(900.0, 325.0)),
                Rect::from_min_max(egui::pos2(300.0, 100.0), egui::pos2(700.0, 325.0)),
            ),
        ];
        for (bounds, expected_image) in cases {
            let output = frame(&context, &mut comparison, bounds, 0.0, vec![]);
            let images: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::Shape::Mesh(mesh)
                        if mesh.texture_id == LEFT_TEXTURE || mesh.texture_id == RIGHT_TEXTURE =>
                    {
                        Some((shape.clip_rect, mesh))
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(images.len(), 2);
            for (_, mesh) in &images {
                assert_eq!(mesh.calc_bounds(), expected_image);
                let uv_bounds =
                    Rect::from_points(&mesh.vertices.iter().map(|v| v.uv).collect::<Vec<_>>());
                assert_eq!(
                    uv_bounds,
                    Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0))
                );
            }
            assert_eq!(images[0].1.vertices, images[1].1.vertices);
            let split_x = expected_image.left() + expected_image.width() * 0.25;
            assert_eq!(images[0].0.right(), split_x);
            assert_eq!(images[1].0.left(), split_x);
            assert_eq!(images[0].0.left(), expected_image.left());
            assert_eq!(images[1].0.right(), expected_image.right());
            assert_eq!(comparison.split, 0.25);
        }
    }
}
