mod distribution;
mod file_browser;
mod gpu;
mod image_io;

use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
};

use eframe::egui::{self, Color32, RichText, Vec2};

use crate::distribution::{ColorSpace, DistributionRenderer};
use crate::file_browser::{FolderBrowser, Thumbnail, ThumbnailLoader};
use crate::gpu::DrtGpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceView {
    Image,
    Analysis,
}

fn main() -> eframe::Result {
    let mut initial_image = None;
    let mut initial_folder = None;
    let mut initial_view = WorkspaceView::Image;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--analysis" {
            initial_view = WorkspaceView::Analysis;
        } else if argument == "--folder" {
            initial_folder = arguments.next().map(PathBuf::from);
        } else if initial_image.is_none() {
            initial_image = Some(PathBuf::from(argument));
        }
    }
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 760.0])
            .with_min_inner_size([800.0, 520.0])
            .with_title("Oklab DRT Bench"),
        renderer: eframe::Renderer::Wgpu,
        centered: true,
        ..Default::default()
    };

    eframe::run_native(
        "Oklab DRT Bench",
        native_options,
        Box::new(move |context| {
            Ok(Box::new(DrtApp::new(
                context,
                initial_image.as_deref(),
                initial_folder.as_deref(),
                initial_view,
            )?))
        }),
    )
}

struct DrtApp {
    gpu: DrtGpu,
    exposure_ev: f32,
    overexposure: f32,
    image_name: String,
    status: String,
    status_error: bool,
    shader_path: PathBuf,
    shader_modified: Option<std::time::SystemTime>,
    last_shader_check: std::time::Instant,
    workspace_view: WorkspaceView,
    color_space: ColorSpace,
    distribution_yaw: f32,
    distribution_pitch: f32,
    image_path: Option<PathBuf>,
    folder_browser: Option<FolderBrowser>,
    thumbnail_loader: ThumbnailLoader,
    pending_image: Option<PendingImage>,
}

struct PendingImage {
    path: PathBuf,
    receiver: Receiver<Result<image_io::LinearImage, String>>,
}

impl DrtApp {
    fn new(
        context: &eframe::CreationContext<'_>,
        initial_image: Option<&Path>,
        initial_folder: Option<&Path>,
        initial_view: WorkspaceView,
    ) -> anyhow::Result<Self> {
        let render_state = context
            .wgpu_render_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("eframe did not create a wgpu render state"))?;
        let shader_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("oklab_drt.slang");
        let image = match initial_image {
            Some(path) => image_io::load(path)?,
            None => image_io::test_pattern(1280, 720),
        };
        let image_name = initial_image.and_then(Path::file_name).map_or_else(
            || "Built-in AP0 HDR test pattern".to_owned(),
            |name| name.to_string_lossy().into(),
        );
        let gpu = DrtGpu::new(render_state, image)?;

        let folder_browser = initial_folder
            .map(|path| FolderBrowser::scan(path.to_path_buf()))
            .transpose()?;
        let mut thumbnail_loader = ThumbnailLoader::default();
        if let Some(browser) = &folder_browser {
            thumbnail_loader.start(browser.paths(), context.egui_ctx.clone());
        }

        Ok(Self {
            gpu,
            exposure_ev: 0.0,
            overexposure: 1.1,
            image_name,
            status: folder_browser.as_ref().map_or_else(
                || "Oklab DRT ready".to_owned(),
                |browser| format!("Folder opened · {} images", browser.entries.len()),
            ),
            status_error: false,
            shader_modified: modified_time(&shader_path),
            shader_path,
            last_shader_check: std::time::Instant::now(),
            workspace_view: initial_view,
            color_space: ColorSpace::Srgb,
            distribution_yaw: 0.75,
            distribution_pitch: -0.35,
            image_path: initial_image.map(Path::to_path_buf),
            folder_browser,
            thumbnail_loader,
            pending_image: None,
        })
    }

    fn open_image(&mut self, context: &egui::Context) {
        let selected = rfd::FileDialog::new()
            .add_filter(
                "HDR / EXR / common images",
                &["exr", "hdr", "png", "jpg", "jpeg", "webp"],
            )
            .pick_file();
        let Some(path) = selected else { return };
        self.begin_image_load(path, context);
    }

    fn open_folder(&mut self, context: &egui::Context) {
        let selected = rfd::FileDialog::new().pick_folder();
        let Some(path) = selected else { return };
        match FolderBrowser::scan(path) {
            Ok(browser) => {
                let count = browser.entries.len();
                self.thumbnail_loader
                    .start(browser.paths(), context.clone());
                self.folder_browser = Some(browser);
                self.set_status(format!("Folder opened · {count} images"), false);
            }
            Err(error) => self.set_status(format!("Folder opening failed: {error:#}"), true),
        }
    }

    fn begin_image_load(&mut self, path: PathBuf, context: &egui::Context) {
        if self
            .pending_image
            .as_ref()
            .is_some_and(|pending| pending.path == path)
            || (self.pending_image.is_none() && self.image_path.as_ref() == Some(&path))
        {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        let worker_path = path.clone();
        let repaint = context.clone();
        std::thread::spawn(move || {
            let result = image_io::load(&worker_path).map_err(|error| format!("{error:#}"));
            let _ = sender.send(result);
            repaint.request_repaint();
        });
        self.set_status(format!("Loading {}…", path.display()), false);
        self.pending_image = Some(PendingImage { path, receiver });
    }

    fn poll_image_load(&mut self) {
        let result = match self
            .pending_image
            .as_ref()
            .map(|pending| pending.receiver.try_recv())
        {
            Some(Ok(result)) => Some(result),
            Some(Err(TryRecvError::Disconnected)) => Some(Err("loader stopped".to_owned())),
            Some(Err(TryRecvError::Empty)) | None => None,
        };
        let Some(result) = result else { return };
        let pending = self.pending_image.take().expect("pending image exists");
        match result.and_then(|image| {
            self.gpu
                .set_image(image)
                .map_err(|error| format!("{error:#}"))
        }) {
            Ok(()) => {
                self.image_name = pending.path.file_name().map_or_else(
                    || pending.path.display().to_string(),
                    |name| name.to_string_lossy().into(),
                );
                self.image_path = Some(pending.path.clone());
                self.set_status(format!("Loaded {}", self.image_name), false);
            }
            Err(error) => self.set_status(format!("Image loading failed: {error:#}"), true),
        }
    }

    fn load_test_pattern(&mut self) {
        match self.gpu.set_image(image_io::test_pattern(1280, 720)) {
            Ok(()) => {
                self.image_name = "Built-in AP0 HDR test pattern".to_owned();
                self.image_path = None;
                self.pending_image = None;
                self.set_status("Built-in test pattern restored".to_owned(), false);
            }
            Err(error) => self.set_status(format!("Test-pattern creation failed: {error:#}"), true),
        }
    }

    fn reload_shader(&mut self) {
        let result = self.gpu.reload_shader(&self.shader_path);
        // A failed edit is retried only after the file changes again.
        self.shader_modified = modified_time(&self.shader_path);
        match result {
            Ok(()) => {
                self.set_status("Slang to SPIR-V hot reload succeeded".to_owned(), false);
            }
            Err(error) => self.set_status(
                format!("Shader compilation failed; keeping the last valid pipeline: {error:#}"),
                true,
            ),
        }
    }

    fn poll_shader(&mut self, context: &egui::Context) {
        if self.last_shader_check.elapsed() < std::time::Duration::from_millis(400) {
            context.request_repaint_after(std::time::Duration::from_millis(400));
            return;
        }
        self.last_shader_check = std::time::Instant::now();
        let current = modified_time(&self.shader_path);
        if current.is_some() && current != self.shader_modified {
            self.reload_shader();
        }
        context.request_repaint_after(std::time::Duration::from_millis(400));
    }

    fn set_status(&mut self, status: String, is_error: bool) {
        self.status = status;
        self.status_error = is_error;
    }
}

impl eframe::App for DrtApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_shader(ui.ctx());
        self.poll_image_load();
        if let Some(browser) = &mut self.folder_browser {
            self.thumbnail_loader.poll(ui.ctx(), browser);
        }
        let context = ui.ctx().clone();

        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open image…  F3").clicked() {
                        self.open_image(&context);
                        ui.close();
                    }
                    if ui.button("Open folder…  F4").clicked() {
                        self.open_folder(&context);
                        ui.close();
                    }
                    if ui.button("Built-in test pattern").clicked() {
                        self.load_test_pattern();
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Exit  Esc").clicked() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("Shader", |ui| {
                    if ui.button("Recompile  F5").clicked() {
                        self.reload_shader();
                        ui.close();
                    }
                    ui.label("Auto-reloads after shaders/oklab_drt.slang changes");
                });
                ui.separator();
                ui.selectable_value(&mut self.workspace_view, WorkspaceView::Image, "Image");
                ui.selectable_value(
                    &mut self.workspace_view,
                    WorkspaceView::Analysis,
                    "Image + Distribution",
                );
                ui.separator();
                ui.label(
                    RichText::new("wgpu 30 · Slang 2025.13+ · egui 0.36")
                        .monospace()
                        .weak(),
                );
            });
        });

        egui::Panel::left("controls")
            .default_size(292.0)
            .resizable(false)
            .show(ui, |ui| {
                ui.heading("Oklab DRT");
                ui.label("Input: scene-linear ACES2065-1 / AP0");
                ui.label("Output: SDR sRGB");
                if self.workspace_view == WorkspaceView::Analysis {
                    ui.separator();
                    ui.label("Distribution space");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.color_space, ColorSpace::Srgb, "sRGB");
                        ui.selectable_value(&mut self.color_space, ColorSpace::Oklab, "Oklab");
                    });
                    ui.label(
                        RichText::new("Every mapped pixel is rendered · drag to rotate")
                            .small()
                            .weak(),
                    );
                }
                ui.separator();
                ui.label("Exposure");
                let exposure_changed = ui
                    .add(
                        egui::Slider::new(&mut self.exposure_ev, -20.0..=20.0)
                            .step_by(0.25)
                            .suffix(" EV"),
                    )
                    .changed();
                ui.label("Highlight asymptote");
                let overexposure_changed = ui
                    .add(egui::Slider::new(&mut self.overexposure, 0.5..=2.0).step_by(0.05))
                    .changed();
                if exposure_changed || overexposure_changed {
                    self.gpu.set_parameters(self.exposure_ev, self.overexposure);
                }
                if ui.button("Reset parameters").clicked() {
                    self.exposure_ev = 0.0;
                    self.overexposure = 1.1;
                    self.gpu.set_parameters(self.exposure_ev, self.overexposure);
                }
                ui.separator();
                ui.label("Current image");
                ui.label(RichText::new(&self.image_name).strong());
                ui.label(format!("{} × {}", self.gpu.width(), self.gpu.height()));
                ui.horizontal(|ui| {
                    if ui.button("Open…").clicked() {
                        self.open_image(&context);
                    }
                    if ui.button("Folder…").clicked() {
                        self.open_folder(&context);
                    }
                    if ui.button("Test pattern").clicked() {
                        self.load_test_pattern();
                    }
                });

                let mut selected_path = None;
                if let Some(browser) = &mut self.folder_browser {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Folder");
                        ui.label(
                            RichText::new(
                                browser
                                    .root
                                    .file_name()
                                    .unwrap_or(browser.root.as_os_str())
                                    .to_string_lossy(),
                            )
                            .small()
                            .weak(),
                        )
                        .on_hover_text(browser.root.display().to_string());
                    });
                    ui.label(
                        RichText::new(format!("{} images", browser.entries.len()))
                            .small()
                            .weak(),
                    );
                    egui::ScrollArea::vertical()
                        .id_salt("folder-image-list")
                        .auto_shrink([false, false])
                        .max_height((ui.available_height() - 90.0).max(100.0))
                        .show(ui, |ui| {
                            for entry in &browser.entries {
                                let selected = self.image_path.as_ref() == Some(&entry.path)
                                    || self
                                        .pending_image
                                        .as_ref()
                                        .is_some_and(|pending| pending.path == entry.path);
                                let response = match &entry.thumbnail {
                                    Thumbnail::Ready(texture) => ui.add(
                                        egui::Button::image_and_text(
                                            egui::Image::new(texture)
                                                .fit_to_exact_size(Vec2::new(72.0, 48.0)),
                                            &entry.name,
                                        )
                                        .selected(selected)
                                        .min_size(Vec2::new(ui.available_width(), 54.0)),
                                    ),
                                    Thumbnail::Pending => ui.add_sized(
                                        [ui.available_width(), 40.0],
                                        egui::Button::new(format!("◌  {}", entry.name))
                                            .selected(selected),
                                    ),
                                    Thumbnail::Failed => ui.add_sized(
                                        [ui.available_width(), 40.0],
                                        egui::Button::new(format!("◇  {}", entry.name))
                                            .selected(selected),
                                    ),
                                };
                                if response.clicked() {
                                    selected_path = Some(entry.path.clone());
                                }
                                response.on_hover_text(entry.path.display().to_string());
                            }
                        });
                }
                if let Some(path) = selected_path {
                    self.begin_image_load(path, &context);
                }
                ui.separator();
                ui.label("Shortcuts");
                ui.label("F3 image · F4 folder · F5 compile · Esc exit");
                ui.add_space(8.0);
                let color = if self.status_error {
                    Color32::LIGHT_RED
                } else {
                    Color32::LIGHT_GREEN
                };
                ui.colored_label(color, &self.status);
            });

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("GPU: {}", self.gpu.adapter_name()));
                ui.separator();
                ui.label(format!("Backend: {:?}", self.gpu.backend()));
                ui.separator();
                ui.label(format!(
                    "Exposure multiplier: {:.3}",
                    2.0_f32.powf(self.exposure_ev)
                ));
            });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(ui.style()).fill(Color32::from_gray(18)))
            .show(ui, |ui| match self.workspace_view {
                WorkspaceView::Image => show_image(
                    ui,
                    self.gpu.texture_id(),
                    self.gpu.width(),
                    self.gpu.height(),
                ),
                WorkspaceView::Analysis => {
                    let point_count = exact_point_count(self.gpu.width(), self.gpu.height());
                    let available_rect = ui.available_rect_before_wrap();
                    let available = available_rect.size();
                    let gap = 14.0;
                    let content_width = (available.x - gap).max(0.0);
                    let left_width = content_width * 0.43;
                    let left_rect = egui::Rect::from_min_size(
                        available_rect.min,
                        Vec2::new(left_width, available.y),
                    );
                    let right_rect = egui::Rect::from_min_max(
                        egui::pos2(left_rect.max.x + gap, available_rect.min.y),
                        available_rect.max,
                    );

                    let header_height = 42.0;
                    paint_centered_label(
                        ui,
                        egui::pos2(left_rect.center().x, left_rect.top()),
                        "Mapped image",
                        egui::TextStyle::Body,
                        ui.visuals().strong_text_color(),
                    );
                    paint_image_in_rect(
                        ui,
                        self.gpu.texture_id(),
                        self.gpu.width(),
                        self.gpu.height(),
                        egui::Rect::from_min_max(
                            egui::pos2(left_rect.left(), left_rect.top() + header_height),
                            left_rect.max,
                        ),
                    );

                    paint_centered_label(
                        ui,
                        egui::pos2(right_rect.center().x, right_rect.top()),
                        &format!("{} distribution", self.color_space.label()),
                        egui::TextStyle::Body,
                        ui.visuals().strong_text_color(),
                    );
                    paint_centered_label(
                        ui,
                        egui::pos2(right_rect.center().x, right_rect.top() + 21.0),
                        &format!(
                            "{} points · complete",
                            format_point_count(u64::from(point_count))
                        ),
                        egui::TextStyle::Monospace,
                        ui.visuals().weak_text_color(),
                    );
                    let canvas_rect = egui::Rect::from_min_max(
                        egui::pos2(right_rect.left(), right_rect.top() + header_height),
                        right_rect.max,
                    );
                    let response = DistributionRenderer::paint(
                        ui,
                        canvas_rect,
                        &mut self.distribution_yaw,
                        &mut self.distribution_pitch,
                        self.color_space,
                        point_count,
                    );
                    response.on_hover_text("Drag to rotate the complete pixel distribution");
                    ui.advance_cursor_after_rect(available_rect);
                }
            });

        let open = context.input(|input| input.key_pressed(egui::Key::F3));
        let open_folder = context.input(|input| input.key_pressed(egui::Key::F4));
        let reload = context.input(|input| input.key_pressed(egui::Key::F5));
        let close = context.input(|input| input.key_pressed(egui::Key::Escape));
        if open {
            self.open_image(&context);
        }
        if open_folder {
            self.open_folder(&context);
        }
        if reload {
            self.reload_shader();
        }
        if close {
            context.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

fn show_image(ui: &mut egui::Ui, texture_id: egui::TextureId, width: u32, height: u32) {
    let available = ui.available_size();
    ui.centered_and_justified(|ui| {
        show_image_in_size(ui, texture_id, width, height, available);
    });
}

fn show_image_in_size(
    ui: &mut egui::Ui,
    texture_id: egui::TextureId,
    width: u32,
    height: u32,
    available: Vec2,
) {
    let source = Vec2::new(width as f32, height as f32);
    let scale = (available.x / source.x)
        .min(available.y / source.y)
        .max(0.01);
    ui.add(
        egui::Image::new(egui::load::SizedTexture::new(texture_id, source))
            .fit_to_exact_size(source * scale),
    );
}

fn paint_image_in_rect(
    ui: &mut egui::Ui,
    texture_id: egui::TextureId,
    width: u32,
    height: u32,
    bounds: egui::Rect,
) {
    let source = Vec2::new(width as f32, height as f32);
    let scale = (bounds.width() / source.x)
        .min(bounds.height() / source.y)
        .max(0.01);
    let image_rect = egui::Rect::from_center_size(bounds.center(), source * scale);
    ui.painter().image(
        texture_id,
        image_rect,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        Color32::WHITE,
    );
}

fn paint_centered_label(
    ui: &egui::Ui,
    position: egui::Pos2,
    text: &str,
    text_style: egui::TextStyle,
    color: Color32,
) {
    ui.painter().text(
        position,
        egui::Align2::CENTER_TOP,
        text,
        text_style.resolve(ui.style()),
        color,
    );
}

fn format_point_count(count: u64) -> String {
    let digits = count.to_string();
    let mut result = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            result.push(',');
        }
        result.push(character);
    }
    result
}

fn exact_point_count(width: u32, height: u32) -> u32 {
    width
        .checked_mul(height)
        .expect("mapped image exceeds wgpu's u32 vertex range")
}

fn modified_time(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

#[cfg(test)]
mod tests {
    use super::{exact_point_count, format_point_count};

    #[test]
    fn reports_the_complete_pixel_count() {
        let count = exact_point_count(1280, 720);
        assert_eq!(count, 921_600);
        assert_eq!(format_point_count(u64::from(count)), "921,600");
    }
}
