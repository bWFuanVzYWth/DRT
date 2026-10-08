mod aces2_data;
mod comparison;
#[cfg(test)]
mod comparison_validation;
mod distribution;
mod drt;
mod file_browser;
mod gpu;
mod image_io;
mod lpm_data;
mod presenter;
mod reference;
mod tone_curve;

use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
};

use eframe::egui::{self, Color32, RichText, Vec2};

use crate::comparison::ComparisonView;
use crate::distribution::{ColorSpace, DEFAULT_PITCH, DEFAULT_YAW, DistributionRenderer};
use crate::file_browser::{FolderBrowser, Thumbnail, ThumbnailLoader};
use crate::gpu::{
    DrtGpu, DrtKind, LinearLogSigmoidParameters, LogSigmoidParameters, OklabChromaParameters,
    ReinhardParameters,
};
use crate::presenter::{DisplayOutput, StartupOptions};
use crate::tone_curve::ToneCurveRenderer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceView {
    Image,
    Analysis,
}

fn main() -> anyhow::Result<()> {
    let mut initial_image = None;
    let mut initial_folder = None;
    let mut initial_view = WorkspaceView::Image;
    let mut initial_show_anomalies = false;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--analysis" {
            initial_view = WorkspaceView::Analysis;
        } else if argument == "--show-anomalies" {
            initial_show_anomalies = true;
        } else if argument == "--folder" {
            initial_folder = arguments.next().map(PathBuf::from);
        } else if initial_image.is_none() {
            initial_image = Some(PathBuf::from(argument));
        }
    }
    presenter::run(StartupOptions {
        initial_image,
        initial_folder,
        initial_view,
        initial_show_anomalies,
    })
}

struct DrtApp {
    gpu: DrtGpu,
    exposure_ev: f32,
    agx_s2o3_parameters: LogSigmoidParameters,
    rgb_log_sigmoid_parameters: LinearLogSigmoidParameters,
    oklab_log_sigmoid_parameters: LinearLogSigmoidParameters,
    rgb_log_sigmoid_hue_retention: f32,
    oklab_reinhard_parameters: ReinhardParameters,
    oklab_reinhard_chroma: OklabChromaParameters,
    oklab_log_sigmoid_chroma: OklabChromaParameters,
    rgb_reinhard_parameters: ReinhardParameters,
    show_anomalies: bool,
    image_name: String,
    status: String,
    status_error: bool,
    shader_paths: [PathBuf; DrtKind::ALL.len()],
    shader_modified: [Option<std::time::SystemTime>; DrtKind::ALL.len()],
    last_shader_check: std::time::Instant,
    workspace_view: WorkspaceView,
    comparison: ComparisonView,
    color_space: ColorSpace,
    distribution_yaw: f32,
    distribution_pitch: f32,
    image_path: Option<PathBuf>,
    folder_browser: Option<FolderBrowser>,
    thumbnail_loader: ThumbnailLoader,
    pending_image: Option<PendingImage>,
    display_output: DisplayOutput,
    hdr_target_headroom: f32,
    hdr_target_user_set: bool,
}

struct PendingImage {
    path: PathBuf,
    receiver: Receiver<Result<image_io::LinearImage, String>>,
}

impl DrtApp {
    fn new(
        render_state: &eframe::egui_wgpu::RenderState,
        egui_context: &egui::Context,
        initial_image: Option<&Path>,
        initial_folder: Option<&Path>,
        initial_view: WorkspaceView,
        initial_show_anomalies: bool,
        display_output: DisplayOutput,
    ) -> anyhow::Result<Self> {
        let shader_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let shader_paths = DrtKind::ALL.map(|drt| shader_directory.join(drt.shader_file()));
        let shader_modified = shader_paths.each_ref().map(|path| modified_time(path));
        let image = match initial_image {
            Some(path) => image_io::load(path)?,
            None => image_io::test_pattern(1280, 720),
        };
        let image_name = initial_image.and_then(Path::file_name).map_or_else(
            || "Built-in AP0 HDR test pattern".to_owned(),
            |name| name.to_string_lossy().into(),
        );
        let mut gpu = DrtGpu::new(render_state, image)?;
        gpu.set_show_anomalies(initial_show_anomalies);
        let hdr_target_headroom = display_output.detected_headroom.unwrap_or(1.0).max(1.0);
        gpu.set_hdr_headroom(if display_output.hdr_surface {
            hdr_target_headroom
        } else {
            1.0
        });

        let folder_browser = initial_folder
            .map(|path| FolderBrowser::scan(path.to_path_buf()))
            .transpose()?;
        let mut thumbnail_loader = ThumbnailLoader::default();
        if let Some(browser) = &folder_browser {
            thumbnail_loader.start(browser.paths(), egui_context.clone());
        }

        Ok(Self {
            gpu,
            exposure_ev: 0.0,
            agx_s2o3_parameters: LogSigmoidParameters::s2o3_reference(),
            rgb_log_sigmoid_parameters: LinearLogSigmoidParameters::default(),
            oklab_log_sigmoid_parameters: LinearLogSigmoidParameters::default(),
            rgb_log_sigmoid_hue_retention: 0.5,
            oklab_reinhard_parameters: ReinhardParameters::oklab_default(),
            oklab_reinhard_chroma: OklabChromaParameters::default(),
            oklab_log_sigmoid_chroma: OklabChromaParameters::default(),
            rgb_reinhard_parameters: ReinhardParameters::default(),
            show_anomalies: initial_show_anomalies,
            image_name,
            status: folder_browser.as_ref().map_or_else(
                || "DRT bench ready".to_owned(),
                |browser| format!("Folder opened · {} images", browser.entries.len()),
            ),
            status_error: false,
            shader_paths,
            shader_modified,
            last_shader_check: std::time::Instant::now(),
            workspace_view: initial_view,
            comparison: ComparisonView::default(),
            color_space: ColorSpace::Srgb,
            distribution_yaw: DEFAULT_YAW,
            distribution_pitch: DEFAULT_PITCH,
            image_path: initial_image.map(Path::to_path_buf),
            folder_browser,
            thumbnail_loader,
            pending_image: None,
            display_output,
            hdr_target_headroom,
            hdr_target_user_set: false,
        })
    }

    fn set_display_output(&mut self, display_output: DisplayOutput) {
        if self.display_output == display_output {
            return;
        }
        self.display_output = display_output;
        if !self.hdr_target_user_set {
            self.hdr_target_headroom = display_output.detected_headroom.unwrap_or(1.0).max(1.0);
        }
        let effective = if display_output.hdr_surface {
            self.hdr_target_headroom
        } else {
            1.0
        };
        self.gpu.set_hdr_headroom(effective);
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

    fn reload_shader(&mut self, drt: DrtKind) {
        let shader_path = self.shader_paths[drt.index()].clone();
        let result = self.gpu.reload_shader(drt, &shader_path);
        // A failed edit is retried only after the file changes again.
        self.shader_modified[drt.index()] = modified_time(&shader_path);
        match result {
            Ok(()) => {
                self.set_status(format!("{} WGSL hot reload succeeded", drt.label()), false);
            }
            Err(error) => self.set_status(
                format!("Shader compilation failed; keeping the last valid pipeline: {error:#}"),
                true,
            ),
        }
    }

    fn reload_visible_shaders(&mut self) {
        self.reload_shader(self.gpu.active_drt());
        if self.comparison.enabled && self.comparison.right_drt != self.gpu.active_drt() {
            let left_error = self.status_error.then(|| self.status.clone());
            self.reload_shader(self.comparison.right_drt);
            if let Some(error) = left_error {
                if self.status_error {
                    self.set_status(format!("{error}\n{}", self.status), true);
                } else {
                    self.set_status(error, true);
                }
            }
        }
    }

    fn poll_shader(&mut self, context: &egui::Context) {
        if self.last_shader_check.elapsed() < std::time::Duration::from_millis(400) {
            context.request_repaint_after(std::time::Duration::from_millis(400));
            return;
        }
        self.last_shader_check = std::time::Instant::now();
        for drt in DrtKind::ALL {
            let current = modified_time(&self.shader_paths[drt.index()]);
            if current.is_some() && current != self.shader_modified[drt.index()] {
                self.reload_shader(drt);
            }
        }
        context.request_repaint_after(std::time::Duration::from_millis(400));
    }

    fn set_status(&mut self, status: String, is_error: bool) {
        self.status = status;
        self.status_error = is_error;
    }
}

impl DrtApp {
    fn ui(&mut self, ui: &mut egui::Ui) {
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
                        self.reload_visible_shaders();
                        ui.close();
                    }
                    ui.label("Auto-reloads every DRT shader");
                });
                ui.separator();
                ui.selectable_value(&mut self.workspace_view, WorkspaceView::Image, "Image");
                ui.selectable_value(
                    &mut self.workspace_view,
                    WorkspaceView::Analysis,
                    "Image + Distribution",
                );
                ui.separator();
                if ui
                    .checkbox(&mut self.comparison.enabled, "Compare")
                    .changed()
                    && self.comparison.enabled
                    && self.comparison.right_drt == self.gpu.active_drt()
                {
                    self.comparison.right_drt = DrtKind::ALL
                        .into_iter()
                        .find(|drt| *drt != self.gpu.active_drt())
                        .expect("multiple DRTs available");
                }
                ui.separator();
                ui.label(RichText::new("wgpu · WGSL · egui").monospace().weak());
            });
        });

        egui::Panel::left("controls")
            .default_size(292.0)
            .resizable(false)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("controls-scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                ui.heading("Display Rendering Transform");
                ui.label("Input: scene-linear ACES2065-1 / AP0");
                match (
                    self.display_output.hdr_surface,
                    self.display_output.detected_headroom,
                ) {
                    (true, Some(headroom)) if headroom > 1.0 => {
                        ui.label("Output: linear scRGB HDR (active)");
                    }
                    (true, Some(_)) => {
                        ui.label("Output: linear scRGB (display currently SDR)");
                    }
                    (true, None) => {
                        ui.label("Output: linear scRGB (HDR headroom unknown)");
                    }
                    (false, _) => {
                        ui.label("Output: SDR sRGB (HDR surface unavailable)");
                    }
                }
                ui.separator();
                ui.label(if self.comparison.enabled { "Left DRT" } else { "DRT" });
                let mut selected_drt = self.gpu.active_drt();
                drt_selector(ui, "left-drt", &mut selected_drt);
                if selected_drt != self.gpu.active_drt() {
                    match selected_drt {
                        DrtKind::AgxS2O3 => {
                            self.gpu.set_agx_s2o3_parameters(self.agx_s2o3_parameters)
                        }
                        DrtKind::OklabLogSigmoid => {
                            self.gpu.set_oklab_log_sigmoid_parameters(self.oklab_log_sigmoid_parameters)
                        }
                        DrtKind::RgbLogSigmoid => {
                            self.gpu.set_rgb_log_sigmoid_parameters(self.rgb_log_sigmoid_parameters)
                        }
                        DrtKind::RgbReinhard => self
                            .gpu
                            .set_rgb_reinhard_parameters(self.rgb_reinhard_parameters),
                        _ => {}
                    }
                    self.gpu.set_drt(selected_drt);
                    self.set_status(format!("Switched to {}", selected_drt.label()), false);
                }
                if self.comparison.enabled {
                    ui.horizontal(|ui| {
                        ui.label("Right DRT");
                        drt_selector(ui, "comparison-right-drt", &mut self.comparison.right_drt);
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Swap sides").clicked() {
                            let left = self.gpu.active_drt();
                            self.gpu.set_drt(self.comparison.right_drt);
                            self.comparison.right_drt = left;
                        }
                        if ui.button("Center divider").clicked() {
                            self.comparison.split = 0.5;
                        }
                    });
                    ui.label(RichText::new("Shared image and exposure · swap to adjust the other DRT")
                        .small().weak());
                    ui.label(RichText::new("Drag the divider · double-click to center")
                        .small().weak());
                }
                self.gpu.set_comparison_drt(self.comparison.enabled.then_some(self.comparison.right_drt));
                if let Some(source) = self.gpu.active_drt().source_url() {
                    ui.hyperlink_to("Reference source", source);
                }
                ui.separator();
                if self.comparison.enabled {
                    ui.label(RichText::new("Left DRT curve and parameters").small().weak());
                }
                ToneCurveRenderer::paint(ui, self.gpu.active_drt().label());
                ui.separator();
                ui.label(RichText::new("Display range").strong());
                if self.display_output.hdr_surface {
                    let detected = self.display_output.detected_headroom;
                    let maximum = detected.unwrap_or(16.0).max(1.0);
                    let mut headroom = self.hdr_target_headroom.clamp(1.0, maximum);
                    let changed = ui
                        .add(
                            egui::Slider::new(&mut headroom, 1.0..=maximum)
                                .step_by(0.05)
                                .custom_formatter(|value, _| format!("{value:.2}×"))
                                .text("HDR headroom"),
                        )
                        .on_hover_text(
                            "Controls the output range of HDR-capable DRTs; 1.0× exactly reproduces SDR output",
                        )
                        .changed();
                    if changed {
                        self.hdr_target_headroom = headroom;
                        self.hdr_target_user_set = true;
                        self.gpu.set_hdr_headroom(headroom);
                    }
                    ui.horizontal(|ui| {
                        let detected_label = detected.map_or_else(
                            || "Detected: unknown".to_owned(),
                            |value| format!("Detected: {value:.2}× SDR white"),
                        );
                        ui.label(RichText::new(detected_label).small().weak());
                        if ui.small_button("Auto").clicked() {
                            self.hdr_target_headroom = detected.unwrap_or(1.0).max(1.0);
                            self.hdr_target_user_set = false;
                            self.gpu.set_hdr_headroom(self.hdr_target_headroom);
                        }
                    });
                    if let (Some(peak), Some(sdr_white)) = (
                        self.display_output.max_nits,
                        self.display_output.sdr_white_nits,
                    ) {
                        ui.label(
                            RichText::new(format!(
                                "Display {peak:.0} nit peak · {sdr_white:.0} nit SDR white"
                            ))
                            .small()
                            .weak(),
                        );
                    }
                    let peak_label = match self.gpu.active_drt() {
                        DrtKind::None => "None",
                        DrtKind::RgbLogSigmoid => "RGB Log Sigmoid",
                        DrtKind::RgbReinhard => "RGB Reinhard",
                        _ => "HDR target",
                    };
                    ui.label(
                        RichText::new(format!(
                            "{peak_label} encoded peak {:.3}; 1.0 remains SDR white",
                            self.gpu.output_peak()
                        ))
                        .small()
                        .weak(),
                    );
                } else {
                    ui.label(
                        RichText::new(
                            "OS/display exposes no HDR scRGB surface; all DRTs use their SDR ranges",
                        )
                        .small()
                        .weak(),
                    );
                }
                if self.workspace_view == WorkspaceView::Analysis {
                    ui.separator();
                    ui.label("Distribution space");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.color_space, ColorSpace::Srgb, "sRGB");
                        ui.selectable_value(&mut self.color_space, ColorSpace::Oklab, "Oklab");
                    });
                    ui.label(
                        RichText::new("Every mapped pixel · left-drag rotate · right-click reset")
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
                let active_drt = self.gpu.active_drt();
                if exposure_changed {
                    self.gpu.set_exposure(self.exposure_ev);
                }
                if matches!(active_drt, DrtKind::RgbReinhard | DrtKind::OklabReinhard) {
                    let using_oklab = active_drt.is_oklab();
                    let mut linear_parameters = if using_oklab {
                        self.oklab_reinhard_parameters
                    } else {
                        self.rgb_reinhard_parameters
                    };
                    ui.separator();
                    let mut linear_changed = false;
                    ui.label(RichText::new(format!("{} tone scale", active_drt.label())).strong());
                    if using_oklab {
                        ui.label(RichText::new("Independent curve in Oklab L^3 · SDR output").small().weak());
                    }
                    linear_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut linear_parameters.linear_slope,
                                0.1..=4.0,
                            )
                            .step_by(0.01)
                            .text("Linear slope"),
                        )
                        .on_hover_text(
                            "Display-linear gain below Compression start; default 1",
                        )
                        .changed();
                    linear_parameters.constrain();
                    let maximum_start = linear_parameters.maximum_compression_start();
                    linear_changed |= ui.add(
                        egui::Slider::new(&mut linear_parameters.compression_start, 0.0..=maximum_start)
                            .step_by(0.001).max_decimals(3).text("Compression start")
                    ).on_hover_text("Scene-linear RGB or Oklab L^3 at the linear/shoulder join; default 0.18").changed();
                    linear_parameters.constrain();
                    let minimum_reach = linear_parameters.minimum_highlight_reach_ev();
                    linear_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut linear_parameters.highlight_reach_ev,
                                minimum_reach..=20.0,
                            )
                            .step_by(0.25)
                            .suffix(" EV")
                            .text("Highlight reach"),
                        )
                        .on_hover_text(
                            "Scene stops above 18% gray that first reach the display peak; shorter reach permits more curve overexposure and preserves stronger mid-highlight color",
                        )
                        .changed();
                    if !using_oklab {
                        linear_changed |= ui.add(
                            egui::Slider::new(&mut linear_parameters.gamut_compression, 0.0..=0.8)
                                .step_by(0.01).text("Gamut compression")
                        ).on_hover_text("Virtual RGB inset toward the neutral axis; default 0.04").changed();
                    }
                    if !using_oklab {
                        linear_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut linear_parameters.hue_retention,
                                    0.0..=1.0,
                                )
                                .step_by(0.01)
                                .custom_formatter(|value, _| format!("{:.0}%", value * 100.0))
                                .text("Hue retention"),
                            )
                            .on_hover_text(
                                "Moves the mapped hue toward the original sRGB hue along the shortest angular path; mapped saturation and value stay unchanged",
                            )
                            .changed();
                    }
                    linear_parameters.constrain();
                    let curve_headroom = if !using_oklab {
                        self.gpu.output_headroom()
                    } else {
                        1.0
                    };
                    let curve = linear_parameters.curve_for_headroom(curve_headroom);
                    let mapped_gray = curve.map_linear(0.18);
                    let effective_reach = curve.highlight_reach_ev;
                    let curve_details = format!(
                        "Curve asymptote {:.3}× · display clamp {:.3}× · linear slope {:.3} · shoulder rate {:.3}",
                        curve.curve_peak, curve_headroom, curve.linear_slope, curve.shoulder_scale,
                    );
                    ui.label(
                        RichText::new(format!(
                            "18% gray → {:.3} · SDR reach {:+.2} EV · effective reach {:+.2} EV",
                            mapped_gray,
                            linear_parameters.highlight_reach_ev,
                            effective_reach,
                        ))
                        .small()
                        .weak(),
                    );
                    ui.label(RichText::new(curve_details).small().weak());
                    if linear_changed {
                        if using_oklab {
                            self.oklab_reinhard_parameters = linear_parameters;
                            self.gpu.set_oklab_reinhard_parameters(linear_parameters);
                        } else {
                            self.rgb_reinhard_parameters = linear_parameters;
                            self.gpu.set_rgb_reinhard_parameters(linear_parameters);
                        }
                    }
                }
                if active_drt == DrtKind::AgxS2O3 {
                    let mut log_sigmoid_parameters = self.agx_s2o3_parameters;
                    ui.separator();
                    ui.label(RichText::new("AgX-S2O3 reference scale").strong());

                    let mut log_sigmoid_changed = ui
                        .add(
                            egui::Slider::new(
                                &mut log_sigmoid_parameters.shadow_ev,
                                -20.0..=-1.0,
                            )
                            .step_by(0.25)
                            .suffix(" EV")
                            .text("Shadow reach"),
                        )
                        .on_hover_text("Scene stops below 18% gray mapped into the output range")
                        .changed();
                    log_sigmoid_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut log_sigmoid_parameters.highlight_ev,
                                1.0..=20.0,
                            )
                            .step_by(0.25)
                            .suffix(" EV")
                            .text("Highlight reach"),
                        )
                        .on_hover_text(
                            "Scene stops above 18% gray mapped to SDR white",
                        )
                        .changed();
                    log_sigmoid_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut log_sigmoid_parameters.output_pivot,
                                0.1..=0.9,
                            )
                            .step_by(0.01)
                            .text("Output middle gray"),
                        )
                        .on_hover_text("Display-encoded signal value assigned to scene-linear 18% gray")
                        .changed();

                    log_sigmoid_parameters.constrain();
                    let minimum_slope = log_sigmoid_parameters.minimum_pivot_slope() + 1.0e-3;
                    log_sigmoid_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut log_sigmoid_parameters.pivot_slope,
                                minimum_slope..=32.0,
                            )
                            .step_by(0.05)
                            .text("Mid contrast"),
                        )
                        .on_hover_text("Slope at middle gray in normalized log2 space")
                        .changed();
                    log_sigmoid_changed |= ui
                        .add(
                            egui::Slider::new(&mut log_sigmoid_parameters.toe_power, 1.0..=8.0)
                                .step_by(0.05)
                                .text("Toe power"),
                        )
                        .on_hover_text("Higher values retain a straighter midrange, then enter black more sharply")
                        .changed();
                    log_sigmoid_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut log_sigmoid_parameters.shoulder_power,
                                1.0..=8.0,
                            )
                            .step_by(0.05)
                            .text("Shoulder power"),
                        )
                        .on_hover_text("Higher values delay highlight compression and approach white more sharply")
                        .changed();
                    log_sigmoid_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut log_sigmoid_parameters.gamut_compression,
                                0.0..=0.8,
                            )
                            .step_by(0.01)
                            .text("Gamut compression"),
                        )
                        .on_hover_text("Inset toward the neutral axis before the per-channel curve; 0.2 is the AgX-S2O3 reference setting")
                        .changed();

                    log_sigmoid_parameters.constrain();
                    ui.label(RichText::new(format!(
                        "{:.2} stops · input pivot {:.3} · slope min {:.3}",
                        log_sigmoid_parameters.highlight_ev - log_sigmoid_parameters.shadow_ev,
                        log_sigmoid_parameters.input_pivot(),
                        log_sigmoid_parameters.minimum_pivot_slope(),
                    )).small().weak());
                    self.agx_s2o3_parameters = log_sigmoid_parameters;
                    if log_sigmoid_changed {
                        self.gpu.set_agx_s2o3_parameters(log_sigmoid_parameters);
                    }
                }
                if active_drt.uses_linear_log_sigmoid() {
                    let using_oklab = active_drt == DrtKind::OklabLogSigmoid;
                    let mut source = if using_oklab {
                        self.oklab_log_sigmoid_parameters
                    } else {
                        self.rgb_log_sigmoid_parameters
                    };
                    ui.separator();
                    ui.label(RichText::new(format!("{} tone scale", active_drt.label())).strong());
                    if using_oklab {
                        ui.label(RichText::new("Independent curve in Oklab L^3 · SDR output").small().weak());
                    }
                    let mut changed = ui
                        .add(
                            egui::Slider::new(&mut source.linear_slope, 0.1..=4.0)
                                .step_by(0.01)
                                .text("Linear slope"),
                        )
                        .on_hover_text("Display-linear gain below Compression start; default 1")
                        .changed();
                    source.constrain();
                    let minimum_start = source.minimum_compression_start();
                    let maximum_start = source.maximum_compression_start();
                    changed |= ui.add(
                        egui::Slider::new(&mut source.compression_start, minimum_start..=maximum_start)
                            .step_by(0.001).max_decimals(3).text("Compression start")
                    ).on_hover_text("Linear/sigmoid join in scene-linear RGB or Oklab L^3; range preserves a tangent-matched shoulder within the 20 EV reach limit").changed();
                    source.constrain();
                    let minimum_reach = source.minimum_highlight_ev();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut source.highlight_ev, minimum_reach..=20.0)
                                .step_by(0.25)
                                .suffix(" EV")
                                .text("Highlight reach"),
                        )
                        .on_hover_text(if using_oklab {
                            "Stops in L^3 above 18% gray mapped to SDR white"
                        } else {
                            "Scene stops above 18% gray mapped to SDR white; HDR extends the shoulder automatically"
                        })
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut source.shoulder_power, 1.0..=8.0)
                                .step_by(0.05)
                                .text("Shoulder power"),
                        )
                        .on_hover_text("Higher values delay highlight compression and approach white more sharply")
                        .changed();
                    if !using_oklab {
                        changed |= ui.add(
                            egui::Slider::new(&mut source.gamut_compression, 0.0..=0.8)
                                .step_by(0.01).text("Gamut compression")
                        ).on_hover_text("Virtual RGB inset toward the neutral axis; default 0.04").changed();
                    }
                    source.constrain();
                    let curve = source.tone_scale();
                    ui.label(RichText::new(format!(
                        "Linear through {:.3} · join output {:.3}", source.compression_start, curve.output_pivot
                    )).small().weak());
                    if !using_oklab && self.gpu.output_peak() > 1.0 {
                        ui.label(RichText::new(format!(
                            "HDR highlight reach {:+.2} EV", source.output_highlight_ev(self.gpu.output_peak())
                        )).small().weak());
                    }
                    if using_oklab {
                        self.oklab_log_sigmoid_parameters = source;
                        if changed {
                            self.gpu.set_oklab_log_sigmoid_parameters(source);
                        }
                        ui.label(RichText::new("Fixed Oklab hue with soft chroma compression.").small().weak());
                    } else {
                        self.rgb_log_sigmoid_parameters = source;
                        if changed {
                            self.gpu.set_rgb_log_sigmoid_parameters(source);
                        }
                        ui.label(RichText::new("Neutral shadows stay linear; color processing can change colored shadows.").small().weak());
                    }
                }
                if active_drt == DrtKind::RgbLogSigmoid {
                    ui.separator();
                    ui.label(RichText::new("HSV hue repair").strong());
                    if ui
                        .add(
                            egui::Slider::new(&mut self.rgb_log_sigmoid_hue_retention, 0.0..=1.0)
                                .step_by(0.01)
                                .custom_formatter(|value, _| format!("{:.0}%", value * 100.0))
                                .text("Hue retention"),
                        )
                        .on_hover_text("Uniform original-hue retention across all brightness levels; saturation and value stay unchanged")
                        .changed()
                    {
                        self.gpu.set_rgb_log_sigmoid_hue_retention(self.rgb_log_sigmoid_hue_retention);
                    }
                }
                if active_drt.is_oklab() {
                    let mut source = if active_drt == DrtKind::OklabReinhard {
                        self.oklab_reinhard_chroma
                    } else {
                        self.oklab_log_sigmoid_chroma
                    };
                    ui.separator();
                    ui.label(RichText::new("Oklab chroma").strong());
                    let mut changed = ui
                        .add(
                            egui::Slider::new(&mut source.highlight_chroma_power, 1.0..=32.0)
                                .step_by(0.25).text("Highlight chroma power"),
                        )
                        .on_hover_text("Higher powers keep chroma longer before fading to white; default 12")
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut source.gamut_rounding_power, 1.0..=8.0)
                                .step_by(0.1).text("Gamut rounding power"),
                        )
                        .on_hover_text("Higher powers round the cusp less and retain more chroma near the gamut boundary; default 4")
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut source.endpoint_compression_power, 1.0..=64.0)
                                .step_by(0.5).text("Endpoint chroma power"),
                        )
                        .on_hover_text("Chroma compression power near black and white; higher retains more chroma, default 32")
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut source.midtone_compression_power, 1.0..=64.0)
                                .step_by(0.5).text("Midtone chroma power"),
                        )
                        .on_hover_text("Chroma compression power at Oklab L=0.5, blended smoothly toward the endpoint power; default 16")
                        .changed();
                    source.constrain();
                    if active_drt == DrtKind::OklabReinhard {
                        self.oklab_reinhard_chroma = source;
                    } else {
                        self.oklab_log_sigmoid_chroma = source;
                    }
                    if changed {
                        self.gpu.set_oklab_chroma_parameters(active_drt, source);
                    }
                }
                if ui
                    .checkbox(&mut self.show_anomalies, "Show anomalies")
                    .on_hover_text(
                        "NaN magenta · +Inf yellow · -Inf cyan · mixed ±Inf orange · negative blue · above target range red",
                    )
                    .changed()
                {
                    self.gpu.set_show_anomalies(self.show_anomalies);
                }
                if self.show_anomalies {
                    egui::Grid::new("anomaly_legend")
                        .num_columns(4)
                        .spacing([5.0, 2.0])
                        .show(ui, |ui| {
                            anomaly_key(ui, Color32::MAGENTA, "NaN");
                            anomaly_key(ui, Color32::YELLOW, "+Inf");
                            ui.end_row();
                            anomaly_key(ui, Color32::CYAN, "-Inf");
                            anomaly_key(ui, Color32::from_rgb(255, 89, 0), "±Inf");
                            ui.end_row();
                            anomaly_key(ui, Color32::from_rgb(0, 64, 255), "< 0");
                            anomaly_key(ui, Color32::from_rgb(255, 13, 0), "> range");
                            ui.end_row();
                        });
                }
                if ui.button("Reset parameters").clicked() {
                    self.exposure_ev = 0.0;
                    self.agx_s2o3_parameters = LogSigmoidParameters::s2o3_reference();
                    self.rgb_log_sigmoid_parameters = LinearLogSigmoidParameters::default();
                    self.oklab_log_sigmoid_parameters = LinearLogSigmoidParameters::default();
                    self.rgb_log_sigmoid_hue_retention = 0.5;
                    self.oklab_reinhard_parameters = ReinhardParameters::oklab_default();
                    self.oklab_reinhard_chroma = OklabChromaParameters::default();
                    self.oklab_log_sigmoid_chroma = OklabChromaParameters::default();
                    self.gpu.set_oklab_chroma_parameters(DrtKind::OklabReinhard, self.oklab_reinhard_chroma);
                    self.gpu.set_oklab_chroma_parameters(DrtKind::OklabLogSigmoid, self.oklab_log_sigmoid_chroma);
                    self.rgb_reinhard_parameters = ReinhardParameters::default();
                    self.show_anomalies = false;
                    self.gpu.set_exposure(self.exposure_ev);
                    self.gpu.set_agx_s2o3_parameters(self.agx_s2o3_parameters);
                    self.gpu.set_rgb_log_sigmoid_parameters(self.rgb_log_sigmoid_parameters);
                    self.gpu.set_oklab_log_sigmoid_parameters(self.oklab_log_sigmoid_parameters);
                    self.gpu.set_rgb_log_sigmoid_hue_retention(
                        self.rgb_log_sigmoid_hue_retention,
                    );
                    self.gpu
                        .set_oklab_reinhard_parameters(self.oklab_reinhard_parameters);
                    self.gpu
                        .set_rgb_reinhard_parameters(self.rgb_reinhard_parameters);
                    self.gpu.set_show_anomalies(false);
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
                        .max_height(300.0)
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
            });

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("GPU: {}", self.gpu.adapter_name()));
                ui.separator();
                ui.label(format!("Backend: {:?}", self.gpu.backend()));
                ui.separator();
                let drt_label = if self.comparison.enabled {
                    format!(
                        "Left: {} · Right: {}",
                        self.gpu.active_drt().label(),
                        self.comparison.right_drt.label()
                    )
                } else {
                    format!("DRT: {}", self.gpu.active_drt().label())
                };
                ui.label(drt_label);
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
                WorkspaceView::Image => {
                    if self.comparison.enabled {
                        let bounds = ui.available_rect_before_wrap();
                        self.paint_workspace_image(ui, bounds);
                        ui.advance_cursor_after_rect(bounds);
                    } else {
                        show_image(
                            ui,
                            self.gpu.texture_id(),
                            self.gpu.width(),
                            self.gpu.height(),
                        );
                    }
                }
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
                        if self.comparison.enabled {
                            "DRT comparison"
                        } else {
                            "Mapped image"
                        },
                        egui::TextStyle::Body,
                        ui.visuals().strong_text_color(),
                    );
                    self.paint_workspace_image(
                        ui,
                        egui::Rect::from_min_max(
                            egui::pos2(left_rect.left(), left_rect.top() + header_height),
                            left_rect.max,
                        ),
                    );

                    paint_centered_label(
                        ui,
                        egui::pos2(right_rect.center().x, right_rect.top()),
                        &if self.comparison.enabled {
                            format!(
                                "{} · {} distribution (left)",
                                self.gpu.active_drt().label(),
                                self.color_space.label()
                            )
                        } else {
                            format!("{} distribution", self.color_space.label())
                        },
                        egui::TextStyle::Body,
                        ui.visuals().strong_text_color(),
                    );
                    paint_centered_label(
                        ui,
                        egui::pos2(right_rect.center().x, right_rect.top() + 21.0),
                        &format!(
                            "{} points · 0–{:.2}× SDR white",
                            format_point_count(u64::from(point_count)),
                            self.gpu.active_output_headroom(),
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
                        self.gpu.active_output_headroom(),
                        point_count,
                    );
                    let hint = if self.color_space == ColorSpace::Srgb
                        && self.gpu.active_output_headroom() > 1.0
                    {
                        "Left-drag to rotate · right-click to reset\nOuter cube: HDR output range · Inner cube: SDR 0–1"
                    } else {
                        "Left-drag to rotate · right-click to restore the default view"
                    };
                    response.on_hover_text(hint);
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
            self.reload_visible_shaders();
        }
        if close {
            context.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn paint_workspace_image(&mut self, ui: &mut egui::Ui, bounds: egui::Rect) {
        if let Some(right_texture) = self.gpu.comparison_texture_id() {
            self.comparison.paint(
                ui,
                bounds,
                self.gpu.texture_id(),
                right_texture,
                Vec2::new(self.gpu.width() as f32, self.gpu.height() as f32),
                self.gpu.active_drt().label(),
                self.comparison.right_drt.label(),
            );
        } else {
            paint_image_in_rect(
                ui,
                self.gpu.texture_id(),
                self.gpu.width(),
                self.gpu.height(),
                bounds,
            );
        }
    }
}

fn drt_selector(ui: &mut egui::Ui, id: &str, selected: &mut DrtKind) {
    egui::ComboBox::from_id_salt(id)
        .width(195.0)
        .selected_text(selected.label())
        .show_ui(ui, |ui| {
            ui.selectable_value(selected, DrtKind::None, DrtKind::None.label());
            ui.separator();
            ui.label(RichText::new("Third-party references").strong());
            for kind in DrtKind::REFERENCES {
                ui.selectable_value(selected, kind, kind.label())
                    .on_hover_text(kind.description());
            }
            ui.separator();
            ui.label(RichText::new("Research transforms").strong());
            for kind in DrtKind::RESEARCH {
                ui.selectable_value(selected, kind, kind.label())
                    .on_hover_text(kind.description());
            }
        });
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

fn anomaly_key(ui: &mut egui::Ui, color: Color32, label: &str) {
    ui.colored_label(color, "■");
    ui.label(RichText::new(label).small());
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
