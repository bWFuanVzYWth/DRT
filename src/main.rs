mod distribution;
mod file_browser;
mod gpu;
mod image_io;
mod presenter;
mod tone_curve;

use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
};

use eframe::egui::{self, Color32, RichText, Vec2};

use crate::distribution::{ColorSpace, DEFAULT_PITCH, DEFAULT_YAW, DistributionRenderer};
use crate::file_browser::{FolderBrowser, Thumbnail, ThumbnailLoader};
use crate::gpu::{AgxParameters, DrtGpu, DrtKind, ReinhardAgxParameters, ReinhardParameters};
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
    agx_s2o3_parameters: AgxParameters,
    agx_hsv_parameters: AgxParameters,
    agx_black_hue_retention: f32,
    agx_white_hue_retention: f32,
    oklab_reinhard_parameters: ReinhardParameters,
    reinhard_parameters: ReinhardParameters,
    reinhard_agx_parameters: ReinhardAgxParameters,
    show_anomalies: bool,
    image_name: String,
    status: String,
    status_error: bool,
    shader_paths: [PathBuf; DrtKind::ALL.len()],
    shader_modified: [Option<std::time::SystemTime>; DrtKind::ALL.len()],
    last_shader_check: std::time::Instant,
    workspace_view: WorkspaceView,
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
            agx_s2o3_parameters: AgxParameters::s2o3_reference(),
            agx_hsv_parameters: AgxParameters::hsv_default(),
            agx_black_hue_retention: 1.0,
            agx_white_hue_retention: 0.5,
            oklab_reinhard_parameters: ReinhardParameters::oklab_default(),
            reinhard_parameters: ReinhardParameters::default(),
            reinhard_agx_parameters: ReinhardAgxParameters::default(),
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
                        self.reload_shader(self.gpu.active_drt());
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
                ui.label("DRT");
                let mut selected_drt = self.gpu.active_drt();
                ui.horizontal_wrapped(|ui| {
                    for drt in DrtKind::ALL {
                        ui.selectable_value(&mut selected_drt, drt, drt.label());
                    }
                });
                if selected_drt != self.gpu.active_drt() {
                    match selected_drt {
                        DrtKind::AgxS2O3 => {
                            self.gpu.set_agx_parameters(self.agx_s2o3_parameters)
                        }
                        DrtKind::AgxHsv => {
                            self.gpu.set_agx_parameters(self.agx_hsv_parameters)
                        }
                        DrtKind::ReinhardGamut => self
                            .gpu
                            .set_reinhard_parameters(self.reinhard_parameters),
                        DrtKind::ReinhardAgx => self
                            .gpu
                            .set_reinhard_agx_parameters(self.reinhard_agx_parameters),
                        DrtKind::None | DrtKind::Oklab => {}
                    }
                    self.gpu.set_drt(selected_drt);
                    self.set_status(format!("Switched to {}", selected_drt.label()), false);
                }
                ui.separator();
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
                        DrtKind::AgxHsv => "AgX-HSV",
                        DrtKind::ReinhardGamut => "Reinhard-Gamut",
                        DrtKind::ReinhardAgx => "Reinhard AgX",
                        DrtKind::Oklab | DrtKind::AgxS2O3 => "HDR target",
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
                if matches!(active_drt, DrtKind::ReinhardGamut | DrtKind::ReinhardAgx | DrtKind::Oklab) {
                    let using_agx = active_drt == DrtKind::ReinhardAgx;
                    let mut reinhard_agx = self.reinhard_agx_parameters;
                    let mut reinhard_parameters = if active_drt == DrtKind::Oklab {
                        self.oklab_reinhard_parameters
                    } else if using_agx {
                        reinhard_agx.base
                    } else {
                        self.reinhard_parameters
                    };
                    ui.separator();
                    let mut reinhard_changed = false;
                    if active_drt != DrtKind::Oklab {
                        ui.label(RichText::new(active_drt.label()).strong());
                        if using_agx {
                            ui.label(RichText::new("Linear shadows · AgX log shoulder · independent settings").small().weak());
                        }
                        reinhard_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut reinhard_parameters.gamut_expansion,
                                    0.0..=0.8,
                                )
                                .step_by(0.01)
                                .custom_formatter(|value, _| format!("{:.0}%", value * 100.0))
                                .text("Virtual gamut expansion"),
                            )
                            .on_hover_text(
                                "Moves the virtual primaries outward, so Rec.709 coordinates contract toward the neutral axis before the tone curve",
                            )
                            .changed();
                    } else {
                        ui.label(RichText::new("Oklab Reinhard curve").strong());
                        ui.label(
                            RichText::new("Oklab has independent curve settings and uses the SDR range.")
                                .small()
                                .weak(),
                        );
                    }
                    reinhard_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut reinhard_parameters.input_scale,
                                0.1..=8.0,
                            )
                            .step_by(0.01)
                            .text("Input scale"),
                        )
                        .on_hover_text(
                            "Defines the linear segment's slope using 18% gray; gray is compressed when Compression start is below 0.18",
                        )
                        .changed();
                    reinhard_parameters.constrain();
                    let maximum_start = reinhard_parameters.maximum_compression_start();
                    reinhard_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut reinhard_parameters.compression_start,
                                0.0..=maximum_start,
                            )
                            .step_by(0.001)
                            .max_decimals(3)
                            .text("Compression start"),
                        )
                        .on_hover_text(
                            "Scene-linear input where the straight segment joins the shoulder. The maximum keeps the join below SDR white",
                        )
                        .changed();
                    let minimum_reach = if using_agx {
                        reinhard_agx.base = reinhard_parameters;
                        reinhard_agx.minimum_highlight_reach_ev()
                    } else {
                        reinhard_parameters.minimum_highlight_reach_ev()
                    };
                    reinhard_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut reinhard_parameters.highlight_reach_ev,
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
                    if using_agx {
                        reinhard_changed |= ui
                            .add(egui::Slider::new(&mut reinhard_agx.shoulder_power, 1.0..=8.0)
                                .step_by(0.05)
                                .text("Shoulder power"))
                            .on_hover_text("AgX shoulder shape: higher values delay compression and approach white more sharply; the linear segment and its join stay fixed")
                            .changed();
                    }
                    if active_drt != DrtKind::Oklab {
                        reinhard_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut reinhard_parameters.hue_retention,
                                    0.0..=1.0,
                                )
                                .step_by(0.01)
                                .custom_formatter(|value, _| format!("{:.0}%", value * 100.0))
                                .text("HSV hue retention"),
                            )
                            .on_hover_text(
                                "Moves the mapped hue toward the original sRGB hue along the shortest angular path; mapped saturation and value stay unchanged",
                            )
                            .changed();
                    }
                    reinhard_parameters.constrain();
                    if using_agx {
                        reinhard_agx.base = reinhard_parameters;
                        reinhard_agx.constrain();
                        reinhard_parameters = reinhard_agx.base;
                    }
                    let curve_headroom = if active_drt != DrtKind::Oklab {
                        self.gpu.output_headroom()
                    } else {
                        1.0
                    };
                    let (mapped_gray, effective_reach, curve_details) = if using_agx {
                        let curve = reinhard_agx.curve_for_headroom(curve_headroom);
                        (curve.map_linear(0.18), curve.highlight_reach_ev, format!(
                            "Curve asymptote {:.3}× · display clamp {:.3}× · linear slope {:.3}",
                            curve.curve_peak, curve_headroom, curve.linear_slope,
                        ))
                    } else {
                        let curve = reinhard_parameters.curve_for_headroom(curve_headroom);
                        (curve.map_linear(0.18), curve.highlight_reach_ev, format!(
                            "Curve asymptote {:.3}× · display clamp {:.3}× · linear slope {:.3} · shoulder rate {:.3}",
                            curve.curve_peak, curve_headroom, curve.linear_slope, curve.shoulder_scale,
                        ))
                    };
                    ui.label(
                        RichText::new(format!(
                            "18% gray → {:.3} · SDR reach {:+.2} EV · effective reach {:+.2} EV",
                            mapped_gray,
                            reinhard_parameters.highlight_reach_ev,
                            effective_reach,
                        ))
                        .small()
                        .weak(),
                    );
                    ui.label(RichText::new(curve_details).small().weak());
                    if reinhard_changed {
                        if active_drt == DrtKind::Oklab {
                            self.oklab_reinhard_parameters = reinhard_parameters;
                            self.gpu.set_oklab_reinhard_parameters(reinhard_parameters);
                        } else if using_agx {
                            self.reinhard_agx_parameters = reinhard_agx;
                            self.gpu.set_reinhard_agx_parameters(reinhard_agx);
                        } else {
                            self.reinhard_parameters = reinhard_parameters;
                            self.gpu.set_reinhard_parameters(reinhard_parameters);
                        }
                    }
                }
                if active_drt.uses_agx() {
                    let mut agx_parameters = match active_drt {
                        DrtKind::AgxS2O3 => self.agx_s2o3_parameters,
                        DrtKind::AgxHsv => self.agx_hsv_parameters,
                        DrtKind::None | DrtKind::Oklab | DrtKind::ReinhardGamut | DrtKind::ReinhardAgx => unreachable!(),
                    };
                    ui.separator();
                    let heading = match active_drt {
                        DrtKind::AgxS2O3 => "AgX-S2O3 reference scale",
                        DrtKind::AgxHsv => "AgX-HSV tone scale",
                        DrtKind::None | DrtKind::Oklab | DrtKind::ReinhardGamut | DrtKind::ReinhardAgx => unreachable!(),
                    };
                    ui.label(RichText::new(heading).strong());

                    let mut agx_changed = ui
                        .add(
                            egui::Slider::new(
                                &mut agx_parameters.shadow_ev,
                                -20.0..=-1.0,
                            )
                            .step_by(0.25)
                            .suffix(" EV")
                            .text("Shadow reach"),
                        )
                        .on_hover_text("Scene stops below 18% gray mapped into the output range")
                        .changed();
                    agx_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut agx_parameters.highlight_ev,
                                1.0..=20.0,
                            )
                            .step_by(0.25)
                            .suffix(" EV")
                            .text("Highlight reach"),
                        )
                        .on_hover_text(
                            "Scene stops above 18% gray mapped into SDR white at 1×; HDR headroom extends this reach automatically",
                        )
                        .changed();
                    agx_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut agx_parameters.output_pivot,
                                0.1..=0.9,
                            )
                            .step_by(0.01)
                            .text("Output middle gray"),
                        )
                        .on_hover_text("Display-encoded signal value assigned to scene-linear 18% gray")
                        .changed();

                    agx_parameters.constrain();
                    let minimum_slope = agx_parameters.minimum_pivot_slope() + 1.0e-3;
                    agx_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut agx_parameters.pivot_slope,
                                minimum_slope..=32.0,
                            )
                            .step_by(0.05)
                            .text("Mid contrast"),
                        )
                        .on_hover_text("Slope at middle gray in normalized log2 space")
                        .changed();
                    agx_changed |= ui
                        .add(
                            egui::Slider::new(&mut agx_parameters.toe_power, 1.0..=8.0)
                                .step_by(0.05)
                                .text("Toe power"),
                        )
                        .on_hover_text("Higher values retain a straighter midrange, then enter black more sharply")
                        .changed();
                    agx_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut agx_parameters.shoulder_power,
                                1.0..=8.0,
                            )
                            .step_by(0.05)
                            .text("Shoulder power"),
                        )
                        .on_hover_text("Higher values delay highlight compression and approach white more sharply")
                        .changed();
                    agx_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut agx_parameters.gamut_compression,
                                0.0..=0.8,
                            )
                            .step_by(0.01)
                            .text("Gamut compression"),
                        )
                        .on_hover_text("Inset toward the neutral axis before the per-channel curve; 0.2 is the original value")
                        .changed();

                    agx_parameters.constrain();
                    let range_description = if active_drt == DrtKind::AgxHsv
                        && self.gpu.output_peak() > 1.0
                    {
                        format!(
                            "SDR {:+.2}/{:+.2} EV · HDR highlight {:+.2} EV · pivot {:.3}",
                            agx_parameters.shadow_ev,
                            agx_parameters.highlight_ev,
                            agx_parameters.output_highlight_ev(self.gpu.output_peak()),
                            agx_parameters.input_pivot(),
                        )
                    } else {
                        format!(
                            "{:.2} stops · input pivot {:.3} · slope min {:.3}",
                            agx_parameters.highlight_ev - agx_parameters.shadow_ev,
                            agx_parameters.input_pivot(),
                            agx_parameters.minimum_pivot_slope(),
                        )
                    };
                    ui.label(RichText::new(range_description).small().weak());
                    match active_drt {
                        DrtKind::AgxS2O3 => self.agx_s2o3_parameters = agx_parameters,
                        DrtKind::AgxHsv => self.agx_hsv_parameters = agx_parameters,
                        DrtKind::None | DrtKind::Oklab | DrtKind::ReinhardGamut | DrtKind::ReinhardAgx => unreachable!(),
                    }
                    if agx_changed {
                        self.gpu.set_agx_parameters(agx_parameters);
                    }
                    if active_drt == DrtKind::AgxHsv {
                        ui.separator();
                        ui.label(RichText::new("HSV hue repair").strong());
                        let mut retention_changed = ui
                            .add(
                                egui::Slider::new(
                                    &mut self.agx_black_hue_retention,
                                    0.0..=1.0,
                                )
                                    .step_by(0.01)
                                    .custom_formatter(|value, _| format!("{:.0}%", value * 100.0))
                                    .text("Black retention"),
                            )
                            .on_hover_text(
                                "Original-hue retention applied where the AgX output value is zero",
                            )
                            .changed();
                        retention_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut self.agx_white_hue_retention,
                                    0.0..=1.0,
                                )
                                .step_by(0.01)
                                .custom_formatter(|value, _| format!("{:.0}%", value * 100.0))
                                .text("White retention"),
                            )
                            .on_hover_text(
                                "Original-hue retention applied where the AgX output value is one",
                            )
                            .changed();
                        ui.label(
                            RichText::new(
                                "Retention is linearly interpolated by the clamped AgX HSV value",
                            )
                            .small()
                            .weak(),
                        );
                        if retention_changed {
                            self.gpu.set_agx_hue_retention(
                                self.agx_black_hue_retention,
                                self.agx_white_hue_retention,
                            );
                        }
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
                    self.agx_s2o3_parameters = AgxParameters::s2o3_reference();
                    self.agx_hsv_parameters = AgxParameters::hsv_default();
                    self.agx_black_hue_retention = 1.0;
                    self.agx_white_hue_retention = 0.5;
                    self.oklab_reinhard_parameters = ReinhardParameters::oklab_default();
                    self.reinhard_parameters = ReinhardParameters::default();
                    self.reinhard_agx_parameters = ReinhardAgxParameters::default();
                    self.show_anomalies = false;
                    self.gpu.set_exposure(self.exposure_ev);
                    let active_agx_parameters = match self.gpu.active_drt() {
                        DrtKind::AgxS2O3 => Some(self.agx_s2o3_parameters),
                        DrtKind::AgxHsv => Some(self.agx_hsv_parameters),
                        DrtKind::None | DrtKind::Oklab | DrtKind::ReinhardGamut | DrtKind::ReinhardAgx => None,
                    };
                    if let Some(parameters) = active_agx_parameters {
                        self.gpu.set_agx_parameters(parameters);
                    }
                    self.gpu.set_agx_hue_retention(
                        self.agx_black_hue_retention,
                        self.agx_white_hue_retention,
                    );
                    self.gpu
                        .set_oklab_reinhard_parameters(self.oklab_reinhard_parameters);
                    self.gpu
                        .set_reinhard_parameters(self.reinhard_parameters);
                    self.gpu
                        .set_reinhard_agx_parameters(self.reinhard_agx_parameters);
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
                ui.label(format!("DRT: {}", self.gpu.active_drt().label()));
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
                    response.on_hover_text(
                        "Left-drag to rotate · right-click to restore the default view",
                    );
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
            self.reload_shader(self.gpu.active_drt());
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
