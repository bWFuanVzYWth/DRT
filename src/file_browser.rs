use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
};

use anyhow::{Context, Result};
use eframe::egui;
use image::DynamicImage;

const THUMBNAIL_WIDTH: u32 = 192;
const THUMBNAIL_HEIGHT: u32 = 128;

pub struct FolderBrowser {
    pub root: PathBuf,
    pub entries: Vec<FolderEntry>,
}

pub struct FolderEntry {
    pub path: PathBuf,
    pub name: String,
    pub thumbnail: Thumbnail,
}

pub enum Thumbnail {
    Pending,
    Ready(egui::TextureHandle),
    Failed,
}

impl FolderBrowser {
    pub fn scan(root: PathBuf) -> Result<Self> {
        let mut paths = std::fs::read_dir(&root)
            .with_context(|| format!("cannot read folder {}", root.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_file() && is_supported_image(path))
            .collect::<Vec<_>>();
        paths.sort_by_cached_key(|path| {
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase()
        });

        let entries = paths
            .into_iter()
            .map(|path| FolderEntry {
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                path,
                thumbnail: Thumbnail::Pending,
            })
            .collect();
        Ok(Self { root, entries })
    }

    pub fn paths(&self) -> Vec<PathBuf> {
        self.entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect()
    }
}

#[derive(Default)]
pub struct ThumbnailLoader {
    receiver: Option<Receiver<ThumbnailResult>>,
    cancel: Option<Arc<AtomicBool>>,
    texture_generation: u64,
}

impl ThumbnailLoader {
    pub fn start(&mut self, paths: Vec<PathBuf>, context: egui::Context) {
        self.cancel();
        self.texture_generation += 1;
        let generation = self.texture_generation;
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let (sender, receiver) = mpsc::channel();
        self.cancel = Some(cancel);
        self.receiver = Some(receiver);

        std::thread::spawn(move || {
            for (index, path) in paths.into_iter().enumerate() {
                if worker_cancel.load(Ordering::Relaxed) {
                    break;
                }
                let image = load_thumbnail(&path).ok();
                if sender.send(ThumbnailResult { index, path, image }).is_err() {
                    break;
                }
                context.request_repaint();
            }
        });

        // Keep texture names unique when a folder is reopened so egui never
        // aliases a stale image with the same filename.
        self.texture_generation = generation;
    }

    pub fn poll(&mut self, context: &egui::Context, browser: &mut FolderBrowser) {
        loop {
            let message = match self.receiver.as_ref().map(Receiver::try_recv) {
                Some(Ok(message)) => message,
                Some(Err(TryRecvError::Empty)) | None => break,
                Some(Err(TryRecvError::Disconnected)) => {
                    self.receiver = None;
                    break;
                }
            };
            let Some(entry) = browser.entries.get_mut(message.index) else {
                continue;
            };
            if entry.path != message.path {
                continue;
            }
            entry.thumbnail = match message.image {
                Some(image) => Thumbnail::Ready(context.load_texture(
                    format!("folder-thumb-{}-{}", self.texture_generation, message.index),
                    image,
                    egui::TextureOptions::LINEAR,
                )),
                None => Thumbnail::Failed,
            };
        }
    }

    fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.receiver = None;
    }
}

impl Drop for ThumbnailLoader {
    fn drop(&mut self) {
        self.cancel();
    }
}

struct ThumbnailResult {
    index: usize,
    path: PathBuf,
    image: Option<egui::ColorImage>,
}

pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "exr" | "hdr" | "png" | "jpg" | "jpeg" | "webp"
            )
        })
}

fn load_thumbnail(path: &Path) -> Result<egui::ColorImage> {
    let image = image::ImageReader::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .with_guessed_format()
        .context("cannot infer thumbnail format")?
        .decode()
        .with_context(|| format!("cannot decode thumbnail {}", path.display()))?;
    let is_hdr = matches!(
        image,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
    );
    let thumbnail = image.thumbnail(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT);
    if is_hdr {
        hdr_thumbnail(thumbnail)
    } else {
        let rgba = thumbnail.to_rgba8();
        Ok(egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        ))
    }
}

fn hdr_thumbnail(image: DynamicImage) -> Result<egui::ColorImage> {
    let rgba = image.to_rgba32f();
    let mut colors = Vec::with_capacity(rgba.width() as usize * rgba.height() as usize);
    let mut log_luminance = 0.0_f32;
    let mut measured = 0_u32;
    for pixel in rgba.pixels() {
        let rgb = ap0_to_rec709([pixel[0], pixel[1], pixel[2]]);
        let luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).max(1.0e-6);
        if luminance.is_finite() {
            log_luminance += luminance.ln();
            measured += 1;
        }
        colors.push((rgb, pixel[3]));
    }
    let average = if measured == 0 {
        0.18
    } else {
        (log_luminance / measured as f32).exp()
    };
    let exposure = (0.18 / average.max(1.0e-6)).clamp(1.0 / 1024.0, 1024.0);
    let mut bytes = Vec::with_capacity(colors.len() * 4);
    for (rgb, alpha) in colors {
        for channel in rgb {
            let mapped = (channel * exposure).max(0.0);
            let mapped = mapped / (1.0 + mapped);
            bytes.push((linear_to_srgb(mapped).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        bytes.push((alpha.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    Ok(egui::ColorImage::from_rgba_unmultiplied(
        [rgba.width() as usize, rgba.height() as usize],
        &bytes,
    ))
}

fn ap0_to_rec709(rgb: [f32; 3]) -> [f32; 3] {
    [
        2.521_649 * rgb[0] - 1.136_888 * rgb[1] - 0.384_761 * rgb[2],
        -0.275_213 * rgb[0] + 1.369_705 * rgb[1] - 0.094_492 * rgb[2],
        -0.015_925 * rgb[0] - 0.147_806 * rgb[1] + 1.163_731 * rgb[2],
    ]
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_supported_extensions_case_insensitively() {
        assert!(is_supported_image(Path::new("frame.EXR")));
        assert!(is_supported_image(Path::new("frame.jpeg")));
        assert!(!is_supported_image(Path::new("notes.txt")));
    }
}
