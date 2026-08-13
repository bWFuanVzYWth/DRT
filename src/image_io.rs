use std::path::Path;

use anyhow::{Context, Result};
use image::DynamicImage;

#[derive(Clone, Debug)]
pub struct LinearImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<f32>,
}

pub fn load(path: &Path) -> Result<LinearImage> {
    let image = image::ImageReader::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .with_guessed_format()
        .context("cannot infer image format")?
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))?;
    Ok(dynamic_to_linear_ap0(image))
}

fn dynamic_to_linear_ap0(image: DynamicImage) -> LinearImage {
    let hdr = matches!(
        image,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
    );
    let rgba = image.to_rgba32f();
    let (width, height) = rgba.dimensions();
    let data = rgba
        .pixels()
        .flat_map(|pixel| {
            let rgb = if hdr {
                [pixel[0], pixel[1], pixel[2]]
            } else {
                [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                ]
            };
            // Standard SDR files are interpreted as display-linear Rec.709 and converted to AP0.
            let ap0 = rec709_to_ap0(rgb);
            [ap0[0], ap0[1], ap0[2], pixel[3]]
        })
        .collect();
    LinearImage {
        width,
        height,
        rgba: data,
    }
}

pub fn test_pattern(width: u32, height: u32) -> LinearImage {
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / (width - 1).max(1) as f32;
            let v = y as f32 / (height - 1).max(1) as f32;
            let exposure = 2.0_f32.powf(-10.0 + 16.0 * u);
            let band = (v * 7.0).floor().min(6.0) as usize;
            let colors = [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ];
            let linear709 = colors[band].map(|channel| channel * exposure);
            let ap0 = rec709_to_ap0(linear709);
            rgba.extend_from_slice(&[ap0[0], ap0[1], ap0[2], 1.0]);
        }
    }
    LinearImage {
        width,
        height,
        rgba,
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

// Keep the original development-harness transform coefficients verbatim.
#[allow(clippy::excessive_precision)]
fn rec709_to_ap0(rgb: [f32; 3]) -> [f32; 3] {
    [
        0.4396430040 * rgb[0] + 0.3830054714 * rgb[1] + 0.1773993089 * rgb[2],
        0.0897157319 * rgb[0] + 0.8134750538 * rgb[1] + 0.0967822524 * rgb[2],
        0.0175127205 * rgb[0] + 0.1115514385 * rgb[1] + 0.8708827930 * rgb[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_endpoints_are_exact() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn rec709_white_stays_ap0_white() {
        let white = rec709_to_ap0([1.0; 3]);
        assert!(white.into_iter().all(|value| (value - 1.0).abs() < 6.0e-5));
    }

    #[test]
    fn pattern_has_expected_shape() {
        let pattern = test_pattern(16, 8);
        assert_eq!((pattern.width, pattern.height), (16, 8));
        assert_eq!(pattern.rgba.len(), 16 * 8 * 4);
        assert!(pattern.rgba.iter().all(|value| value.is_finite()));
    }
}
