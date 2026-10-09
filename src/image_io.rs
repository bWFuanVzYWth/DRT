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
    let linear_ap0 = matches!(
        image,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
    );
    let rgba = image.to_rgba32f();
    let (width, height) = rgba.dimensions();
    let data = rgba
        .pixels()
        .flat_map(|pixel| {
            let ap0 = if linear_ap0 {
                [pixel[0], pixel[1], pixel[2]]
            } else {
                // Standard SDR files are interpreted as display-encoded sRGB.
                let linear_rec709 = [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                ];
                rec709_to_ap0(linear_rec709)
            };
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

const COLOR_TRAJECTORY_LEVELS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const COLOR_TRAJECTORY_BANDS: usize = 61;
const COLOR_TRAJECTORY_MAX_GAIN: f64 = 16384.0;

fn color_trajectory_colors() -> [[f32; 3]; COLOR_TRAJECTORY_BANDS] {
    let mut colors = [[0.0; 3]; COLOR_TRAJECTORY_BANDS];
    let mut index = 0;
    for r in COLOR_TRAJECTORY_LEVELS {
        for g in COLOR_TRAJECTORY_LEVELS {
            for b in COLOR_TRAJECTORY_LEVELS {
                if r == 1.0 || g == 1.0 || b == 1.0 {
                    colors[index] = [r, g, b];
                    index += 1;
                }
            }
        }
    }
    colors
}

/// 61 discrete linear-sRGB colors, each following the same brightness ray.
/// The black-inclusive exponential ramp spans shadows through scene gain 2^14.
/// All pixels are converted to the workbench's scene-linear AP0 input space.
pub fn color_trajectory_pattern(width: u32, height: u32) -> LinearImage {
    let colors = color_trajectory_colors().map(rec709_to_ap0);
    let denominator = f64::from(width.saturating_sub(1).max(1));
    let gains: Vec<f32> = (0..width)
        .map(|x| {
            let u = f64::from(x) / denominator;
            // Shift the usual -10..+14 EV ramp down to exact black, then
            // normalize its endpoint. Computing in f64 avoids cancellation
            // near black and rounding the peak above 2^14.
            ((2.0_f64.powf(24.0 * u) - 1.0) / (2.0_f64.powi(24) - 1.0) * COLOR_TRAJECTORY_MAX_GAIN)
                as f32
        })
        .collect();
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        // Integer allocation gives all 61 bands equal height at the default
        // 976 rows, without a one-pixel final band or floating-point seams.
        let band = (u64::from(y) * COLOR_TRAJECTORY_BANDS as u64 / u64::from(height)) as usize;
        let color = colors[band];
        for gain in &gains {
            rgba.extend_from_slice(&[color[0] * gain, color[1] * gain, color[2] * gain, 1.0]);
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
    fn floating_point_ap0_is_not_converted_again() {
        let source = [2.0, -0.25, 0.5];
        let image =
            DynamicImage::ImageRgb32F(image::Rgb32FImage::from_pixel(1, 1, image::Rgb(source)));

        let loaded = dynamic_to_linear_ap0(image);

        assert_eq!(loaded.rgba, [source[0], source[1], source[2], 1.0]);
    }

    #[test]
    fn pattern_has_expected_shape() {
        let pattern = test_pattern(16, 8);
        assert_eq!((pattern.width, pattern.height), (16, 8));
        assert_eq!(pattern.rgba.len(), 16 * 8 * 4);
        assert!(pattern.rgba.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn color_trajectories_cover_the_requested_grid_and_hdr_rays() {
        let colors = color_trajectory_colors();
        let mut seen = std::collections::HashSet::new();
        for rgb in colors {
            assert!(rgb.contains(&1.0));
            assert!(
                rgb.into_iter()
                    .all(|channel| COLOR_TRAJECTORY_LEVELS.contains(&channel))
            );
            assert!(seen.insert(rgb.map(|channel| (channel * 4.0) as u8)));
        }
        assert_eq!(seen.len(), 5_usize.pow(3) - 4_usize.pow(3));

        let width = 65;
        let height = COLOR_TRAJECTORY_BANDS as u32 * 2;
        let image = color_trajectory_pattern(width, height);
        assert_eq!(image.rgba.len(), width as usize * height as usize * 4);
        for (band, color) in colors.into_iter().enumerate() {
            let start = band * 2 * width as usize * 4;
            let row = &image.rgba[start..start + width as usize * 4];
            assert_eq!(
                row,
                &image.rgba[start + width as usize * 4..start + 2 * width as usize * 4]
            );
            assert_eq!(&row[..4], &[0.0, 0.0, 0.0, 1.0]);
            let endpoint =
                rec709_to_ap0(color).map(|channel| channel * COLOR_TRAJECTORY_MAX_GAIN as f32);
            assert_eq!(&row[row.len() - 4..row.len() - 1], &endpoint);
            for pair in row.as_chunks::<4>().0.windows(2) {
                assert!(
                    pair[1][..3]
                        .iter()
                        .zip(&pair[0][..3])
                        .all(|(next, prev)| next >= prev)
                );
            }
            for pixel in row.as_chunks::<4>().0 {
                assert_eq!(pixel[3], 1.0);
                assert!(
                    pixel[..3]
                        .iter()
                        .all(|value| value.is_finite() && *value >= 0.0 && *value < 65504.0)
                );
                // Every intermediate pixel lies on the same AP0 ray; there
                // is no interpolation between neighboring discrete colors.
                let gain = pixel[0] / endpoint[0];
                for channel in 1..3 {
                    assert!((pixel[channel] - endpoint[channel] * gain).abs() < 0.003);
                }
            }
        }
    }

    #[test]
    fn color_trajectory_pattern_handles_small_and_empty_dimensions() {
        for (width, height) in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 3), (3, 67)] {
            let image = color_trajectory_pattern(width, height);
            assert_eq!((image.width, image.height), (width, height));
            assert_eq!(image.rgba.len(), width as usize * height as usize * 4);
            assert!(image.rgba.iter().all(|value| value.is_finite()));
        }
    }
}
