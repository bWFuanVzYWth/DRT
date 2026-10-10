// Calibrate the log-input / display-linear-output plot in screen coordinates.

#[test]
fn tone_curve_linear_y_has_exact_black_and_equal_light_intervals() {
    let rect = egui::Rect::from_min_size(egui::pos2(27.0, 41.0), egui::vec2(510.0, 204.0));
    for peak in [1.0_f32, 4.0, 64.0] {
        let positions =
            [0.0, 0.25, 0.5, 0.75, 1.0].map(|fraction| map_y(rect, fraction * peak, peak));
        assert_eq!(positions[0], rect.bottom(), "black must map to the bottom");
        assert_eq!(
            positions[4],
            rect.top(),
            "actual HDR/SDR peak must map to the top"
        );
        for pair in positions.windows(2) {
            assert!(
                (pair[0] - pair[1] - rect.height() / 4.0).abs() < 0.0001,
                "linear output intervals are not equally spaced at peak {peak}: {positions:?}"
            );
        }
        let middle_gray = map_y(rect, 0.18, peak);
        assert!(
            (middle_gray - (rect.bottom() - rect.height() * 0.18 / peak)).abs() < 0.0001,
            "18% display-linear gray is misplaced at peak {peak}"
        );
    }
}

#[test]
fn tone_curve_identity_is_exponential_and_stops_at_the_real_peak_crossing() {
    let rect = egui::Rect::from_min_size(egui::pos2(27.0, 41.0), egui::vec2(510.0, 204.0));
    for (ev, fraction) in [(-12.0, 0.0), (0.0, 0.5), (12.0, 1.0)] {
        assert!((map_x(rect, ev) - (rect.left() + fraction * rect.width())).abs() < 0.0001);
    }
    for peak in [1.0_f32, 4.0, 64.0] {
        let points = identity_points(rect, peak);
        assert!(points.len() > 32);
        assert!((points[0].x - rect.left()).abs() < 0.0001);
        let last = points.last().unwrap();
        let peak_ev = (peak / 0.18).log2();
        // The screen calibration stays fixed at ±12 EV for every display peak.
        let crossing = rect.left() + rect.width() * (peak_ev + 12.0) / 24.0;
        assert!((last.x - crossing).abs() < 0.0001);
        assert!((last.y - rect.top()).abs() < 0.0001);
        assert!(
            last.x < rect.right(),
            "reference must stop before the plot's maximum input EV"
        );
        assert!(
            points[..points.len() - 1]
                .iter()
                .all(|point| point.y > rect.top() + 0.05),
            "reference contains a clamped horizontal tail"
        );
        // Deep HDR shadows can differ by less than one screen-coordinate
        // f32 ULP. Allow tied y positions there; the visible anchors below
        // still verify distinct, calibrated linear-light intervals.
        assert!(
            points
                .windows(2)
                .all(|pair| pair[1].x > pair[0].x && pair[1].y <= pair[0].y)
        );

        // Independent anchors: half the peak is one stop below its crossing,
        // quarter is two stops below. These screen positions would be wrong
        // if the reference remained a diagonal or the y axis were logarithmic.
        for fraction in [0.25_f32, 0.5, 0.75] {
            let ev = peak_ev + fraction.log2();
            let x = rect.left() + rect.width() * (ev + 12.0) / 24.0;
            let pair = points
                .windows(2)
                .find(|pair| pair[0].x <= x && pair[1].x >= x)
                .unwrap();
            let t = (x - pair[0].x) / (pair[1].x - pair[0].x);
            let interpolated_y = pair[0].y + t * (pair[1].y - pair[0].y);
            let expected_y = rect.bottom() - fraction * rect.height();
            assert!(
                (interpolated_y - expected_y).abs() < 0.05,
                "identity misses {fraction}× peak {peak}: expected y {expected_y}, got {interpolated_y}"
            );
        }
        assert!(
            points[points.len() / 2].y > rect.bottom() - 0.01 * rect.height(),
            "identity still resembles a diagonal instead of an exponential"
        );
    }
}
