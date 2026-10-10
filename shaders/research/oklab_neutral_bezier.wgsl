// SPDX-License-Identifier: GPL-3.0-only
// Experimental highlight-color retention helpers, compiled by the opt-in GPU
// audit in src/oklab_bezier_validation.rs. The production DRT is unchanged.
// tail is the scalar shoulder's remaining fraction, decreasing from 1 to 0.
// These are parametric Bezier curves with uniformly spaced x controls, so the
// x coordinate is tail itself and no inverse curve search is needed.

// Cubic retention controls [0, 1, 1, 1]. Equivalent to quartic k = 3.
fn bezierHighlightRetentionCubic(tail: f32) -> f32 {
    return tail * (3.0 + tail * (-3.0 + tail));
}

// Quartic retention controls [0, k/4, 1, 1, 1], 0 <= k <= 4.
// k = 1 reproduces the current DRT. k controls the white-end tangent;
// the first two derivatives at the linear join remain zero.
fn bezierHighlightRetentionQuartic(tail: f32, k: f32) -> f32 {
    return tail * (k + tail * ((6.0 - 3.0 * k)
        + tail * ((3.0 * k - 8.0) + (3.0 - k) * tail)));
}

// Quintic fade controls [0, 0, 0, a, b, 1], 0 <= a <= b <= 1.
// Retention controls are [0, 1-b, 1-a, 1, 1, 1]. The white-end tangent
// is 5*(1-b); a independently changes the interior shape. a=.3, b=.8
// is a degree elevation of the current quartic, not a different curve.
fn bezierHighlightRetentionQuintic(tail: f32, a: f32, b: f32) -> f32 {
    let c1 = 5.0 * (1.0 - b);
    let c2 = 10.0 * (2.0 * b - a - 1.0);
    let c3 = 10.0 * (1.0 + 3.0 * a - 3.0 * b);
    let c4 = 5.0 * (-1.0 - 6.0 * a + 4.0 * b);
    let c5 = 1.0 + 10.0 * a - 5.0 * b;
    return tail * (c1 + tail * (c2 + tail * (c3 + tail * (c4 + tail * c5))));
}
