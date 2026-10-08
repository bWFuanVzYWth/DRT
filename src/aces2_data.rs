// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project.
// Rust port of aces-core Lib.Academy.OutputTransform a2.v1.
// Source commit 069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80; see references/entries/aces_20.md.
//! CPU initialization for the complete ACES 2 output transform.
//!
//! Storage layout (f32 indices): 0..41 input JMh parameters, 41..82 limiting
//! JMh parameters, 82..93 TSParams, 93..103 compression constants,
//! 103..465 padded reach M, 465..1551 padded cusp J/M/h, 1551..1913 gamma.
//! Each JMh block contains four column-major 3x3 matrices and F_L_n, cz,
//! inv_cz, A_w_J, inv_A_w_J. The ten constants are limit_J_max,
//! model_gamma_inv, sat, sat_thr, compr, chroma_compress_scale, mid_J,
//! focus_dist, lower_hull_gamma_inv, peakLuminance.

type V = [f64; 3];
type M = [[f64; 3]; 3];
const N: usize = 360;
const PAD: usize = N + 2;
const GAMMA: f64 = 0.59 * (1.48 + 0.4472135954999579);
const AP0: [[f64; 2]; 4] = [
    [0.7347, 0.2653],
    [0., 1.],
    [0.0001, -0.077],
    [0.32168, 0.33767],
];
const AP1: [[f64; 2]; 4] = [
    [0.713, 0.293],
    [0.165, 0.830],
    [0.128, 0.044],
    [0.32168, 0.33767],
];
const REC709: [[f64; 2]; 4] = [[0.64, 0.33], [0.3, 0.6], [0.15, 0.06], [0.3127, 0.329]];
const CAM16: [[f64; 2]; 4] = [
    [0.8336, 0.1735],
    [2.3854, -1.4659],
    [0.087, -0.125],
    [0.333, 0.333],
];

fn mv(m: M, v: V) -> V {
    m.map(|r| r[0] * v[0] + r[1] * v[1] + r[2] * v[2])
}
fn mm(a: M, b: M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn scale(v: V, s: f64) -> V {
    v.map(|x| x * s)
}
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
fn lerpv(a: V, b: V, t: f64) -> V {
    std::array::from_fn(|i| lerp(a[i], b[i], t))
}
fn inv(a: M) -> M {
    let d = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d,
        ],
    ]
}
fn rgb_xyz(pri: [[f64; 2]; 4]) -> M {
    let base: M = std::array::from_fn(|i| {
        std::array::from_fn(|j| match i {
            0 => pri[j][0],
            1 => pri[j][1],
            _ => 1. - pri[j][0] - pri[j][1],
        })
    });
    let white = [
        pri[3][0] / pri[3][1],
        1.,
        (1. - pri[3][0] - pri[3][1]) / pri[3][1],
    ];
    let s = mv(inv(base), white);
    std::array::from_fn(|i| std::array::from_fn(|j| base[i][j] * s[j]))
}
fn cone_f(v: f64) -> f64 {
    let x = v.abs().powf(0.42);
    (x / (27.13 + x)).copysign(v)
}
fn cone_i(v: f64) -> f64 {
    let x = v.abs().min(0.99);
    (27.13 * x / (1. - x)).powf(1. / 0.42).copysign(v)
}

#[derive(Clone, Copy)]
struct Jmh {
    rgb_cone: M,
    cone_rgb: M,
    cone_aab: M,
    aab_cone: M,
    fl: f64,
    cz: f64,
    ic: f64,
    aw: f64,
    iaw: f64,
}
impl Jmh {
    fn new(pri: [[f64; 2]; 4]) -> Self {
        let xyz = rgb_xyz(pri);
        let sharpen = inv(rgb_xyz(CAM16));
        let xyz_w = mv(xyz, [100.; 3]);
        let rgb_w = mv(sharpen, xyz_w);
        let k = 1.0_f64 / 501.;
        let k4 = k.powi(4);
        let fl = 0.2 * k4 * 500. + 0.1 * (1. - k4).powi(2) * 500.0_f64.powf(1. / 3.);
        let fl_n = fl / 100.;
        let d = rgb_w.map(|x| fl_n * xyz_w[1] / x);
        let adapted = std::array::from_fn(|i| cone_f(d[i] * rgb_w[i]));
        let base = [
            [2., 1., 1. / 20.],
            [1., -12. / 11., 1. / 11.],
            [1. / 9., 1. / 9., -2. / 9.],
        ];
        let mut cone_aab = base.map(|r| r.map(|x| x * 400.));
        let aw = mv(cone_aab, adapted)[0];
        cone_aab[0] = cone_aab[0].map(|x| x / aw);
        for row in &mut cone_aab[1..] {
            *row = row.map(|x| x * 43. * 0.9);
        }
        let mut rgb_cone = mm(sharpen, xyz);
        for i in 0..3 {
            rgb_cone[i] = rgb_cone[i].map(|x| x * 100. * d[i]);
        }
        let aw_j = cone_f(fl);
        Self {
            rgb_cone,
            cone_rgb: inv(rgb_cone),
            cone_aab,
            aab_cone: inv(cone_aab),
            fl: fl_n,
            cz: GAMMA,
            ic: 1. / GAMMA,
            aw: aw_j,
            iaw: 1. / aw_j,
        }
    }
    fn aab(self, v: V) -> V {
        mv(self.cone_aab, mv(self.rgb_cone, v).map(cone_f))
    }
    fn to_jmh(self, v: V) -> V {
        let a = self.aab(v);
        if a[0] <= 0. {
            return [0.; 3];
        }
        [
            100. * a[0].powf(self.cz),
            a[1].hypot(a[2]),
            a[2].atan2(a[1]).to_degrees().rem_euclid(360.),
        ]
    }
    fn to_rgb(self, v: V) -> V {
        let h = v[2].to_radians();
        let a = [(v[0] / 100.).powf(self.ic), v[1] * h.cos(), v[1] * h.sin()];
        mv(self.cone_rgb, mv(self.aab_cone, a).map(cone_i))
    }
    fn y_j(self, y: f64) -> f64 {
        (100. * (cone_f(y.abs() * self.fl) * self.iaw).powf(self.cz)).copysign(y)
    }
    fn pack(self, out: &mut Vec<f32>) {
        for mat in [self.rgb_cone, self.cone_rgb, self.cone_aab, self.aab_cone] {
            for j in 0..3 {
                for row in mat {
                    out.push(row[j] as f32);
                }
            }
        }
        out.extend([self.fl, self.cz, self.ic, self.aw, self.iaw].map(|x| x as f32));
    }
}

#[derive(Clone, Copy)]
struct Ts {
    n: f64,
    nr: f64,
    g: f64,
    toe: f64,
    ct: f64,
    s2: f64,
    u2: f64,
    m2: f64,
    limit: f64,
    inverse: f64,
    lp: f64,
}
impl Ts {
    fn new(n: f64) -> Self {
        let nr = 100.;
        let g = 1.15;
        let toe = 0.04;
        let hit = 128. + 768. * (n / nr).ln() / (100.0_f64).ln();
        let m0 = n / nr;
        let m1 = 0.5 * (m0 + (m0 * (m0 + 4. * toe)).sqrt());
        let u = ((hit / m1) / (hit / m1 + 1.)).powf(g);
        let m = m1 / u;
        let ct = 10.013 / nr * (1. + (n / 100.).log2() * 0.14);
        let gip = 0.5 * (ct + (ct * (ct + 4. * toe)).sqrt());
        let q = (gip / m).powf(1. / g);
        let gipp2 = -m1 * q / (q - 1.);
        let w2 = 0.18 / gipp2;
        let s2 = w2 * m1;
        let u2 = ((hit / m1) / (hit / m1 + w2)).powf(g);
        let m2 = m1 / u2;
        Self {
            n,
            nr,
            g,
            toe,
            ct,
            s2,
            u2,
            m2,
            limit: 8. * hit,
            inverse: n / (u2 * nr),
            lp: (n / nr).log10(),
        }
    }
    fn pack(self, out: &mut Vec<f32>) {
        out.extend(
            [
                self.n,
                self.nr,
                self.g,
                self.toe,
                self.ct,
                self.s2,
                self.u2,
                self.m2,
                self.limit,
                self.inverse,
                self.lp,
            ]
            .map(|x| x as f32),
        );
    }
}
fn corner(i: usize) -> V {
    [
        if (i + 1) % 6 < 3 { 1. } else { 0. },
        if (i + 5) % 6 < 3 { 1. } else { 0. },
        if (i + 3) % 6 < 3 { 1. } else { 0. },
    ]
}
fn rotate_corners(rgb: [V; 6], jmh: [V; 6]) -> ([V; 8], [V; 8]) {
    let first = (0..6)
        .min_by(|&a, &b| jmh[a][2].total_cmp(&jmh[b][2]))
        .unwrap();
    let mut r = [[0.; 3]; 8];
    let mut j = r;
    for i in 0..6 {
        r[i + 1] = rgb[(i + first) % 6];
        j[i + 1] = jmh[(i + first) % 6];
    }
    r[0] = r[6];
    r[7] = r[1];
    j[0] = j[6];
    j[7] = j[1];
    j[0][2] -= 360.;
    j[7][2] += 360.;
    (r, j)
}
fn limiting_corners(params: Jmh, n: f64) -> ([V; 8], [V; 8]) {
    let rgb = std::array::from_fn(|i| scale(corner(i), n / 100.));
    let jmh = rgb.map(|v| params.to_jmh(v));
    rotate_corners(rgb, jmh)
}
fn reach_corners(params: Jmh, limit_j: f64, upper: f64) -> [V; 8] {
    let limit_a = (limit_j / 100.).powf(params.ic);
    let rgb = std::array::from_fn(|i| {
        let (mut lo, mut hi) = (0., upper);
        while hi - lo > 1e-3 {
            let mid = (lo + hi) / 2.;
            if params.aab(scale(corner(i), mid))[0] < limit_a {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        scale(corner(i), hi)
    });
    let jmh = rgb.map(|v| params.to_jmh(v));
    rotate_corners(rgb, jmh).1
}
fn hue_table(reach: [V; 8], limit: [V; 8]) -> [f64; PAD] {
    let mut hues: Vec<f64> = reach[1..7]
        .iter()
        .chain(limit[1..7].iter())
        .map(|v| v[2])
        .collect();
    hues.sort_by(f64::total_cmp);
    hues.dedup();
    let mut positions = Vec::with_capacity(hues.len());
    let mut min_index = usize::from(hues[0] != 0.);
    let mut last = usize::MAX;
    for (i, &h) in hues.iter().enumerate() {
        let mut idx = (h.round() as usize).max(min_index).min(N - 1);
        if last == idx {
            if i > 1 && positions[i - 2] != positions[i - 1] - 1 {
                positions[i - 1] -= 1;
            } else {
                idx += 1;
            }
        }
        positions.push(idx.min(N - 1));
        min_index = idx;
        last = idx;
    }
    let mut out = [0.; PAD];
    let mut total = 0;
    for i in 0..=hues.len() {
        let lower = if i == 0 { 0. } else { hues[i - 1] };
        let upper = if i == hues.len() { 360. } else { hues[i] };
        let next = if i == hues.len() { N } else { positions[i] };
        let samples = next - total;
        for j in 0..samples {
            out[1 + total + j] = lower + j as f64 * (upper - lower) / samples as f64;
        }
        total = next;
    }
    out[0] = out[N] - 360.;
    out[N + 1] = out[1] + 360.;
    out
}
fn cusp(h: f64, rgb: [V; 8], jmh: [V; 8], params: Jmh) -> [f64; 2] {
    let hi = (1..8).find(|&i| jmh[i][2] > h).unwrap_or(7);
    let lo = hi - 1;
    if jmh[lo][2] == h {
        return [jmh[lo][0], jmh[lo][1]];
    }
    let (mut lt, mut ut) = (0., 1.);
    while ut - lt > 1e-7 {
        let t = (lt + ut) / 2.;
        let sample = params.to_jmh(lerpv(rgb[lo], rgb[hi], t));
        if sample[2] < jmh[lo][2] || (sample[2] < jmh[hi][2] && sample[2] > h) {
            ut = t;
        } else {
            lt = t;
        }
    }
    let sample = params.to_jmh(lerpv(rgb[lo], rgb[hi], (lt + ut) / 2.));
    [sample[0], sample[1]]
}
fn reach_table(params: Jmh, j: f64) -> [f64; PAD] {
    let mut out = [0.; PAD];
    for i in 0..N {
        let (mut lo, mut hi) = (0., 50.);
        while hi < 1300. && !params.to_rgb([j, hi, i as f64]).iter().any(|x| *x < 0.) {
            lo = hi;
            hi += 50.;
        }
        while hi - lo > 1e-2 {
            let mid = (hi + lo) / 2.;
            if params.to_rgb([j, mid, i as f64]).iter().any(|x| *x < 0.) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        out[i + 1] = hi;
    }
    out[0] = out[N];
    out[N + 1] = out[1];
    out
}
fn focus_j(cusp: f64, mid: f64, max: f64) -> f64 {
    lerp(cusp, mid, (1.3 - cusp / max).min(1.))
}
fn focus_gain(j: f64, threshold: f64, max: f64, dist: f64) -> f64 {
    let gain = max * dist;
    if j > threshold {
        let g = ((max - threshold) / (max - j).max(0.0001)).log10();
        gain * (g * g + 1.)
    } else {
        gain
    }
}
fn intersect(j: f64, m: f64, focus: f64, max: f64, gain: f64) -> f64 {
    let ms = m / gain;
    let a = ms / focus;
    if j < focus {
        let b = 1. - ms;
        let c = -j;
        -2. * c / (b + (b * b - 4. * a * c).sqrt())
    } else {
        let b = -(1. + ms + max * a);
        let c = max * ms + j;
        -2. * c / (b - (b * b - 4. * a * c).sqrt())
    }
}
fn slope(j: f64, focus: f64, max: f64, gain: f64) -> f64 {
    let scalar = if j < focus { j } else { max - j };
    scalar * (j - focus) / (focus * gain)
}
fn estimate(axis: f64, slope: f64, gamma: f64, max: f64, mmax: f64, reference: f64) -> f64 {
    reference * (axis / reference).powf(gamma) * mmax / (max - slope * mmax)
}
fn boundary(
    cusp: [f64; 2],
    max: f64,
    gt: f64,
    gb: f64,
    axis: f64,
    slope: f64,
    reference: f64,
) -> f64 {
    let lower = estimate(axis, slope, gb, cusp[0], cusp[1], reference);
    let upper = estimate(
        max - axis,
        -slope,
        gt,
        max - cusp[0],
        cusp[1],
        max - reference,
    );
    let s = 0.12 * cusp[1];
    let h = (s - (lower - upper).abs()).max(0.) / s;
    lower.min(upper) - h * h * h * s / 6.
}
fn gamma_table(
    cusps: [V; PAD],
    params: Jmh,
    n: f64,
    max: f64,
    mid: f64,
    dist: f64,
    gb: f64,
) -> [f64; PAD] {
    let mut out = [0.; PAD];
    for i in 1..=N {
        let c = [cusps[i][0], cusps[i][1]];
        let hue = cusps[i][2];
        let focus = focus_j(c[0], mid, max);
        let threshold = lerp(c[0], max, 0.3);
        let tests = [0.01, 0.1, 0.5, 0.8, 0.99].map(|t| {
            let j = lerp(c[0], max, t);
            let gain = focus_gain(j, threshold, max, dist);
            let axis = intersect(j, c[1], focus, max, gain);
            let s = slope(axis, focus, max, gain);
            let reference = intersect(c[0], c[1], focus, max, gain);
            (axis, s, reference)
        });
        let fits = |gamma: f64| {
            tests.iter().all(|&(axis, s, reference)| {
                let m = boundary(c, max, 1. / gamma, gb, axis, s, reference);
                params
                    .to_rgb([axis + s * m, m, hue])
                    .iter()
                    .any(|x| *x > n / 100.)
            })
        };
        let (mut low, mut high) = (0., 0.4);
        while high < 5. && !fits(high) {
            low = high;
            high += 0.4;
        }
        while high - low > 1e-5 {
            let test = (low + high) / 2.;
            if fits(test) {
                high = test;
            } else {
                low = test;
            }
        }
        out[i] = 1. / high;
    }
    out[0] = out[N];
    out[N + 1] = out[1];
    out
}

/// Initialize the official a2.v1 transform with a Rec.709 / D65 limiting gamut.
/// A headroom of 1 is the official 100nit sRGB preset; HDR is the same parametric
/// transform at `100 * headroom` nits, represented in linear scRGB units.
pub fn generate(headroom: f32) -> Vec<f32> {
    let n = 100. * f64::from(headroom.clamp(1., 100.));
    let input = Jmh::new(AP0);
    let reach = Jmh::new(AP1);
    let limit = Jmh::new(REC709);
    let ts = Ts::new(n);
    let max = input.y_j(n);
    let mid = input.y_j(ts.ct * 100.);
    let dist = 1.35 + 1.35 * 1.75 * ts.lp;
    let gb = 1. / (1.14 + 0.07 * ts.lp);
    let reach_m = reach_table(reach, max);
    let (r, j) = limiting_corners(limit, n);
    let rc = reach_corners(reach, max, ts.limit);
    let hues = hue_table(rc, j);
    let mut cusps = [[0.; 3]; PAD];
    for i in 1..PAD {
        let c = cusp(hues[i], r, j, limit);
        cusps[i] = [c[0], c[1] * (1. + 0.27 * 0.12), hues[i]];
    }
    cusps[0] = [cusps[N][0], cusps[N][1], hues[0]];
    cusps[N + 1] = [cusps[1][0], cusps[1][1], hues[N + 1]];
    let upper = gamma_table(cusps, limit, n, max, mid, dist, gb);
    let mut out = Vec::with_capacity(1913);
    input.pack(&mut out);
    limit.pack(&mut out);
    ts.pack(&mut out);
    out.extend(
        [
            max,
            1. / GAMMA,
            (1.3 - 1.3 * 0.69 * ts.lp).max(0.2),
            0.5 / n,
            2.4 + 2.4 * 3.3 * ts.lp,
            (0.03379 * n).powf(0.30596) - 0.45135,
            mid,
            dist,
            gb,
            n,
        ]
        .map(|x| x as f32),
    );
    out.extend(reach_m.map(|x| x as f32));
    for v in cusps {
        out.extend(v.map(|x| x as f32));
    }
    out.extend(upper.map(|x| x as f32));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_layout_and_all_hdr_tables_are_finite() {
        for peak in [1., 2., 10., 40., 100.] {
            let p = generate(peak);
            assert_eq!(p.len(), 1913);
            assert!(p.iter().all(|x| x.is_finite()));
            assert_eq!(p[102], 100. * peak);
            assert!(p[103..465].iter().all(|x| *x > 0.));
            assert!(p[1551..].iter().all(|x| *x > 0.));
        }
    }
    #[test]
    fn cam_preserves_neutral_lightness_and_inverts_colors() {
        for pri in [AP0, AP1, REC709] {
            let p = Jmh::new(pri);
            for v in [[0.18; 3], [1., 0., 0.], [0.1, 0.8, 0.05], [10.; 3]] {
                let out = p.to_rgb(p.to_jmh(v));
                for i in 0..3 {
                    assert!((out[i] - v[i]).abs() < 1e-10);
                }
            }
            let neutral = p.to_jmh([1.; 3]);
            assert!((neutral[0] - 100.).abs() < 1e-10);
            assert!(neutral[1] < 1e-9);
        }
    }
}
