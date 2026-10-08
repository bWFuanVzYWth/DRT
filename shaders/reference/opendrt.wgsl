// SPDX-License-Identifier: GPL-3.0-only
// OpenDRT v1.1.0, Standard look; Jed Smith, GPL-3.0.
// Port of OpenDRT.dctl at af683323e2a8a63501f02c0a724ec538e3228ad0.
// AP0 / scene-linear input; Rec.709 / D65 / display-linear output.
// Rec.1886 dim surround; final EOTF is performed by the workbench.
// See references/entries/opendrt.md for the pinned source and output contract.

fn odrt_spow(x: f32, p: f32) -> f32 {
    if (x <= 0.0) { return x; }
    return pow(x, p);
}

fn odrt_toe(x: f32, toe: f32, inverse: bool) -> f32 {
    if (toe == 0.0) { return x; }
    if (inverse) { return (x + sqrt(x * (4.0 * toe + x))) / 2.0; }
    return odrt_spow(x, 2.0) / (x + toe);
}

fn odrt_hyperbolic(x: f32, s: f32, p: f32) -> f32 {
    return odrt_spow(x / (x + s), p);
}

fn odrt_opponent(rgb: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(rgb.r - rgb.b, rgb.g - (rgb.r + rgb.b) / 2.0);
}

// DCTL _fmod is the C remainder operation, including a negative dividend.
fn odrt_fmod(x: f32, y: f32) -> f32 { return x - y * trunc(x / y); }

fn odrt_hue_offset(h: f32, o: f32) -> f32 {
    return odrt_fmod(h - o + 3.14159265358979323846, 6.28318530717958647692) - 3.14159265358979323846;
}

fn odrt_gauss(x: f32, w: f32) -> f32 { return exp(-x * x / w); }

fn odrt_softplus(x: f32, s: f32) -> f32 {
    if (x > 10.0 * s || s < 0.0001) { return x; }
    return s * log(max(0.0, 1.0 + exp(x / s)));
}

fn reference_transform(ap0: vec3<f32>, headroom: f32) -> vec3<f32> {
    let peak = max(headroom, 1.0);
    // All parameters are the upstream Standard look. Contrast Low and
    // Contrast High are disabled in that preset, so their branches are elided.
    let tn_con = 1.66;
    let tn_toe = 0.003;
    let tn_off = 0.005;
    let rs_sa = 0.35;
    let rs_w = vec3<f32>(0.25, 1.0 - 0.25 - 0.55, 0.55);

    // Tonescale constraints: tn_sh=.5, tn_Lg=10 nits, tn_gb=.13,
    // tn_Lp=100*peak nits, surround=Dim (1), HDR Purity=.5.
    let ts_x1 = pow(2.0, 6.0 * 0.5 + 4.0);
    let ts_y1 = peak;
    let ts_x0 = 0.18 + tn_off;
    let ts_y0 = 0.1 * (1.0 + 0.13 * log2(ts_y1));
    let ts_s0 = odrt_toe(ts_y0, tn_toe, true);
    let ts_p = tn_con / 1.05;
    let ts_s10 = ts_x0 * (pow(ts_s0, -1.0 / tn_con) - 1.0);
    let ts_m1 = ts_y1 / pow(ts_x1 / (ts_x1 + ts_s10), tn_con);
    let ts_m2 = odrt_toe(ts_m1, tn_toe, true);
    let ts_s = ts_x0 * (pow(ts_s0 / ts_m2, -1.0 / tn_con) - 1.0);
    let pt_cmp_Lf = 0.5 * min(1.0, (100.0 * peak - 100.0) / 900.0);
    let s_Lp100 = ts_x0 * (pow(0.1, -1.0 / tn_con) - 1.0);
    let ts_s1 = ts_s * pt_cmp_Lf + s_Lp100 * (1.0 - pt_cmp_Lf);

    // Original AP0-to-XYZ-D65 CAT02 matrix, then XYZ-D65-to-P3-D65.
    // Negative input values are preserved as in the original DCTL.
    let xyz = vec3<f32>(
        dot(ap0, vec3<f32>(0.938630948750273197, -0.00574192055037397141, 0.017566898851772296)),
        dot(ap0, vec3<f32>(0.338093594922021567, 0.72721390281143572, -0.0653074977334571899)),
        dot(ap0, vec3<f32>(0.000723121511341165988, 0.000818441849244731985, 1.08751618739929268)));
    var rgb = vec3<f32>(
        dot(xyz, vec3<f32>(2.49349691194142542, -0.93138361791912383, -0.402710784450716841)),
        dot(xyz, vec3<f32>(-0.829488969561574696, 1.76266406031834655, 0.0236246858419435941)),
        dot(xyz, vec3<f32>(0.0358458302437844531, -0.0761723892680418041, 0.956884524007687309)));
    var sat_L = dot(rgb, rs_w);
    rgb = vec3<f32>(sat_L * rs_sa) + rgb * (1.0 - rs_sa);
    rgb += vec3<f32>(tn_off);
    var tsn = sqrt(max(0.0, dot(rgb, rgb))) / 1.73205080756887729353;
    if (tsn != 0.0) { rgb /= tsn; } else { rgb = vec3<f32>(0.0); }
    let opp = odrt_opponent(rgb);
    var ach_d = sqrt(max(0.0, dot(opp, opp))) / 2.0;
    ach_d = 1.25 * odrt_toe(ach_d, 0.25, false);
    let hue = odrt_fmod(atan2(opp.x, opp.y) + 3.14159265358979323846 + 1.10714931, 6.28318530717958647692);
    let ha_rgb = vec3<f32>(
        odrt_gauss(odrt_hue_offset(hue, 0.1), 0.66),
        odrt_gauss(odrt_hue_offset(hue, 4.3), 0.66),
        odrt_gauss(odrt_hue_offset(hue, 2.3), 0.66));
    let ha_rgb_hs = vec3<f32>(
        odrt_gauss(odrt_hue_offset(hue, -0.4), 0.66), ha_rgb.y,
        odrt_gauss(odrt_hue_offset(hue, 2.5), 0.66));
    let ha_cmy = vec3<f32>(
        odrt_gauss(odrt_hue_offset(hue, 3.3), 0.5),
        odrt_gauss(odrt_hue_offset(hue, 1.3), 0.5),
        odrt_gauss(odrt_hue_offset(hue, -1.15), 0.5));

    // Brilliance: brl=0, R=-2.5, G=B=-1.5, range=.5, strength=.35.
    let brl_tsf = pow(tsn / (tsn + 1.0), 0.5);
    let brl_exf = dot(vec3<f32>(-2.5, -1.5, -1.5), ha_rgb) * pow(ach_d, 1.0 / 0.35);
    var brl_factor = 1.0 - brl_tsf;
    if (brl_exf < 0.0) { brl_factor = brl_tsf; }
    tsn *= pow(2.0, brl_exf * brl_factor);
    let tsn_pt = odrt_hyperbolic(tsn, ts_s1, ts_p);
    let tsn_const = odrt_hyperbolic(tsn, s_Lp100, ts_p);
    tsn = odrt_hyperbolic(tsn, ts_s, ts_p);

    // Hue Contrast R: amount=1, range=.3.
    var hc_ts = 1.0 - tsn_const;
    var hc_c = hc_ts * (1.0 - ach_d) + ach_d * (1.0 - hc_ts);
    hc_c *= ach_d * ha_rgb.x;
    hc_ts = pow(hc_ts, 1.0 / 0.3);
    let hc_f = hc_c - 2.0 * hc_c * hc_ts + 1.0;
    rgb = vec3<f32>(rgb.r, rgb.g * hc_f, rgb.b * hc_f);

    // Hue Shift RGB: amounts (.6, .35, .66), ranges (.6, 1, 1).
    let hs_rgb = ha_rgb_hs * ach_d * vec3<f32>(pow(tsn_pt, 1.0 / 0.6), tsn_pt, tsn_pt);
    var hsf = hs_rgb * vec3<f32>(0.6, -0.35, -0.66);
    rgb += vec3<f32>(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
    // Hue Shift CMY: amounts (.25, 0, 0), ranges (1, 1, 1).
    let hs_cmy = ha_cmy * ach_d * (1.0 - tsn_pt);
    hsf = hs_cmy * vec3<f32>(-0.25, 0.0, 0.0);
    rgb += vec3<f32>(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);

    // Purity compression, including both low and high limits.
    let pt_lml_p = 1.0 + 4.0 * (1.0 - tsn_pt) * (0.25 + 0.5 * ha_rgb_hs.x + 0.1 * ha_rgb_hs.z);
    var ptf = 1.0 - pow(tsn_pt, pt_lml_p);
    let pt_lmh_p = (1.0 - ach_d * 0.5 * ha_rgb_hs.x) * (1.0 - 0.25 * ach_d);
    ptf = pow(ptf, pt_lmh_p);
    let ptm_low_f = 1.0 + 0.4 * exp(-2.0 * ach_d * ach_d / 0.5) * pow(1.0 - tsn_const, 1.0 / 0.25);
    let ptm_high_f = 1.0 - 0.8 * exp(-2.0 * ach_d * ach_d / 0.4) * pow(tsn_pt, 1.0 / (4.0 * 0.35));
    ptf *= ptm_low_f * ptm_high_f;
    rgb = rgb * ptf + vec3<f32>(1.0 - ptf);
    sat_L = dot(rgb, rs_w);
    rgb = (vec3<f32>(sat_L * rs_sa) - rgb) / (rs_sa - 1.0);

    // Display gamut / creative white: Standard uses D65, so CAT and
    // creative-white interpolation are identities; retain original matrices.
    let display_xyz = vec3<f32>(
        dot(rgb, vec3<f32>(0.486570948648216151, 0.265667693169093, 0.198217285234362467)),
        dot(rgb, vec3<f32>(0.228974564069748754, 0.691738521836506193, 0.079286914093744984)),
        dot(rgb, vec3<f32>(-4e-17, 0.0451133818589026167, 1.04394436890097575)));
    rgb = vec3<f32>(
        dot(display_xyz, vec3<f32>(3.24096994190452348, -1.53738317757009435, -0.498610760293003552)),
        dot(display_xyz, vec3<f32>(-0.969243636280879506, 1.87596750150771996, 0.0415550574071755843)),
        dot(display_xyz, vec3<f32>(0.0556300796969936354, -0.20397695888897649, 1.05697151424287816)));

    // Post Brilliance: -.5 + RGB (-1.25,-1.25,-.25).
    let brlp_opp = odrt_opponent(rgb);
    var brlp_ach_d = sqrt(max(0.0, dot(brlp_opp, brlp_opp))) / 4.0;
    brlp_ach_d = 1.1 * brlp_ach_d * brlp_ach_d / (brlp_ach_d + 0.1);
    let brlp_m = -0.5 + dot(vec3<f32>(-1.25, -1.25, -0.25), ach_d * ha_rgb);
    rgb *= pow(2.0, brlp_m * brlp_ach_d * tsn);
    rgb = vec3<f32>(odrt_softplus(rgb.r, 0.06), odrt_softplus(rgb.g, 0.08), odrt_softplus(rgb.b, 0.06));
    tsn *= ts_m2;
    tsn = odrt_toe(tsn, tn_toe, false);
    // Original Rec.1886 output is divided by peak, clipped to [0,1],
    // and encoded. Undo only the peak normalization to return absolute
    // display-linear values in the workbench's SDR-white units.
    return clamp(rgb * tsn, vec3<f32>(0.0), vec3<f32>(peak));
}
