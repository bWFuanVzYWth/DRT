// Copyright (c) 2015 Academy of Motion Picture Arts and Sciences (A.M.P.A.S.).
// Derived from aces-aswf/aces-core v1.3, commit 1256fee50ee35548c6eab8eca854ff3349008489.
// Academy ACES license: THIRD_PARTY_LICENSES/references/aces_13/LICENSE.md.
// Full RRT + sRGB 100nit dim ODT; see references/entries/aces_13.md.

const A13_AP0_AP1 = mat3x3f(
    vec3f(1.4514393161, -0.0765537734, 0.0083161484),
    vec3f(-0.2365107469, 1.1762296998, -0.0060324498),
    vec3f(-0.2149285693, -0.0996759264, 0.9977163014));
const A13_AP1_Y = vec3f(0.2722287168, 0.6740817658, 0.0536895174);
const A13_AP1_XYZ = mat3x3f(
    vec3f(0.6624541811, 0.2722287168, -0.0055746495),
    vec3f(0.1340042065, 0.6740817658, 0.0040607335),
    vec3f(0.1561876870, 0.0536895174, 1.0103391003));
const A13_CAT = mat3x3f(
    vec3f(0.9872240087, -0.0075983718, 0.0030725771),
    vec3f(-0.0061132286, 1.0018614847, -0.0050959615),
    vec3f(0.0159532883, 0.0053300358, 1.0816806031));
const A13_XYZ_709 = mat3x3f(
    vec3f(3.2409699419, -0.9692436363, 0.0556300797),
    vec3f(-1.5373831776, 1.8759675015, -0.2039769589),
    vec3f(-0.4986107603, 0.0415550574, 1.0569715142));

fn a13_spline_piece(t: f32, c: vec3f) -> f32 {
    return t*t*(0.5*c.x-c.y+0.5*c.z) + t*(-c.x+c.y) + 0.5*(c.x+c.y);
}
fn a13_rrt_curve(x: f32) -> f32 {
    let low = array<f32,6>(-4.0,-4.0,-3.1573765773,-0.4852499958,1.8477324706,1.8477324706);
    let high = array<f32,6>(-0.7185482425,2.0810307172,3.6681241237,4.0,4.0,4.0);
    let lx = log2(max(x, 0.00006103515625)) * 0.3010299956639812;
    let lmin = log2(0.18 * exp2(-15.0)) * 0.3010299956639812;
    let lmid = log2(0.18) * 0.3010299956639812;
    let lmax = log2(0.18 * exp2(18.0)) * 0.3010299956639812;
    if lx <= lmin { return 0.0001; }
    if lx >= lmax { return 10000.0; }
    var coord: f32;
    var coeff: vec3f;
    if lx < lmid {
        coord = 3.0 * (lx-lmin)/(lmid-lmin);
        let j = min(u32(coord), 2u);
        coeff = vec3f(low[j],low[j+1u],low[j+2u]);
        coord -= f32(j);
    } else {
        coord = 3.0 * (lx-lmid)/(lmax-lmid);
        let j = min(u32(coord), 2u);
        coeff = vec3f(high[j],high[j+1u],high[j+2u]);
        coord -= f32(j);
    }
    return pow(10.0,a13_spline_piece(coord,coeff));
}
fn a13_odt_curve(x: f32) -> f32 {
    let low = array<f32,10>(-1.6989700043,-1.6989700043,-1.4779,-1.2291,-0.8648,-0.448,0.00518,0.4511080334,0.9113744414,0.9113744414);
    let high = array<f32,10>(0.5154386965,0.8470437783,1.1358,1.3802,1.5197,1.5985,1.6467,1.6746091357,1.6878733390,1.6878733390);
    let lx = log2(max(x,0.00006103515625)) * 0.3010299956639812;
    let lmin = log2(a13_rrt_curve(0.18*exp2(-6.5))) * 0.3010299956639812;
    let lmid = log2(4.8) * 0.3010299956639812;
    let lmax = log2(a13_rrt_curve(0.18*exp2(6.5))) * 0.3010299956639812;
    if lx <= lmin { return pow(10.0,-1.6989700043360187); }
    if lx >= lmax { return pow(10.0,0.04*(lx-lmax)+log2(48.0)*0.3010299956639812); }
    var coord: f32;
    var coeff: vec3f;
    if lx < lmid {
        coord = 7.0*(lx-lmin)/(lmid-lmin);
        let j = min(u32(coord),6u);
        coeff = vec3f(low[j],low[j+1u],low[j+2u]);
        coord -= f32(j);
    } else {
        coord = 7.0*(lx-lmid)/(lmax-lmid);
        let j = min(u32(coord),6u);
        coeff = vec3f(high[j],high[j+1u],high[j+2u]);
        coord -= f32(j);
    }
    return pow(10.0,a13_spline_piece(coord,coeff));
}
fn a13_red_weight(hue: f32) -> f32 {
    let x = hue;
    if x <= -67.5 || x >= 67.5 { return 0.0; }
    let coord = (x+67.5)/33.75;
    let j = u32(coord);
    let t = coord-f32(j);
    var y: f32;
    switch j {
        case 0u: { y = t*t*t/6.0; }
        case 1u: { y = (-3.0*t*t*t+3.0*t*t+3.0*t+1.0)/6.0; }
        case 2u: { y = (3.0*t*t*t-6.0*t*t+4.0)/6.0; }
        default: { y = (-t*t*t+3.0*t*t-3.0*t+1.0)/6.0; }
    }
    return 1.5*y;
}
fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    var aces = ap0;
    let maxc = max(max(aces.r,aces.g),aces.b);
    let minc = min(min(aces.r,aces.g),aces.b);
    let sat = (max(maxc,1e-10)-max(minc,1e-10))/max(maxc,0.01);
    let chroma = sqrt(max(0.0,aces.b*(aces.b-aces.g)+aces.g*(aces.g-aces.r)+aces.r*(aces.r-aces.b)));
    let yc = (aces.r+aces.g+aces.b+1.75*chroma)/3.0;
    let sigx = (sat-0.4)/0.2;
    let sigt = max(1.0-abs(sigx/2.0),0.0);
    let sig = 0.5*(1.0+sign(sigx)*(1.0-sigt*sigt));
    var glow = 0.0;
    if yc <= (2.0/3.0)*0.08 { glow = 0.05*sig; }
    else if yc < 0.16 { glow = 0.05*sig*(0.08/yc-0.5); }
    aces *= 1.0+glow;
    var centered_hue = 0.0;
    if aces.r != aces.g || aces.g != aces.b {
        centered_hue = degrees(atan2(sqrt(3.0)*(aces.g-aces.b),2.0*aces.r-aces.g-aces.b));
    }
    aces.r += a13_red_weight(centered_hue)*sat*(0.03-aces.r)*0.18;
    var rgb = clamp(A13_AP0_AP1 * max(aces,vec3f(0.0)),vec3f(0.0),vec3f(65504.0));
    rgb = mix(vec3f(dot(rgb,A13_AP1_Y)),rgb,0.96);
    rgb = vec3f(a13_rrt_curve(rgb.r),a13_rrt_curve(rgb.g),a13_rrt_curve(rgb.b));
    // RRT AP1 -> AP0 and ODT AP0 -> AP1 are inverse matrices: combine them.
    rgb = vec3f(a13_odt_curve(rgb.r),a13_odt_curve(rgb.g),a13_odt_curve(rgb.b));
    let black = pow(10.0,-1.6989700043360187);
    rgb = (rgb-vec3f(black))/(48.0-black);
    // Dark-to-dim adjustment changes only XYZ Y, retaining x/y chromaticity.
    let Y = max(dot(rgb,A13_AP1_Y),0.0);
    if Y > 0.0 { rgb *= pow(Y,0.9811)/Y; } else { rgb = vec3f(0.0); }
    rgb = mix(vec3f(dot(rgb,A13_AP1_Y)),rgb,0.93);
    return clamp(A13_XYZ_709*A13_CAT*A13_AP1_XYZ*rgb,vec3f(0.0),vec3f(1.0));
}
