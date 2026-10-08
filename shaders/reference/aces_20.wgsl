// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project.
// Analytic WGSL port of aces-core Lib.Academy.OutputTransform a2.v1.
// Core commit 069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80.
// Host initialization in src/aces2_data.rs; provenance in references/entries/aces_20.md.

const A20_AP0_AP1 = mat3x3f(
    vec3f(1.4514393161,-0.0765537734,0.0083161484),
    vec3f(-0.2365107469,1.1762296998,-0.0060324498),
    vec3f(-0.2149285693,-0.0996759264,0.9977163014));
const A20_AP1_AP0 = mat3x3f(
    vec3f(0.6954522414,0.0447945634,-0.0055258826),
    vec3f(0.1406786965,0.8596711185,0.0040252103),
    vec3f(0.1638690622,0.0955343182,1.0015006723));

fn a20_matrix(base: u32) -> mat3x3f {
    return mat3x3f(
        vec3f(reference_data[base],reference_data[base+1u],reference_data[base+2u]),
        vec3f(reference_data[base+3u],reference_data[base+4u],reference_data[base+5u]),
        vec3f(reference_data[base+6u],reference_data[base+7u],reference_data[base+8u]));
}
fn a20_cone_f(v: vec3f) -> vec3f {
    let x = pow(abs(v),vec3f(0.42));
    return sign(v)*x/(vec3f(27.13)+x);
}
fn a20_cone_i(v: vec3f) -> vec3f {
    let x = min(abs(v),vec3f(0.99));
    return sign(v)*pow(27.13*x/(vec3f(1.0)-x),vec3f(1.0/0.42));
}
fn a20_y_j(Y: f32) -> f32 {
    let x = pow(abs(Y)*reference_data[36u],0.42);
    return sign(Y)*100.0*pow(x/(27.13+x)*reference_data[40u],reference_data[37u]);
}
fn a20_j_y(J: f32) -> f32 {
    let Ra = min(reference_data[39u]*pow(abs(J)/100.0,reference_data[38u]),0.99);
    return pow(27.13*Ra/(1.0-Ra),1.0/0.42)/reference_data[36u];
}
fn a20_rgb_jmh(rgb: vec3f) -> vec3f {
    let aab = a20_matrix(18u)*a20_cone_f(a20_matrix(0u)*rgb);
    if aab.x <= 0.0 { return vec3f(0.0); }
    var h = degrees(atan2(aab.z,aab.y));
    if h < 0.0 { h += 360.0; }
    return vec3f(100.0*pow(aab.x,reference_data[37u]),length(aab.yz),h);
}
fn a20_jmh_rgb(jmh: vec3f) -> vec3f {
    let h = radians(jmh.z);
    let aab = vec3f(pow(jmh.x/100.0,reference_data[79u]),jmh.y*cos(h),jmh.y*sin(h));
    return a20_matrix(50u)*a20_cone_i(a20_matrix(68u)*aab);
}
fn a20_reach(h: f32) -> f32 {
    let base = min(u32(h),359u);
    return mix(reference_data[104u+base],reference_data[105u+base],h-f32(base));
}
fn a20_toe(x: f32, limit: f32, k1in: f32, k2in: f32) -> f32 {
    if x > limit { return x; }
    let k2 = max(k2in,0.001);
    let k1 = sqrt(k1in*k1in+k2*k2);
    let k3 = (limit+k1)/(limit+k2);
    let mb = k3*x-k1;
    let mc = k2*k3*x;
    return 0.5*(mb+sqrt(mb*mb+4.0*mc));
}
fn a20_chroma_norm(h: f32) -> f32 {
    let hr = radians(h);
    let a = cos(hr);
    let b = sin(hr);
    let cos2 = a*a-b*b;
    let sin2 = 2.0*a*b;
    let cos3 = 4.0*a*a*a-3.0*a;
    let sin3 = 3.0*b-4.0*b*b*b;
    return (11.34072*a+16.46899*cos2+7.88380*cos3+14.66441*b-6.37224*sin2+9.19364*sin3+77.12896)*reference_data[98u];
}
fn a20_tonemap_compress(jmh: vec3f) -> vec3f {
    let linear = a20_j_y(jmh.x)/100.0;
    let f = reference_data[89u]*pow(max(0.0,linear)/(linear+reference_data[87u]),reference_data[84u]);
    let Y = max(0.0,f*f/(f+reference_data[85u]))*100.0;
    let J = a20_y_j(Y);
    var M = jmh.y;
    if M != 0.0 {
        let nJ = J/reference_data[93u];
        let snJ = max(0.0,1.0-nJ);
        let norm = a20_chroma_norm(jmh.z);
        let limit = pow(nJ,reference_data[94u])*a20_reach(jmh.z)/norm;
        M *= pow(J/jmh.x,reference_data[94u]);
        M /= norm;
        M = limit-a20_toe(limit-M,limit-0.001,snJ*reference_data[95u],sqrt(nJ*nJ+reference_data[96u]));
        M = a20_toe(M,limit,nJ*reference_data[97u],snJ);
        M *= norm;
    }
    return vec3f(J,M,jmh.z);
}
fn a20_solve_intersect(J: f32, M: f32, focus: f32, maxJ: f32, gain: f32) -> f32 {
    let ms = M/gain;
    let a = ms/focus;
    if J < focus {
        let b = 1.0-ms;
        let c = -J;
        return -2.0*c/(b+sqrt(b*b-4.0*a*c));
    }
    let b = -(1.0+ms+maxJ*a);
    let c = maxJ*ms+J;
    return -2.0*c/(b-sqrt(b*b-4.0*a*c));
}
fn a20_estimate(axis: f32, slope: f32, gamma: f32, Jmax: f32, Mmax: f32, refJ: f32) -> f32 {
    return refJ*pow(axis/refJ,gamma)*Mmax/(Jmax-slope*Mmax);
}
fn a20_cusp(row: u32) -> vec3f {
    let base = 465u+3u*row;
    return vec3f(reference_data[base],reference_data[base+1u],reference_data[base+2u]);
}
fn a20_gamut_compress(jmh: vec3f) -> vec3f {
    let J = jmh.x;
    let M = jmh.y;
    let h = jmh.z;
    let maxJ = reference_data[93u];
    if J <= 0.0 { return vec3f(0.0,0.0,h); }
    if M < 0.0 || J > maxJ { return vec3f(J,0.0,h); }
    // Same interval search as upstream, over the full table rather than the
    // optional optimized search range. This has identical interpolation.
    var lo = 0u;
    var hi = 361u;
    while lo+1u < hi {
        let mid = (lo+hi)/2u;
        if h > a20_cusp(mid).z { lo = mid; } else { hi = mid; }
    }
    let c0 = a20_cusp(hi-1u);
    let c1 = a20_cusp(hi);
    let cusp = mix(c0.xy,c1.xy,(h-c0.z)/(c1.z-c0.z));
    // CTL a2.v1 interpolation_weight deliberately returns h-h_lo, without
    // dividing by interval width. Preserve this for upper-hull gamma lookup.
    let gammaTop = mix(reference_data[1551u+hi-1u],reference_data[1551u+hi],h-c0.z);
    let focus = mix(cusp.x,reference_data[99u],min(1.0,1.3-cusp.x/maxJ));
    let threshold = mix(cusp.x,maxJ,0.3);
    var gain = maxJ*reference_data[100u];
    if J > threshold {
        let adjustment = log2((maxJ-threshold)/max(0.0001,maxJ-J))*0.3010299956639812;
        gain *= adjustment*adjustment+1.0;
    }
    let axis = a20_solve_intersect(J,M,focus,maxJ,gain);
    var scalar = maxJ-axis;
    if axis < focus { scalar = axis; }
    let slope = scalar*(axis-focus)/(focus*gain);
    let cuspAxis = a20_solve_intersect(cusp.x,cusp.y,focus,maxJ,gain);
    let lower = a20_estimate(axis,slope,reference_data[101u],cusp.x,cusp.y,cuspAxis);
    let upper = a20_estimate(maxJ-axis,-slope,gammaTop,maxJ-cusp.x,cusp.y,maxJ-cuspAxis);
    let smoothing = 0.12*cusp.y;
    let smoothH = max(smoothing-abs(lower-upper),0.0)/smoothing;
    let boundary = min(lower,upper)-smoothH*smoothH*smoothH*smoothing/6.0;
    if boundary <= 0.0 { return vec3f(J,0.0,h); }
    let reach = a20_estimate(axis,slope,reference_data[94u],maxJ,a20_reach(h),maxJ);
    let proportion = max(boundary/reach,0.75);
    let compressThreshold = proportion*boundary;
    var remapped = M;
    if M > compressThreshold && proportion < 1.0 {
        let mo = M-compressThreshold;
        let go = boundary-compressThreshold;
        let ro = reach-compressThreshold;
        let scale = ro/(ro/go-1.0);
        let nd = mo/scale;
        remapped = compressThreshold+scale*nd/(1.0+nd);
    }
    return vec3f(axis+remapped*slope,remapped,h);
}
fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    let ap1 = clamp(A20_AP0_AP1*ap0,vec3f(0.0),vec3f(reference_data[90u]));
    let jmh = a20_rgb_jmh(A20_AP1_AP0*ap1);
    let mapped = a20_tonemap_compress(jmh);
    let gamut = a20_gamut_compress(mapped);
    return clamp(a20_jmh_rgb(gamut),vec3f(0.0),vec3f(reference_data[102u]/100.0));
}
