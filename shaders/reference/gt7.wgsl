// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Polyphony Digital Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// WGSL port of the official GT7 Tone Mapping sample 1.0 (2025-08-10).
// Uses its default ICtCp branch, component curve, chroma fade, and RGB blend.
// Provenance, unit conversion and input/output adapters: references/entries/gt7.md.
// reference_data stores the original curve initialization, not a LUT.

fn gt7_inverse_pq(value: f32) -> f32 {
    // Source frame-buffer 1 = 100 cd/m²; ST-2084 normalizes to 10,000 cd/m².
    let normalized = value * 100.0 / 10000.0;
    let power = pow(normalized, 0.1593017578125);
    return exp2(78.84375 * (log2(0.8359375 + 18.8515625 * power)
        - log2(1.0 + 18.6875 * power)));
}

fn gt7_pq(value: f32) -> f32 {
    let power = pow(clamp(value, 0.0, 1.0), 1.0 / 78.84375);
    let numerator = max(power - 0.8359375, 0.0);
    let normalized = pow(numerator / (18.8515625 - 18.6875 * power),
        1.0 / 0.1593017578125);
    return normalized * 10000.0 / 100.0;
}

fn gt7_rgb_to_ictcp(rgb: vec3f) -> vec3f {
    let l = (1688.0 * rgb.r + 2146.0 * rgb.g + 262.0 * rgb.b) / 4096.0;
    let m = (683.0 * rgb.r + 2951.0 * rgb.g + 462.0 * rgb.b) / 4096.0;
    let s = (99.0 * rgb.r + 309.0 * rgb.g + 3688.0 * rgb.b) / 4096.0;
    let l_pq = gt7_inverse_pq(l);
    let m_pq = gt7_inverse_pq(m);
    let s_pq = gt7_inverse_pq(s);
    return vec3f(
        (2048.0 * l_pq + 2048.0 * m_pq) / 4096.0,
        (6610.0 * l_pq - 13613.0 * m_pq + 7003.0 * s_pq) / 4096.0,
        (17933.0 * l_pq - 17390.0 * m_pq - 543.0 * s_pq) / 4096.0);
}

fn gt7_ictcp_to_rgb(ictcp: vec3f) -> vec3f {
    let l = gt7_pq(ictcp.x + 0.00860904 * ictcp.y + 0.11103 * ictcp.z);
    let m = gt7_pq(ictcp.x - 0.00860904 * ictcp.y - 0.11103 * ictcp.z);
    let s = gt7_pq(ictcp.x + 0.560031 * ictcp.y - 0.320627 * ictcp.z);
    return max(vec3f(0.0), vec3f(
        3.43661 * l - 2.50645 * m + 0.0698454 * s,
        -0.79133 * l + 1.9836 * m - 0.192271 * s,
        -0.0259499 * l - 0.0989137 * m + 1.12486 * s));
}

fn gt7_curve(input: f32) -> f32 {
    if (input < 0.0) { return 0.0; }
    let peak = reference_data[0];
    if (input < 0.444 * peak) {
        let linear_weight = smoothstep(0.0, 0.538, input);
        let toe = 0.538 * pow(input / 0.538, 1.280);
        return (1.0 - linear_weight) * toe + linear_weight * input;
    }
    return reference_data[2] + reference_data[3] * exp(input * reference_data[4]);
}

fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    // Workbench AP0/D60 -> Rec.2020/D65. The published sample requires
    // nonnegative Rec.2020 for its PQ powers; clip at that input boundary.
    let rec709 = reference_ap0_to_rec709(ap0);
    let color = max(vec3f(0.0), vec3f(
        dot(rec709, vec3f(0.627403895934699, 0.329283038377884, 0.043313065687417)),
        dot(rec709, vec3f(0.069097289358232, 0.919540395075458, 0.011362315566310)),
        dot(rec709, vec3f(0.016391438875150, 0.088013307877226, 0.895595253247624))));

    let ucs = gt7_rgb_to_ictcp(color);
    let skewed_rgb = vec3f(gt7_curve(color.r), gt7_curve(color.g), gt7_curve(color.b));
    let skewed_ucs = gt7_rgb_to_ictcp(skewed_rgb);
    let chroma_scale = 1.0 - smoothstep(0.98, 1.16, ucs.x / reference_data[1]);
    let scaled_ucs = vec3f(skewed_ucs.x, ucs.yz * chroma_scale);
    let scaled_rgb = gt7_ictcp_to_rgb(scaled_ucs);
    let mapped = reference_data[5] * min(
        0.4 * skewed_rgb + 0.6 * scaled_rgb, vec3f(reference_data[0]));

    // Published output is linear Rec.2020. The workbench presents linear
    // Rec.709 and clips for display only in its shared presentation adapter.
    return vec3f(
        dot(mapped, vec3f(1.660491002108434, -0.587641138788551, -0.072849863319884)),
        dot(mapped, vec3f(-0.124550474521591, 1.132899897125961, -0.008349422604371)),
        dot(mapped, vec3f(-0.018150763354905, -0.100578898008008, 1.118729661362913)));
}
