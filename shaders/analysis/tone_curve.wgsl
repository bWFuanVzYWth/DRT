struct PlotParameters {
    sample_count: u32,
    target_is_srgb: u32,
    output_peak: f32,
    _padding: u32,
}

@group(0) @binding(0) var curve_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> parameters: PlotParameters;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

fn srgb_to_linear(encoded: vec3<f32>) -> vec3<f32> {
    let low = encoded / 12.92;
    let high = pow((encoded + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, encoded <= vec3<f32>(0.04045));
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let encoded = textureLoad(curve_texture, vec2<u32>(index, 0u), 0).r;
    let display_linear = srgb_to_linear(vec3<f32>(encoded)).r;
    let x = f32(index) / f32(parameters.sample_count - 1u);
    let y = clamp(display_linear / parameters.output_peak, 0.0, 1.0);
    var output: VertexOutput;
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    let gamma_color = vec3<f32>(0.84);
    let color = select(
        gamma_color,
        srgb_to_linear(gamma_color),
        parameters.target_is_srgb != 0u,
    );
    return vec4<f32>(color, 1.0);
}
