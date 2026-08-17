struct CompositorParameters {
    decode_to_linear: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> parameters: CompositorParameters;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    output.uv = uv;
    return output;
}

fn extended_srgb_to_linear_channel(encoded: f32) -> f32 {
    let magnitude = abs(encoded);
    let linear = select(
        magnitude / 12.92,
        pow((magnitude + 0.055) / 1.055, 2.4),
        magnitude > 0.04045,
    );
    return sign(encoded) * linear;
}

fn extended_srgb_to_linear(encoded: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        extended_srgb_to_linear_channel(encoded.r),
        extended_srgb_to_linear_channel(encoded.g),
        extended_srgb_to_linear_channel(encoded.b),
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let encoded = textureSample(source_texture, source_sampler, input.uv);
    if parameters.decode_to_linear != 0u {
        return vec4<f32>(extended_srgb_to_linear(encoded.rgb), encoded.a);
    }
    return encoded;
}
