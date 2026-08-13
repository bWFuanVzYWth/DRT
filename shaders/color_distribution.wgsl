struct DistributionParameters {
    rotation : vec4<f32>,
    image_size : vec2<u32>,
    space : u32,
    target_is_srgb : u32,
}

@group(0) @binding(0) var output_image : texture_2d<f32>;
@group(0) @binding(1) var<uniform> parameters : DistributionParameters;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
    @location(1) alpha : f32,
}

fn srgb_to_linear(encoded : vec3<f32>) -> vec3<f32> {
    let low = encoded / 12.92;
    let high = pow((encoded + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, encoded <= vec3<f32>(0.04045));
}

fn linear_rgb_to_oklab(color : vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    let m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    let s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;
    let root = pow(max(vec3<f32>(l, m, s), vec3<f32>(0.0)), vec3<f32>(1.0 / 3.0));
    return vec3<f32>(
        0.2104542553 * root.x + 0.7936177850 * root.y - 0.0040720468 * root.z,
        1.9779984951 * root.x - 2.4285922050 * root.y + 0.4505937099 * root.z,
        0.0259040371 * root.x + 0.7827717662 * root.y - 0.8086757660 * root.z
    );
}

fn rotate(position : vec3<f32>) -> vec3<f32> {
    let yaw = parameters.rotation.x;
    let pitch = parameters.rotation.y;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let cp = cos(pitch);
    let sp = sin(pitch);
    let around_y = vec3<f32>(
        cy * position.x + sy * position.z,
        position.y,
        -sy * position.x + cy * position.z
    );
    return vec3<f32>(
        around_y.x,
        cp * around_y.y - sp * around_y.z,
        sp * around_y.y + cp * around_y.z
    );
}

fn project(position : vec3<f32>) -> vec4<f32> {
    let transformed = rotate(position);
    let fit = select(0.62, 0.52, parameters.space == 0u);
    // WebGPU clips depth to 0..w. Bias the rotated depth into that interval so
    // the back half of the distribution is not discarded.
    return vec4<f32>(transformed.xy * fit, 0.5 + transformed.z * 0.1, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    let pixel = vec2<u32>(vertex_index % parameters.image_size.x, vertex_index / parameters.image_size.x);
    let encoded = textureLoad(output_image, pixel, 0).rgb;

    var distribution_position : vec3<f32>;
    if (parameters.space == 0u) {
        distribution_position = encoded * 2.0 - 1.0;
    } else {
        let lab = linear_rgb_to_oklab(srgb_to_linear(encoded));
        distribution_position = vec3<f32>(lab.y * 2.5, lab.x * 2.0 - 1.0, lab.z * 2.5);
    }

    // Keep the complete point cloud inside clip space at every rotation. A
    // perspective divide can push points across or beyond a clip plane; this
    // orthographic fit only changes the viewing direction.
    var output : VertexOutput;
    output.position = project(distribution_position);
    output.color = encoded;
    output.alpha = 0.24;
    return output;
}

fn cube_edge(edge : u32) -> vec2<u32> {
    switch edge {
        case 0u: { return vec2<u32>(0u, 1u); }
        case 1u: { return vec2<u32>(0u, 2u); }
        case 2u: { return vec2<u32>(0u, 4u); }
        case 3u: { return vec2<u32>(1u, 3u); }
        case 4u: { return vec2<u32>(1u, 5u); }
        case 5u: { return vec2<u32>(2u, 3u); }
        case 6u: { return vec2<u32>(2u, 6u); }
        case 7u: { return vec2<u32>(3u, 7u); }
        case 8u: { return vec2<u32>(4u, 5u); }
        case 9u: { return vec2<u32>(4u, 6u); }
        case 10u: { return vec2<u32>(5u, 7u); }
        default: { return vec2<u32>(6u, 7u); }
    }
}

fn cube_corner(index : u32) -> vec3<f32> {
    let encoded = vec3<f32>(
        f32(index & 1u),
        f32((index >> 1u) & 1u),
        f32((index >> 2u) & 1u)
    );
    return encoded * 2.0 - 1.0;
}

@vertex
fn vs_guide(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var position : vec3<f32>;
    var color = vec3<f32>(0.34, 0.38, 0.45);
    var alpha = 0.42;
    if (vertex_index < 24u) {
        let edge = cube_edge(vertex_index / 2u);
        let corner = select(edge.x, edge.y, (vertex_index & 1u) == 1u);
        position = cube_corner(corner);
    } else {
        let axis = (vertex_index - 24u) / 2u;
        let endpoint = (vertex_index & 1u) == 1u;
        position = vec3<f32>(-1.0);
        if (endpoint) {
            if (axis == 0u) {
                position.x = 1.0;
                color = vec3<f32>(1.0, 0.025, 0.015);
            } else if (axis == 1u) {
                position.y = 1.0;
                color = vec3<f32>(0.015, 1.0, 0.025);
            } else {
                position.z = 1.0;
                color = vec3<f32>(0.015, 0.12, 1.0);
            }
        } else if (axis == 0u) {
            color = vec3<f32>(1.0, 0.025, 0.015);
        } else if (axis == 1u) {
            color = vec3<f32>(0.015, 1.0, 0.025);
        } else {
            color = vec3<f32>(0.015, 0.12, 1.0);
        }
        alpha = 0.95;
    }

    var output : VertexOutput;
    output.position = project(position);
    output.color = color;
    output.alpha = alpha;
    return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    // egui prefers a gamma-space UNORM framebuffer, but some platforms expose
    // only an sRGB framebuffer. In that fallback case the attachment performs
    // the final OETF, so feed it linear values.
    let color = select(input.color, srgb_to_linear(input.color), parameters.target_is_srgb != 0u);
    return vec4<f32>(color, input.alpha);
}
