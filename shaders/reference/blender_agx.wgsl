// Workbench adapter for Blender's official AgX Base Rec.1886 view.
// Pinned Blender source: 8b5bd560cf16070d10f361d0a9f00b39436dc379.
// Uses the original 57^3 AgX_Base_sRGB.cube grid, without a polynomial fit.
// Original assets, licenses and adapter details: references/entries/blender_agx.md.
fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    // Composite of the pinned config's AP0 -> XYZ D65 Bradford -> E-Gamut.
    let egamut = vec3f(
        dot(ap0, vec3f(1.3242028951644897, -0.23679925501346588, -0.08740367740392685)),
        dot(ap0, vec3f(-0.02774209901690483, 0.9744444489479065, 0.05329766497015953)),
        dot(ap0, vec3f(0.10790395736694336, 0.03378080576658249, 0.8583151698112488)));
    let allocated = clamp((log2(max(egamut, vec3f(1e-20))) + 12.47393) / 24.9999988117,
        vec3f(0.0), vec3f(1.0));
    let encoded = reference_sample_cube(allocated, u32(reference_data[0]), u32(reference_data[1]));
    // The LUT produces Rec.1886 code values; return linear Rec.709 to the
    // common adapter, which performs the workbench's sRGB/scRGB presentation.
    return pow(max(encoded, vec3f(0.0)), vec3f(2.4));
}
