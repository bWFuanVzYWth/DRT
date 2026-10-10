"""Compile Polyphony Digital's pinned GT7 C++ sample to regenerate GPU goldens.

Run from the repository root: python references/validation/generate_gt7_vectors.py
Requires clang++ (or --compiler) and the ignored official source cache populated
as documented in references/entries/gt7.md. The tonemapper body is included
unchanged; this script only adapts the workbench input gamut, framebuffer units,
and Rec.2020 output gamut outside it. Generated C++ and executables are ignored.
No download occurs here, so regeneration always uses the audited source hash.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "third_party/reference_sources/gt7"
SOURCE_HASH = "94df7e7e310b9423ebb5aaaa29416f376ebbae66e0948fb259f8688314b29822"
SOURCE_URL = "https://blog.selfshadow.com/publications/s2025-shading-course/pdi/supplemental/gt7_tone_mapping.cpp"
HEADROOMS = (1.0, 2.0, 4.0, 16.0, 40.0, 64.0)
AP0_TO_REC709 = (
    (2.5214008886, -1.1339957494, -0.3875618568),
    (-0.2762140616, 1.3725955663, -0.0962823557),
    (-0.0153202001, -0.1529925618, 1.1683871996),
)
REC2020_TO_REC709 = (
    (1.660491002108434, -.587641138788551, -.072849863319884),
    (-.124550474521591, 1.132899897125961, -.008349422604371),
    (-.018150763354905, -.100578898008008, 1.118729661362913),
)
AUTHOR_INPUTS = ((.5, 1.23, .75), (12.3, 34.3, 56.9), (1504.7, 64.51, .5))
INPUTS = (
    ("black", (0.0, 0.0, 0.0)),
    ("near_black", (0.0001, 0.0001, 0.0001)),
    ("shadow_gray", (0.01, 0.01, 0.01)),
    ("middle_gray", (0.18, 0.18, 0.18)),
    ("curve_midpoint", (0.538, 0.538, 0.538)),
    ("reference_white", (1.0, 1.0, 1.0)),
    ("sdr_shoulder_join", (1.11, 1.11, 1.11)),
    ("sdr_paper_white", (2.5, 2.5, 2.5)),
    ("highlight_gray", (16.0, 16.0, 16.0)),
    ("very_bright_gray", (4096.0, 4096.0, 4096.0)),
    ("red", (1.0, 0.0, 0.0)),
    ("green", (0.0, 1.0, 0.0)),
    ("blue", (0.0, 0.0, 1.0)),
    ("cyan", (0.0, 1.0, 1.0)),
    ("magenta", (1.0, 0.0, 1.0)),
    ("yellow", (1.0, 1.0, 0.0)),
    ("bright_red", (64.0, 0.0, 0.0)),
    ("bright_green", (0.0, 64.0, 0.0)),
    ("bright_blue", (0.0, 0.0, 64.0)),
    ("bright_yellow", (64.0, 64.0, 0.0)),
    ("bright_cyan", (0.0, 64.0, 64.0)),
    ("bright_magenta", (64.0, 0.0, 64.0)),
    ("blue_highlight", (0.1, 0.4, 4.0)),
    ("warm_highlight", (100.0, 10.0, 1.0)),
    ("near_white_highlight", (16.0, 14.0, 12.0)),
    ("signed_rec709", (-0.1, 0.2, 0.1)),
)

WRAPPER = r'''
#define _CRT_SECURE_NO_WARNINGS
#include <cmath>
#include <cstdio>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
// The published sample uses std::*f names. LLVM's Windows C++ headers expose
// these names only in the global namespace; add aliases without editing it.
namespace std { using ::powf; using ::expf; using ::logf;
                using ::log2f; using ::exp2f; }
#define main gt7_published_sample_main
#include "gt7_tone_mapping.cpp"
#undef main

int main(int argc, char**) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
    if (argc > 1) {
        // Solve the original ICtCp intensity on a chromatic ray. This finds
        // scene inputs bracketing the two original chroma-fade thresholds.
        float headrooms[] = {1.f,2.f,4.f,16.f,40.f,64.f};
        float ratios[] = {.978f,.982f,1.158f,1.162f};
        for (float h:headrooms) {
            GT7ToneMapping mapper;
            if(h == 1.f) mapper.initializeAsSDR();
            else mapper.initializeAsHDR(250.f*std::min(h,40.f));
            for (float ratio:ratios) {
                float low=0.f,high=1.f;
                for(int i=0;i<30;i++) {
                    float rgb[]={.25f*high,high,.5f*high},ucs[3];
                    rgbToUcs(rgb,ucs);
                    if(ucs[0]>=ratio*mapper.framebufferLuminanceTargetUcs_) break;
                    high*=2.f;
                }
                for(int i=0;i<48;i++) {
                    float middle=(low+high)*.5f;
                    float rgb[]={.25f*middle,middle,.5f*middle},ucs[3];
                    rgbToUcs(rgb,ucs);
                    if(ucs[0]<ratio*mapper.framebufferLuminanceTargetUcs_) low=middle;
                    else high=middle;
                }
                printf("%.9g %.9g %.9g %.9g %.9g\n",h,ratio,.25f*high,high,.5f*high);
            }
        }
        return 0;
    }
    int index;
    float headroom, ap0[3];
    while (scanf("%d %f %f %f %f", &index, &headroom, &ap0[0], &ap0[1], &ap0[2]) == 5) {
        // Workbench adapter only: AP0 D60 -> Rec.709 D65 -> Rec.2020 D65.
        const float r = ap0[0]*2.5214008886f + ap0[1]*-1.1339957494f + ap0[2]*-.3875618568f;
        const float g = ap0[0]*-.2762140616f + ap0[1]*1.3725955663f + ap0[2]*-.0962823557f;
        const float b = ap0[0]*-.0153202001f + ap0[1]*-.1529925618f + ap0[2]*1.1683871996f;
        const float rgb[3] = {
            std::max(0.f,r*.627403895934699f + g*.329283038377884f + b*.043313065687417f),
            std::max(0.f,r*.069097289358232f + g*.919540395075458f + b*.011362315566310f),
            std::max(0.f,r*.016391438875150f + g*.088013307877226f + b*.895595253247624f)
        };
        GT7ToneMapping toneMapper;
        const float effectiveHeadroom = std::min(headroom,40.f);
        if (headroom == 1.f) toneMapper.initializeAsSDR();
        else toneMapper.initializeAsHDR(250.f*effectiveHeadroom);
        float output[3];
        toneMapper.applyToneMapping(rgb,output);
        // SDR's original .4 correction is already applied. HDR framebuffer
        // values use 100 nits/unit; convert to 250-nit SDR-white units here.
        const float scale = headroom == 1.f ? 1.f : .4f;
        for(float& c:output) c*=scale;
        const float R = output[0]*1.660491002108434f - output[1]*.58764113878855f - output[2]*.072849863319884f;
        const float G = -output[0]*.124550474521591f + output[1]*1.13289989712596f - output[2]*.008349422604369f;
        const float B = -output[0]*.018150763354905f - output[1]*.100578898008007f + output[2]*1.118729661362912f;
        printf("%d %.9g %.9g %.9g\n",index,R,G,B);
    }
}
'''


def inverse(matrix):
    x, y, z = matrix
    determinant = (x[0] * (y[1] * z[2] - y[2] * z[1])
                   - x[1] * (y[0] * z[2] - y[2] * z[0])
                   + x[2] * (y[0] * z[1] - y[1] * z[0]))
    cofactors = (
        (y[1] * z[2] - y[2] * z[1], x[2] * z[1] - x[1] * z[2], x[1] * y[2] - x[2] * y[1]),
        (y[2] * z[0] - y[0] * z[2], x[0] * z[2] - x[2] * z[0], x[2] * y[0] - x[0] * y[2]),
        (y[0] * z[1] - y[1] * z[0], x[1] * z[0] - x[0] * z[1], x[0] * y[1] - x[1] * y[0]),
    )
    return tuple(tuple(c / determinant for c in row) for row in cofactors)


def half(value):
    return struct.unpack("e", struct.pack("e", value))[0]


def convert(matrix, rgb):
    return tuple(sum(row[i] * rgb[i] for i in range(3)) for row in matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=shutil.which("clang++") or "C:/Program Files/LLVM/bin/clang++.exe")
    args = parser.parse_args()
    source = DIRECTORY / "gt7_tone_mapping.cpp"
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != SOURCE_HASH:
        raise RuntimeError(f"GT7 source hash mismatch: expected {SOURCE_HASH}, found {actual_hash}")
    cpp = DIRECTORY / "validate_reference.cpp"
    executable = DIRECTORY / "validate_reference.exe"
    cpp.write_text(WRAPPER, encoding="utf-8")
    subprocess.run([args.compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)],
                   check=True, timeout=60)
    rec709_to_ap0 = inverse(AP0_TO_REC709)
    inputs = []
    for name, rgb in INPUTS:
        ap0 = tuple(half(sum(row[i] * rgb[i] for i in range(3))) for row in rec709_to_ap0)
        inputs.append({"name": name, "input_rec709": rgb, "input_ap0": ap0})
    # Direct AP0 colors additionally exercise wide-gamut and negative inputs.
    inputs.extend({"name": name, "input_ap0": rgb} for name, rgb in (
        ("ap0_red", (16.0, 0.0, 0.0)),
        ("ap0_green", (0.0, 16.0, 0.0)),
        ("ap0_blue", (0.0, 0.0, 16.0)),
        ("negative_ap0", (-1.0, -1.0, -1.0)),
    ))
    def rec2020_case(name, rgb):
        ap0 = tuple(half(x) for x in convert(rec709_to_ap0, convert(REC2020_TO_REC709, rgb)))
        return {"name": name, "source_input_rec2020": rgb, "input_ap0": ap0}

    inputs.extend(rec2020_case(f"official_example_{index + 1}", rgb)
                  for index, rgb in enumerate(AUTHOR_INPUTS))
    fade_lines = subprocess.check_output([str(executable), "--fade-inputs"], text=True,
                                        timeout=30).splitlines()
    inputs_by_headroom = {h: inputs.copy() for h in HEADROOMS}
    for line in fade_lines:
        h, ratio, r, g, b = map(float, line.split())
        case = rec2020_case(f"chroma_fade_ratio_{ratio:.3f}", (r, g, b))
        case["source_ictcp_intensity_ratio_before_f16"] = ratio
        inputs_by_headroom[h].append(case)
    indexed_inputs = [(h, case) for h in HEADROOMS for case in inputs_by_headroom[h]]
    request = "".join(f"{index} {h} " + " ".join(str(x) for x in case["input_ap0"]) + "\n"
                      for index, (h, case) in enumerate(indexed_inputs))
    lines = subprocess.check_output([str(executable)], input=request, text=True,
                                    timeout=30).splitlines()
    if len(lines) != len(indexed_inputs):
        raise RuntimeError("Compiled upstream did not return every test vector")
    records = []
    offset = 0
    for headroom in HEADROOMS:
        vectors = []
        for case_index, case in enumerate(inputs_by_headroom[headroom]):
            index = offset + case_index
            result = lines[index].split()
            if int(result[0]) != index:
                raise RuntimeError(f"Unexpected compiled upstream vector order at {index}")
            output = [float(x) for x in result[1:]]
            if len(output) != 3 or not all(math.isfinite(x) for x in output):
                raise RuntimeError(f"Compiled upstream returned invalid output at {index}: {output}")
            vectors.append({**case, "expected_linear_rec709": output})
        offset += len(vectors)
        records.append({
            "algorithm": "gt7", "headroom": headroom,
            "source_url": SOURCE_URL, "source_sha256": SOURCE_HASH,
            "source_version": "1.0 (2025-08-10)",
            "oracle": "Compiled unmodified official GT7ToneMapping C++ ICtCp implementation",
            "input_space": "scene-linear AP0 D60 quantized to rgba16float",
            "output_space": "display-linear Rec.709 D65, GT 250-nit SDR white = 1",
            "effective_headroom": min(headroom, 40.0),
            "mode": "initializeAsSDR()" if headroom == 1.0 else f"initializeAsHDR({250 * min(headroom, 40.0):g} nits), output divided by 2.5",
            "documented_peak_range": "250..10000 nits; requested headroom 64 is capped at 40 to preserve the official input contract",
            "defaults": {"alpha": .25, "gray_point": .538, "linear_section": .444,
                         "toe_strength": 1.28, "blend_ratio": .6,
                         "chroma_fade_start": .98, "chroma_fade_end": 1.16},
            "vectors": vectors,
        })
    destination = Path(__file__).with_name("gt7_reference_vectors.json")
    destination.write_text(json.dumps({"format_version": 1, "records": records}, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Wrote {len(indexed_inputs)} compiled-official GT7 vectors to {destination}")


if __name__ == "__main__":
    main()
