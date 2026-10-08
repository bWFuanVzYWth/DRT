"""Regenerate ACES reference vectors with official OpenColorIO CPU transforms.

Requires PyOpenColorIO 2.6.0 and NumPy. The project's ignored local installation
at .cache/reference-python is preferred when present. No helper executables are
compiled or launched. Inputs are quantized to the application's float16 source
texture precision *before* evaluating the independent CPU oracle.
"""
from pathlib import Path
import ctypes
import json
import sys

if hasattr(ctypes, "windll"):
    ctypes.windll.kernel32.SetErrorMode(0x8003)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".cache/reference-python"))
import numpy as np
import PyOpenColorIO as ocio

ACES13_COMMIT = "1256fee50ee35548c6eab8eca854ff3349008489"
ACES20_COMMIT = "069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80"
XYZ_TO_709 = np.array([
    [3.240969941904521, -1.537383177570093, -0.498610760293004],
    [-0.96924363628088, 1.87596750150772, 0.041555057407175],
    [0.055630079696994, -0.203976958888977, 1.056971514242878],
])
INPUTS = [[0., 0., 0.], [0.00001] * 3, [0.001] * 3, [0.01] * 3, [0.18] * 3,
          [1.] * 3, [4.] * 3, [16.] * 3, [256.] * 3, [4096.] * 3,
          [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
          [1., 1., 0.], [0., 1., 1.], [1., 0., 1.],
          [10., 0.1, 0.01], [0.01, 10., 0.1], [0.1, 0.01, 10.],
          [100., 10., 1.], [0.1, 0.4, 4.], [-0.1, 0.2, 0.05],
          [-1., -1., -1.], [1., -0.2, 0.1], [0., 0.0001, 0.],
          [0.037, 0.071, 0.055], [0.56, 0.13, 0.022], [0.014, 0.47, 0.23],
          [2.7, 0.031, 1.4], [0.011, 1.7, 3.8], [21., 12., 0.46],
          [0.42, 35., 7.3], [12., 0.07, 62.]]


def rgb_xyz(primaries):
    """Derive standard RGB/XYZ matrices from official ACES chromaticities."""
    basis = np.array([[x, y, 1. - x - y] for x, y in primaries[:3]]).T
    x, y = primaries[3]
    white = np.array([x / y, 1., (1. - x - y) / y])
    return basis @ np.diag(np.linalg.solve(basis, white))


AP0_XYZ = rgb_xyz([[.7347, .2653], [0., 1.], [.0001, -.077], [.32168, .33767]])
AP1_XYZ = rgb_xyz([[.713, .293], [.165, .830], [.128, .044], [.32168, .33767]])
AP0_TO_AP1 = np.linalg.solve(AP1_XYZ, AP0_XYZ)
AP1_TO_AP0 = np.linalg.inv(AP0_TO_AP1)


def record(algorithm, headroom, cpu):
    result = {
        "algorithm": algorithm,
        "commit": ACES13_COMMIT if algorithm == "aces_13" else ACES20_COMMIT,
        "headroom": headroom,
        "input_space": "scene-linear AP0 D60, quantized to float16 before oracle",
        "output_space": "display-linear Rec709 D65, SDR white = 1",
        "vectors": [],
    }
    for value in INPUTS:
        ap0 = np.array(value, dtype=np.float16).astype(np.float32).tolist()
        oracle_input = ap0
        if algorithm == "aces_20":
            # The OCIO fixed function is the JMh output-transform core. The
            # official CTL outputTransform_fwd additionally begins with AP1
            # clamping; supply that explicit stage to evaluate the full CTL.
            peak = 100. * headroom
            forward_limit = 8. * (128. + 768. * np.log(peak / 100.) / np.log(100.))
            ap1 = np.clip(AP0_TO_AP1 @ np.array(ap0), 0., forward_limit)
            oracle_input = (AP1_TO_AP0 @ ap1).tolist()
        expected = np.array(cpu.applyRGB(oracle_input), dtype=np.float64)
        if algorithm == "aces_13":
            # This builtin implements the complete SDR video RRT/ODT through
            # the D65 XYZ output stage; the original sRGB ODT then converts
            # XYZ -> linear Rec709, clips, and applies sRGB encoding.
            expected = XYZ_TO_709 @ expected
        expected = np.clip(expected, 0., 1. if algorithm == "aces_13" else headroom)
        if not np.isfinite(expected).all():
            raise ValueError(f"Non-finite oracle output for {algorithm}: {ap0}")
        result["vectors"].append({"input_ap0": ap0, "expected_linear_rec709": expected.tolist()})
    return result


def main():
    if ocio.__version__ != "2.6.0":
        raise RuntimeError(f"Expected pinned PyOpenColorIO 2.6.0, found {ocio.__version__}")
    config = ocio.Config.CreateRaw()
    builtin = ocio.BuiltinTransform()
    builtin.setStyle("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-VIDEO_1.0")
    records = [record("aces_13", 1., config.getProcessor(builtin).getDefaultCPUProcessor())]
    for headroom in [1., 2., 10., 40., 64.]:
        transform = ocio.FixedFunctionTransform(ocio.FIXED_FUNCTION_ACES_OUTPUT_TRANSFORM_20,
                                               [100. * headroom, .64, .33, .3, .6, .15, .06, .3127, .329])
        records.append(record("aces_20", headroom, config.getProcessor(transform).getDefaultCPUProcessor()))
    result = {
        "format_version": 1,
        "oracle": {
            "implementation": "Official ASWF OpenColorIO CPU processor",
            "package": "PyOpenColorIO 2.6.0",
            "package_url": "https://pypi.org/project/opencolorio/2.6.0/",
            "source_url": "https://github.com/AcademySoftwareFoundation/OpenColorIO/tree/v2.6.0",
            "aces_13_style": builtin.getStyle(),
            "aces_20_style": "Official CTL AP1 clamp followed by FIXED_FUNCTION_ACES_OUTPUT_TRANSFORM_20, Rec709 D65 limiting primaries",
            "note": "Independent OCIO implementation of official ACES algorithms; f32 CPU implementation and initialization precision may differ from CTL and WGSL.",
        },
        "records": records,
    }
    dest = ROOT / "references/validation/aces_reference_vectors.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {sum(len(r['vectors']) for r in records)} official OCIO reference vectors to {dest}")


if __name__ == "__main__":
    main()
