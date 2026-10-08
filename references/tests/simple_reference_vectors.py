#!/usr/bin/env python3
"""Independent double-precision reference fixtures for the five analytical ports.

This evaluates the upstream GLSL/author mathematical definitions, not WGSL text.
Input samples are linear Rec.709 before the workbench AP0 adapter. GPU tests
must convert them to AP0, run reference_transform, then compare linear output.
Run with --check to verify checked-in fixtures or without arguments to regenerate.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Exactly the project common.wgsl adapter. The GPU reads AP0 through rgba16float;
# fixtures supply the actual stored AP0 and evaluate the author math after
# reconstructing Rec.709 from that quantized input, not from the requested RGB.
AP0_TO_REC709 = np.array([
    [2.5214008886, -1.1339957494, -0.3875618568],
    [-0.2762140616, 1.3725955663, -0.0962823557],
    [-0.0153202001, -0.1529925618, 1.1683871996],
], dtype=np.float64)
REC709_TO_AP0 = np.linalg.inv(AP0_TO_REC709)


def quantized_input(rgb):
    ap0 = (REC709_TO_AP0 @ rgb).astype(np.float16).astype(np.float32)
    reconstructed = AP0_TO_REC709 @ ap0.astype(np.float64)
    return ap0.tolist(), reconstructed.tolist()


def nonnegative(rgb):
    return [max(0.0, channel) for channel in rgb]


def pbr_neutral(rgb, peak):
    rgb = nonnegative(rgb)
    x = min(rgb)
    offset = x - 6.25 * x * x if x < 0.08 else 0.04
    color = [v - offset for v in rgb]
    maximum = max(color)
    if maximum < 0.76:
        return color
    d = 0.24
    new_maximum = 1.0 - d * d / (maximum + d - 0.76)
    whitening = 1.0 - 1.0 / (0.15 * (maximum - new_maximum) + 1.0)
    return [v * new_maximum / maximum * (1.0 - whitening)
            + new_maximum * whitening for v in color]


def hable_curve(x):
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (a * x + c * b) + d * e)
            / (x * (a * x + b) + d * f)) - e / f


def hable(rgb, peak):
    return [hable_curve(2.0 * x) / hable_curve(11.2) for x in nonnegative(rgb)]


def lottes_coefficients():
    # Original nested ColToneB expression retained as the independent oracle
    # for the algebraically rearranged WGSL coefficient calculation.
    hdr_max, contrast, shoulder, mid_in, mid_out = 16.0, 2.0, 1.0, 0.18, 0.18
    hi = hdr_max ** contrast
    hs = hdr_max ** (contrast * shoulder)
    mi = mid_in ** contrast
    ms = mid_in ** (contrast * shoulder)
    b = -((-mi + mid_out * (hs * mi - hi * ms * mid_out)
           / (hs * mid_out - ms * mid_out)) / (ms * mid_out))
    c = (hs * mi - hi * ms * mid_out) / (hs * mid_out - ms * mid_out)
    return b, c


def lottes(rgb, peak):
    rgb = nonnegative(rgb)
    b, c = lottes_coefficients()
    maximum = max(1.0e-6, max(rgb))
    z = maximum ** 2.0
    tone_maximum = z / (z * b + c)
    return [tone_maximum * abs((abs(x / maximum) ** (2.0 / 32.0))
                              * (1.0 - tone_maximum ** 4.0)
                              + tone_maximum ** 4.0) ** 32.0 for x in rgb]


def uchimura_scalar(x, peak):
    # Author Desmos definitions, including the intermediate C2/P rather than
    # the simplified WGSL rate. Mathematical continuity supplies the join.
    x = max(x, 0.0)
    p, a, m, length, c, b = max(peak, 1.0), 1.0, 0.22, 0.40, 1.33, 0.0
    l0 = (p - m) * length / a
    s0, s1 = m + l0, m + a * l0
    c2 = a * p / (p - s1)
    t = m * (x / m) ** c + b
    linear = m + a * (x - m)
    shoulder = p - (p - s1) * math.exp(-c2 * (x - s0) / p)
    q = min(max(x / m, 0.0), 1.0)
    w0 = 1.0 - q * q * (3.0 - 2.0 * q)
    w2 = 0.0 if x < s0 else 1.0
    w1 = 1.0 - w0 - w2
    return t * w0 + linear * w1 + shoulder * w2


def uchimura(rgb, peak):
    return [uchimura_scalar(x, peak) for x in rgb]


def aces_fitted(rgb, peak):
    return [min(max((x * (2.51 * x + 0.03))
                    / (x * (2.43 * x + 0.59) + 0.14), 0.0), 1.0)
            for x in nonnegative(rgb)]


ALGORITHMS = {
    "pbr_neutral": (pbr_neutral, {
        "start_compression": 0.76, "desaturation": 0.15,
        "fresnel_reflection": 0.04,
    }),
    "hable": (hable, {
        "a": 0.15, "b": 0.50, "c": 0.10, "d": 0.20, "e": 0.02,
        "f": 0.30, "white": 11.2, "exposure_bias": 2.0,
    }),
    "lottes": (lottes, {
        "hdr_max": 16.0, "contrast": 2.0, "shoulder": 1.0,
        "mid_in": 0.18, "mid_out": 0.18, "crosstalk": 4.0,
        "saturation": 2.0, "cross_saturation": 32.0,
    }),
    "uchimura": (uchimura, {
        "p": "headroom", "a": 1.0, "m": 0.22, "l": 0.40,
        "c": 1.33, "b": 0.0,
    }),
    "aces_fitted": (aces_fitted, {
        "a": 2.51, "b": 0.03, "c": 2.43, "d": 0.59, "e": 0.14,
        "additional_exposure_multiplier": 1.0,
    }),
}

SAMPLES = [
    ("black", [0.0, 0.0, 0.0]),
    ("deep_shadow", [1.0e-5, 2.0e-5, 5.0e-6]),
    ("neutral_fresnel_toe_boundary", [0.08, 0.08, 0.08]),
    ("middle_gray", [0.18, 0.18, 0.18]),
    ("uchimura_linear_start", [0.22, 0.22, 0.22]),
    ("uchimura_sdr_shoulder_start", [0.532, 0.532, 0.532]),
    ("neutral_compression_boundary", [0.80, 0.08, 0.08]),
    ("diffuse_white", [1.0, 1.0, 1.0]),
    ("warm_highlight", [4.0, 0.7, 0.2]),
    ("blue_highlight", [0.1, 0.4, 3.0]),
    ("primary_red", [3.0, 0.0, 0.0]),
    ("hable_white_point_with_bias", [5.6, 5.6, 5.6]),
    ("lottes_input_peak", [16.0, 16.0, 16.0]),
    ("over_range", [100.0, 12.0, 1.0]),
    ("negative_rec709_adapter", [-0.2, 0.3, 1.0]),
]


def fixtures():
    records = []
    for name, (transform, defaults) in ALGORITHMS.items():
        for headroom in ([1.0, 4.0] if name == "uchimura" else [1.0]):
            records.append({
                "algorithm": name,
                "headroom": headroom,
                "input_space": "scene-linear Rec.709 D65 before AP0 adapter",
                "gpu_input_space": "scene-linear AP0 quantized to rgba16float",
                "output_space": "display-linear Rec.709, SDR white = 1",
                "defaults": defaults,
                "vectors": [],
            })
            for tag, rgb in SAMPLES:
                ap0, reconstructed = quantized_input(rgb)
                records[-1]["vectors"].append({
                    "name": tag,
                    "input_rec709": rgb,
                    "input_ap0": ap0,
                    "reconstructed_rec709": reconstructed,
                    "expected_linear_rec709": transform(reconstructed, headroom),
                    "unquantized_expected_linear_rec709": transform(rgb, headroom),
                })
    return {"format_version": 1, "records": records}


def invariant_checks():
    assert math.isclose(hable([5.6] * 3, 1.0)[0], 1.0, abs_tol=1e-14)
    assert math.isclose(lottes([0.18] * 3, 1.0)[0], 0.18, abs_tol=1e-14)
    assert math.isclose(lottes([16.0] * 3, 1.0)[0], 1.0, abs_tol=1e-14)
    for peak in [1.0, 4.0, 16.0]:
        assert uchimura_scalar(0.22, peak) == 0.22
        assert math.isclose(uchimura_scalar(0.4, peak), 0.4, abs_tol=1e-14)
        assert math.isclose(uchimura_scalar(1.0e4, peak), peak, abs_tol=1e-14)
    for transform, _ in ALGORITHMS.values():
        assert max(abs(x) for x in transform([0.0] * 3, 1.0)) < 1e-14
        values = [transform([2.0 ** (stop / 8.0)] * 3, 1.0)[0]
                  for stop in range(-160, 129)]
        assert all(math.isfinite(x) for x in values)
        assert all(a <= b + 1e-14 for a, b in zip(values, values[1:]))
    for _, rgb in SAMPLES:
        assert all(0.0 <= x <= 1.0 for x in pbr_neutral(rgb, 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    invariant_checks()
    target = Path(__file__).with_suffix(".json")
    expected = fixtures()
    if args.check:
        actual = json.loads(target.read_text(encoding="utf-8"))
        assert actual == expected, "Fixtures changed; regenerate and review"
        print(f"Verified {sum(len(r['vectors']) for r in expected['records'])} reference vectors")
    else:
        target.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
