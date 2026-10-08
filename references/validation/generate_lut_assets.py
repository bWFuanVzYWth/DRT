"""Pack pinned AgX/Filmic grids and generate independent OCIO oracle vectors.

Original source trees stay in third_party/reference_sources (ignored by Git).
Runtime float32 tables and reference vectors are deterministic tracked ports.
Requires numpy and PyOpenColorIO; root's local OCIO wheel lives in .cache.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import sys

if sys.platform == "win32":
    ctypes.windll.kernel32.SetErrorMode(0x8003)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".cache/reference-python"))
import numpy as np
import PyOpenColorIO as ocio

BLENDER = ROOT / "third_party/reference_sources/blender/release/datafiles/colormanagement"
FILMIC = ROOT / "third_party/reference_sources/filmic"
ASSETS = ROOT / "shaders/reference/assets"
BLENDER_COMMIT = "8b5bd560cf16070d10f361d0a9f00b39436dc379"
FILMIC_COMMIT = "84bf836a4d9e130045c962c47ac4206395d4393b"


def cube(path):
    size = None
    values = []
    for line in path.read_text().splitlines():
        items = line.split()
        if not items or items[0].startswith("#"):
            continue
        if items[0] == "LUT_3D_SIZE":
            size = int(items[1])
        elif len(items) == 3:
            try:
                values.extend(float(x) for x in items)
            except ValueError:
                pass
    assert len(values) == size**3 * 3
    return size, np.asarray(values, dtype="<f4")


def spi3d(path):
    lines = path.read_text().splitlines()
    sizes = [int(x) for x in lines[2].split()]
    assert sizes[0] == sizes[1] == sizes[2]
    size = sizes[0]
    values = np.empty((size**3, 3), dtype="<f4")
    count = 0
    for line in lines[3:]:
        items = line.split()
        if len(items) == 6:
            r, g, b = [int(x) for x in items[:3]]
            values[r + size * (g + size * b)] = [float(x) for x in items[3:]]
            count += 1
    assert count == size**3
    return size, values.ravel()


def spi1d(path):
    text = path.read_text()
    values = np.asarray([float(x) for x in text.split("{", 1)[1].split("}", 1)[0].split()], dtype="<f4")
    length = int(next(line.split()[1] for line in text.splitlines() if line.startswith("Length")))
    assert len(values) == length
    return values


def write_or_check(path, data):
    if "--check" in sys.argv:
        assert path.read_bytes() == data, f"Stale generated asset: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def decode(rgb):
    x = np.asarray(rgb, dtype=float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def main():
    config = ocio.Config.CreateFromFile(str(BLENDER / "config.ocio"))
    matrix_processor = config.getProcessor("ACES2065-1", "Linear FilmLight E-Gamut").getDefaultCPUProcessor()
    matrix = np.asarray([matrix_processor.applyRGB(x.tolist()) for x in np.eye(3)]).T
    linear_processor = config.getProcessor("ACES2065-1", "Linear Rec.709").getDefaultCPUProcessor()
    rec709_matrix = np.asarray([linear_processor.applyRGB(x.tolist()) for x in np.eye(3)]).T
    agx_path = BLENDER / "luts/AgX_Base_sRGB.cube"
    filmic_path = FILMIC / "luts/desat65cube.spi3d"
    look_path = FILMIC / "looks/Filmic_to_0-70_1-03.spi1d"
    agx_size, agx_values = cube(agx_path)
    filmic_size, filmic_values = spi3d(filmic_path)
    look = spi1d(look_path)
    # Header: dimension, cube offset, curve offset, curve length, followed by
    # optional metadata/reserved values. Grids are red-fast RGB triplets.
    agx_header = np.zeros(16, dtype="<f4")
    agx_header[:4] = [agx_size, 16, 0, 0]
    filmic_header = np.zeros(16, dtype="<f4")
    filmic_header[:4] = [filmic_size, 16, 16 + filmic_values.size, look.size]
    write_or_check(ASSETS / "blender_agx.bin", np.concatenate([agx_header, agx_values]).tobytes())
    write_or_check(ASSETS / "filmic.bin", np.concatenate([filmic_header, filmic_values, look]).tobytes())

    # Evaluate the official Blender AgX display view, without looks.
    agx = config.getProcessor(ocio.DisplayViewTransform(src="ACES2065-1", display="sRGB", view="AgX")).getDefaultCPUProcessor()
    # Compose the original author's Filmic Log + Base Contrast assets. This
    # preserves the original 65^3 grid rather than Blender's resampled 33^3 LUT.
    row_major = np.eye(4)
    row_major[:3, :3] = rec709_matrix
    group = ocio.GroupTransform([
        ocio.MatrixTransform(matrix=row_major.ravel().tolist()),
        ocio.AllocationTransform(allocation=ocio.ALLOCATION_LG2, vars=[-12.473931188, 12.526068812]),
        ocio.FileTransform(src=str(filmic_path), interpolation=ocio.INTERP_TETRAHEDRAL),
        ocio.AllocationTransform(allocation=ocio.ALLOCATION_UNIFORM, vars=[0.0, 0.66]),
        ocio.FileTransform(src=str(look_path), interpolation=ocio.INTERP_LINEAR),
    ])
    filmic = ocio.Config.CreateRaw().getProcessor(group).getDefaultCPUProcessor()
    colors = [np.zeros(3), np.full(3, 0.18), np.ones(3)]
    colors += [np.full(3, 0.18 * 2.0**ev) for ev in [-16, -10, -5, 3, 6, 10, 16]]
    colors += [v * level for level in [0.18, 1.0, 16.0, 128.0] for v in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0.3, 0.01]), np.array([0.01, 0.3, 1])]]
    colors += [np.array([-0.01, 0.1, 0.2]), np.array([1.0, -0.1, 4.0]), np.array([1e-8, 2e-8, 3e-8])]
    records = []
    for algorithm, processor in [("blender_agx", agx), ("filmic", filmic)]:
        vectors = []
        for rgb in colors:
            ap0 = np.asarray(np.linalg.solve(rec709_matrix, rgb), dtype=np.float16).astype(float).tolist()
            expected = np.clip(decode(processor.applyRGB(ap0)), 0.0, 1.0).tolist()
            vectors.append({"input_ap0": ap0, "expected_linear_rec709": expected})
        records.append({"algorithm": algorithm, "headroom": 1.0, "vectors": vectors})
    result = {"format_version": 1, "oracle": "PyOpenColorIO " + ocio.__version__, "records": records}
    write_or_check(ROOT / "references/validation/lut_reference_vectors.json", (json.dumps(result, indent=2) + "\n").encode())
    source_paths = [("blender_agx", agx_path, BLENDER_COMMIT), ("filmic_gamut", filmic_path, FILMIC_COMMIT), ("filmic_base_contrast", look_path, FILMIC_COMMIT)]
    metadata = {"format_version": 1, "ap0_to_e_gamut": matrix.tolist(), "ap0_to_rec709": rec709_matrix.tolist(), "assets": [{"id": name, "commit": commit, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "local_source": str(path.relative_to(ROOT)).replace("\\", "/")} for name, path, commit in source_paths]}
    write_or_check(ROOT / "references/validation/lut_assets.json", (json.dumps(metadata, indent=2) + "\n").encode())
    print(f"AgX {agx_size}^3, Filmic {filmic_size}^3 + {look.size}; {sum(len(r['vectors']) for r in records)} OCIO oracle vectors")
    print("AP0 to E-Gamut:", matrix.tolist())


if __name__ == "__main__":
    main()
