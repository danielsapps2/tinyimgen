"""Render one clip of a rigged test character as a lit filmstrip (for checking motion by eye).

    .venv3d/bin/python tools3d/clip_strip.py out/3d/<job> Run [--az 90] [--frames 10] [-o strip.png]
"""
import argparse
import os
import sys

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

import rig  # noqa: E402
from rig_check import skin  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("clip")
    ap.add_argument("--az", type=float, nargs="+", default=[90, 30])
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("-o", default=None)
    a = ap.parse_args()
    import cv2
    import torch
    import tripo
    from tinyimgen.gpulock import gpu_lock
    keep = {}
    rig.rig(a.job, keep=keep)
    clip = next(c for c in keep["clips"] if c["name"] == a.clip)
    mesh = keep["mesh"]
    V, F = mesh["V"].astype(np.float64), mesh["F"]
    UV, tex = rig.render_texture(mesh)
    H = V[:, 1].max()
    n = len(clip["poses"]) - 1
    rows = []
    for az in a.az:
        tiles = []
        for i in range(a.frames):
            Vp = skin(V, keep["idx"], keep["w"], keep["J"], clip["poses"][int(i * n / a.frames)])
            with gpu_lock():
                img = tripo.render(Vp, F, UV, tex, np.array([0, 0.5 * H, 0]), 0.55 * H, az, size=360, W=260,
                                   bg=0.86).copy()
                torch.cuda.empty_cache()
            img[int(360 / 2 + 360 / (1.1 * H) * 0.5 * H) - 1:, :] = (img[int(360 / 2 + 360 / (1.1 * H) * 0.5 * H) - 1:, :]
                                                                      * 0.8).astype(np.uint8)   # floor band
            tiles.append(img)
        rows.append(np.concatenate(tiles, 1))
    out = a.o or os.path.join(a.job, f"strip_{a.clip}.png")
    cv2.imwrite(out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(out, clip["extras"])
