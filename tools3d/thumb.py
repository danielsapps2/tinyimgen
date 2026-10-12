"""Render a thumbnail for an imported 3D model (Explorer import):

    .venv3d/bin/python tools3d/thumb.py out/3d/<job>      # model.glb -> thumb.png, recorded in job.json

A lit, textured three-quarter view framed to the model's bounds (Y up, as glTF is).
"""
import json
import os
import sys

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402


def main(d):
    import torch
    from PIL import Image
    import rig
    import tripo
    from tinyimgen.gpulock import gpu_lock
    with open(os.path.join(d, "job.json")) as f:
        job = json.load(f)
    mesh = rig.load_glb(os.path.join(d, job["outputs"]["game"]))
    V, F = mesh["V"].astype(np.float64), mesh["F"]
    UV, tex = rig.render_texture(mesh)
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    half = 0.55 * max(np.ptp(V[:, 1]), np.linalg.norm([np.ptp(V[:, 0]), np.ptp(V[:, 2])]) * 768 / 512)
    with gpu_lock():
        img = tripo.render(V, F, UV, tex, c, half, 30, size=768, W=512, el=12, bg=0.86)
        torch.cuda.empty_cache()
    Image.fromarray(img).save(os.path.join(d, "thumb.png"))
    with open(os.path.join(d, "job.json")) as f:     # re-read: the user may have renamed it meanwhile
        job = json.load(f)
    job.setdefault("outputs", {})["thumb"] = "thumb.png"
    job.setdefault("stats", {})["triangles"] = int(len(F))
    with open(os.path.join(d, "job.json.tmp"), "w") as f:
        json.dump(job, f, indent=1)
    os.replace(os.path.join(d, "job.json.tmp"), os.path.join(d, "job.json"))


if __name__ == "__main__":
    main(sys.argv[1])
