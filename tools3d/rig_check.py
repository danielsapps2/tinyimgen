"""Rig + animation check for one or more 3D job folders (each with a game.glb): rigs it, measures foot slide per clip,
and renders anim_sheet.png (8 lit frames per clip; side view, front view for strafes).

    .venv3d/bin/python tools3d/rig_check.py out/3d/<job> [out/3d/<job> ...]

Foot slide is the ground-frame speed of each planted foot's pivot (heel while the toes are up, else the ball) during
stance, with the clip played in place at its own speed. It should print 0.0 for every clip.
"""
import os
import sys

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple silicon: run ops MPS lacks on the CPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rig  # noqa: E402


def skin(V, idx, w, J, pose):
    """Linear blend skinning of the bind-pose vertices into `pose`."""
    out = np.zeros_like(V)
    R = np.stack(pose.R)
    X = np.stack([pose.pos(n) for n in rig.NAMES])
    for k in range(4):
        j = idx[:, k]
        out += w[:, k:k + 1] * (np.einsum("nij,nj->ni", R[j], V - J[j]) + X[j])
    return out


def foot_low(sk, pose, side):
    """The character's foot point nearest the floor (heel or ball: the toes bend at the ball) in a pose."""
    f = sk.foot[side]
    R, A = pose.get(side + "Foot"), pose.pos(side + "Foot")
    pts = [A + R @ (q - f["ankle"]) for q in (f["heel"], f["ball"])]
    j = int(np.argmin([q[1] for q in pts]))
    return j, pts[j]


def slide(keep):
    """Per clip: the ground-frame speed (m/s) of each foot's lowest point while it is on the floor (within 2.5 mm)
    in consecutive frames, with the in-place clip moved at its own speed. 0 = no foot sliding."""
    sk, res = keep["skel"], {}
    for clip in keep["clips"]:
        ex = clip["extras"]
        if not ex.get("speed") and clip["name"] != "Idle":
            continue
        v = ex.get("speed", 0) * np.array(ex.get("direction", [0, 0, 0]))
        dt, worst = 1 / clip["fps"], 0.0
        for side in ("Left", "Right"):
            prev = None
            for p in clip["poses"]:
                j, q = foot_low(sk, p, side)
                down = q[1] < 0.0025
                if down and prev is not None and prev[0] == j:
                    worst = max(worst, np.linalg.norm(((q - prev[1]) + v * dt)[[0, 2]]) / dt)
                prev = (j, q) if down else None
        res[clip["name"]] = round(float(worst), 3)
    return res


def sheet(path, keep, cols=8):
    import cv2
    from tinyimgen.device import empty_cache
    import tripo
    from tinyimgen.gpulock import gpu_lock
    mesh = keep["mesh"]
    V, F = mesh["V"].astype(np.float64), mesh["F"]
    UV, tex = rig.render_texture(mesh)
    H = V[:, 1].max()
    rows = []
    for clip in keep["clips"]:
        poses, az = clip["poses"], 0 if clip["name"].startswith("Strafe") else 90
        tiles = []
        for i in range(cols):
            Vp = skin(V, keep["idx"], keep["w"], keep["J"], poses[int(round(i * (len(poses) - 1) / cols))])
            with gpu_lock():
                img = tripo.render(Vp, F, UV, tex, np.array([0, 0.5 * H, 0]), 0.56 * H, az, size=256, bg=0.86)
                empty_cache()
            tiles.append(img)
        row = np.concatenate(tiles, 1)
        cv2.putText(row, clip["name"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        rows.append(row)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        keep = {}
        try:
            info = rig.rig(d, keep=keep)
        except rig.RigError as e:
            print(d, "not rigged:", e)
            continue
        print(d, info["seconds"], "contact split:", info["skin"]["contact_faces_split"])
        print("  speeds:", {c["name"]: c.get("speed") for c in info["clips"]})
        print("  foot slide (m/s):", slide(keep))
        sheet(os.path.join(d, "anim_sheet.png"), keep)
