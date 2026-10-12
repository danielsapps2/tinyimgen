"""Auto-rig a generated character: fit a humanoid skeleton, skin the game mesh to it, and write a GLB that carries the
skeleton and the clip library (anim.py).

    .venv3d/bin/python tools3d/rig.py out/3d/<job_id>          # reads game.glb, writes rigged.glb + rig.json
    .venv3d/bin/python tools3d/rig.py --job out/3d/<job_id>    # same, and records the result in job.json

Input is the export pipeline's game.glb: front facing +Z, feet at y=0, so the character's left is +X.

- Skeleton: 22 joints named like Unity's / Godot's humanoid profile (Hips, Spine, Chest, UpperChest, Neck, Head,
  Left/Right Shoulder, UpperArm, LowerArm, Hand, UpperLeg, LowerLeg, Foot, Toes), plus 3 bones per finger
  (Left/Right Thumb..Little Proximal/Intermediate/Distal, fitted by MediaPipe Hands on a hand close-up; 52 in all),
  so engines auto-map it for retargeting. Joints come from MediaPipe Pose on a lit front render of the mesh; depth comes from casting a ray
  through the mesh at each joint and taking the middle of the limb it hits. Feet are measured on the mesh itself.
- Skinning: voxel geodesic binding. The mesh is voxelised solid, each bone's distance is grown through the inside of
  the body only (so a hand resting on the thigh doesn't bind to the leg, a sleeve doesn't bind to the torso), and each
  vertex blends its nearest bones by inverse distance, then the weights are smoothed over the surface.
- Bind pose: every joint has identity rotation (glTF/VRM style), so a clip is just per-joint rotations and engines
  can retarget it from the joint positions alone.
"""
import json
import os
import struct
import sys

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")   # see make3d.py: pinned server RAM stalls numpy's huge pages

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
POSE_TASK = os.path.join(os.path.dirname(HERE), "third_party", "models", "pose_landmarker_heavy.task")
# per-clip playback rates chosen on the Play page ({"Run": 1.2, "StrafeLeft": 2.0, ...}) over the shipped defaults
CLIP_RATES = os.path.join(os.path.dirname(HERE), "out", "clip_rates.json")
CLIP_RATES_DEFAULT = os.path.join(HERE, "clip_rates_default.json")


def clip_rates():
    rates = {}
    for path in (CLIP_RATES_DEFAULT, CLIP_RATES):
        try:
            with open(path) as f:
                rates.update({k: float(v) for k, v in json.load(f).items()})
        except (OSError, ValueError):
            pass
    return rates

# (name, parent). Order is the skin's joint order (Hips first).
JOINTS = [
    ("Hips", None), ("Spine", "Hips"), ("Chest", "Spine"), ("UpperChest", "Chest"), ("Neck", "UpperChest"),
    ("Head", "Neck"),
    ("LeftShoulder", "UpperChest"), ("LeftUpperArm", "LeftShoulder"), ("LeftLowerArm", "LeftUpperArm"),
    ("LeftHand", "LeftLowerArm"),
    ("RightShoulder", "UpperChest"), ("RightUpperArm", "RightShoulder"), ("RightLowerArm", "RightUpperArm"),
    ("RightHand", "RightLowerArm"),
    ("LeftUpperLeg", "Hips"), ("LeftLowerLeg", "LeftUpperLeg"), ("LeftFoot", "LeftLowerLeg"), ("LeftToes", "LeftFoot"),
    ("RightUpperLeg", "Hips"), ("RightLowerLeg", "RightUpperLeg"), ("RightFoot", "RightLowerLeg"),
    ("RightToes", "RightFoot"),
]
BODY = [j for j, _ in JOINTS]                      # the 22 body joints (fit by fit_skeleton)
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
PHALANX = ("Proximal", "Intermediate", "Distal")
for _side in ("Left", "Right"):                    # 15 finger joints per hand (Unity / Godot humanoid names)
    for _f in FINGERS:
        for _k, _ph in enumerate(PHALANX):
            JOINTS.append((_side + _f + _ph, _side + "Hand" if _k == 0 else _side + _f + PHALANX[_k - 1]))
NAMES = [j for j, _ in JOINTS]
INDEX = {j: i for i, j in enumerate(NAMES)}
PARENT = [None if p is None else INDEX[p] for _, p in JOINTS]
# end points of the leaf bones (for skinning segments and the viewer), not joints
TIPS = {"Head": "HeadTop", "LeftToes": "LeftToeTip", "RightToes": "RightToeTip"}
TIPS.update({s_ + f + "Distal": s_ + f + "Tip" for s_ in ("Left", "Right") for f in FINGERS})
FINGER_BONES = {s_: [INDEX[s_ + f + ph] for f in FINGERS for ph in PHALANX] for s_ in ("Left", "Right")}
# MediaPipe hand landmarks per finger: Proximal, Intermediate, Distal joints, then the tip
MP_FINGER = {"Thumb": (1, 2, 3, 4), "Index": (5, 6, 7, 8), "Middle": (9, 10, 11, 12), "Ring": (13, 14, 15, 16),
             "Little": (17, 18, 19, 20)}


class RigError(RuntimeError):
    pass


# --- mesh I/O -------------------------------------------------------------------------------------------------------

def load_glb(path):
    """The game mesh exactly as stored, all parts merged: V, N, UV (float32), F (int, into the merged V), and
    "parts": one per primitive with its vertex/face ranges, texture (encoded bytes, mime) and material name. Tripo's
    export has one part; the local pipeline's has body and hair."""
    import pygltflib
    g = pygltflib.GLTF2().load(path)
    blob = g.binary_blob()

    def acc(i):
        a = g.accessors[i]
        bv = g.bufferViews[a.bufferView]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
        dt = np.dtype({5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a.componentType])
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        stride = bv.byteStride or n * dt.itemsize
        raw = np.frombuffer(blob, np.uint8, count=stride * (a.count - 1) + n * dt.itemsize, offset=off)
        if stride == n * dt.itemsize:
            return raw.view(dt).reshape(a.count, n).copy()
        return np.stack([raw[i * stride:i * stride + n * dt.itemsize].view(dt) for i in range(a.count)])

    def local(node):
        if node.matrix is not None:
            return np.array(node.matrix, np.float64).reshape(4, 4).T
        M = np.eye(4)
        if node.scale is not None:
            M = np.diag([*node.scale, 1.0]) @ M
        if node.rotation is not None:
            x, y, z, w = node.rotation
            M = np.r_[np.c_[np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]),
                            np.zeros(3)], [[0, 0, 0, 1]]] @ M
        if node.translation is not None:
            T = np.eye(4)
            T[:3, 3] = node.translation
            M = T @ M
        return M
    parent = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}

    def world(i):
        M = local(g.nodes[i])
        while i in parent:
            i = parent[i]
            M = local(g.nodes[i]) @ M
        return M
    Vs, Ns, UVs, Fs, parts = [], [], [], [], []
    nv = nf = 0
    for ni, node in enumerate(g.nodes):
        if node.mesh is None:
            continue
        M = world(ni)
        for p in g.meshes[node.mesh].primitives:
            if p.mode not in (None, 4):
                continue
            V = acc(p.attributes.POSITION).astype(np.float64)
            V = V @ M[:3, :3].T + M[:3, 3]
            F = (acc(p.indices).reshape(-1, 3) if p.indices is not None else np.arange(len(V)).reshape(-1, 3))
            N = acc(p.attributes.NORMAL) if p.attributes.NORMAL is not None else None
            if N is not None:
                N = N @ np.linalg.inv(M[:3, :3]).T
                N /= np.maximum(1e-12, np.linalg.norm(N, axis=1, keepdims=True))
            UV = acc(p.attributes.TEXCOORD_0) if p.attributes.TEXCOORD_0 is not None else np.zeros((len(V), 2))
            image, name = None, node.name or "Character"
            if p.material is not None:
                mat = g.materials[p.material]
                name = mat.name or name
                bct = mat.pbrMetallicRoughness.baseColorTexture if mat.pbrMetallicRoughness else None
                if bct is not None:
                    img = g.images[g.textures[bct.index].source]
                    bv = g.bufferViews[img.bufferView]
                    image = (bytes(blob[bv.byteOffset or 0:(bv.byteOffset or 0) + bv.byteLength]), img.mimeType)
            parts.append({"name": name, "v": (nv, nv + len(V)), "f": (nf, nf + len(F)), "image": image,
                          "has_normals": N is not None})
            Vs.append(V.astype(np.float32))
            Ns.append(N if N is not None else np.zeros_like(V, np.float32))
            UVs.append(UV.astype(np.float32))
            Fs.append(F.astype(np.int64) + nv)
            nv += len(V)
            nf += len(F)
    if not parts:
        raise RigError("no triangle mesh in " + os.path.basename(path))
    return {"V": np.concatenate(Vs), "N": np.concatenate(Ns).astype(np.float32), "UV": np.concatenate(UVs),
            "F": np.concatenate(Fs), "parts": parts}


def render_texture(mesh):
    """One texture for preview renders: the parts' textures side by side, with UVs remapped to it (trimesh's v-up
    convention, which tripo.render uses)."""
    import io
    from PIL import Image
    texs = []
    for p in mesh["parts"]:
        t = (np.asarray(Image.open(io.BytesIO(p["image"][0])).convert("RGB")) if p["image"] is not None
             else np.full((64, 64, 3), 180, np.uint8))
        texs.append(np.asarray(Image.fromarray(t).resize((1024, 1024))))
    n = len(texs)
    UV = np.c_[mesh["UV"][:, 0], 1 - mesh["UV"][:, 1]].astype(np.float64)
    for i, p in enumerate(mesh["parts"]):
        a, b = p["v"]
        UV[a:b, 0] = (i + np.clip(UV[a:b, 0], 0, 1)) / n
    return UV, np.concatenate(texs, 1)


def weld(V, F):
    _, inv = np.unique(np.round(V.astype(np.float64), 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((inv.max() + 1, 3))
    Vw[inv] = V
    return Vw, inv[F], inv


# --- skeleton fitting -----------------------------------------------------------------------------------------------

def front_render(V, F, UV, tex, size=1024):
    """Lit front view (camera on +Z) with the pixel mapping: px = W/2 + S (x - cx), py = size/2 - S (y - cy).
    UV, tex: from render_texture."""
    import tripo
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    half = 0.56 * max(hi[1] - lo[1], hi[0] - lo[0])
    img = tripo.render(V.astype(np.float64), F, UV, tex, c, half, 0, size=size, bg=0.86)
    return img, {"c": c, "S": size / (2 * half), "size": size}


def to_world(px, py, m):
    return np.array([(px - m["size"] / 2) / m["S"] + m["c"][0], (m["size"] / 2 - py) / m["S"] + m["c"][1]])


def detect_pose(img):
    """33 MediaPipe pose landmarks (px, py, visibility) of the most prominent person, or None."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if not os.path.exists(POSE_TASK):
        raise RigError(f"missing {POSE_TASK} (MediaPipe pose_landmarker_heavy.task)")
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_TASK), num_poses=1, min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3))
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    det.close()
    if not r.pose_landmarks:
        return None
    H, W = img.shape[:2]
    return np.array([[p.x * W, p.y * H, p.visibility if p.visibility is not None else 1.0] for p in r.pose_landmarks[0]])


def z_hits(Vw, Fw, x, y):
    """Sorted z (front first) of the mesh surface along the line through (x, y) parallel to z."""
    tri = Vw[Fw]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    sel = ((np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0]) <= x) & (np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0]) >= x)
           & (np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1]) <= y) & (np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1]) >= y))
    a, b, c = a[sel], b[sel], c[sel]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1)
    w0 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / den
    w1 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / den
    w2 = 1 - w0 - w1
    inside = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    z = (w0 * a[:, 2] + w1 * b[:, 2] + w2 * c[:, 2])[inside]
    z = np.sort(z)[::-1]
    if len(z) > 1:   # drop near-duplicate hits on shared edges
        z = z[np.r_[True, np.abs(np.diff(z)) > 1e-5]]
    return z


def limb_depth(Vw, Fw, x, y, r):
    """Middle of the first solid the ray at (x, y) enters (the front-most limb there). Searches a small ring when the
    point misses the mesh. Returns z or None."""
    for rad in (0.0, 0.3 * r, 0.6 * r, r):
        for ang in ((0,) if rad == 0 else np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            z = z_hits(Vw, Fw, x + rad * np.cos(ang), y + rad * np.sin(ang))
            if len(z) >= 2:
                return float((z[0] + z[1]) / 2)
    return None


def torso_depth(Vw, Fw, x, y, r):
    """Middle of the thickest solid at (x, y): the body, not a pouch hanging in front of it or hair behind it."""
    best = None
    for dx in (0.0, -0.3 * r, 0.3 * r):
        z = z_hits(Vw, Fw, x + dx, y)
        for i in range(0, len(z) - 1, 2):
            t = z[i] - z[i + 1]
            if best is None or t > best[0]:
                best = (t, (z[i] + z[i + 1]) / 2)
    return None if best is None else float(best[1])


def fit_skeleton(mesh, debug_dir=None):
    """Joint positions (len(NAMES) x 3) and tip positions {name: xyz}, in the mesh's space."""
    V, F = mesh["V"].astype(np.float64), mesh["F"]
    UV, tex = render_texture(mesh)
    Vw, Fw, _ = weld(V, F)
    from tinyimgen.gpulock import gpu_lock
    with gpu_lock():
        img, m = front_render(V, F, UV, tex)
        import torch
        torch.cuda.empty_cache()
    L = detect_pose(img)
    if debug_dir:
        from PIL import Image
        Image.fromarray(img).save(os.path.join(debug_dir, "rig_front.png"))
    if L is None:
        raise RigError("no humanoid found on the front render (rigging needs a character with two arms and two legs)")
    need = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    weak = [i for i in need if L[i, 2] < 0.2]
    if len(weak) > 2:
        raise RigError(f"pose landmarks too uncertain ({len(weak)} of 12 limb joints); is this a humanoid?")
    # MediaPipe's "left" is the person's left, which on a front view is image right (+X). Swap if it disagrees.
    if L[11, 0] < L[12, 0] and L[23, 0] < L[24, 0]:
        swap = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
        for a, b in swap:
            L[[a, b]] = L[[b, a]]
    H = V[:, 1].max() - V[:, 1].min()
    w = lambda i: to_world(L[i, 0], L[i, 1], m)  # noqa: E731
    P = {}
    r_limb = 0.03 * H

    def put(name, xy, z):
        P[name] = np.array([xy[0], xy[1], z])

    # legs and arms, depth from the limb under each joint
    for side, s in (("Left", 0), ("Right", 1)):
        for name, i in (("UpperArm", 11), ("LowerArm", 13), ("Hand", 15), ("LowerLeg", 25)):
            xy = w(i + s)
            z = limb_depth(Vw, Fw, xy[0], xy[1], r_limb)
            put(side + name, xy, 0.0 if z is None else z)
    # shoulder joints sit a little inside and below MediaPipe's shoulder points (which are on the deltoid)
    for side in ("Left", "Right"):
        p = P[side + "UpperArm"]
        p[1] -= 0.012 * H
    # hip joints: MediaPipe's hips are near the femoral heads; depth from the torso
    hipL, hipR = w(23), w(24)
    hip_y = (hipL[1] + hipR[1]) / 2
    z_pelvis = torso_depth(Vw, Fw, (hipL[0] + hipR[0]) / 2, hip_y, r_limb)
    if z_pelvis is None:
        raise RigError("could not find the pelvis inside the mesh")
    put("LeftUpperLeg", hipL, z_pelvis)
    put("RightUpperLeg", hipR, z_pelvis)
    # feet, measured on the mesh: heel/toe extent of the vertices under each ankle
    tips = {}
    for side, s in (("Left", 0), ("Right", 1)):
        ank = w(27 + s)
        knee = P[side + "LowerLeg"]
        ank_y = min(ank[1], knee[1] - 0.05 * H)
        near = Vw[(np.abs(Vw[:, 0] - ank[0]) < 0.07 * H) & (Vw[:, 1] < ank_y + 0.01 * H)]
        if len(near) < 20:
            raise RigError(f"no {side.lower()} foot found under the ankle")
        heel_z, toe_z = np.percentile(near[:, 2], 1), np.percentile(near[:, 2], 99.5)
        flen = toe_z - heel_z
        fx = float(np.median(near[:, 0]))
        ank_y = float(np.clip(ank_y, 0.03 * H, 0.07 * H))
        P[side + "Foot"] = np.array([fx, ank_y, heel_z + 0.25 * flen])
        P[side + "Toes"] = np.array([fx, 0.3 * ank_y, heel_z + 0.74 * flen])
        tips[side + "ToeTip"] = np.array([fx, 0.25 * ank_y, toe_z])
    # knees: keep MediaPipe's x/y but put the knee between hip and ankle in depth when the ray missed
    for side in ("Left", "Right"):
        k = P[side + "LowerLeg"]
        if abs(k[2]) < 1e-9:
            k[2] = (P[side + "UpperLeg"][2] + P[side + "Foot"][2]) / 2
    # trunk
    sh_mid = (P["LeftUpperArm"] + P["RightUpperArm"]) / 2
    top = Vw[:, 1].max()
    ear_y = (w(7)[1] + w(8)[1]) / 2 if min(L[7, 2], L[8, 2]) > 0.2 else w(0)[1]
    head_x = (w(7)[0] + w(8)[0]) / 2 if min(L[7, 2], L[8, 2]) > 0.2 else w(0)[0]
    head_y = ear_y - 0.33 * (top - ear_y)
    hips = np.array([(hipL[0] + hipR[0]) / 2, hip_y + 0.03 * H, z_pelvis])
    neck_y = sh_mid[1] + 0.25 * (head_y - sh_mid[1])
    trunk = {"Hips": hips[1], "Spine": hips[1] + 0.18 * (neck_y - hips[1]),
             "Chest": hips[1] + 0.45 * (neck_y - hips[1]), "UpperChest": hips[1] + 0.72 * (neck_y - hips[1]),
             "Neck": neck_y, "Head": head_y}
    for name, y in trunk.items():
        t = (y - hips[1]) / max(1e-6, head_y - hips[1])
        x = hips[0] + np.clip(t, 0, 1) * (head_x - hips[0])
        z = torso_depth(Vw, Fw, x, y, r_limb)
        P[name] = np.array([x, y, z_pelvis if z is None else z])
    tips["HeadTop"] = np.array([P["Head"][0], top, P["Head"][2]])
    # clavicles: from beside the neck out to the shoulder joint
    for side in ("Left", "Right"):
        ua = P[side + "UpperArm"]
        P[side + "Shoulder"] = np.array([P["UpperChest"][0] + 0.2 * (ua[0] - P["UpperChest"][0]),
                                         ua[1] - 0.01 * H, (P["UpperChest"][2] + P["Neck"][2]) / 2])
    # hand tips: past the knuckles (pose points 17/19 = pinky/index knuckles), along the forearm direction
    for side, s in (("Left", 0), ("Right", 1)):
        wr = P[side + "Hand"]
        kn = (w(17 + s) + w(19 + s)) / 2
        d = np.r_[kn - wr[:2], 0.0]
        fore = wr - P[side + "LowerArm"]
        fore /= np.linalg.norm(fore)
        reach = max(np.linalg.norm(d) * 2.0, 0.07 * H)
        tips[side + "HandTip"] = wr + fore * reach
    # mirror-average the trunk-symmetric parts when the character is (nearly) symmetric about its centre line
    cx = hips[0]
    for a, b in (("LeftUpperLeg", "RightUpperLeg"), ("LeftLowerLeg", "RightLowerLeg"), ("LeftFoot", "RightFoot"),
                 ("LeftToes", "RightToes")):
        pa, pb = P[a], P[b]
        if abs(pa[1] - pb[1]) < 0.04 * H and abs((pa[0] - cx) + (pb[0] - cx)) < 0.04 * H:
            dx = ((pa[0] - cx) - (pb[0] - cx)) / 2
            y, z = (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2
            P[a], P[b] = np.array([cx + dx, y, z]), np.array([cx - dx, y, z])
    J = np.array([P[n] for n in BODY])
    sanity(J, tips, H)
    return J, tips, {"landmarks": L.tolist(), "map": {"c": m["c"].tolist(), "S": m["S"], "size": m["size"]}}


def sanity(J, tips, H):
    g = lambda n: J[INDEX[n]]  # noqa: E731
    if not g("LeftUpperArm")[0] > g("Neck")[0] > g("RightUpperArm")[0]:
        raise RigError("arms are not on either side of the neck")
    if not g("Head")[1] > g("Neck")[1] > g("Hips")[1] > g("LeftLowerLeg")[1] > g("LeftFoot")[1]:
        raise RigError("joint heights out of order (is the character upright?)")
    for side in ("Left", "Right"):
        thigh = np.linalg.norm(g(side + "LowerLeg") - g(side + "UpperLeg"))
        shin = np.linalg.norm(g(side + "Foot") - g(side + "LowerLeg"))
        if not (0.12 * H < thigh < 0.4 * H and 0.12 * H < shin < 0.4 * H):
            raise RigError(f"implausible {side.lower()} leg proportions")


# --- fingers --------------------------------------------------------------------------------------------------------

def hand_frame(J, tips, side):
    """(wrist, unit finger direction, unit thumb-side direction, hand length) from the body fit."""
    w, tip = J[INDEX[side + "Hand"]], tips[side + "HandTip"]
    z = (tip - w) / np.linalg.norm(tip - w)
    y = np.array([0, 0, 1.0]) - z[2] * z                       # thumbs point forward in T- and A-poses
    y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-6 else np.array([0, 1.0, 0])
    return w, z, y, float(np.linalg.norm(tip - w))


def fingers_geometric(Vw, J, tips, side, H):
    """A plausible five-finger layout when the landmarks can't be found: knuckles across the palm's width at half
    the hand's length, fingers straight to the hand's end, the thumb out from the thumb side."""
    w, z, y, L = hand_frame(J, tips, side)
    rel = Vw - w
    along, across = rel @ z, rel @ y
    band = (along > 0.3 * L) & (along < 0.7 * L) & (np.abs(across) < 0.6 * L) & (np.linalg.norm(rel, axis=1) < L)
    hw = float(np.percentile(np.abs(across[band]), 90)) if band.sum() > 20 else 0.22 * L
    pts = {}
    for f, lat, length in (("Index", 0.72, 0.97), ("Middle", 0.24, 1.0), ("Ring", -0.24, 0.96), ("Little", -0.72, 0.86)):
        mcp = w + z * 0.52 * L + y * lat * hw
        tipf = w + z * (0.52 + 0.48 * length) * L + y * lat * hw * 1.05
        seg = tipf - mcp
        pts[f] = [mcp, mcp + 0.45 * seg, mcp + 0.75 * seg, tipf]
    pts["Thumb"] = [w + z * 0.12 * L + y * 0.55 * hw, w + z * 0.3 * L + y * 1.05 * hw,
                    w + z * 0.45 * L + y * 1.3 * hw, w + z * 0.6 * L + y * 1.45 * hw]
    return pts


def fit_fingers(mesh, J, tips, debug_dir=None):
    """Finger joints for both hands: MediaPipe Hands on a lit close-up render looking at the flat of each hand,
    each landmark placed in the middle of the finger under it (as the body joints are). Falls back to
    fingers_geometric when no plausible hand is found. Returns ({joint: xyz}, {tip: xyz}, info)."""
    import tripo
    import landmarks
    from tinyimgen.gpulock import gpu_lock
    V, F = mesh["V"].astype(np.float64), mesh["F"]
    UV, tex = render_texture(mesh)
    Vw, Fw, _ = weld(V, F)
    H = np.ptp(Vw[:, 1])
    hips = J[INDEX["Hips"]]
    pos, tipd, info = {}, {}, {}
    for side in ("Left", "Right"):
        w, z, y, L = hand_frame(J, tips, side)
        c = w + z * 0.5 * L
        near = Vw[(np.linalg.norm(Vw - c, axis=1) < 0.7 * L) & ((Vw - w) @ z > -0.1 * L)]
        cam = None
        if len(near) > 30:
            P = near - near.mean(0)
            cam = np.linalg.eigh(P.T @ P)[1][:, 0]                  # the hand's flat axis
            out = (c - hips) * np.array([1, 0, 1])
            cam = cam * (np.sign(cam[1]) if abs(cam[1]) > 0.7 else np.sign(cam @ out) or 1.0)
        pts, how = None, "geometric"
        if cam is not None:
            el = float(np.degrees(np.arcsin(np.clip(cam[1], -0.999, 0.999))))
            el = float(np.clip(el, -88, 88))
            az = float(np.degrees(np.arctan2(cam[0], cam[2])))
            a, e = np.radians(az), np.radians(el)
            camv = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
            right = np.cross([0, 1, 0], camv); right /= np.linalg.norm(right)
            up = np.cross(camv, right)
            size, half = 512, 0.75 * L
            with gpu_lock():
                img = tripo.render(V, F, UV, tex, c, half, az, size=size, el=el, bg=0.86)
                import torch
                torch.cuda.empty_cache()
            if debug_dir:
                from PIL import Image
                Image.fromarray(img).save(os.path.join(debug_dir, f"rig_hand_{side.lower()}.png"))
            S = size / (2 * half)
            wpx = np.array([size / 2 + S * ((w - c) @ right), size / 2 - S * ((w - c) @ up)])
            found = landmarks.hands(img)
            if found:
                lm = min(found, key=lambda q: np.linalg.norm(q[0, :2] - wpx))
                Vc = np.c_[Vw @ right, Vw @ up, Vw @ camv]
                P3 = []
                for px, py, _ in lm:
                    xx, yy = (px - size / 2) / S + c @ right, (size / 2 - py) / S + c @ up
                    zz = limb_depth(Vc, Fw, xx, yy, 0.02 * L)
                    zz = c @ camv if zz is None else zz
                    P3.append(right * xx + up * yy + camv * zz)
                P3 = np.array(P3)
                mid_len = np.linalg.norm(P3[9] - P3[0])
                ok = (np.linalg.norm(P3[0] - w) < 0.35 * L and 0.3 * L < mid_len < 0.75 * L
                      and np.linalg.norm(P3[12] - P3[9]) > 0.25 * L)
                if ok:
                    pts = {f: [P3[i] for i in MP_FINGER[f]] for f in FINGERS}
                    how = "mediapipe"
        if pts is None:
            pts = fingers_geometric(Vw, J, tips, side, H)
        for f in FINGERS:
            for k, ph in enumerate(PHALANX):
                pos[side + f + ph] = np.asarray(pts[f][k], float)
            tipd[side + f + "Tip"] = np.asarray(pts[f][3], float)
        info[side] = how
    return pos, tipd, info


def palm_normal(J, side):
    """Unit normal out of the palm (the side the fingers curl toward)."""
    w = J[INDEX[side + "Hand"]]
    n = np.cross(J[INDEX[side + "IndexProximal"]] - w, J[INDEX[side + "LittleProximal"]] - w)
    n = n / max(1e-9, np.linalg.norm(n))
    return -n if side == "Left" else n


# finger curl limits (degrees) per joint: a full fist; the thumb bends less at its base
CURL_MAX = {"Thumb": (35.0, 50.0, 70.0), "Index": (85.0, 100.0, 70.0), "Middle": (85.0, 100.0, 70.0),
            "Ring": (85.0, 100.0, 70.0), "Little": (85.0, 100.0, 70.0)}


def curl_axes(J, tips):
    """Per finger joint, the axis (bind / hand frame) a positive rotation about which bends the finger toward the
    palm; the thumb bends across the palm toward the little finger."""
    out = {}
    for side in ("Left", "Right"):
        n = palm_normal(J, side)
        across = J[INDEX[side + "LittleProximal"]] - J[INDEX[side + "IndexProximal"]]
        across /= max(1e-9, np.linalg.norm(across))
        for f in FINGERS:
            for k, ph in enumerate(PHALANX):
                name = side + f + ph
                a = J[INDEX[name]]
                b = J[INDEX[side + f + PHALANX[k + 1]]] if k < 2 else tips[side + f + "Tip"]
                d = (b - a) / max(1e-9, np.linalg.norm(b - a))
                toward = n if f != "Thumb" else (n + 0.8 * across)
                ax = np.cross(d, toward)
                out[name] = ax / max(1e-9, np.linalg.norm(ax))
    return out


# --- skinning -------------------------------------------------------------------------------------------------------

def segments(J, tips):
    """One segment per joint (its bone), from the joint to its child (or tip)."""
    segs = []
    for i, n in enumerate(NAMES):
        kids = [k for k, p in enumerate(PARENT) if p == i]
        if n in TIPS:
            end = tips[TIPS[n]]
        elif n.endswith("Hand"):
            end = J[INDEX[n[:-4] + "MiddleProximal"]]       # the palm: wrist to the middle knuckle
        elif n == "Hips":
            end = J[INDEX["Spine"]]
        elif n == "UpperChest":
            end = J[INDEX["Neck"]]
        else:
            end = J[kids[0]]
        segs.append((J[i], end))
    return segs


def voxelize(Vw, Fw, pitch):
    """Solid occupancy grid: surface voxels from dense surface samples, plus everything the outside can't reach."""
    from scipy import ndimage
    lo = Vw.min(0) - 2 * pitch
    dims = np.ceil((Vw.max(0) + 2 * pitch - lo) / pitch).astype(int) + 1
    tri = Vw[Fw]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = np.maximum(1, np.ceil(area / (0.25 * pitch) ** 2)).astype(int)
    n = np.minimum(n, 4000)
    fi = np.repeat(np.arange(len(Fw)), n)
    r1, r2 = np.random.default_rng(0).random((2, len(fi)))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * tri[fi, 0] + (s * (1 - r2))[:, None] * tri[fi, 1] + (s * r2)[:, None] * tri[fi, 2]
    pts = np.r_[pts, Vw]
    ijk = np.floor((pts - lo) / pitch).astype(int)
    shell = np.zeros(dims, bool)
    shell[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    shell = ndimage.binary_closing(shell, iterations=1) | shell
    outside, _ = ndimage.label(~shell)
    border = np.unique(np.r_[outside[0].ravel(), outside[-1].ravel(), outside[:, 0].ravel(), outside[:, -1].ravel(),
                             outside[:, :, 0].ravel(), outside[:, :, -1].ravel()])
    solid = ~np.isin(outside, border[border > 0])
    return solid, lo


# limb radius (fraction of the height) within which a hand / forearm keeps whatever it touches
ARM_REACH = {"Hand": 0.035, "LowerArm": 0.045}


def release_contacts(D, E, H):
    """Generated meshes fuse a hand resting on the thigh, a pouch or an apron into one surface, so the volume can't
    tell them apart and a swinging arm drags the cloth along. A vertex near a hand / forearm bone loses that bone
    (D set to inf, in place) when it lies outside the limb's own radius (E: straight-line distance to the bone) and a
    non-arm bone is about as close through the body (D: geodesic distance) - i.e. it belongs to whatever the arm
    touches. A T-pose sleeve or gauntlet is far from every torso bone through the volume, so it keeps its arm weights.
    Returns the mask of vertices that lost an arm bone."""
    arm = [INDEX[s + b] for s in ("Left", "Right") for b in ("UpperArm", "LowerArm", "Hand")] + [
        b for bs in FINGER_BONES.values() for b in bs]
    other = [i for i in range(len(NAMES)) if i not in arm and "Shoulder" not in NAMES[i]]
    d_other = D[:, other].min(1)
    d_min = D.min(1)
    mask = np.zeros(D.shape, bool)
    for s in ("Left", "Right"):
        for b, r in ARM_REACH.items():
            j = INDEX[s + b]
            m = (E[:, j] > r * H) & (d_other < 2.5 * D[:, j]) & np.isfinite(d_other)
            mask[:, j] = m & (D[:, j] < 2.0 * d_min)    # only where the bone would carry real weight
    D[mask] = np.inf
    return mask.any(1)


FOREARM = [INDEX[s + b] for s in ("Left", "Right") for b in ("LowerArm", "Hand")] + [
    b for bs in FINGER_BONES.values() for b in bs]
ARM_ALL = FOREARM + [INDEX[s + b] for s in ("Left", "Right") for b in ("UpperArm", "Shoulder")]


def contact_side(dominant):
    """1 = forearm / hand, 2 = body (not any arm bone), 0 = upper arm and clavicle (blends with both)."""
    side = np.where(np.isin(dominant, FOREARM), 1, np.where(np.isin(dominant, ARM_ALL), 0, 2))
    return side


def split_contacts(mesh, joints, weights):
    """Tear the surface where a hand or forearm is fused to the body (a hand resting on the thigh): every triangle
    with both forearm-side and body-side corners goes to the side most of its corners are on, and its other corners
    are duplicated with that side's weights. The two sides then separate cleanly when the arm moves (the opening faces
    the body) instead of stretching a skin membrane between them. Returns (mesh, joints, weights, faces split)."""
    dom = joints[np.arange(len(joints)), np.argmax(weights, 1)]
    side = contact_side(dom)
    F = mesh["F"]
    fs = side[F]
    mixed = np.nonzero((fs == 1).any(1) & (fs == 2).any(1))[0]
    if not len(mixed):
        return mesh, joints, weights, 0
    V, N, UV = [mesh["V"]], [mesh["N"]], [mesh["UV"]]
    J, W = [joints], [weights]
    F = F.copy()
    dup = {}                        # (vertex, side) -> new vertex
    new = len(mesh["V"])
    extra = []                      # (original vertex, donor vertex)
    for f in mixed:
        cls = 1 if (fs[f] == 1).sum() > (fs[f] == 2).sum() else 2
        donor = F[f][fs[f] == cls][0]
        for k in range(3):
            v = F[f, k]
            if side[v] in (1, 2) and side[v] != cls:
                key = (v, cls)
                if key not in dup:
                    dup[key] = new
                    extra.append((v, donor))
                    new += 1
                F[f, k] = dup[key]
    src = np.array([a for a, _ in extra])
    don = np.array([b for _, b in extra])
    V.append(mesh["V"][src]); N.append(mesh["N"][src]); UV.append(mesh["UV"][src])
    J.append(joints[don]); W.append(weights[don])
    V, N, UV, J, W = (np.concatenate(x) for x in (V, N, UV, J, W))
    # keep each part's vertices contiguous: rebuild part by part (duplicates belong to their source's part)
    owner = np.zeros(len(V), int)
    for i, p in enumerate(mesh["parts"]):
        owner[p["v"][0]:p["v"][1]] = i
    owner[len(mesh["V"]):] = owner[src]
    order, parts, Fs, nf = [], [], [], 0
    for i, p in enumerate(mesh["parts"]):
        verts = np.nonzero(owner == i)[0]
        start = len(order)
        order.extend(verts.tolist())
        faces = F[p["f"][0]:p["f"][1]]
        Fs.append(faces)
        parts.append(dict(p, v=(start, start + len(verts)), f=(nf, nf + len(faces))))
        nf += len(faces)
    order = np.array(order)
    remap = np.empty(len(V), int)
    remap[order] = np.arange(len(V))
    out = dict(mesh, V=V[order], N=N[order], UV=UV[order], F=remap[np.concatenate(Fs)], parts=parts)
    return out, J[order], W[order], len(mixed)


def finger_weights(Vw, Fw, W, J, tips, H, power=4.0, smooth_iters=3):
    """Share each hand's weight between the palm (Hand bone) and its finger bones by distance along the surface:
    fingers meet only at the palm, so a vertex on one finger is far from its neighbours' bones even where the
    fingers touch in space."""
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    e = np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]
    for side in ("Left", "Right"):
        hb = INDEX[side + "Hand"]
        wrist = J[hb]
        reach = 1.6 * np.linalg.norm(tips[side + "MiddleTip"] - wrist)
        region = np.nonzero((W[:, hb] > 0.02) & (np.linalg.norm(Vw - wrist, axis=1) < reach))[0]
        if len(region) < 30:
            continue
        loc = -np.ones(len(Vw), int)
        loc[region] = np.arange(len(region))
        ee = e[(loc[e[:, 0]] >= 0) & (loc[e[:, 1]] >= 0)]
        lens = np.linalg.norm(Vw[ee[:, 0]] - Vw[ee[:, 1]], axis=1) + 1e-6
        G = sparse.coo_matrix((lens, (loc[ee[:, 0]], loc[ee[:, 1]])), shape=(len(region),) * 2).tocsr()
        G = G.maximum(G.T)
        bones = [hb] + FINGER_BONES[side]
        P = Vw[region]
        Dg = np.full((len(region), len(bones)), np.inf)
        for k, b in enumerate(bones):
            name = NAMES[b]
            a = J[b]
            if k == 0:
                end = J[INDEX[side + "MiddleProximal"]]
            elif name.endswith("Distal"):
                end = tips[TIPS[name]]
            else:
                end = J[[c for c, pp in enumerate(PARENT) if pp == b][0]]
            t = np.clip(((P - a) @ (end - a)) / max(1e-12, (end - a) @ (end - a)), 0, 1)
            dseg = np.linalg.norm(P - (a + t[:, None] * (end - a)), axis=1)
            src = np.nonzero(dseg <= dseg.min() * 1.3 + 0.002 * H)[0]       # the surface right around the bone
            geo = dijkstra(G, indices=src, min_only=True)
            Dg[:, k] = geo + dseg[src].min()
        Dg = np.maximum(Dg, 0.002 * H)
        reach = np.isfinite(Dg).any(1)
        f = np.zeros_like(Dg)
        f[reach] = np.where(np.isfinite(Dg[reach]), (Dg[reach].min(1, keepdims=True) / Dg[reach]) ** power, 0.0)
        f[~reach, 0] = 1.0                       # cut off from every finger (a hand fused to the body): palm only
        f[f < 0.02] = 0
        f /= np.maximum(f.sum(1, keepdims=True), 1e-12)
        Gu = (G > 0).astype(np.float64)
        Pm = sparse.diags(1 / np.maximum(np.asarray(Gu.sum(1)).ravel(), 1)) @ Gu
        for _ in range(smooth_iters):
            f = 0.5 * f + 0.5 * (Pm @ f)
        share = W[region, hb].copy()
        W[np.ix_(region, bones)] = 0.0
        W[np.ix_(region, bones)] += share[:, None] * f
    return W


def skin_weights(V, F, J, tips, pitch_frac=1 / 160, power=4.0, keep=4, smooth_iters=6):
    """(N, 4) joint indices and (N, 4) weights for the vertices of V (unwelded; seams get identical weights)."""
    from scipy import ndimage, sparse
    from scipy.spatial import cKDTree
    from skimage.graph import MCP_Geometric
    Vw, Fw, inv = weld(V, F)
    H = np.ptp(Vw[:, 1])
    pitch = H * pitch_frac
    solid, lo = voxelize(Vw, Fw, pitch)
    centers = lambda idx: lo + (np.asarray(idx) + 0.5) * pitch  # noqa: E731
    solid_idx = np.argwhere(solid)
    tree = cKDTree(centers(solid_idx))
    # each vertex -> its nearest solid voxel
    dv, vv = tree.query(Vw)
    vox_of_vert = solid_idx[vv]
    cost = np.where(solid, 1.0, np.inf)
    D = np.full((len(Vw), len(NAMES)), np.inf)
    E = np.full_like(D, np.inf)
    finger = {b for bs in FINGER_BONES.values() for b in bs}
    for b, (a, e) in enumerate(segments(J, tips)):
        if b in finger:            # fingers are too thin for the voxels: shared out of the hand's weight below
            continue
        L = max(2, int(np.ceil(np.linalg.norm(e - a) / (0.5 * pitch))))
        pts = a + np.linspace(0, 1, L + 1)[:, None] * (e - a)
        _, si = tree.query(pts)       # bones may graze the surface: start from the nearest solid voxels
        starts = [tuple(x) for x in np.unique(solid_idx[si], axis=0)]
        mcp = MCP_Geometric(cost, fully_connected=True)
        dist, _ = mcp.find_costs(starts)
        d = dist[vox_of_vert[:, 0], vox_of_vert[:, 1], vox_of_vert[:, 2]] * pitch + dv
        # distance to the segment itself where it is closer (thin parts the voxels merge)
        t = np.clip(((Vw - a) @ (e - a)) / max(1e-12, (e - a) @ (e - a)), 0, 1)
        de = np.linalg.norm(Vw - (a + t[:, None] * (e - a)), axis=1)
        D[:, b] = np.where(np.isfinite(d), np.maximum(d, de), np.inf)
        E[:, b] = de
    # fragments no bone reaches through the volume (floaters, a prop not touching the body): straight-line distance
    lost = ~np.isfinite(D).any(1)
    D[lost] = E[lost]
    D = np.maximum(D, 0.25 * pitch)
    contact = release_contacts(D, E, H)
    W = np.where(np.isfinite(D), (D.min(1, keepdims=True) / D) ** power, 0.0)
    W[W < 0.01] = 0
    W /= W.sum(1, keepdims=True)
    # smooth over the surface (uniform Laplacian on the welded mesh), but not across a hand / forearm resting on the
    # body: those edges are cut later (split_contacts), and blending over them would smear a band of weights
    e = np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]
    side = contact_side(np.argmax(W, 1))
    e = e[~(((side[e[:, 0]] == 1) & (side[e[:, 1]] == 2)) | ((side[e[:, 0]] == 2) & (side[e[:, 1]] == 1)))]
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(Vw),) * 2).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    Pm = sparse.diags(1 / np.maximum(deg, 1)) @ A
    for _ in range(smooth_iters):
        W = 0.5 * W + 0.5 * (Pm @ W)
    W0 = W.copy()
    W = finger_weights(Vw, Fw, W, J, tips, H)
    lost_w = W.sum(1) < 0.5                    # fused hands can strand a vertex: keep its weights from before
    W[lost_w] = W0[lost_w]
    idx = np.argsort(-W, 1)[:, :keep]
    w = np.take_along_axis(W, idx, 1)
    w[w < 0.02] = 0
    empty = w.sum(1) <= 0                      # nothing usable left: all on the strongest bone
    w[empty, 0] = 1.0
    w /= w.sum(1, keepdims=True)
    order = np.argsort(idx, 1)   # deterministic layout
    idx, w = np.take_along_axis(idx, order, 1), np.take_along_axis(w, order, 1)
    idx[w == 0] = 0              # glTF: unused influence slots point at joint 0
    if not np.isfinite(w).all():
        raise RigError("skinning produced invalid weights")
    sinfo_contact = int(contact.sum())
    return idx[inv].astype(np.uint8), w[inv].astype(np.float32), {"pitch": pitch, "voxels": int(solid.sum()),
                                                                   "unreached_vertices": int(lost.sum()),
                                                                   "contact_released": sinfo_contact}


def unit_normals(N, V, F):
    """Normals as glTF requires them (unit length): zero ones (degenerate source triangles) get the area-weighted
    normal of their faces, or +Y."""
    N = N.astype(np.float64).copy()
    bad = np.linalg.norm(N, axis=1) < 1e-6
    if bad.any():
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]).astype(np.float64)
        acc = np.zeros_like(N)
        for k in range(3):
            np.add.at(acc, F[:, k], fn)
        N[bad] = acc[bad]
        still = np.linalg.norm(N, axis=1) < 1e-12
        N[still] = [0, 1, 0]
    return (N / np.linalg.norm(N, axis=1, keepdims=True)).astype(np.float32)


# --- glTF -----------------------------------------------------------------------------------------------------------

def hand_sockets(J, tips):
    """Where a held prop attaches, per hand: (position, rotation quaternion xyzw), in the bind pose (hand joints
    have identity rotation, so this is also the socket's rotation relative to its hand).
    Socket axes: +Z along the fingers (wrist -> middle knuckle), +Y out of the thumb side of the fist (little ->
    index knuckle: where a sword's blade points), +X = Y x Z. Origin: the middle of the tunnel a closed fist makes,
    under the first finger segments just past the knuckles, so curling the fingers (auto wrap) closes them around
    the handle. A prop's Grip node uses the same axes, so a game lines the two up (see static/props.js)."""
    import anim
    H = tips["HeadTop"][1]
    out = {}
    for side in ("Left", "Right"):
        w = J[INDEX[side + "Hand"]]
        i, m, l = (J[INDEX[side + f + "Proximal"]] for f in ("Index", "Middle", "Little"))
        z = (m - w) / np.linalg.norm(m - w)
        y = i - l
        y = y - (y @ z) * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        palm = palm_normal(J, side)
        palm = palm - (palm @ z) * z
        palm /= max(1e-9, np.linalg.norm(palm))
        knuckles = (i + l) / 2
        p = np.linalg.norm(J[INDEX[side + "MiddleIntermediate"]] - m)     # first finger segment's length
        pos = knuckles + z * 0.2 * p + palm * 0.62 * p
        out[side] = (pos, anim.quat(np.stack([x, y, z], 1)))
    return out


def write_glb(path, mesh, J, joints, weights, clips, extras=None, tips=None):
    """Skinned GLB: mesh + skeleton (identity-rotation bind pose) + animations.
    clips: list of {"name", "fps", "rot": (frames, joints, 4) xyzw local, "hips": (frames, 3) Hips translation,
    "extras": {...}}."""
    buf = bytearray()
    views, accessors = [], []

    def add(arr, target=None, comp=5126, typ="VEC3", minmax=False):
        arr = np.ascontiguousarray(arr)
        while len(buf) % 4:
            buf.append(0)
        off = len(buf)
        buf.extend(arr.tobytes())
        v = {"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes}
        if target:
            v["target"] = target
        views.append(v)
        count = arr.shape[0] if typ != "SCALAR" else arr.size
        a = {"bufferView": len(views) - 1, "componentType": comp, "count": int(count), "type": typ}
        if minmax:
            flat = arr.reshape(count, -1)
            a["min"], a["max"] = flat.min(0).tolist(), flat.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    nJ = len(NAMES)
    ibm = np.tile(np.eye(4, dtype=np.float32), (nJ, 1, 1))
    ibm[:, 3, :3] = -J              # column-major: translation lives in the last column -> row 3 when flattened
    ibm_acc = add(ibm.reshape(nJ, 16), typ="MAT4")
    images, textures, materials, samplers, prims = [], [], [], [], []
    for part in mesh["parts"]:
        a, b = part["v"]
        fa, fb = part["f"]
        attrs = {"POSITION": add(mesh["V"][a:b].astype(np.float32), 34962, minmax=True)}
        if part.get("has_normals", True):
            attrs["NORMAL"] = add(unit_normals(mesh["N"][a:b], mesh["V"][a:b], mesh["F"][fa:fb] - a), 34962)
        attrs["TEXCOORD_0"] = add(mesh["UV"][a:b].astype(np.float32), 34962, typ="VEC2")
        attrs["JOINTS_0"] = add(joints[a:b].astype(np.uint8), 34962, comp=5121, typ="VEC4")
        attrs["WEIGHTS_0"] = add(weights[a:b].astype(np.float32), 34962, typ="VEC4")
        F = mesh["F"][fa:fb] - a
        big = b - a > 65535
        ind = add(F.astype(np.uint32 if big else np.uint16).ravel(), 34963, comp=5125 if big else 5123, typ="SCALAR")
        pbr = {"metallicFactor": 0.0, "roughnessFactor": 0.85}
        if part["image"] is not None:
            data, mime = part["image"]
            while len(buf) % 4:
                buf.append(0)
            off = len(buf)
            buf.extend(data)
            views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data)})
            images.append({"bufferView": len(views) - 1, "mimeType": mime})
            if not samplers:
                samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071})
            textures.append({"source": len(images) - 1, "sampler": 0})
            pbr["baseColorTexture"] = {"index": len(textures) - 1}
        materials.append({"name": part["name"], "pbrMetallicRoughness": pbr})
        prims.append({"attributes": attrs, "indices": ind, "material": len(materials) - 1})

    nodes = [{"name": "Character", "mesh": 0, "skin": 0}, {"name": "Armature", "children": [2]}]
    jnode = lambda j: 2 + j  # noqa: E731
    axes = curl_axes(J, tips) if tips and len(J) == len(NAMES) else {}
    for j, n in enumerate(NAMES):
        p = PARENT[j]
        t = J[j] - (J[p] if p is not None else 0)
        node = {"name": n, "translation": [float(x) for x in t]}
        if n in axes:     # finger joints: how to curl them (props.js, Prop Editor), in the joint's own frame
            f = next(f for f in FINGERS if f in n)
            node["extras"] = {"curl_axis": [round(float(x), 5) for x in axes[n]],
                              "curl_max_deg": CURL_MAX[f][PHALANX.index(n[len(n.split(f)[0]) + len(f):])]}
            if n in TIPS:   # the fingertip, for contact tests (Prop Editor's auto wrap)
                node["extras"]["tip"] = [round(float(x), 5) for x in tips[TIPS[n]] - J[j]]
        kids = [jnode(k) for k, pp in enumerate(PARENT) if pp == j]
        if kids:
            node["children"] = kids
        nodes.append(node)
    # hand sockets for holdable props (prop editor): children of the hand joints
    for side, (pos, rot) in (hand_sockets(J, tips) if tips else {}).items():
        h = INDEX[side + "Hand"]
        nodes.append({"name": side + "Hand_Socket", "translation": [float(x) for x in pos - J[h]],
                      "rotation": [float(x) for x in rot]})
        nodes[jnode(h)].setdefault("children", []).append(len(nodes) - 1)

    animations = []
    for clip in clips:
        n = clip["rot"].shape[0]
        times = (np.arange(n) / clip["fps"]).astype(np.float32)
        t_acc = add(times, typ="SCALAR", minmax=True)
        accessors[t_acc]["min"], accessors[t_acc]["max"] = [float(times[0])], [float(times[-1])]
        samplers_, channels = [], []
        for j in range(nJ):
            r_acc = add(clip["rot"][:, j].astype(np.float32), typ="VEC4")
            samplers_.append({"input": t_acc, "output": r_acc, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers_) - 1, "target": {"node": jnode(j), "path": "rotation"}})
        h_acc = add(clip["hips"].astype(np.float32))
        samplers_.append({"input": t_acc, "output": h_acc, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers_) - 1, "target": {"node": jnode(0), "path": "translation"}})
        a = {"name": clip["name"], "samplers": samplers_, "channels": channels}
        if clip.get("extras"):
            a["extras"] = clip["extras"]
        animations.append(a)

    while len(buf) % 4:
        buf.append(0)
    gltf = {"asset": {"version": "2.0", "generator": "tinyimgen rig.py"}, "scene": 0,
            "scenes": [{"nodes": [0, 1]}], "nodes": nodes,
            "meshes": [{"name": "Character", "primitives": prims}],
            "materials": materials, "skins": [{"name": "Humanoid", "joints": [jnode(j) for j in range(nJ)],
                                               "inverseBindMatrices": ibm_acc, "skeleton": jnode(0)}],
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(buf)}],
            "animations": animations}
    if images:
        gltf.update(images=images, textures=textures, samplers=samplers)
    if extras:
        gltf["scenes"][0]["extras"] = extras
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(buf), 0x004E4942) + bytes(buf))


def blender():
    """A Blender with a working glTF importer (the distro build lacks numpy): TINYIMGEN_BLENDER, else ~/Tools."""
    for b in (os.environ.get("TINYIMGEN_BLENDER"), os.path.expanduser("~/Tools/blender-build/bin/blender")):
        if b and os.path.exists(b):
            return b
    return None


def to_fbx(glb, fbx):
    """rigged.glb -> rigged.fbx through Blender (~2 s). Returns False when Blender is missing or the export fails."""
    import subprocess
    b = blender()
    if not b:
        return False
    if os.path.exists(fbx):
        os.remove(fbx)
    r = subprocess.run([b, "-b", "-P", os.path.join(HERE, "blender_fbx.py"), "--", glb, fbx], capture_output=True,
                       text=True, timeout=300)
    return r.returncode == 0 and "FBX_OK" in r.stdout and os.path.exists(fbx)


# --- driver ---------------------------------------------------------------------------------------------------------

def rig(job_dir, src="game.glb", out="rigged.glb", progress=lambda msg: None, keep=None):
    import time
    import anim
    t0 = time.time()
    mesh = load_glb(os.path.join(job_dir, src))
    progress("fitting skeleton")
    J, tips, dbg = fit_skeleton(mesh, debug_dir=job_dir)
    progress("fitting fingers")
    fpos, ftips, finfo = fit_fingers(mesh, J, tips, debug_dir=job_dir)
    J = np.vstack([J, [fpos[n] for n in NAMES[len(BODY):]]])
    tips.update(ftips)
    progress("skinning")
    t1 = time.time()
    idx, w, sinfo = skin_weights(mesh["V"].astype(np.float64), mesh["F"], J, tips)
    mesh, idx, w, sinfo["contact_faces_split"] = split_contacts(mesh, idx, w)
    t2 = time.time()
    progress("baking animations")
    abd = anim.arm_abduction(mesh["V"].astype(np.float64), idx, w, J, tips)
    fyaw = anim.foot_rest_yaw(mesh["V"].astype(np.float64), idx, w)
    skel = anim.Skeleton(J, tips, abd, fyaw)
    rates = clip_rates()
    clips = [anim.bake(skel, c, rates.get(c["name"], 1.0)) for c in anim.LIBRARY]
    write_glb(os.path.join(job_dir, out), mesh, J, idx, w, clips, tips=tips,
              extras={"tinyimgen_rig": 1, "joints": NAMES, "sockets": ["LeftHand_Socket", "RightHand_Socket"]})
    progress("FBX export")
    fbx = to_fbx(os.path.join(job_dir, out), os.path.join(job_dir, os.path.splitext(out)[0] + ".fbx"))
    warnings = []
    if sinfo["contact_faces_split"] > 50:
        warnings.append("The hands touch the body in this pose, so the mesh was split where they meet; moving arms may "
                        "show small gaps there. Rebuild with Pose: T-pose for clean arm animation.")
    info = {"warnings": warnings, "joints": {n: [round(float(x), 4) for x in J[i]] for i, n in enumerate(NAMES)},
            "tips": {k: [round(float(x), 4) for x in v] for k, v in tips.items()},
            "clips": [{"name": c["name"], "seconds": round((c["rot"].shape[0] - 1) / c["fps"], 3),
                       **(c.get("extras") or {})} for c in clips],
            "arm_abduction_deg": {k: round(float(np.degrees(v)), 1) for k, v in abd.items()}, "fingers": finfo,
            "foot_yaw_deg": {k: round(float(np.degrees(v)), 1) for k, v in fyaw.items()},
            "fbx": os.path.splitext(out)[0] + ".fbx" if fbx else None, "skin": sinfo,
            "seconds": {"fit": round(t1 - t0, 1), "skin": round(t2 - t1, 1), "total": round(time.time() - t0, 1)}}
    if keep is not None:   # for tests: the in-memory results
        keep.update(mesh=mesh, J=J, tips=tips, idx=idx, w=w, clips=clips, skel=skel)
    with open(os.path.join(job_dir, "rig.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def rig_job(job_dir):
    """Rig a finished 3D job in place (server: "Rig + animate" on an existing model). Progress and the result go to
    job.json's "rig" field; the job's own status stays "done"."""
    path = os.path.join(job_dir, "job.json")

    def update(**kw):
        with open(path) as f:
            data = json.load(f)
        for k, v in kw.items():
            if k == "outputs":
                data.setdefault("outputs", {}).update(v)
            else:
                data[k] = v
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    update(rig={"state": "running", "detail": "starting"})
    try:
        info = rig(job_dir, progress=lambda msg: update(rig={"state": "running", "detail": msg}))
    except RigError as e:
        update(rig={"state": "skipped", "detail": str(e)})
        return
    except Exception as e:
        update(rig={"state": "error", "detail": f"{type(e).__name__}: {e}"})
        raise
    import time
    update(rig={"state": "done", "clips": [c["name"] for c in info["clips"]], "warnings": info["warnings"],
                "stamp": int(time.time())},
           outputs={"rigged": "rigged.glb", "rig": "rig.json", **({"fbx": info["fbx"]} if info["fbx"] else {})})


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--job":
        rig_job(sys.argv[2])
    else:
        print(json.dumps(rig(sys.argv[1], *sys.argv[2:]), indent=1))
