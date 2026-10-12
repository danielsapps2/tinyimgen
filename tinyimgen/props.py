"""Holdable props: grip points on a 3D object (set in the prop editor, /prop-editor) and the game files made from them.

    out/3d/<job>/grip.json   the editor's data (below)
    out/3d/<job>/prop.glb    game.glb under a "Prop" root, plus Grip_Primary (and Grip_Secondary) nodes

Grip node axes match the characters' hand sockets (rig.hand_sockets): +Z along the fingers, +Y out of the thumb
side of the fist, +X = Y x Z. "upright" holds (swords, axes, torches, tools) point the prop along the grip's +Y,
"forward" holds (guns, wands, torches held out) along its +Z. To hold a prop, a game makes the grip node coincide
with the hand socket:

    prop_root.world = hand_socket.world * inverse(grip_node.local)

(grip nodes sit directly under the Prop root and carry no scale; the model's size lives on the "Model" node).

grip.json:
    {"version": 1, "hands": 1 | 2, "hold": "upright" | "forward", "scale": s, "hand": "right" | "left",
     "primary":   {"position": [x, y, z], "rotation": [x, y, z, w]},      # in the Prop root, metres (scaled)
     "secondary": {...} | null,                                        # the second hand (two-handed props)
     "direction": [x, y, z], "roll_deg": r, "secondary_roll_deg": r2,   # the editor's inputs (model space)
     "points_model": {"primary": [...], "secondary": [...]},           # grip points in the unscaled model
     "length_m": the prop's longest side after scaling,
     "fingers": {"primary": {"Thumb": c, ...}, "secondary": {...}},   # curl around the grip: c = 0..1 for the
                                       # whole finger, or [knuckle, middle, end] per joint (props.js setFingers)
     "arms": "animated" | "static",
     "carry": {"position": [x, y, z] (character heights), "rotation": [x, y, z, w], "of": "grip"} | null}
                                       # static hold: where the holding hand's grip (Grip_Primary = the hand socket)
                                       # sits relative to the UpperChest bone (props.js carryPose). "of": "root"
                                       # (older saves) = where the Prop root sits instead.
"""
import json
import math
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, "out", "3d")


def _vec(v, n):
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                                                 for x in v):
        raise ValueError(f"expected {n} numbers")
    return [float(x) for x in v]


def _quat(q):
    q = _vec(q, 4)
    n = math.sqrt(sum(x * x for x in q))
    if n < 1e-6:
        raise ValueError("zero rotation")
    return [x / n for x in q]


def clean(grip):
    """Validate the editor's payload."""
    out = {"version": 1, "hands": 2 if int(grip.get("hands", 1)) == 2 else 1,
           "hold": "forward" if grip.get("hold") == "forward" else "upright",
           "scale": float(grip.get("scale", 1.0)), "hand": "left" if grip.get("hand") == "left" else "right"}
    if not (0.01 <= out["scale"] <= 100):
        raise ValueError("scale out of range")
    for k in ("primary", "secondary"):
        g = grip.get(k)
        if k == "secondary" and (out["hands"] == 1 or not g):
            out[k] = None
            continue
        if not g:
            raise ValueError("the primary grip is missing")
        out[k] = {"position": _vec(g.get("position"), 3), "rotation": _quat(g.get("rotation"))}
    out["direction"] = _vec(grip.get("direction", [0, 1, 0]), 3)
    out["roll_deg"] = float(grip.get("roll_deg", 0))
    out["secondary_roll_deg"] = float(grip.get("secondary_roll_deg", 0))
    pm = grip.get("points_model") or {}
    out["points_model"] = {k: _vec(v, 3) for k, v in pm.items() if k in ("primary", "secondary") and v}
    if grip.get("length_m") is not None:
        out["length_m"] = float(grip["length_m"])
    # finger curls per hand (0 straight .. 1 fist) and the optional static carry pose
    fingers = {}
    for k in ("primary", "secondary"):
        f = (grip.get("fingers") or {}).get(k)
        if f:
            clamp = lambda v: min(1.0, max(0.0, float(v)))  # noqa: E731
            fingers[k] = {n: ([clamp(x) for x in f[n]][:3] if isinstance(f[n], (list, tuple)) else clamp(f[n]))
                          for n in ("Thumb", "Index", "Middle", "Ring", "Little") if n in f}
    out["fingers"] = fingers
    c = grip.get("carry")
    out["carry"] = ({"position": _vec(c.get("position"), 3), "rotation": _quat(c.get("rotation")),
                     "of": "grip" if c.get("of") == "grip" else "root"}
                    if c and c.get("position") else None)
    out["arms"] = "static" if out["carry"] and grip.get("arms") == "static" else "animated"
    return out


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB")
    off, js, binc = 12, None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk)
        elif ctype == 0x004E4942:
            binc = chunk
        off += 8 + clen
    return js, binc


def write_glb(path, js, binc):
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    binc += b"\0" * (-len(binc) % 4)
    total = 12 + 8 + len(j) + (8 + len(binc) if binc else 0)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A) + j)
        if binc:
            f.write(struct.pack("<II", len(binc), 0x004E4942) + binc)
    os.replace(path + ".tmp", path)


def build_prop(job_dir, grip):
    """prop.glb: the object's game.glb under Prop -> Model (scaled), with the grip nodes beside Model."""
    js, binc = read_glb(os.path.join(job_dir, "game.glb"))
    scene = js["scenes"][js.get("scene", 0)]
    nodes = js["nodes"]
    model = len(nodes)
    nodes.append({"name": "Model", "scale": [grip["scale"]] * 3, "children": list(scene["nodes"])})
    kids = [model]
    for k, name in (("primary", "Grip_Primary"), ("secondary", "Grip_Secondary")):
        g = grip.get(k)
        if g:
            nodes.append({"name": name, "translation": g["position"], "rotation": g["rotation"]})
            kids.append(len(nodes) - 1)
    nodes.append({"name": "Prop", "children": kids,
                  "extras": {"tinyimgen_prop": 1, "hands": grip["hands"], "hold": grip["hold"]}})
    scene["nodes"] = [len(nodes) - 1]
    js.setdefault("asset", {})["generator"] = "tinyimgen props.py"
    write_glb(os.path.join(job_dir, "prop.glb"), js, binc)


def save(job_id, grip):
    d = os.path.join(JOBS, os.path.basename(job_id))
    if not os.path.isfile(os.path.join(d, "game.glb")):
        raise ValueError("that 3D object has no game.glb")
    grip = clean(grip)
    with open(os.path.join(d, "grip.json.tmp"), "w") as f:
        json.dump(grip, f, indent=1)
    os.replace(os.path.join(d, "grip.json.tmp"), os.path.join(d, "grip.json"))
    build_prop(d, grip)
    jp = os.path.join(d, "job.json")
    with open(jp) as f:
        job = json.load(f)
    job.setdefault("outputs", {}).update(prop="prop.glb", grip="grip.json")
    job["grip"] = {"hands": grip["hands"], "hold": grip["hold"]}
    with open(jp + ".tmp", "w") as f:
        json.dump(job, f, indent=1)
    os.replace(jp + ".tmp", jp)
    return grip


def load(job_id):
    p = os.path.join(JOBS, os.path.basename(job_id), "grip.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)
