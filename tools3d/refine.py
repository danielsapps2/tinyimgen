"""Pass 2: geometry-aligned texture views painted by klein, re-projected onto the mesh.

    .venv3d/bin/python tools3d/refine.py OUTDIR --describe "short outfit description" [--server http://127.0.0.1:7860]

For each orbit angle: render the pass-1 textured mesh (pixel-aligned with the bake projection), ask the
running TinyImGen server to repaint it with klein using [render, original front] as references, cut it
out, check it still lines up with the mesh, and bake everything again with the original front on top.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from texture import dev  # noqa: F401  (first: picks the device and sets the MPS fallback switch before torch loads)
import numpy as np
import torch
from PIL import Image

from texture import Model, align_to_render, camera, empty_cache, load_rgba, repaint_check

ANGLES = {"front_left": 45, "left": 90, "back_left": 135, "back": 180, "back_right": 225, "right": 270,
          "front_right": 315}
MIRROR = {"front_left": "front_right", "front_right": "front_left", "left": "right", "right": "left",
          "back_left": "back_right", "back_right": "back_left"}
VIEW_WORDS = {"front_left": "three-quarter front view from the character's left", "left": "left side profile view",
              "back_left": "three-quarter back view from the character's left", "back": "back view",
              "back_right": "three-quarter back view from the character's right", "right": "right side profile view",
              "front_right": "three-quarter front view from the character's right"}


def klein(server, prompt, refs, w, h, seed, retries=1):
    body = json.dumps({"model": "klein", "prompt": prompt, "refs": refs, "width": w, "height": h,
                       "seed": seed, "hidden": True}).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(server + "/api/generate", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                detail = json.load(e).get("detail", "")
            except ValueError:
                detail = ""
            if attempt == retries:
                raise RuntimeError(f"klein request failed ({e.code}): {detail or e.reason}") from None
            time.sleep(5)  # usually another app briefly holding VRAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--describe", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:7860")
    ap.add_argument("--faces", type=int, default=30000)
    ap.add_argument("--power", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--front-ref", help="gallery file name of the original klein front image "
                                        "(default: the cutout in OUTDIR/views/front.png)")
    a = ap.parse_args()
    t0 = time.time()
    rdir = os.path.join(a.out, "refined")
    os.makedirs(rdir, exist_ok=True)
    model = Model(a.out, faces=a.faces)
    p1 = json.load(open(os.path.join(a.out, "texture_pass1.json")))
    tex1 = np.asarray(Image.open(os.path.join(a.out, "albedo_pass1.png")).convert("RGB"))
    front = load_rgba(os.path.join(a.out, "views", "front.png"))
    H, W = front.shape[:2]
    ffit = p1["fits"]["front"]
    fs, fty = ffit[0], ffit[2]
    front_ref = a.front_ref or os.path.relpath(os.path.join(a.out, "views", "front.png"), "out")
    from rembg import new_session, remove
    cut = new_session("isnet-general-use")

    stats, repaints = {"views": {}}, {}
    cam, right = camera(0)
    front_layer, own, _ = model.front_layer(front, cam, right, ffit, a.power, 1.5)
    layers = {"front": front_layer}
    for name, az in ANGLES.items():
        cam, right = camera(az)
        # same scale as the front fit; centre the orbiting silhouette horizontally
        u = model.V @ right
        fit = (fs, W / 2 - fs * (u.min() + u.max()) / 2, fty)
        render, sil = model.render(tex1, cam, right, fit, H, W)
        rpath = os.path.join(rdir, f"{name}_render.png")
        Image.fromarray(render).save(rpath)
        prompt = (f"Image 1 is a rough 3D render of a game character seen in {VIEW_WORDS[name]}. Repaint image 1 "
                  f"as a clean, sharp, high-quality game character render with soft even lighting. Keep exactly the "
                  f"same pose, silhouette, size and position as image 1. The character's design, colours and "
                  f"materials must match image 2: {a.describe}. Plain light grey background.")
        t = time.time()
        empty_cache()  # hand our rasterizer scratch back before the server needs the GPU
        meta = klein(a.server, prompt, [os.path.relpath(rpath, "out"), front_ref], W, H, a.seed)
        gen = Image.open(os.path.join("out", "gallery", meta["file"])).convert("RGB")
        rgba = np.asarray(remove(gen, session=cut)).astype(np.float32) / 255
        Image.fromarray((rgba * 255).astype(np.uint8)).save(os.path.join(rdir, f"{name}.png"))
        ok, why = repaint_check(render, rgba, az)   # a repaint turned toward the camera paints a second face
        rgba, drift = align_to_render(rgba, render)
        vstats = {}
        layers.update(model.repaint_layers(name, rgba, ok, sil, cam, right, fit, a.power, vstats))
        repaints[name] = (rgba, render, fit, vstats)   # aligned: the mirror inherits the drift correction
        stats["views"][name] = {"klein_s": meta["seconds"], "drift_px": round(drift, 1), "check": why, **vstats}
        print(name, stats["views"][name], f"({time.time() - t:.1f}s)", flush=True)
    for name, (_, render, fit, _) in repaints.items():   # no usable head: borrow the mirrored opposite side's
        other = MIRROR.get(name)
        if other is None or name + "_head" in layers or other + "_head" not in layers:
            continue
        rgba_o, _, fit_o, vstats_o = repaints[other]
        layer, why = model.mirrored_head(rgba_o, camera(ANGLES[other])[1], fit_o, render, *camera(ANGLES[name]), fit,
                                         a.power, ANGLES[name], vstats_o[f"{other}_head_trust"])
        stats["views"][name]["head"] = why
        if layer is not None:
            layers[name + "_mirror"] = layer

    tex8 = model.bake(layers, stats, own=own)
    Image.fromarray(tex8).save(os.path.join(a.out, "albedo.png"))
    model.export(tex8, os.path.join(a.out, "textured.glb"))
    stats["seconds_total"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "texture_pass2.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "views"}))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
