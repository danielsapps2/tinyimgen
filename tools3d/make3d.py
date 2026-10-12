"""One image -> textured, game-ready GLB. Launched by server.py (3D tab) inside .venv3d:

    .venv3d/bin/python tools3d/make3d.py out/3d/<job_id>

Reads <job>/job.json (config written by the server), writes progress back into it after every stage.
klein work goes through the running image server; Hunyuan3D and the texture baker run here. Both sides
take the cross-process GPU lock, so they never share the 6 GB card at the same time.
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# numpy asks the kernel for transparent huge pages on every large array. With the image server's ~21 GB of pinned
# host memory fragmenting RAM, each request can stall seconds in failed compaction: the Tripo export's texture
# transfer took 8 min instead of ~1 s. Must be set before numpy is imported.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from tinyimgen.gpulock import gpu_lock  # noqa: E402
import parts  # noqa: E402
from refine import ANGLES, MIRROR, VIEW_WORDS, klein  # noqa: E402
from shape import cutout, generate_shape  # noqa: E402
from texture import (PASS1, Model, align_to_render, camera, detect_face, export_scene,  # noqa: E402
                     render_scene, repaint_check)

SERVER = os.environ.get("TINYIMGEN_SERVER", "http://127.0.0.1:7860")
QUALITY = {"draft": [], "standard": ["front_left", "left", "right", "front_right"], "high": list(ANGLES)}
# Seen from the side, T-pose arms point at the camera and klein redraws them hanging down. Use angles that keep the
# arms visibly outstretched instead.
QUALITY_TPOSE = {"draft": [], "standard": ["front_left", "back_left", "back_right", "front_right"],
                 "high": ["front_left", "back_left", "back", "back_right", "front_right"]}
POWER = 6.0
ENHANCE = 3          # the views carrying close-ups are baked at 3x the source resolution


def ref_path(ref):
    return os.path.join(OUT, "gallery", ref) if "/" not in ref else os.path.join(OUT, ref)


TPOSE = ("Show this exact same character standing in a T-pose: both arms stretched straight out to the sides at "
         "shoulder height, palms facing down, both hands open and flat with all five fingers straight and slightly "
         "spread so there is a small gap between each finger, thumbs relaxed beside the hands, nothing held in the "
         "hands, legs straight and slightly apart. Front view, "
         "facing the camera, whole body visible head to toe with both hands fully in frame, centred. Keep exactly the "
         "same face, hair, outfit, materials and colours. Plain light grey studio background, soft even lighting.")
# the body is built bald and the hair added as its own mesh (parts.hair_layer): the scalp, ears and face come out
# clean, and the hair stays one piece. A separate, minimal edit keeps the person and the framing (asking for a bald
# character in the T-pose prompt turned women into men and added the accessories it named).
BALD = ("Remove all of the hair and everything worn on this character's head (hats, caps, helmets, goggles, glasses, "
        "headbands, hair accessories), leaving a smooth bald scalp with the ears visible. Keep everything from the "
        "chin down exactly as it is. Change nothing else: keep exactly the same person, face, gender, age, "
        "expression, pose, clothing, colours, framing and background.")
HAIR_SIDE = ("Image 1 is a rough 3D render of a game character's head and shoulders seen in {view}. Repaint image 1 "
             "as a clean, sharp, high-quality game character render with soft even lighting. Keep exactly the same "
             "head angle, silhouette, size and position as image 1: do not turn the head toward the camera. The hair, "
             "headwear, face, skin and colours must match image 2; the hair is one solid, smooth, sculpted mass with "
             "no loose strands. Plain light grey background.")
BALD_NOTE = (" Show the head completely bald: a bare, smooth scalp with nothing on it, whatever the description says "
             "about the hair or anything worn on the head.")
BACK = ("Show this exact same character from directly behind (back view). Keep exactly the same pose as image 1 seen "
        "from behind: every arm and leg in the same position{arms}. Same outfit, same colours, full body head to toe, "
        "centred, plain light grey studio background, soft even lighting. The character: {describe}.")


BACK_OBJECT = ("Show this exact same object from directly behind (back view), seen from the same height and distance as "
               "image 1. Same shape, proportions, materials and colours, the whole object in frame, centred, plain light "
               "grey studio background, soft even lighting. The object: {describe}.")
TURNED = (" The character is turned exactly as in image 1: do not turn the head or body toward the camera; where "
          "image 1 shows the back or side of the head, show the back or side of the head.")


BALD_MAX_CHANGE = 9.0    # mean |difference| below the chin; faithful bald edits measure ~3, an added scarf ~20


def below_chin_change(a, b, face):
    """How much the bald edit changed the neck and shoulders (it should only touch the head): mean absolute
    difference of the blurred images in a window below the chin."""
    if face is None:
        return 0.0
    x, y, w, h = face[:4]
    H, W = a.shape[:2]
    y0, y1 = int(min(H, y + 1.15 * h)), int(min(H, y + 2.6 * h))
    x0, x1 = int(max(0, x - 0.8 * w)), int(min(W, x + 1.8 * w))
    if y1 <= y0 or x1 <= x0:
        return 0.0
    import cv2
    fa, fb = (cv2.GaussianBlur(z[y0:y1, x0:x1].astype(np.float32), (0, 0), 2.0) for z in (a, b))
    return float(np.abs(fa - fb).mean())


def mask_of(img):
    return np.asarray(img.convert("RGBA"))[..., 3] > 127


def norm_mask(m, H=512, W=800):
    """Crop a silhouette to its bounding box and scale it to a fixed height, so masks compare by shape only."""
    ys, xs = np.nonzero(m)
    c = Image.fromarray(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255)
    c = np.asarray(c.resize((min(W, max(1, round(c.width * H / c.height))), H))) > 127
    out = np.zeros((H, W), bool)
    x0 = (W - c.shape[1]) // 2
    out[:, x0:x0 + c.shape[1]] = c
    return out


def back_match(front_cut, back_cut):
    """A back view's outline is close to the front's mirrored: 0.91-0.93 for good klein back views, ~0.67 when it
    changed the pose (e.g. dropped T-pose arms)."""
    a, b = norm_mask(mask_of(front_cut))[:, ::-1], norm_mask(mask_of(back_cut))
    return float((a & b).sum() / max(1, (a | b).sum()))


class Job:
    def __init__(self, d):
        self.dir, self.path = d, os.path.join(d, "job.json")
        with open(self.path) as f:
            self.data = json.load(f)
        self.t0 = time.time()

    def save(self):
        self.data["elapsed"] = round(time.time() - self.t0, 1)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def stage(self, key, state, detail=None):
        st = next(s for s in self.data["stages"] if s["key"] == key)
        if state == "running" and "_t" not in st:  # multi-step stages report detail updates; keep the start
            st["_t"] = time.time()
        if state in ("done", "skipped") and "_t" in st:
            st["seconds"] = round(time.time() - st.pop("_t"), 1)
        st["state"] = state
        if detail is not None:
            st["detail"] = detail
        self.data["stage"] = key
        self.save()


def main(job_dir):
    job = Job(job_dir)
    cfg = job.data["config"]
    job.data["status"] = "running"
    job.save()
    os.makedirs(os.path.join(job_dir, "views"), exist_ok=True)
    human = cfg.get("kind", "human") == "human"     # objects skip T-pose, bald base, head / hands / hair and faces
    noun = "character" if human else "object"
    describe = cfg.get("describe") or f"the same {noun}, design and colours as image 2"
    source = cfg["source"]
    tpose = human and cfg.get("pose") == "tpose"
    has_pose_stage = any(s["key"] == "pose" for s in job.data["stages"])  # jobs created before the pose option

    # 0. optional T-pose: arms away from the body, so no surface hides another and nothing bleeds between them
    if tpose:
        job.stage("pose", "running", "klein: T-pose")
        best = None
        for seed in (3, 17):
            meta = klein(SERVER, f"{TPOSE} The character: {describe}.", [source], 1216, 1216, seed)
            img = Image.open(ref_path(meta["file"])).convert("RGB")
            m = mask_of(cutout(img))
            ys, xs = np.nonzero(m)
            spread = (xs.max() - xs.min()) / max(1, ys.max() - ys.min())   # ~0.35 arms down, ~0.9+ arms out
            ok = spread > 0.7 and detect_face(np.asarray(img)) is not None
            if best is None or (ok, spread) > (best[0], best[1]):
                best = (ok, spread, meta)
            if ok:
                break
        source = best[2]["file"]
        job.data["config"]["posed_source"] = source
        job.stage("pose", "done", f"{best[2]['seconds']}s" + ("" if best[0] else " · arms may not be fully out"))
    elif has_pose_stage:
        job.stage("pose", "skipped", "keeping the source pose")

    if cfg.get("engine", "local") == "tripo":   # jobs from before the engine option ran the local pipeline
        return run_tripo(job, cfg, source, tpose)

    src = Image.open(ref_path(source)).convert("RGB")
    W = min(1216, (src.width // 64) * 64) or 832
    H = min(1216, (src.height // 64) * 64) or 1216

    # 0b. bald base: the body is built without hair, which is added back as its own mesh after the shape
    haired = None
    if human and any(s["key"] == "bald" for s in job.data["stages"]) and cfg.get("bald", True):
        job.stage("bald", "running", "klein: removing the hair")
        why = "no face found"
        for seed in (3, 17):
            meta = klein(SERVER, BALD, [source], W, H, seed)
            img = np.asarray(Image.open(ref_path(meta["file"])).convert("RGB"))
            ref = np.asarray(src.resize((img.shape[1], img.shape[0])))
            M, info = parts.register(ref, img)
            face = detect_face(img)
            below = below_chin_change(ref, img, face)
            if (M is not None and face is not None and info.get("register_shift_px", 99) < 6
                    and below < BALD_MAX_CHANGE):
                haired, source = source, meta["file"]
                job.data["config"]["haired_source"], job.data["config"]["bald_source"] = haired, source
                break
            why = (info.get("register", "no face found") if M is None else "no face found" if face is None
                   else f"the edit changed the clothes ({below:.0f})" if below >= BALD_MAX_CHANGE
                   else "the edit moved the character")
        job.stage("bald", "done" if haired else "skipped",
                  f"{meta['seconds']}s" if haired else f"kept the hair ({why})")
        if haired:
            describe += BALD_NOTE
    refs = {"front": source}
    refs.update({k: v for k, v in (cfg.get("views") or {}).items() if v})
    if tpose or haired:  # side/back views supplied for the original pose (or with hair) don't match the base image
        refs = {"front": source}

    # 1. reference views: only the back is generated; sides come later from mesh-aligned renders.
    # klein sometimes changes the pose for the back view; check its outline against the mirrored front and retry.
    cuts = {"front": cutout(Image.open(ref_path(source)))}
    if "back" not in refs and cfg.get("auto_back", True):
        arms = " (arms stretched straight out to the sides, exactly as in image 1)" if tpose else ""
        best = None
        for attempt, seed in enumerate((21, 5, 33)):
            job.stage("views", "running", "klein: back view" + (f" (retry {attempt})" if attempt else ""))
            meta = klein(SERVER, (BACK if human else BACK_OBJECT).format(arms=arms, describe=describe), [source], W, H,
                         seed)
            cut = cutout(Image.open(ref_path(meta["file"])))
            score = back_match(cuts["front"], cut)
            if best is None or score > best[0]:
                best = (score, meta, cut)
            if score >= 0.82:
                break
        score, meta, cuts["back"] = best
        refs["back"] = meta["file"]
        job.data["stats"] = {"back_match": round(score, 3)}
        job.stage("views", "done", f"back view {meta['seconds']}s" + (f" · {attempt + 1} tries" if attempt else ""))
    else:
        job.stage("views", "skipped", "using your reference views" if len(refs) > 1 else "front only")

    # 2. cut-outs (CPU)
    job.stage("cutout", "running")
    views = {}
    for k, r in refs.items():
        im = cuts[k] if k in cuts else cutout(Image.open(ref_path(r)))
        im.save(os.path.join(job_dir, "views", f"{k}.png"))
        views[k] = im
    if haired:   # the same pose with its hair, for the hair layer (not a projection view of the body)
        cutout(Image.open(ref_path(haired))).save(os.path.join(job_dir, "views", "haired.png"))
    job.stage("cutout", "done", ", ".join(views))

    # 3. shape
    job.stage("shape", "running", "Hunyuan3D-2mv" if len(views) > 1 else "Hunyuan3D-2mini")
    with gpu_lock():
        mesh, sstats = generate_shape(views)
    mesh.export(os.path.join(job_dir, "shape.glb"))
    job.data.setdefault("stats", {})["shape"] = sstats
    job.stage("shape", "done", f"{sstats['model'].split('/')[1]} · {sstats['peak_vram_gb']} GB")

    stats = job.data["stats"]
    front_rgba = np.asarray(views["front"].convert("RGBA")).astype(np.float32) / 255

    # 3b. head and hands rebuilt from close-ups (parts.py): they are the worst-resolved parts of a full-body image
    mesh_name, replaced, pieces, hair = "shape.glb", None, {}, None
    if human and any(s["key"] == "parts" for s in job.data["stages"]) and cfg.get("parts", True):
        job.stage("parts", "running")

        def kfn(prompt, paths, seed):
            meta = klein(SERVER, prompt, [os.path.relpath(p, OUT) for p in paths], parts.CLOSE, parts.CLOSE, seed)
            return os.path.join(OUT, "gallery", meta["file"])
        pviews = {k: np.asarray(v.convert("RGBA")).astype(np.float32) / 255 for k, v in views.items()
                  if k in ("front", "back")}
        if haired:
            pviews["haired"] = np.asarray(Image.open(os.path.join(job_dir, "views", "haired.png")).convert("RGBA")
                                          ).astype(np.float32) / 255
        pstats = stats.setdefault("parts", {})
        steps, t_last = pstats.setdefault("seconds", {}), [time.time(), "start"]

        def progress(msg):   # also times each step, for tuning
            steps[t_last[1]] = round(time.time() - t_last[0], 1)
            t_last[:] = [time.time(), msg]
            job.stage("parts", "running", msg)
        mesh_name, replaced, pieces, hair = parts.run(job_dir, pviews, kfn, gpu_lock, progress, pstats)
        progress("done")
        done = ([k for k in ("head",) if replaced and replaced.get(k)] +
                (["hands"] if replaced and replaced.get("hands") else []) + (["hair"] if hair else []))
        job.stage("parts", "done", " + ".join(done) if done else "kept the generator's head and hands")

    # 4. texture pass 1: project the reference views
    job.stage("texture", "running")
    with gpu_lock():
        # the face and hands are re-sculpted from the source image before decimation (sculpt.py), unless the parts
        # pass rebuilt them
        t_tex = time.time()
        model = Model(job_dir, mesh_name, int(cfg.get("faces", 30000)), 2048, front=front_rgba if human else None,
                      replaced=replaced)
        stats.setdefault("texture_seconds", {})["model"] = round(time.time() - t_tex, 1)
        stats["sculpt"] = {k: v for k, v in model.sculpt_info.items() if k != "hands"}
        layers, fits, own = {}, {}, None
        for name, rgba_img in views.items():
            rgba = np.asarray(rgba_img.convert("RGBA")).astype(np.float32) / 255
            cam, right = camera(PASS1[name])
            fit, iou = model.fit(cam, right, rgba[..., 3] > 0.5)
            fits[name] = fit
            stats[f"{name}_iou"] = round(iou, 3)
            if pieces.get(name):   # bake from the view upscaled with the close-ups pasted in (same fit, x ENHANCE)
                rgba = parts.enhance(rgba, pieces[name], ENHANCE)
                fit = (fit[0] * ENHANCE, fit[1] * ENHANCE, fit[2] * ENHANCE, fit[3])
            trust = 1.5 if name == "front" else min(1.0, iou / 0.85) ** 2
            if name == "front" and human:  # the source alone paints the face and hands (own): others misplace them
                layer, own, info = model.front_layer(rgba, cam, right, fit, POWER, trust)
                stats.update(info)
                own = own.cpu()
            else:
                layer = model.project(rgba, cam, right, fit, POWER, trust)
            layers[name] = tuple(t.cpu() for t in layer)  # park in RAM: klein needs the VRAM between our stages
        stats["texture_seconds"]["project"] = round(time.time() - t_tex - stats["texture_seconds"]["model"], 1)
        tex8 = model.bake({k: v for k, v in layers.items()}, stats, own=own)
        torch.cuda.empty_cache()
    Image.fromarray(tex8).save(os.path.join(job_dir, "albedo_pass1.png"))
    job.stage("texture", "done", f"{len(model.F):,} tris · {stats['texels_seen_pct']}% seen")

    # 5. refine: klein repaints mesh-aligned renders, which are re-projected
    angles = (QUALITY_TPOSE if tpose else QUALITY)[cfg.get("quality", "standard")]
    if angles:
        front = np.asarray(views["front"].convert("RGBA")).astype(np.float32) / 255
        fH, fW = front.shape[:2]
        fs, fty = fits["front"][0], fits["front"][2]
        final_layers = dict(layers)  # mesh-aligned repaints replace same-named pass-1 views
        repaints = {}                # name -> (cut-out, passed the face check, render, fit)
        for i, name in enumerate(angles):
            job.stage("refine", "running", f"{VIEW_WORDS[name]} ({i + 1}/{len(angles)})")
            cam, right = camera(ANGLES[name])
            u = model.V @ right
            fit = (fs, fW / 2 - fs * (u.min() + u.max()) / 2, fty)
            with gpu_lock():
                render, sil = model.render(tex8, cam, right, fit, fH, fW)
                torch.cuda.empty_cache()
            rpath = os.path.join(job_dir, "views", f"{name}_render.png")
            Image.fromarray(render).save(rpath)
            asset = "game character" if human else "game asset"
            prompt = (f"Image 1 is a rough 3D render of a {asset} seen in {VIEW_WORDS[name]}. Repaint image 1 "
                      f"as a clean, sharp, high-quality {asset} render with soft even lighting. Keep exactly "
                      f"the same {'pose, ' if human else ''}silhouette, size and position as image 1. The {noun}'s design, colours and "
                      f"materials must match image 2: {describe}. Plain light grey background.")
            # klein sometimes turns the character to face the camera (a second face on the cheek or the back of the
            # head): check the repaint's face against the render's, retry once, and drop it if still wrong
            for seed in (7, 19):
                meta = klein(SERVER, prompt if seed == 7 or not human else prompt + TURNED, [os.path.relpath(rpath, OUT), source],
                             fW, fH, seed)
                rgba_img = cutout(Image.open(os.path.join(OUT, "gallery", meta["file"])))
                rgba = np.asarray(rgba_img).astype(np.float32) / 255
                ok, why = repaint_check(render, rgba, ANGLES[name]) if human else (True, "object")
                if ok:
                    break
            stats[f"{name}_check"] = why + ("" if seed == 7 else " (retried)")
            rgba_img.save(os.path.join(job_dir, "views", f"{name}.png"))
            rgba, drift = align_to_render(rgba, render)  # undo klein's few-pixel drift from the render
            stats[f"{name}_drift_px"] = round(drift, 1)
            with gpu_lock():
                for k, layer in model.repaint_layers(name, rgba, ok, sil, cam, right, fit, POWER, stats).items():
                    final_layers[k] = tuple(t.cpu() for t in layer)
                torch.cuda.empty_cache()
            repaints[name] = (rgba, render, fit)   # aligned: the mirror fallback inherits the drift correction
        # a side without a usable head (face turned, or misaligned) borrows it from the mirrored opposite side
        with gpu_lock():
            for name, (_, render, fit) in repaints.items():
                other = MIRROR.get(name)
                if not human or other is None or name + "_head" in final_layers or other + "_head" not in final_layers:
                    continue
                rgba_o, _, fit_o = repaints[other]
                layer, why = model.mirrored_head(rgba_o, camera(ANGLES[other])[1], fit_o, render, *camera(ANGLES[name]),
                                                 fit, POWER, ANGLES[name], stats[f"{other}_head_trust"])
                stats[f"{name}_head"] = why
                if layer is not None:
                    final_layers[name + "_mirror"] = tuple(t.cpu() for t in layer)
            tex8 = model.bake(final_layers, stats, own=own)
            torch.cuda.empty_cache()
        job.stage("refine", "done", f"{len(angles)} views repainted")
    else:
        job.stage("refine", "skipped", "draft quality")

    # 5b. hair layer texture: the haired front (with its close-up) and the hair added to the back view, hidden by the
    # body where it is behind it
    items = [("body", model, tex8)]
    if hair:
        job.stage("export", "running", "hair texture")
        hV, hF, hpieces = hair
        with gpu_lock():
            hm = Model(job_dir, "hair.glb", len(hF) + 10, 1024, cache_name="uvmesh_hair.npz", raw=True,
                       occluder=(model.V, model.F))
            hl = {}
            hfront = parts.enhance(pviews["haired"], hpieces["front"], ENHANCE)
            f0 = fits["front"]
            hl["front"] = hm.project(hfront, *camera(0), (f0[0] * ENHANCE, f0[1] * ENHANCE, f0[2] * ENHANCE, f0[3]),
                                     POWER, 1.5)
            if "back" in hpieces and "back" in fits:
                hback = parts.enhance(pviews["back"], hpieces["back"], ENHANCE)
                fb = fits["back"]
                hl["back"] = hm.project(hback, *camera(180), (fb[0] * ENHANCE, fb[1] * ENHANCE, fb[2] * ENHANCE, fb[3]),
                                        POWER, 1.0)
            htex = hm.bake(hl, stats.setdefault("hair_bake", {}))
            torch.cuda.empty_cache()
        # the sides of the hair: nothing above sees them. klein repaints head-framed renders of body + hair seen
        # from each side (with the haired close-up for the look), registered back onto the render and projected
        head = model.sculpt_info.get("head")
        hair_close = os.path.join(job_dir, "parts", "hair_close.png")
        if head and os.path.exists(hair_close):
            c, r = np.asarray(head["c"], np.float32), float(head["r"])
            sides = {}
            for name, az in (("hair_left", 90), ("hair_right", 270)):
                job.stage("export", "running", f"klein: hair, {VIEW_WORDS[name.replace('hair_', '')]}")
                cam, right = camera(az)
                S = parts.CLOSE / (2.6 * r)
                fit = (S, parts.CLOSE / 2 - S * float(c @ right), parts.CLOSE / 2 + S * float(c[1]))
                with gpu_lock():
                    render = render_scene([(model, tex8), (hm, htex)], cam, right, fit, parts.CLOSE, parts.CLOSE)
                    torch.cuda.empty_cache()
                rpath = os.path.join(job_dir, "views", f"{name}_render.png")
                Image.fromarray(render).save(rpath)
                meta = klein(SERVER, HAIR_SIDE.format(view=VIEW_WORDS[name.replace("hair_", "")]),
                             [os.path.relpath(rpath, OUT), os.path.relpath(hair_close, OUT)], parts.CLOSE, parts.CLOSE, 7)
                rgba_img = cutout(Image.open(os.path.join(OUT, "gallery", meta["file"])))
                rgba_img.save(os.path.join(job_dir, "views", f"{name}.png"))
                rgba = np.asarray(rgba_img).astype(np.float32) / 255
                ok, why = repaint_check(render, rgba, az)
                stats[f"{name}_check"] = why
                if ok:
                    rgba, drift = align_to_render(rgba, render)
                    stats[f"{name}_drift_px"] = round(drift, 1)
                    sides[name] = (rgba, render, fit, az)
            # a side whose repaint turned the face to the camera borrows the other side's, mirrored (hair is close to
            # symmetric; the same fallback as the body's heads)
            for name, other in (("hair_left", "hair_right"), ("hair_right", "hair_left")):
                if name in sides or other not in sides:
                    continue
                o_rgba, _, o_fit, o_az = sides[other]
                az = (360 - o_az) % 360
                cam, right = camera(az)
                fit = (S, parts.CLOSE / 2 - S * float(c @ right), parts.CLOSE / 2 + S * float(c[1]))
                with gpu_lock():
                    render = render_scene([(model, tex8), (hm, htex)], cam, right, fit, parts.CLOSE, parts.CLOSE)
                M = np.float32([[1, 0, fit[1] - (parts.CLOSE - o_fit[1])], [0, 1, 0]])
                mirrored = cv2.warpAffine(np.ascontiguousarray(o_rgba[:, ::-1]), M, (parts.CLOSE, parts.CLOSE),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mirrored, _ = align_to_render(mirrored, render)
                sides[name] = (mirrored, render, fit, az)
                stats[f"{name}_check"] += " -> mirrored the other side"
            for name, (rgba, _, fit, az) in sides.items():
                cam, right = camera(az)
                with gpu_lock():
                    hl[name] = hm.project(rgba, cam, right, fit, POWER, 1.0)
            if sides:
                with gpu_lock():
                    htex = hm.bake(hl, stats["hair_bake"])
                    torch.cuda.empty_cache()
        Image.fromarray(htex).save(os.path.join(job_dir, "hair_albedo.png"))
        items.append(("hair", hm, htex))
        # hair cap: the body's scalp takes the hair colour where the images show hair (seen through any gap)
        cap = []
        if hpieces.get("cap_front") is not None:
            m = cv2.erode(hpieces["cap_front"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
            hr = pviews["haired"].copy()
            hr[..., 3] = m
            cap.append((hr, 0, fits["front"]))
        if hpieces.get("cap_back") is not None and "back" in fits:
            bb, hbc, bmask = hpieces["cap_back"]
            H_, W_ = pviews["back"].shape[:2]
            rgb = parts.uncrop(hbc.astype(np.float32) / 255, bb, H_, W_)
            a = parts.uncrop(cv2.erode(bmask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)[..., None],
                             bb, H_, W_)
            cap.append((np.dstack([rgb, a]), 180, fits["back"]))
        if cap:
            with gpu_lock():
                lay = [model.project(img, *camera(az), f, POWER, 1.0) for img, az, f in cap]
                tex8, n = model.paint_cap(tex8, lay)
                items[0] = ("body", model, tex8)
                stats["hair_cap_texels"] = n
                torch.cuda.empty_cache()

    # 6. export + thumbnail
    job.stage("export", "running")
    Image.fromarray(tex8).save(os.path.join(job_dir, "albedo.png"))
    if len(items) > 1:
        export_scene(items, os.path.join(job_dir, "textured.glb"))
        export_scene(items, os.path.join(job_dir, "game.glb"), height=float(cfg.get("height", 1.8)))
    else:
        model.export(tex8, os.path.join(job_dir, "textured.glb"))
        model.export(tex8, os.path.join(job_dir, "game.glb"), height=float(cfg.get("height", 1.8)))
    with gpu_lock():
        cam, right = camera(30)
        thumb = render_scene([(m, tx) for _, m, tx in items], cam, right, model.orbit_fit(right, 768, 512), 768, 512,
                             bg=(0.86, 0.86, 0.85))
        torch.cuda.empty_cache()
    Image.fromarray(thumb).save(os.path.join(job_dir, "thumb.png"))
    job.data["outputs"] = {"game": "game.glb", "textured": "textured.glb", "shape": "shape.glb",
                           "albedo": "albedo.png", "thumb": "thumb.png"}
    stats["triangles"] = len(model.F)
    job.stage("export", "done", f"{cfg.get('height', 1.8)} m · origin at feet")
    run_rig(job)
    job.data["status"] = "done"
    job.save()


def run_tripo(job, cfg, source, tpose):
    """Shape and texture from Tripo H3.1 on fal (tripo.py), then the local game-ready export: front to +Z, scaled to
    the requested height with the origin at the feet, a full-detail GLB and one decimated to the triangle budget."""
    import tripo
    stats = job.data.setdefault("stats", {})
    images = {"front": ref_path(source)}
    views = {k: v for k, v in (cfg.get("views") or {}).items() if v}
    if not tpose and all(views.get(k) for k in ("left", "back", "right")):   # side views made for the source pose
        images.update({k: ref_path(views[k]) for k in ("left", "back", "right")})
    job.stage("views", "skipped", "using your reference views" if len(images) > 1 else
              "front only" if not views else "Tripo needs all three extra views; using the front only")

    job.stage("shape", "running", "Tripo H3.1: uploading")
    raw = os.path.join(job.dir, "tripo_raw.glb")
    info = tripo.generate(images, cfg.get("quality", "standard"), raw,
                          progress=lambda msg: job.stage("shape", "running", msg))
    stats["tripo"] = info
    job.stage("shape", "done", f"Tripo H3.1 · {info['geometry_quality']} geometry · {info['seconds']}s")

    job.stage("export", "running", "orienting and scaling")
    height = float(cfg.get("height", 1.8))
    V, F, UV, tex = tripo.load(raw)
    with gpu_lock():
        az = tripo.front_azimuth(V, F, UV, tex)
        torch.cuda.empty_cache()
    stats["front_azimuth"] = az if az is not None else "no face found (Tripo default, 90)"
    V = tripo.normalise(V, 90 if az is None else az, height)
    tripo.export(V, F, UV, tex, os.path.join(job.dir, "textured.glb"), normals=tripo.smooth_normals(V, F))
    stats["full_triangles"] = len(F)

    job.stage("export", "running", f"game mesh · {int(cfg.get('faces', 30000)):,} triangles")
    gV, gF, gUV, gN, gtex = tripo.game_mesh(V, F, UV, tex, int(cfg.get("faces", 30000)), gpu=gpu_lock)
    tripo.export(gV, gF, gUV, gtex, os.path.join(job.dir, "game.glb"), normals=gN)
    Image.fromarray(gtex).save(os.path.join(job.dir, "albedo.png"))
    stats["triangles"] = len(gF)

    with gpu_lock():   # thumbnail: the full-detail model from 30 degrees, framed like the local pipeline's
        lo, hi = V.min(0), V.max(0)
        a = np.radians(30)
        u = V @ np.array([np.cos(a), 0, -np.sin(a)])                  # image-right axis at azimuth 30
        c = np.array([(u.min() + u.max()) / 2 * np.cos(a), (lo[1] + hi[1]) / 2, -(u.min() + u.max()) / 2 * np.sin(a)])
        half = max(0.5 * (hi[1] - lo[1]) / 0.88, 0.5 * np.ptp(u) / 0.94 * 768 / 512)   # fit height and (T-pose) width
        thumb = tripo.render(V, F, UV, tex, c, half, 30, size=768, W=512, bg=0.86)
        torch.cuda.empty_cache()
    Image.fromarray(thumb).save(os.path.join(job.dir, "thumb.png"))
    # "shape" is what the viewer's second mode and download show: for Tripo, the full-detail textured model (the raw
    # download, tripo_raw.glb, faces +X at Tripo's own scale)
    job.data["outputs"] = {"game": "game.glb", "textured": "textured.glb", "shape": "textured.glb", "full": True,
                           "albedo": "albedo.png", "thumb": "thumb.png"}
    job.stage("export", "done", f"{height} m · origin at feet · {len(gF):,} tris + full detail")
    run_rig(job)
    job.data["status"] = "done"
    job.save()


def run_rig(job):
    """Optional last stage: skeleton, skin and the clip library (rig.py) on game.glb -> rigged.glb. A character that
    can't be rigged (not a humanoid, a prop) skips the stage; the model itself is still done."""
    if not any(s["key"] == "rig" for s in job.data["stages"]):
        return
    import rig
    job.stage("rig", "running", "fitting skeleton")
    try:
        info = rig.rig(job.dir, progress=lambda msg: job.stage("rig", "running", msg))
    except rig.RigError as e:
        job.data["rig"] = {"state": "skipped", "detail": str(e)}
        job.stage("rig", "skipped", str(e))
        return
    job.data["outputs"].update(rigged="rigged.glb", rig="rig.json", **({"fbx": info["fbx"]} if info["fbx"] else {}))
    job.data["rig"] = {"state": "done", "clips": [c["name"] for c in info["clips"]], "warnings": info["warnings"],
                       "stamp": int(time.time())}
    job.data.setdefault("stats", {})["rig_seconds"] = info["seconds"]["total"]
    job.stage("rig", "done", f"{len(info['clips'])} clips · {len(rig.NAMES)} joints")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    d = sys.argv[1]
    try:
        main(d)
    except Exception as e:
        traceback.print_exc()
        j = Job(d)
        j.data["status"] = "error"
        j.data["error"] = f"{type(e).__name__}: {e}"
        for s in j.data["stages"]:
            if s["state"] == "running":
                s["state"] = "error"
        j.save()
        sys.exit(1)
