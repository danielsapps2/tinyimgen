"""TinyImGen: Studio (live generation), 3D (image -> game-ready GLB) and Showcase, on the local GPU.

    .venv/bin/python server.py            # http://127.0.0.1:7860  (FLUX.2 klein 4B only)
    .venv/bin/python server.py --zimage   # also load Z-Image Turbo (better in-image text, ~16 s)
"""
import argparse
import base64
import binascii
import io
import json
import os
import subprocess
import threading
import time
import uuid
from typing import Literal

from tinyimgen import device  # picks CUDA / Apple MPS / CPU; must load before torch (MPS CPU-fallback switch)
import torch
import uvicorn
from PIL import Image, UnidentifiedImageError
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.realpath(os.path.join(ROOT, "out"))
GALLERY = os.path.join(OUT, "gallery")
SHOWCASE = os.path.join(OUT, "showcase")
UPLOADS = os.path.join(OUT, "uploads")
JOBS = os.path.join(OUT, "3d")
VENV3D = os.path.join(ROOT, ".venv3d", "bin", "python")
BLENDER = next((b for b in (os.environ.get("TINYIMGEN_BLENDER"), os.path.expanduser("~/Tools/blender-build/bin/blender"),
                            "/Applications/Blender.app/Contents/MacOS/Blender")
                if b and os.path.exists(b)), None)
MOCAP = os.path.join(OUT, "mocap")
for d in (GALLERY, SHOWCASE, UPLOADS, JOBS):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="TinyImGen")
state = {"engine": None, "error": None, "started": time.time(), "models": ("klein",), "job": None, "fal": None,
         "rigs": {}}
STAGES = [("pose", "T-pose (klein)"), ("bald", "Bald base (klein)"), ("views", "Reference views"), ("cutout", "Background removal"), ("shape", "3D shape"),
          ("parts", "Head & hands detail"), ("texture", "Texture projection"), ("refine", "klein texture repaint"), ("export", "Game-ready export")]
# Tripo H3.1 on fal returns a textured mesh: only the T-pose (klein) and the game-ready export run here
STAGES_TRIPO = [("pose", "T-pose (klein)"), ("views", "Reference views"), ("shape", "3D model (Tripo H3.1)"),
                ("export", "Game-ready export")]
RIG_STAGE = ("rig", "Rig + animations")
HUMAN_ONLY = {"pose", "bald", "parts"}     # T-pose, bald base, head & hands: characters only
FAL_KEY_FILE = os.path.expanduser("~/.config/tinyimgen/fal_key")


def has_fal_key():
    from tinyimgen.falgen import fal_key
    return fal_key() is not None


FAL_LEDGER = os.path.join(GALLERY, "fal_costs.jsonl")   # one line per paid fal image request (estimated cost)


def fal_spend():
    """Estimated fal image spend: today (local time) and all time, in USD."""
    today, total = 0.0, 0.0
    if os.path.exists(FAL_LEDGER):
        day = time.strftime("%Y-%m-%d")
        with open(FAL_LEDGER) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                total += r.get("cost_usd", 0)
                if time.strftime("%Y-%m-%d", time.localtime(r.get("time", 0))) == day:
                    today += r.get("cost_usd", 0)
    return {"today": round(today, 2), "total": round(total, 2)}


def _load():
    try:
        from tinyimgen.engine import Engine
        state["engine"] = Engine(state["models"])
    except Exception as e:  # surface load failures in the UI instead of dying silently
        state["error"] = repr(e)
        raise


def resolve_ref(name):
    """A reference is a gallery file name, or a path under out/ (uploads/…, 3d/<job>/…). Never outside out/."""
    path = os.path.realpath(os.path.join(GALLERY, name) if "/" not in name else os.path.join(OUT, name))
    if not path.startswith(OUT + os.sep) or not os.path.isfile(path):
        raise HTTPException(400, f"Unknown reference image {name}")
    return path


def ref_url(name):
    return "/media/gallery/" + name if "/" not in name else "/media/" + name


class GenRequest(BaseModel):
    model: str = "klein"
    prompt: str = Field(min_length=1, max_length=2000)
    width: int = Field(1024, ge=512, le=1536)
    height: int = Field(1024, ge=512, le=1536)
    steps: int | None = Field(None, ge=1, le=12)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    refs: list[str] = Field(default_factory=list, max_length=4)  # gallery names, or paths under out/
    hidden: bool = False  # tool calls (e.g. 3D texturing): save the image but keep it out of Studio history


@app.get("/api/info")
def info():
    from tinyimgen.falgen import FAL_MODELS
    from tinyimgen.engine import MODELS
    eng = state["engine"]
    return {
        "gpu": device.name(),
        "vram_total_gb": round(device.memory_gb(), 2),
        "ready": eng is not None,
        "error": state["error"],
        "load_seconds": round(eng.load_seconds, 1) if eng else None,
        "loading_for": round(time.time() - state["started"]),
        "models": MODELS,
        "enabled": list(eng.pipes) if eng else list(state["models"]),
        "has_3d": os.path.exists(VENV3D),
        "has_fal": has_fal_key(),
        "fal_models": {k: {f: v[f] for f in ("label", "detail", "price", "max_refs")} for k, v in FAL_MODELS.items()},
        "fal_spend": fal_spend(),
    }


@app.get("/api/progress")
def progress():
    if state["fal"]:   # a cloud request: no steps to report, only its queue state
        return {"busy": True, "step": 0, "steps": 1, "model": state["fal"][0], "remote": state["fal"][1]}
    eng = state["engine"]
    return eng.progress if eng else {"busy": False}


@app.post("/api/generate")
def generate(req: GenRequest):
    from tinyimgen import falgen
    eng = state["engine"]
    if req.model in falgen.FAL_MODELS:
        refs = [Image.open(resolve_ref(name)).convert("RGB") for name in req.refs]
        state["fal"] = (req.model, "queued")
        try:
            img, meta = falgen.generate(req.model, req.prompt, req.width, req.height, req.seed, refs,
                                        progress=lambda st: state.update(fal=(req.model, st)))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:   # fal rejected it (safety filter, bad input, outage): show fal's message
            raise HTTPException(502, f"fal: {getattr(e, 'message', None) or e}")
        finally:
            state["fal"] = None
        with open(FAL_LEDGER, "a") as f:
            f.write(json.dumps({"time": time.time(), "model": req.model, "cost_usd": meta["cost_usd"],
                                "request_id": meta["request_id"]}) + "\n")
    else:
        if eng is None:
            raise HTTPException(503, "Models are still loading")
        if req.model not in eng.pipes:
            raise HTTPException(400, f"Unknown model {req.model}")
        refs = [Image.open(resolve_ref(name)).convert("RGB") for name in req.refs]
        try:
            img, meta = eng.generate(req.model, req.prompt, req.width, req.height, req.steps, req.seed, refs)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if not device.is_oom(e):
                raise
            raise HTTPException(503, "GPU out of memory — another app is using VRAM; close it or retry")
    meta["refs"] = req.refs
    name = f"{int(time.time() * 1000)}_{req.model}.webp"
    img.save(os.path.join(GALLERY, name), quality=92)
    meta["file"] = name
    meta["created"] = time.time()
    if not req.hidden:
        with open(os.path.join(GALLERY, "history.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")
    return meta


def _history():
    path = os.path.join(GALLERY, "history.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()][::-1]


@app.get("/api/history")
def history():
    return _history()[:60]


class Upload(BaseModel):
    name: str = Field("upload", max_length=200)
    data: str = Field(max_length=30_000_000)  # data: URL or bare base64


@app.post("/api/upload")
def upload(req: Upload):
    raw = req.data.split(",", 1)[1] if req.data.startswith("data:") else req.data
    try:
        img = Image.open(io.BytesIO(base64.b64decode(raw, validate=True)))
        img.load()
    except (binascii.Error, UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not an image Pillow can read")
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
    img.thumbnail((2048, 2048), Image.LANCZOS)  # re-encoded: nothing but pixels is kept
    name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.png"
    img.save(os.path.join(UPLOADS, name))
    ref = "uploads/" + name
    return {"ref": ref, "url": ref_url(ref), "width": img.width, "height": img.height}


def _jobs():
    rows = []
    for jid in sorted(os.listdir(JOBS), reverse=True):
        path = os.path.join(JOBS, jid, "job.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                j = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # being rewritten; next poll gets it
        running = state["job"] and state["job"][0] == jid and state["job"][1].poll() is None
        if j.get("status") in ("queued", "running") and not running:
            j["status"] = "error"
            j.setdefault("error", "The 3D job stopped unexpectedly (server restarted?) — see log.txt")
        rigging = jid in state["rigs"] and state["rigs"][jid].poll() is None
        if (j.get("rig") or {}).get("state") == "running" and not rigging and j.get("status") == "done":
            j["rig"] = {"state": "error", "detail": "rigging stopped unexpectedly — see rig_log.txt"}
        rows.append(j)
    return sorted(rows, key=lambda j: j.get("created", 0), reverse=True)


@app.get("/api/library")
def library():
    """Everything usable as a reference: generations, uploads, 3D model renders."""
    gens = [{"ref": r["file"], "url": ref_url(r["file"]), "label": r["prompt"], "prompt": r["prompt"],
             "kind": "generation"} for r in _history()[:120]]
    ups = [{"ref": "uploads/" + n, "url": ref_url("uploads/" + n), "label": "Upload", "kind": "upload"}
           for n in sorted(os.listdir(UPLOADS), reverse=True) if n.endswith(".png")][:120]
    renders = []
    for j in _jobs():
        if j.get("status") == "done":
            ref = f"3d/{j['id']}/thumb.png"
            renders.append({"ref": ref, "url": ref_url(ref), "label": "3D render · " + (j.get("title") or j["id"]),
                            "kind": "render", "job": j["id"]})
    return {"generations": gens, "uploads": ups, "renders": renders}


class Job3D(BaseModel):
    source: str
    back: str | None = None
    left: str | None = None    # camera on the character's left: the figure faces image-left
    right: str | None = None
    describe: str = Field("", max_length=600)
    quality: Literal["draft", "standard", "high"] = "standard"
    pose: Literal["keep", "tpose"] = "keep"
    engine: Literal["tripo", "local"] = "tripo"   # tripo: Tripo H3.1 on fal (paid, ~$0.3-0.4); local: Hunyuan3D + klein
    faces: int = Field(30000, ge=5000, le=100000)
    height: float = Field(1.8, gt=0.1, le=20)
    title: str = Field("", max_length=120)
    rig: bool = True     # rig.py: humanoid skeleton + Idle/Walk/Run/WalkBackward/StrafeLeft/StrafeRight clips
    kind: Literal["human", "object"] = "human"   # Humans page (rigged characters) or Objects page (props, static)


@app.post("/api/3d")
def start_3d(req: Job3D):
    if not os.path.exists(VENV3D):
        raise HTTPException(501, "3D tools are not installed (.venv3d missing) — see README")
    if state["engine"] is None:
        raise HTTPException(503, "Models are still loading")
    if state["job"] and state["job"][1].poll() is None:
        raise HTTPException(409, "A 3D job is already running — wait for it to finish")
    if req.engine == "tripo" and not has_fal_key():
        raise HTTPException(400, f"Tripo needs a fal API key: set FAL_KEY or save it to {FAL_KEY_FILE}")
    for r in (req.source, req.back, req.left, req.right):
        if r:
            resolve_ref(r)
    if req.kind == "object":       # props and static assets: no T-pose, no skeleton
        req.rig, req.pose = False, "keep"
    jid = str(int(time.time() * 1000))
    d = os.path.join(JOBS, jid)
    os.makedirs(d)
    job = {"id": jid, "created": time.time(), "status": "queued", "title": req.title,
           "config": {"source": req.source, "views": {"back": req.back, "left": req.left, "right": req.right},
                      "describe": req.describe, "quality": req.quality, "faces": req.faces, "height": req.height,
                      "pose": req.pose, "engine": req.engine, "kind": req.kind},
           "source_url": ref_url(req.source),
           "stages": [{"key": k, "label": l, "state": "pending"}
                      for k, l in (STAGES_TRIPO if req.engine == "tripo" else STAGES) + ([RIG_STAGE] if req.rig else [])
                      if req.kind == "human" or k not in HUMAN_ONLY]}
    with open(os.path.join(d, "job.json"), "w") as f:
        json.dump(job, f, indent=1)
    log = open(os.path.join(d, "log.txt"), "w")
    proc = subprocess.Popen([VENV3D, os.path.join(ROOT, "tools3d", "make3d.py"), d], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT)
    state["job"] = (jid, proc)
    return job


CLIP_RATES = os.path.join(OUT, "clip_rates.json")
RIG_CLIPS = ("Idle", "Walk", "Run", "WalkBackward", "StrafeLeft", "StrafeRight")


CLIP_RATES_DEFAULT = os.path.join(ROOT, "tools3d", "clip_rates_default.json")   # shipped defaults


@app.get("/api/clip_rates")
def get_clip_rates():
    saved = {}
    for path in (CLIP_RATES_DEFAULT, CLIP_RATES):
        try:
            with open(path) as f:
                saved.update(json.load(f))
        except (OSError, ValueError):
            pass
    return {c: float(saved.get(c, 1.0)) for c in RIG_CLIPS}


@app.post("/api/clip_rates")
def set_clip_rates(rates: dict[str, float]):
    """Playback rate per clip (1 = as captured), used by every rig from now on."""
    clean = {c: round(min(4.0, max(0.25, float(v))), 3) for c, v in rates.items() if c in RIG_CLIPS}
    out = {**get_clip_rates(), **clean}
    with open(CLIP_RATES + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(CLIP_RATES + ".tmp", CLIP_RATES)
    return out


@app.post("/api/3d/{jid}/rig")
def rig_3d(jid: str):
    """Rig an existing model: skeleton, skin weights and the clip library, on its game.glb (~5-10 s)."""
    if not os.path.exists(VENV3D):
        raise HTTPException(501, "3D tools are not installed (.venv3d missing) — see README")
    j = get_3d(jid)
    if j.get("status") != "done" or not os.path.exists(os.path.join(JOBS, jid, "game.glb")):
        raise HTTPException(400, "The model isn't finished yet")
    busy = state["rigs"].get(jid)
    if busy and busy.poll() is None:
        raise HTTPException(409, "Already rigging this model — wait a few seconds")
    path = os.path.join(JOBS, jid, "job.json")   # mark it before the process starts, so the first poll sees it
    with open(path) as f:
        data = json.load(f)
    data["rig"] = {"state": "running", "detail": "starting"}
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)
    log = open(os.path.join(JOBS, jid, "rig_log.txt"), "w")
    proc = subprocess.Popen([VENV3D, os.path.join(ROOT, "tools3d", "rig.py"), "--job", os.path.join(JOBS, jid)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state["rigs"][jid] = proc
    return {"id": jid, "rig": {"state": "running", "detail": "starting"}}


@app.get("/api/3d")
def list_3d(kind: str | None = None):
    from tinyimgen.library import job_kind
    rows = []
    for j in _jobs():
        j["kind"] = job_kind(j)
        if kind in (None, j["kind"]):
            rows.append(j)
    return rows[:80]


@app.get("/api/3d/{jid}")
def get_3d(jid: str):
    for j in _jobs():
        if j["id"] == jid:
            return j
    raise HTTPException(404, "No such 3D job")


# --- Explorer: everything generated, with favorites / names / tags, import, export and delete -----------------------

class ExplorerMeta(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=2000)
    favorite: bool | None = None
    title: str | None = Field(None, max_length=200)
    tags: list[str] | None = Field(None, max_length=30)
    kind: Literal["human", "object"] | None = None


class ExplorerIds(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=2000)
    full: bool = False   # export: include full-detail meshes (tens of MB each)


@app.get("/api/explorer")
def explorer():
    from tinyimgen import library
    items = library.list_items(_jobs())
    counts = {}
    for it in items:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    return {"items": items, "counts": counts}


@app.get("/api/explorer/thumb")
def explorer_thumb(id: str):
    from tinyimgen import library
    try:
        return FileResponse(library.thumb(id), headers={"Cache-Control": "max-age=86400"})
    except Exception as e:
        raise HTTPException(404, f"No thumbnail: {e}")


@app.post("/api/explorer/meta")
def explorer_meta(req: ExplorerMeta):
    from tinyimgen import library
    tags = [t.strip()[:40] for t in req.tags if t.strip()] if req.tags is not None else None
    for i in req.ids:
        try:
            library.set_meta(i, favorite=req.favorite, title=req.title, tags=tags, kind=req.kind)
        except (ValueError, OSError) as e:
            raise HTTPException(400, f"{i}: {e}")
    return {"ok": True}


@app.post("/api/explorer/delete")
def explorer_delete(req: ExplorerIds):
    from tinyimgen import library
    busy = {i for i in req.ids if i.startswith("3d:") and state["job"] and state["job"][0] == i[3:]
            and state["job"][1].poll() is None}
    if busy:
        raise HTTPException(409, "That 3D model is still building")
    return {"moved_to_trash": library.delete(req.ids)}


@app.post("/api/explorer/export")
def explorer_export(req: ExplorerIds, background: BackgroundTasks):
    from tinyimgen import library
    path = library.export_zip(req.ids, library.list_items(_jobs()), full=req.full)
    background.add_task(os.remove, path)
    return FileResponse(path, media_type="application/zip",
                        filename=time.strftime("tinyimgen_export_%Y%m%d_%H%M%S.zip"))


@app.post("/api/explorer/import")
async def explorer_import(request: Request, name: str):
    """Raw file body (one file per request): images, .glb / .fbx / .obj, or a zip from Export."""
    from tinyimgen import library
    data = await request.body()
    if len(data) > 1_500_000_000:
        raise HTTPException(413, "File too large")
    try:
        ids = library.import_file(name, data, venv3d=VENV3D if os.path.exists(VENV3D) else None, blender=BLENDER)
    except (ValueError, OSError, UnidentifiedImageError) as e:
        raise HTTPException(400, str(e))
    return {"ids": ids}


# --- holdable props: grip points on 3D objects (prop editor) ----------------------------------------------------

@app.get("/api/props/{jid}")
def get_prop(jid: str):
    from tinyimgen import props
    j = get_3d(jid)
    return {"job": j, "grip": props.load(jid)}


@app.post("/api/props/{jid}")
def save_prop(jid: str, grip: dict):
    from tinyimgen import props
    j = get_3d(jid)
    if j.get("status") != "done":
        raise HTTPException(400, "That model isn't finished yet")
    try:
        saved = props.save(jid, grip)
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise HTTPException(400, f"Couldn't save the grips: {e}")
    return {"grip": saved, "files": {"prop": f"/media/3d/{jid}/prop.glb", "grip": f"/media/3d/{jid}/grip.json"}}


@app.get("/api/showcase")
def showcase():
    path = os.path.join(SHOWCASE, "showcase.json")
    if not os.path.exists(path):
        return {"items": []}
    with open(path) as f:
        return json.load(f)


app.mount("/media/gallery", StaticFiles(directory=GALLERY), name="gallery")
app.mount("/media/showcase", StaticFiles(directory=SHOWCASE), name="showcase")
app.mount("/media/uploads", StaticFiles(directory=UPLOADS), name="uploads")
app.mount("/media/3d", StaticFiles(directory=JOBS), name="jobs")
os.makedirs(MOCAP, exist_ok=True)
app.mount("/media/mocap", StaticFiles(directory=MOCAP), name="mocap")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")


@app.get("/")
def studio_page():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/3d")
def page_3d():
    return FileResponse(os.path.join(ROOT, "static", "3d.html"))


@app.get("/objects")
def page_objects():
    return FileResponse(os.path.join(ROOT, "static", "3d.html"))


@app.get("/prop-editor")
def page_prop_editor():
    return FileResponse(os.path.join(ROOT, "static", "prop-editor.html"))


@app.get("/explorer")
def page_explorer():
    return FileResponse(os.path.join(ROOT, "static", "explorer.html"))


@app.get("/play")
def play_page():
    return FileResponse(os.path.join(ROOT, "static", "play.html"))


@app.get("/showcase")
def showcase_page():
    return FileResponse(os.path.join(ROOT, "static", "showcase.html"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--zimage", action="store_true", help="also load Z-Image Turbo (INT4 via nunchaku on CUDA, stock bf16 on Apple silicon)")
    a = ap.parse_args()
    if a.zimage:
        state["models"] = ("klein", "zimage")
    threading.Thread(target=_load, daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
