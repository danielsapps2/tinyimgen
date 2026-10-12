"""The Explorer's view of everything under out/: images, humans, 3D objects and the rest, plus the user's own
metadata (favorites, names, tags, Humans/Objects moves) and import / export / delete.

Item ids (stable across restarts):
    gen:<file>     a Studio generation (out/gallery, listed in history.jsonl)
    up:<file>      an uploaded / imported image (out/uploads)
    3d:<job id>    a 3D model (out/3d/<id>): kind "human" or "object"
    tool:<file>    an image the 3D pipeline made on the way (T-pose, back view, repaint...; out/gallery, not in history)
    vid:<clip>     a motion-capture reference video (out/mocap/<clip>/video.mp4)

User metadata lives in out/library.json: {id: {"favorite": bool, "title": str, "tags": [str]}}. Deleting moves the
files to out/trash/<timestamp>/ (nothing is destroyed) and forgets the item.
"""
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.realpath(os.path.join(ROOT, "out"))
GALLERY = os.path.join(OUT, "gallery")
UPLOADS = os.path.join(OUT, "uploads")
JOBS = os.path.join(OUT, "3d")
MOCAP = os.path.join(OUT, "mocap")
TRASH = os.path.join(OUT, "trash")
THUMBS = os.path.join(OUT, ".thumbs")
META = os.path.join(OUT, "library.json")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
MODEL_EXT = (".glb", ".fbx", ".obj")
VENV3D = os.path.join(ROOT, ".venv3d", "bin", "python")
_lock = threading.Lock()

# files of a 3D job worth keeping / exporting (the rest are pipeline intermediates)
JOB_FILES = [("rigged.glb", "Rigged GLB"), ("rigged.fbx", "Rigged FBX"), ("prop.glb", "Prop GLB (grips)"),
             ("grip.json", "Grip JSON"), ("game.glb", "Game GLB"),
             ("albedo.png", "Texture PNG"), ("textured.glb", "Full-detail GLB"), ("model.glb", "Model GLB"),
             ("thumb.png", "Render PNG")]
JOB_EXTRA = ["job.json", "rig.json"]
JOB_BIG = {"textured.glb"}     # left out of exports unless asked for (tens of MB)


# --- metadata -------------------------------------------------------------------------------------------------------

def load_meta():
    try:
        with open(META) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_meta(meta):
    tmp = META + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, META)


def set_meta(item_id, **kw):
    """Update favorite / title / tags / kind for one item. kind moves a 3D model between Humans and Objects."""
    with _lock:
        meta = load_meta()
        m = meta.setdefault(item_id, {})
        for k in ("favorite", "title", "tags"):
            if k in kw and kw[k] is not None:
                m[k] = kw[k]
        save_meta(meta)
    if kw.get("kind") in ("human", "object") and item_id.startswith("3d:"):
        path = os.path.join(JOBS, safe(item_id[3:]), "job.json")
        with _lock, open(path) as f:
            job = json.load(f)
        job.setdefault("config", {})["kind"] = kw["kind"]
        with open(path + ".tmp", "w") as f:
            json.dump(job, f, indent=1)
        os.replace(path + ".tmp", path)


def safe(name):
    """A single path component (no traversal)."""
    name = os.path.basename(name)
    if not name or name.startswith("."):
        raise ValueError("bad name")
    return name


# --- listing --------------------------------------------------------------------------------------------------------

def job_kind(job):
    """human or object. New jobs say so; older ones were all character builds, except where rigging found no
    humanoid."""
    k = (job.get("config") or {}).get("kind")
    if k in ("human", "object"):
        return k
    if (job.get("rig") or {}).get("state") == "skipped":
        return "object"
    return "human"


def _file(url, path, label):
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    return {"label": label, "url": url, "name": os.path.basename(path), "size": size}


def _history():
    path = os.path.join(GALLERY, "history.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def list_items(jobs):
    """Every item, newest first. jobs: the server's 3D job list (with live status)."""
    meta = load_meta()
    items = []
    hist = _history()
    in_hist = set()
    for r in hist:
        f = r.get("file")
        p = os.path.join(GALLERY, f or "")
        if not f or not os.path.isfile(p):
            continue
        in_hist.add(f)
        items.append({"id": "gen:" + f, "kind": "image", "sub": "generation", "title": r.get("prompt", f),
                      "created": r.get("created") or os.path.getmtime(p), "thumb": "/api/explorer/thumb?id=gen:" + f,
                      "preview": "/media/gallery/" + f, "ref": f,
                      "files": [x for x in [_file("/media/gallery/" + f, p, "Image")] if x],
                      "info": {k: r[k] for k in ("prompt", "model", "width", "height", "seed", "steps", "seconds",
                                                 "cost_usd", "refs") if k in r}})
    for f in sorted(os.listdir(UPLOADS)) if os.path.isdir(UPLOADS) else []:
        p = os.path.join(UPLOADS, f)
        if not f.lower().endswith(IMAGE_EXT) or not os.path.isfile(p):
            continue
        items.append({"id": "up:" + f, "kind": "image", "sub": "upload", "title": "Upload", "created": os.path.getmtime(p),
                      "thumb": "/api/explorer/thumb?id=up:" + f, "preview": "/media/uploads/" + f, "ref": "uploads/" + f,
                      "files": [x for x in [_file("/media/uploads/" + f, p, "Image")] if x], "info": {}})
    for f in sorted(os.listdir(GALLERY)) if os.path.isdir(GALLERY) else []:
        p = os.path.join(GALLERY, f)
        if f in in_hist or not f.lower().endswith(IMAGE_EXT) or not os.path.isfile(p):
            continue
        items.append({"id": "tool:" + f, "kind": "other", "sub": "pipeline", "title": "3D pipeline image",
                      "created": os.path.getmtime(p), "thumb": "/api/explorer/thumb?id=tool:" + f,
                      "preview": "/media/gallery/" + f, "ref": f,
                      "files": [x for x in [_file("/media/gallery/" + f, p, "Image")] if x], "info": {}})
    for j in jobs:
        d = os.path.join(JOBS, j["id"])
        outs = j.get("outputs") or {}
        files = [x for x in (_file(f"/media/3d/{j['id']}/{fn}", os.path.join(d, fn), label)
                             for fn, label in JOB_FILES) if x]
        cfg = j.get("config") or {}
        when = time.strftime("%b %d, %H:%M", time.localtime(j.get("created", 0)))
        thumb = f"/media/3d/{j['id']}/{outs['thumb']}" if outs.get("thumb") else (j.get("source_url") or "")
        preview = outs.get("rigged") or outs.get("game")
        items.append({"id": "3d:" + j["id"], "kind": job_kind(j),
                      "sub": "import" if cfg.get("engine") == "import" else cfg.get("engine", "local"),
                      "title": j.get("title") or cfg.get("describe") or ("3D model · " + when),
                      "created": j.get("created", 0), "thumb": thumb, "status": j.get("status"),
                      "preview": f"/media/3d/{j['id']}/{preview}" if preview and j.get("status") == "done" else None,
                      "rigged": bool(outs.get("rigged")), "clips": (j.get("rig") or {}).get("clips", []),
                      "holdable": j.get("grip"),
                      "files": files, "job": j["id"], "ref": f"3d/{j['id']}/thumb.png" if outs.get("thumb") else None,
                      "info": {k: v for k, v in {"engine": cfg.get("engine"), "quality": cfg.get("quality"),
                                                 "height_m": cfg.get("height"), "triangles": (j.get("stats") or {})
                                                 .get("triangles"), "seconds": j.get("elapsed"),
                                                 "source": cfg.get("source"), "describe": cfg.get("describe")}.items()
                               if v not in (None, "")}})
    if os.path.isdir(MOCAP):
        for clip in sorted(os.listdir(MOCAP)):
            p = os.path.join(MOCAP, clip, "video.mp4")
            if os.path.isfile(p):
                items.append({"id": "vid:" + clip, "kind": "other", "sub": "video", "title": "Motion reference · " + clip,
                              "created": os.path.getmtime(p), "thumb": "/api/explorer/thumb?id=vid:" + clip,
                              "preview": f"/media/mocap/{clip}/video.mp4",
                              "files": [x for x in [_file(f"/media/mocap/{clip}/video.mp4", p, "Video MP4")] if x],
                              "info": {}})
    for it in items:
        m = meta.get(it["id"], {})
        it["favorite"] = bool(m.get("favorite"))
        it["tags"] = m.get("tags", [])
        if m.get("title"):
            it["title"] = m["title"]
        it["size"] = sum(f["size"] for f in it["files"])
    items.sort(key=lambda x: x["created"], reverse=True)
    return items


# --- thumbnails -----------------------------------------------------------------------------------------------------

def item_path(item_id):
    """The main file of an item (images, videos, a 3D job's thumbnail)."""
    kind, _, name = item_id.partition(":")
    name = safe(name)
    if kind in ("gen", "tool"):
        return os.path.join(GALLERY, name)
    if kind == "up":
        return os.path.join(UPLOADS, name)
    if kind == "vid":
        return os.path.join(MOCAP, name, "video.mp4")
    if kind == "3d":
        return os.path.join(JOBS, name, "thumb.png")
    raise ValueError("unknown item")


def thumb(item_id, size=384):
    """A cached small WebP of the item (images: downsized; videos: the middle frame)."""
    from PIL import Image
    src = item_path(item_id)
    os.makedirs(THUMBS, exist_ok=True)
    key = item_id.replace(":", "_").replace("/", "_") + f"_{size}.webp"
    out = os.path.join(THUMBS, key)
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    if src.endswith(".mp4"):   # the middle frame, through the 3D environment's OpenCV (the server has none)
        import subprocess
        frame = out + ".png"
        code = ("import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); c.set(1, int(c.get(7)//2)); ok,f=c.read(); "
                "ok and cv2.imwrite(sys.argv[2], f)")
        subprocess.run([VENV3D, "-c", code, src, frame], timeout=60, capture_output=True)
        if not os.path.exists(frame):
            raise ValueError("unreadable video")
        img = Image.open(frame)
        img.load()
        os.remove(frame)
    else:
        img = Image.open(src)
        img.load()
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    img.save(out, quality=85)
    return out


# --- delete ---------------------------------------------------------------------------------------------------------

def delete(ids):
    """Move the items' files to out/trash/<timestamp>/ and forget their metadata. Returns how many were moved."""
    dest = os.path.join(TRASH, time.strftime("%Y%m%d-%H%M%S"))
    n = 0
    with _lock:
        meta = load_meta()
        for item_id in ids:
            try:
                kind, _, name = item_id.partition(":")
                name = safe(name)
                src = os.path.join(JOBS, name) if kind == "3d" else (
                    os.path.join(MOCAP, name) if kind == "vid" else item_path(item_id))
            except ValueError:
                continue
            if not os.path.exists(src):
                continue
            os.makedirs(os.path.join(dest, kind), exist_ok=True)
            shutil.move(src, os.path.join(dest, kind, name))
            meta.pop(item_id, None)
            n += 1
        save_meta(meta)
    return n


# --- export ---------------------------------------------------------------------------------------------------------

def export_zip(ids, items, full=False):
    """A zip of the items (files + manifest.json with titles, favorites, tags and generation info), importable
    again. 3D models bring their game files; full-detail meshes only with full=True."""
    byid = {it["id"]: it for it in items}
    fd, path = tempfile.mkstemp(suffix=".zip", prefix="tinyimgen_")
    os.close(fd)
    manifest = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for item_id in ids:
            it = byid.get(item_id)
            if not it:
                continue
            kind, _, name = item_id.partition(":")
            entry = {k: it.get(k) for k in ("id", "kind", "sub", "title", "favorite", "tags", "created", "info")}
            if kind == "3d":
                d = os.path.join(JOBS, safe(name))
                folder = f"3d/{name}/"
                for fn in [f for f, _ in JOB_FILES] + JOB_EXTRA:
                    p = os.path.join(d, fn)
                    if os.path.isfile(p) and (full or fn not in JOB_BIG):
                        z.write(p, folder + fn)
                entry["path"] = folder
            else:
                src = item_path(item_id)
                arc = f"{'videos' if kind == 'vid' else 'images'}/{name if kind != 'vid' else name + '.mp4'}"
                z.write(src, arc)
                entry["path"] = arc
            manifest.append(entry)
        z.writestr("manifest.json", json.dumps({"tinyimgen_export": 1, "exported": time.time(), "items": manifest},
                                               indent=1))
    return path


# --- import ---------------------------------------------------------------------------------------------------------

def import_file(name, data, venv3d=None, blender=None):
    """Bring a file into the library: images become uploads, GLB / FBX / OBJ become 3D objects (FBX and OBJ go
    through Blender), a zip from export_zip restores its items (other zips: every supported file inside).
    Returns the new item ids."""
    low = name.lower()
    if low.endswith(IMAGE_EXT):
        return [_import_image(name, data)]
    if low.endswith(MODEL_EXT):
        return [_import_model(name, data, venv3d, blender)]
    if low.endswith(".zip"):
        return _import_zip(data, venv3d, blender)
    raise ValueError(f"{name}: unsupported file type (images, .glb, .fbx, .obj or an exported .zip)")


def _import_image(name, data, meta_in=None):
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
    img.thumbnail((2048, 2048), Image.LANCZOS)
    os.makedirs(UPLOADS, exist_ok=True)
    fn = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.png"
    img.save(os.path.join(UPLOADS, fn))
    item_id = "up:" + fn
    m = meta_in or {}
    set_meta(item_id, title=m.get("title") or os.path.splitext(os.path.basename(name))[0],
             favorite=m.get("favorite"), tags=m.get("tags"))
    return item_id


def _new_job_dir():
    jid = str(int(time.time() * 1000))
    while os.path.exists(os.path.join(JOBS, jid)):
        jid = str(int(jid) + 1)
    d = os.path.join(JOBS, jid)
    os.makedirs(d)
    return jid, d


def _import_model(name, data, venv3d, blender, kind="object", meta_in=None):
    import subprocess
    jid, d = _new_job_dir()
    low = name.lower()
    raw = os.path.join(d, "import" + os.path.splitext(low)[1])
    with open(raw, "wb") as f:
        f.write(data)
    glb = os.path.join(d, "game.glb")             # the name the rigger reads (move it to Humans to rig it)
    if low.endswith(".glb"):
        os.replace(raw, glb)
    else:
        if not blender:
            shutil.rmtree(d)
            raise ValueError(f"{name}: converting {os.path.splitext(low)[1]} needs Blender")
        script = os.path.join(ROOT, "tools3d", "blender_convert.py")
        r = subprocess.run([blender, "-b", "-P", script, "--", raw, glb], capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(glb):
            shutil.rmtree(d)
            raise ValueError(f"{name}: Blender couldn't convert it")
    title = (meta_in or {}).get("title") or os.path.splitext(os.path.basename(name))[0]
    job = {"id": jid, "created": time.time(), "status": "done", "title": title,
           "config": {"engine": "import", "kind": kind, "source": None, "file": os.path.basename(name)},
           "stages": [], "outputs": {"game": "game.glb", "shape": "game.glb"}}
    with open(os.path.join(d, "job.json"), "w") as f:
        json.dump(job, f, indent=1)
    if venv3d:   # thumbnail in the background (GPU render through the 3D environment)
        subprocess.Popen([venv3d, os.path.join(ROOT, "tools3d", "thumb.py"), d], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, "thumb_log.txt"), "w"))
    m = meta_in or {}
    set_meta("3d:" + jid, favorite=m.get("favorite"), tags=m.get("tags"))
    return "3d:" + jid


def _import_zip(data, venv3d, blender):
    import io
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    out = []
    if "manifest.json" in names:
        man = json.loads(z.read("manifest.json"))
        for e in man.get("items", []):
            path = e.get("path", "")
            if e.get("kind") in ("human", "object") and path.startswith("3d/"):
                jid, d = _new_job_dir()
                for n in names:
                    if n.startswith(path) and not n.endswith("/"):
                        fn = safe(n[len(path):])
                        with open(os.path.join(d, fn), "wb") as f:
                            f.write(z.read(n))
                jp = os.path.join(d, "job.json")
                job = json.load(open(jp)) if os.path.exists(jp) else {
                    "status": "done", "config": {"engine": "import"}, "outputs": {"game": "game.glb"}}
                job.update(id=jid, created=e.get("created") or time.time(), title=e.get("title") or job.get("title"))
                job.setdefault("config", {})["kind"] = e["kind"]
                job["status"] = "done"
                if os.path.exists(os.path.join(d, "rigged.glb")):
                    job.setdefault("rig", {})["state"] = "done"
                else:
                    job.pop("rig", None)
                with open(jp, "w") as f:
                    json.dump(job, f, indent=1)
                set_meta("3d:" + jid, favorite=e.get("favorite"), tags=e.get("tags"))
                out.append("3d:" + jid)
            elif path in names and path.lower().endswith(IMAGE_EXT):
                out.append(_import_image(path, z.read(path), e))
        return out
    for n in names:
        if n.endswith("/") or n.startswith("__MACOSX"):
            continue
        low = n.lower()
        if low.endswith(IMAGE_EXT):
            out.append(_import_image(n, z.read(n)))
        elif low.endswith(MODEL_EXT):
            out.append(_import_model(n, z.read(n), venv3d, blender))
    return out
