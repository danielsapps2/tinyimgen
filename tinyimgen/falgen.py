"""Cloud image models on fal.ai, next to the local klein / Z-Image engine.

Each model has a text-to-image endpoint and an edit endpoint (used when the Studio sends reference images). Prices
are fal's published rates as of 2026-10-12. `cost()` estimates one request, which is logged with the image so the
Studio can show spend; fal's dashboard has the exact charges.

The fal key comes from FAL_KEY or ~/.config/tinyimgen/fal_key (the same key the 3D tab uses for Tripo).
"""
import io
import math
import os
import tempfile
import time

from PIL import Image

KEY_FILE = os.path.expanduser("~/.config/tinyimgen/fal_key")

FAL_MODELS = {
    "nb_pro": {
        "label": "Nano Banana Pro",
        "detail": "Google · best text, layout and edits",
        "price": "$0.15 / image",
        "t2i": "fal-ai/nano-banana-pro", "edit": "fal-ai/nano-banana-pro/edit",
        "kind": "nano", "resolution": "2K", "max_refs": 14,
    },
    "nb2": {
        "label": "Nano Banana 2",
        "detail": "Google · fast, high quality",
        "price": "$0.08 / image",
        "t2i": "fal-ai/nano-banana-2", "edit": "fal-ai/nano-banana-2/edit",
        "kind": "nano", "resolution": "1K", "max_refs": 14,
    },
    "flux2pro": {
        "label": "FLUX.2 [pro]",
        "detail": "Black Forest Labs · photoreal, klein's big sibling",
        "price": "$0.03 / megapixel",
        "t2i": "fal-ai/flux-2-pro", "edit": "fal-ai/flux-2-pro/edit",
        "kind": "flux", "max_refs": 8,
    },
    "seedream45": {
        "label": "Seedream 4.5",
        "detail": "ByteDance · 2K+ output, strong composition",
        "price": "$0.04 / image",
        "t2i": "fal-ai/bytedance/seedream/v4.5/text-to-image", "edit": "fal-ai/bytedance/seedream/v4.5/edit",
        "kind": "seedream", "max_refs": 10,
    },
}

NANO_RATIOS = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"]


def fal_key():
    key = os.environ.get("FAL_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
    return key or None


def cost(model, width, height, n_refs=0):
    """Estimated USD for one request at fal's published rate."""
    if model == "nb_pro":
        return 0.15
    if model == "nb2":
        return 0.08
    if model == "seedream45":
        return 0.04
    if model == "flux2pro":   # processed megapixels: the output (rounded up) plus each reference (~1 MP after fal's resize)
        return round(0.03 * (math.ceil(width * height / 1e6) + n_refs), 3)
    return 0.0


def _nano_ratio(w, h):
    r = w / h
    return min(NANO_RATIOS, key=lambda s: abs(math.log(r / (int(s.split(":")[0]) / int(s.split(":")[1])))))


def _seedream_size(w, h):
    """Seedream 4.5 needs at least 2560x1440 pixels in total: the same aspect, scaled up to that."""
    k = max(1.0, math.sqrt(2560 * 1440 / (w * h)))
    return {"width": int(math.ceil(w * k / 16) * 16), "height": int(math.ceil(h * k / 16) * 16)}


def generate(model, prompt, width, height, seed, refs, progress=None):
    """refs: PIL images. Returns (PIL image, meta). Raises ValueError for user-fixable problems."""
    import fal_client
    import httpx
    spec = FAL_MODELS[model]
    key = fal_key()
    if not key:
        raise ValueError(f"No fal API key: save it to {KEY_FILE} (or set FAL_KEY)")
    if len(refs) > spec["max_refs"]:
        raise ValueError(f"{spec['label']} takes at most {spec['max_refs']} reference images")
    client = fal_client.SyncClient(key=key, default_timeout=300)
    t = time.time()
    urls = []
    for im in refs:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            im.save(f.name)
            urls.append(client.upload_file(f.name))
    args = {"prompt": prompt}
    if spec["kind"] == "nano":
        args.update(aspect_ratio=_nano_ratio(width, height), resolution=spec["resolution"], output_format="png",
                    num_images=1)
    elif spec["kind"] == "flux":
        args.update(image_size={"width": width, "height": height}, output_format="png")
    elif spec["kind"] == "seedream":
        args.update(image_size=_seedream_size(width, height), num_images=1)
    if seed is not None:
        args["seed"] = int(seed)
    endpoint = spec["edit"] if urls else spec["t2i"]
    if urls:
        args["image_urls"] = urls
    if progress:
        progress("queued")
    handle = client.submit(endpoint, args)
    for ev in handle.iter_events(interval=0.5):
        if progress:
            progress("generating" if type(ev).__name__ == "InProgress" else "queued")
    res = handle.get()
    r = httpx.get(res["images"][0]["url"], timeout=120, follow_redirects=True)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img, {
        "model": model, "prompt": prompt, "width": img.width, "height": img.height, "steps": None,
        "seed": res.get("seed", seed), "seconds": round(time.time() - t, 2), "provider": "fal",
        "endpoint": endpoint, "request_id": handle.request_id, "cost_usd": cost(model, img.width, img.height, len(urls)),
    }
