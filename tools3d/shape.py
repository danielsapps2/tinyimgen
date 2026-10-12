"""Image(s) -> 3D shape with Hunyuan3D-2 mini (single view) or mv (multi-view), fp16 on a 6 GB GPU.

    .venv3d/bin/python tools3d/shape.py OUTDIR --front f.webp [--left l.webp --back b.webp --right r.webp]

View keys follow Hunyuan3D: "left" is the camera on the character's left (figure faces image-left).
Writes OUTDIR/views/*.png (background removed), OUTDIR/shape_<model>.glb and OUTDIR/shape_<model>.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinyimgen import device  # noqa: E402  (before torch: MPS fallback switch)
import torch  # noqa: E402
from PIL import Image

MODELS = {
    "mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"),
    "mv": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo"),
}
_cutter = None


def cutout(img):
    """Background removal. isnet is ~20x faster on CPU than rembg's default bria-rmbg-2.0, fine on plain backdrops."""
    global _cutter
    from rembg import new_session, remove
    if _cutter is None:
        _cutter = new_session("isnet-general-use")
    return remove(img.convert("RGB"), session=_cutter, bgcolor=[255, 255, 255, 0])


def swap_components(pipe):
    """Model-level offload: move a component to the GPU on first use, evicting the others.
    (hy3dgen's own enable_model_cpu_offload expects a diffusers `components` attribute it lacks.)"""
    names = ("conditioner", "model", "vae")
    state = {"active": None}
    for n in names:
        getattr(pipe, n).to("cpu")

    def activate(name):
        if state["active"] != name:
            for n in names:
                if n != name:
                    getattr(pipe, n).to("cpu")
            device.empty_cache()
            getattr(pipe, name).to(device.DEVICE)
            state["active"] = name

    def wrap(obj, attr, name):
        fn = getattr(obj, attr)
        def inner(*args, **kwargs):
            activate(name)
            return fn(*args, **kwargs)
        setattr(obj, attr, inner)

    pipe._swap_state = state
    wrap(pipe.conditioner, "forward", "conditioner")
    wrap(pipe.conditioner, "unconditional_embedding", "conditioner")
    wrap(pipe.model, "forward", "model")
    wrap(pipe.vae, "forward", "vae")
    wrap(pipe.vae, "latents2mesh", "vae")


_pipes = {}


def _pipeline(name):
    """Loaded once per process and kept in RAM: a job runs the shape model up to three times (body, head, hair), and
    loading it is slower than running it."""
    if name not in _pipes:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        repo, sub = MODELS[name]
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo, subfolder=sub, use_safetensors=True,
                                                                device="cpu", dtype=torch.float16)
        # ~4 GB of fp16 weights (DINOv2-giant encoder 2.3 GB): keep only the active component on the GPU.
        # Unified memory (Apple silicon) has room for all of it, so there it simply stays on the device.
        pipe.device = torch.device(device.DEVICE)
        pipe.enable_flashvdm(topk_mode="merge")
        if device.CUDA:
            swap_components(pipe)
        else:
            for n in ("conditioner", "model", "vae"):
                getattr(pipe, n).to(device.DEVICE)
        _pipes[name] = pipe
    return _pipes[name]


def generate_shape(views, name=None, steps=5, octree=380, seed=1234):
    """views: {"front": RGBA, optional "left"/"back"/"right"}. Returns (trimesh, stats)."""
    name = name or ("mv" if len(views) > 1 else "mini")
    repo, sub = MODELS[name]
    pipe = _pipeline(name)
    device.synchronize(); device.reset_peak(); t = time.time()
    mesh = pipe(image=views if name == "mv" else views["front"], num_inference_steps=steps,
                octree_resolution=octree, num_chunks=20000, generator=torch.manual_seed(seed),
                output_type="trimesh")[0]
    device.synchronize()
    stats = {"model": f"{repo}/{sub}", "seconds": round(time.time() - t, 1),
             "peak_vram_gb": device.peak_gb(),
             "vertices": len(mesh.vertices), "faces": len(mesh.faces),
             "views": list(views) if name == "mv" else ["front"]}
    # park every component in RAM between uses: the image server needs the VRAM back
    if device.CUDA:
        for n in ("conditioner", "model", "vae"):
            getattr(pipe, n).to("cpu")
        pipe._swap_state["active"] = None
    del pipe
    import gc
    gc.collect()
    device.empty_cache()
    return mesh, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--front", required=True)
    ap.add_argument("--left"); ap.add_argument("--back"); ap.add_argument("--right")
    ap.add_argument("--models", default="mini,mv")
    ap.add_argument("--octree", type=int, default=380)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/views", exist_ok=True)
    views = {}
    for key in ("front", "left", "back", "right"):
        path = getattr(a, key)
        if not path:
            continue
        cached = f"{a.out}/views/{key}.png"
        if not os.path.exists(cached):
            cutout(Image.open(path)).save(cached)
        views[key] = Image.open(cached)
    for name in a.models.split(","):
        if name == "mv" and len(views) < 2:
            continue
        mesh, stats = generate_shape(views if name == "mv" else {"front": views["front"]}, name,
                                     a.steps, a.octree, a.seed)
        mesh.export(f"{a.out}/shape_{name}.glb")
        with open(f"{a.out}/shape_{name}.json", "w") as f:
            json.dump(stats, f, indent=1)
        print(name, stats, flush=True)


if __name__ == "__main__":
    main()
