"""Two-model text-to-image engine tuned for a 6 GB Turing GPU (RTX 2060).

* Everything runs in fp16 (Turing has no fast bf16: 3.6 vs 23 TFLOPS measured).
* Transformer and text-encoder blocks live in pinned RAM and are streamed through
  the GPU by LayerStreamer, so peak VRAM stays under ~3.7 GB at 1024x1024.
* FLUX.2-klein-4B and Z-Image-Turbo use the identical Qwen3-4B text encoder,
  so one streamed copy serves both.
"""
import ctypes
import gc
import os
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must precede huggingface_hub import
# Desktop apps (browsers especially) grab and release VRAM unpredictably; avoid fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from .gpulock import gpu_lock
from .stream import LayerStreamer, cpu_embedding

torch.set_grad_enabled(False)

DT = torch.float16

MODELS = {
    "klein": {
        "label": "FLUX.2 klein 4B",
        "detail": "fp16 · 4 steps · Black Forest Labs · Apache-2.0",
        "steps": 4,
        "max_steps": 8,
    },
    "zimage": {
        "label": "Z-Image Turbo",
        "detail": "SVDQuant INT4 · 8 steps · Tongyi-MAI · Apache-2.0",
        "steps": 8,
        "max_steps": 12,
    },
}


def _to_cuda_except(module, skip):
    for name, child in module.named_children():
        if name not in skip:
            child.cuda()
    for t in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        t.data = t.data.cuda()


def _fit_ref(img, max_pixels=1024 * 1024):
    """Reference images cost transformer tokens like the output does; cap them at ~1 MP."""
    w, h = img.size
    if w * h > max_pixels:
        k = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(64, int(w * k)), max(64, int(h * k))), Image.LANCZOS)
    return img


class Engine:
    def __init__(self, models=("klein",)):
        self.lock = threading.Lock()
        self.progress = {"busy": False, "step": 0, "steps": 0, "model": None}
        self.pipes = {}
        t0 = time.time()

        from diffusers import Flux2KleinPipeline
        klein = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=DT)
        causal = klein.text_encoder
        causal.lm_head = torch.nn.Identity()  # weight-tied to the CPU embedding; pipelines only read hidden states
        lm = causal.model
        LayerStreamer(lm.layers)
        cpu_embedding(lm.embed_tokens)
        lm.norm.cuda(); lm.rotary_emb.cuda()

        if "klein" in models:
            tr = klein.transformer
            LayerStreamer(list(tr.transformer_blocks) + list(tr.single_transformer_blocks), cyclic=True)
            _to_cuda_except(tr, ("transformer_blocks", "single_transformer_blocks"))
            klein.vae.cuda(); klein.vae.enable_tiling()  # same speed, far lower peak
            self.pipes["klein"] = klein

        if "zimage" in models:
            from diffusers import ZImagePipeline
            from nunchaku import NunchakuZImageTransformer2DModel
            from .compat import patch_nunchaku_zimage
            patch_nunchaku_zimage()
            path = hf_hub_download("nunchaku-ai/nunchaku-z-image-turbo", "svdq-int4_r128-z-image-turbo.safetensors")
            tr = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=DT)
            z = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", transformer=tr, text_encoder=lm,
                                               torch_dtype=DT)
            LayerStreamer(list(tr.noise_refiner) + list(tr.context_refiner) + list(tr.layers), cyclic=True)
            _to_cuda_except(tr, ("noise_refiner", "context_refiner", "layers"))
            z.vae.cuda(); z.vae.enable_tiling()
            self.pipes["zimage"] = z

        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand fp32->fp16 conversion scratch back to the OS
        self.load_seconds = time.time() - t0

    def generate(self, model, prompt, width=1024, height=1024, steps=None, seed=None, refs=None):
        """refs: optional list of PIL reference images (klein multi-reference editing)."""
        pipe = self.pipes[model]
        steps = int(steps or MODELS[model]["steps"])
        seed = int(seed if seed is not None else torch.seed() % 2**31)
        width, height = (int(width) // 64) * 64, (int(height) // 64) * 64

        def on_step(p, i, t, kw):
            self.progress["step"] = i + 1
            return kw

        if refs:
            refs = [_fit_ref(r) for r in refs]
        with self.lock:
            self.progress.update(busy=True, step=0, steps=steps, model=model, waiting=True)
            try:
                with gpu_lock():  # a 3D job may be using the card
                    self.progress["waiting"] = False
                    return self._run(pipe, model, prompt, width, height, steps, seed, refs, on_step)
            finally:
                # Hand cached VRAM back so other GPU apps (e.g. the 3D pipeline) can use it;
                # the next image re-allocates in well under a second.
                torch.cuda.empty_cache()
                self.progress.update(busy=False, waiting=False)

    def _run(self, pipe, model, prompt, width, height, steps, seed, refs, on_step):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        kwargs = dict(prompt=prompt, width=width, height=height, num_inference_steps=steps,
                      generator=torch.Generator("cuda").manual_seed(seed),
                      callback_on_step_end=on_step)
        if model == "zimage":
            kwargs["guidance_scale"] = 0.0
        if refs:
            if model != "klein":
                raise ValueError("reference images are only supported by klein")
            kwargs["image"] = list(refs)
        try:
            img = pipe(**kwargs).images[0]
        except torch.OutOfMemoryError:
            # Another process grew its VRAM mid-run; release our cache and retry once.
            gc.collect(); torch.cuda.empty_cache()
            kwargs["generator"] = torch.Generator("cuda").manual_seed(seed)
            self.progress["step"] = 0
            img = pipe(**kwargs).images[0]
        torch.cuda.synchronize()
        return img, {
            "model": model, "prompt": prompt, "width": width, "height": height, "steps": steps,
            "seed": seed, "seconds": round(time.time() - t, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

