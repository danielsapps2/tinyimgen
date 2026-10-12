"""Benchmark Z-Image-Turbo on a 6 GB Turing GPU (fp16 + nunchaku INT4 transformer)."""
import argparse, time, os
import torch
torch.set_grad_enabled(False)
from diffusers import ZImagePipeline
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--mode", default="int4", choices=["int4", "fp16-offload"])
p.add_argument("--rank", default="r128")
p.add_argument("--size", type=int, default=1024)
p.add_argument("--steps", type=int, default=8)
p.add_argument("--out", default="out/bench")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

dt = torch.float16
t0 = time.time()
if args.mode == "int4":
    from nunchaku import NunchakuZImageTransformer2DModel
    from tinyimgen.compat import patch_nunchaku_zimage; patch_nunchaku_zimage()
    path = hf_hub_download("nunchaku-ai/nunchaku-z-image-turbo", f"svdq-int4_{args.rank}-z-image-turbo.safetensors")
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=dt)
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=dt)
else:
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=dt)
from tinyimgen.stream import LayerStreamer
tr = pipe.transformer
blocks = list(tr.noise_refiner) + list(tr.context_refiner) + list(tr.layers)
LayerStreamer(blocks, cyclic=True)
for n, m in tr.named_children():
    if n not in ("layers", "noise_refiner", "context_refiner"): m.to("cuda")
for n, p in tr.named_parameters(recurse=False): p.data = p.data.cuda()
for n, b in tr.named_buffers(recurse=False): b.data = b.data.cuda()

# Text encoder (Qwen3-4B, ~8 GB fp16) streams from pinned RAM block by block.
from tinyimgen.stream import LayerStreamer, cpu_embedding
te = pipe.text_encoder
LayerStreamer(te.layers); cpu_embedding(te.embed_tokens); te.norm.to("cuda"); te.rotary_emb.to("cuda")
pipe.vae.to("cuda")
pipe.vae.enable_tiling()
print(f"load {time.time()-t0:.1f}s  vram_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")

prompts = [
    "A photorealistic close-up portrait of an elderly fisherman with a weathered face, wearing a yellow raincoat, golden hour light, shallow depth of field",
    "A cozy cyberpunk ramen shop at night in the rain, neon signs that read \"TINY GPU\", reflections on wet pavement, cinematic",
    "A red fox sitting in a snowy birch forest, morning mist, national geographic photo",
]
for i, pr in enumerate(prompts):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    img = pipe(prompt=pr, height=args.size, width=args.size, num_inference_steps=args.steps,
               guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(42 + i)).images[0]
    torch.cuda.synchronize(); el = time.time() - t
    import numpy as np
    a = np.asarray(img)
    print(f"[{i}] {el:.1f}s peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB mean={a.mean():.1f} std={a.std():.1f}")
    img.save(f"{args.out}/zimage_{args.mode}_{args.size}_{i}.png")
