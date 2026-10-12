"""Benchmark FLUX.2-klein-4B (fp16, all weights streamed) on a 6 GB Turing GPU."""
import argparse, os, time
import numpy as np
import torch
torch.set_grad_enabled(False)
from diffusers import Flux2KleinPipeline
from tinyimgen.stream import LayerStreamer, cpu_embedding

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=1024)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--out", default="out/bench")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

dt = torch.float16
t0 = time.time()
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dt)
lm = pipe.text_encoder.model
pipe.text_encoder.lm_head = torch.nn.Identity()  # tied to the CPU embedding; only hidden states are used
LayerStreamer(lm.layers); cpu_embedding(lm.embed_tokens); lm.norm.cuda(); lm.rotary_emb.cuda()
tr = pipe.transformer
LayerStreamer(list(tr.transformer_blocks) + list(tr.single_transformer_blocks), cyclic=True)
for n, m in tr.named_children():
    if n not in ("transformer_blocks", "single_transformer_blocks"):
        m.cuda()
pipe.vae.cuda()
print(f"load {time.time()-t0:.1f}s")

prompts = [
    "A photorealistic close-up portrait of an elderly fisherman with a weathered face, wearing a yellow raincoat, golden hour light, shallow depth of field",
    "A cozy cyberpunk ramen shop at night in the rain, neon signs that read \"TINY GPU\", reflections on wet pavement, cinematic",
    "A red fox sitting in a snowy birch forest, morning mist, national geographic photo",
]
for i, pr in enumerate(prompts):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    img = pipe(prompt=pr, height=args.size, width=args.size, num_inference_steps=args.steps,
               generator=torch.Generator("cuda").manual_seed(42 + i)).images[0]
    torch.cuda.synchronize(); el = time.time() - t
    a = np.asarray(img)
    print(f"[{i}] {el:.1f}s peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB mean={a.mean():.1f} std={a.std():.1f}")
    img.save(f"{args.out}/klein_fp16_{args.size}_{i}.png")
