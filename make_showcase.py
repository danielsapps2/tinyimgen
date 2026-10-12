"""Render the showcase gallery: every prompt through both models, with timings."""
import json
import os

from tinyimgen.engine import Engine

OUT = "out/showcase"
PROMPTS = [
    ("Portrait", 1024, 1024, "Candid portrait of a young woman with freckles laughing in a sunflower field, late afternoon backlight, 85mm lens, film grain"),
    ("Typography", 832, 1216, "A retro 1970s travel poster for Mars that says \"VISIT OLYMPUS MONS\" in bold orange letters, rocket and red dunes, screen-printed texture"),
    ("Cinematic", 1216, 832, "Cinematic still of a lone astronaut walking through a flooded neon Tokyo alley at night, rain, volumetric light, anamorphic lens flare"),
    ("Anime", 832, 1216, "Anime key visual of a girl with silver hair on a rooftop at dusk, school uniform, wind blowing cherry blossoms, city skyline, detailed clouds"),
    ("Product", 1024, 1024, "Studio product photo of a matte black mechanical wristwatch on wet slate, water droplets, softbox lighting, ultra sharp"),
    ("Interior", 1216, 832, "Scandinavian cabin living room with a crackling fireplace, big windows showing snowy mountains, warm lamps, architectural photography"),
    ("Food", 1024, 1024, "Overhead shot of a steaming bowl of tonkotsu ramen with chashu, soft egg, scallions and nori on a dark wooden table, food photography"),
    ("Fantasy", 1216, 832, "Epic fantasy concept art of a colossal stone dragon sleeping around a mountain village, morning fog, painterly, matte painting"),
    ("Macro", 1024, 1024, "Extreme macro photo of a jumping spider with iridescent green fangs on a leaf, dew drops, shallow depth of field"),
    ("Illustration", 1024, 1024, "Cute isometric 3D illustration of a tiny coffee shop on a floating island, pastel colors, soft shadows, clay render"),
    ("Signage", 1024, 1024, "A chalkboard menu outside a bakery reading \"TODAY: Croissant $3, Sourdough $6, Cinnamon Roll $4\" with hand-drawn wheat doodles"),
    ("Wildlife", 1216, 832, "An owl in flight with wings spread over a misty pine forest at dawn, wildlife photography, motion detail in feathers"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    done = {}
    if os.path.exists(f"{OUT}/showcase.json"):  # resume an interrupted run
        with open(f"{OUT}/showcase.json") as f:
            done = {it["id"]: it for it in json.load(f)["items"]}
    eng = Engine(("klein", "zimage"))  # the showcase is a side-by-side comparison
    items = []
    for i, (tag, w, h, prompt) in enumerate(PROMPTS):
        entry = done.get(i) or {"id": i, "tag": tag, "prompt": prompt, "width": w, "height": h, "results": {}}
        for model in ("klein", "zimage"):
            if model in entry["results"]:
                continue
            img, meta = eng.generate(model, prompt, w, h, seed=1000 + i)
            name = f"{i:02d}_{model}.webp"
            img.save(f"{OUT}/{name}", quality=92)
            meta["file"] = name
            entry["results"][model] = meta
            print(tag, model, meta["seconds"], "s", meta["peak_vram_gb"], "GB", flush=True)
        items.append(entry)
        with open(f"{OUT}/showcase.json", "w") as f:
            json.dump({"load_seconds": round(eng.load_seconds, 1), "items": items}, f, indent=1)


if __name__ == "__main__":
    main()
