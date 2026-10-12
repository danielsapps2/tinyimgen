# TinyImGen

High-quality local text-to-image on a **6 GB RTX 2060** (Turing), in under 20 seconds per 1-megapixel image.

| Model | Precision | Steps | 1024² time | Peak VRAM | Strength |
|---|---|---|---|---|---|
| FLUX.2 [klein] 4B | fp16, streamed | 4 | ≈ 9 s | ≈ 3.9 GB | Photorealism, fastest |
| Z-Image Turbo 6B | SVDQuant INT4 (nunchaku), streamed | 8 | ≈ 16 s | ≈ 3.9 GB | Best text rendering & composition |

## Run

```bash
.venv/bin/python server.py           # preview site + live generator, FLUX.2 klein 4B only → http://127.0.0.1:7860
.venv/bin/python server.py --zimage  # also load Z-Image Turbo (exact in-image text, ~16 s/image)
.venv/bin/python make_showcase.py    # re-render the side-by-side showcase (loads both models)
```

Weights are pinned in host RAM.
- klein only (the default): about 23 GB of RAM, about 24 s to load.
- With `--zimage`: about 30 GB of RAM, about 50–60 s to load.

The first image after startup is about 1.5 s slower while the GPU kernels warm up.

## What makes it fit

- **fp16 only.** On this card fp16 matmul measured 23 TFLOPS and bf16 measured 3.6 TFLOPS (bf16 is emulated on Turing).
  - Stock Z-Image in fp16 overflows and produces black images.
  - The nunchaku SVDQuant INT4 build is numerically stable and also about 1.6× faster.
- **Layer streaming** (`tinyimgen/stream.py`).
  - Each transformer and text-encoder block stays in pinned RAM.
  - Block *i+1* is copied over PCIe 3.0 (13 GB/s measured) on a side stream while block *i* computes.
  - The host waits on compute events, so at most about 2 blocks are resident at once.
- **Shared text encoder.** klein and Z-Image ship bit-identical Qwen3-4B weights, so one streamed copy serves both.
  - Prompt encoding takes 0.6 s.
  - The same encoder on the CPU took 12.6 s.
  - The 151k-token embedding table (742 MB) stays on the CPU, and only looked-up rows are shipped to the GPU.

## Layout

- `tinyimgen/engine.py`: loads both pipelines and the shared encoder, and provides `generate()`.
- `tinyimgen/stream.py`: the `LayerStreamer` and the CPU-side embedding.
- `tinyimgen/compat.py`: a shim for the forward signature in nunchaku 1.3.0dev under diffusers 0.41.
- `server.py`: the FastAPI preview site.
  - `/`: the Studio (`static/index.html`), with your generations listed below the generator.
  - `/showcase`: the side-by-side gallery and measurements (`static/showcase.html`).
  - Shared styles and scripts are in `static/style.css` and `static/common.js`.
- `bench_*.py`: the standalone benchmarks used to choose the configuration.

## Environment

- Python 3.12 venv
- torch 2.11+cu128
- diffusers 0.41
- transformers 5.19
- nunchaku 1.3.0.dev20260213 (cu12.8 / torch 2.11 wheel)
- Not in the repo:
  - `.venv` / `.venv3d`, generated output (`out/`) and logs.
  - `third_party/Hunyuan3D-2`: a clone of https://github.com/Tencent-Hunyuan/Hunyuan3D-2 (commit `f8db630`), used only by the local 3D engine.
- In the repo:
  - The MediaPipe and YuNet models (`third_party/models/`).
  - The default clip speeds (`tools3d/clip_rates_default.json`). Speeds saved on the Play page go to `out/clip_rates.json` and override them.

## Site

- **Image Studio** (`/`) generates images, locally (klein, Z-Image) or with a cloud model on fal.ai (`tinyimgen/falgen.py`):
  - Nano Banana Pro ($0.15/image), Nano Banana 2 ($0.08), FLUX.2 [pro] ($0.03/megapixel, references included) and Seedream 4.5 ($0.04, renders at 2K+).
  - Each has an edit endpoint, so references work with all of them.
  - Each image's estimated cost is shown on its card and logged to `out/gallery/fal_costs.jsonl`. The Studio shows today's and all-time spend. Prices are fal's published rates as of 2026-10-12; the fal dashboard has exact charges.
  - The fal key is shared with the 3D tab: `~/.config/tinyimgen/fal_key` or `FAL_KEY`. It accepts up to 4 **references** for klein to edit or combine. A reference can be:
  - a past generation,
  - an upload (drag and drop works),
  - a render of a 3D model.

  Each generation card has **+ Reference** and **Make 3D →** buttons.
- **3D** (`/3d`) builds a model from one source image, with optional back and side views.
  - **Engine:** Tripo H3.1 on fal (the default, paid per model, see below) or the local pipeline (free, all on this GPU).
  - Quality presets: Tripo uses standard geometry, detailed geometry, or detailed geometry plus a detailed texture. Locally they take Draft ≈ 1.5 min, Standard ≈ 3.5–4 min, High ≈ 5.5 min.
  - **Pose: T-pose** has klein re-pose the source first (+25 s). With arms away from the body nothing hides another surface, so hands and sleeves texture without bleeding into clothes, and the model is ready to rig. "Keep as is" uses the source pose.
  - The page shows live stage progress.
  - The viewer rotates and zooms the model (Google model-viewer, loaded from jsDelivr).
  - Downloads: game GLB, texture PNG, and the full-detail GLB (Tripo) or raw shape (local).
  - **Use render as reference** sends the model back to the Studio.
  - **Rig + animations** (on by default, ~6–9 s) adds a humanoid skeleton and six clips; the viewer gets a clip picker. Models built earlier get a **Rig + animate** button. See [Rig and animations](#rig-and-animations).
- **Play** (`/play?id=<job>`): walk a rigged character around in three.js with WASD, using the same locomotion blending a game would.
- 3D jobs run as `tools3d/make3d.py` in `.venv3d` and save to `out/3d/<id>/`. The image server and the job share the card through a cross-process GPU lock (`tinyimgen/gpulock.py`), so Studio requests wait instead of crashing with an out-of-memory error.
- **3D Humans** (`/3d`) and **3D Objects** (`/objects`) are the same creator page in two modes.
  - Humans build characters with the T-pose option and **Rig + animations**.
  - Objects build props, weapons, vehicles and buildings: no T-pose, no skeleton, 1.0 m default size.
  - Each page lists only its own kind (`config.kind` in `job.json`; older jobs count as humans unless rigging found no humanoid).
  - Image Studio cards have **3D human →** and **3D object →**.
- **Prop Editor** (`/prop-editor?id=<object>`; **Make holdable ✋** on 3D Objects and in the Explorer): turns a 3D object into a holdable prop. See [Holdable props](#holdable-props).
- **Explorer** (`/explorer`): everything in `out/`.
  - **Kinds:** images (generations and uploads), 3D humans, 3D objects, and other (motion-capture reference videos, plus the 3D pipeline's intermediate images behind the **Pipeline images** toggle).
  - **Browsing:** search names, prompts and tags; sort by date, name, size or favorites. A detail drawer previews each item (images, animated humans with their clips, videos) and offers rename, tags, favorite, downloads of every file, and links into the Studio, creators and Play.
  - **Select mode** (or Ctrl/Shift-click) applies favorite, move between Humans and Objects, export and delete to many items at once.
  - **Export** writes a .zip of the files plus `manifest.json` (titles, favorites, tags, generation info). Full-detail meshes are optional.
  - **Import** (button, or drop files on the page): images, `.glb`, `.fbx` / `.obj` (converted through Blender), or an exported .zip, which restores the items with their metadata. Imported 3D models get a rendered thumbnail and appear under 3D Objects; move a character to Humans to rig it.
  - **Delete** moves files to `out/trash/<time>/`; nothing is destroyed.
  - User metadata lives in `out/library.json`, and the logic in `tinyimgen/library.py`.
- **Showcase** (`/showcase`): the comparison gallery.

## Image → 3D (`tools3d/`, separate `.venv3d`)

### Tripo H3.1 (default, `tools3d/tripo.py`)

In an October 2026 comparison on the four test characters, Tripo H3.1, Hunyuan 3D v3.1 Pro and Rodin v2.5 (all on fal) were far ahead of the local pipeline on faces, hair and hands. Tripo was the best of the three. It is now the default 3D step:

1. **T-pose** (optional, klein, ~25 s) as before.
2. **Tripo H3.1** (`tripo3d/h3.1/image-to-3d`) on fal builds the textured model from the front image. It returns about 2M triangles with a 2K texture. The `multiview-to-3d` endpoint is used only when left, back and right views are all supplied, because it takes views by position.
3. **Export** (~30 s, mostly CPU, so it doesn't hold the GPU):
   - finds the front on face-detector renders, since Tripo's output faces +X;
   - turns it to +Z, scales it to the requested height and puts the origin at the feet;
   - writes `textured.glb` at full detail.
   - It then decimates a welded copy to the triangle budget and unwraps fresh UVs (xatlas). Decimation runs a fast pre-pass to 10x the budget (`fast-simplification`, ~2 s), then pymeshlab's quadric pass; pymeshlab alone took ~30 s for the same result. Tripo's texture is transferred onto them from the nearest point of the full-detail surface. Decimating with Tripo's own UVs spends the budget holding its texture seams and leaves fingers jagged.
   - The result is `game.glb` plus `albedo.png`. Both GLBs carry smooth vertex normals.

Setup:
- Save a fal API key to `~/.config/tinyimgen/fal_key`, or set `FAL_KEY` before starting the server.
- `fal-client` and `fast-simplification` are installed in `.venv3d`.
- `make3d.py` sets `NUMPY_MADVISE_HUGEPAGE=0` before numpy loads. The image server pins ~21 GB of host RAM, which fragments memory, so numpy's huge-page requests on large arrays stalled for seconds each in failed kernel compaction. That turned a ~1 s texture transfer into 8 minutes.
- Cost is roughly $0.30–0.40 per model at the Standard preset. Check your fal dashboard for exact charges.
- The raw Tripo download is kept as `tripo_raw.glb`.

### Local pipeline (`engine: "local"`)

klein makes the character and its texture views, and Hunyuan3D-2 makes the shape. Measured on the RTX 2060, with the image server running alongside:

1. **Front image.** Generate it with klein in the Studio or through `/api/generate`: full body, A-pose, plain grey background. About 9 s.
2. **Reference views.** Call `/api/generate` with `refs: [front]` and ask for a back view and side views. About 18 s each.
3. **Shape.** About 11 s at 2.6 GB VRAM for multi-view, or about 6.5 s for single view.
   `.venv3d/bin/python tools3d/shape.py OUT --front F --back B --left L --right R`
   Hunyuan3D-2mv takes the four views and 2mini takes the front only. "left" means the camera is on the character's left.
4. **Texture pass 1.** About 30 s: decimate to 30k triangles, unwrap UVs, then project the reference views with occlusion tests and blend.
   `.venv3d/bin/python tools3d/texture.py OUT`
5. **Texture pass 2.** About 4 min for 7 views.
   `.venv3d/bin/python tools3d/refine.py OUT --front-ref <gallery file> --describe "outfit"`
   - Renders the pass-1 model from 7 orbit angles.
   - klein repaints each render using the original front image as a second reference.
   - Each repaint is pixel-aligned with the mesh (silhouette IoU 0.86–0.96), and the result is baked again.
6. **Game-ready export.**
   `blender -b -P <gamedev-tooling>/scripts/blender_gameready.py -- OUT/textured.glb OUT/game.glb --height 1.8`
   Use the `~/Tools/blender-build` Blender: the distro build lacks numpy, which the glTF importer needs.

`.venv3d` pins transformers 4.57 and diffusers 0.35, because Hunyuan3D's DINOv2 weights don't load under transformers 5. It also has mediapipe 1.1 for the face and hand landmarks.

### Mesh cleanup

Before decimating to the triangle budget (10k / 30k / 60k), `Model` repairs the raw Hunyuan3D mesh: it removes floaters and non-manifold edges and vertices, and closes small holes. It then applies two passes of HC Laplacian smoothing, which removes the generator's lumpy banding on faces and cloth but keeps features like noses, pockets and laces. The "Raw shape GLB" download is the untouched generator output.

### Face and hand sculpt (`tools3d/sculpt.py`)

Hunyuan3D leaves faces round and soft (no lip line, shallow eyes, the nose a few pixels off where the image has it) and hands lumpy. Before decimation, the source image's dense MediaPipe landmarks (`tools3d/landmarks.py`: 478 face points, 21 per hand, models in `third_party/models/*.task`) repair them:

- **Face depth:** the front surface inside the face oval is moved along the view axis onto a thin-plate-spline surface through the landmarks' depth. x and y stay put, so the front projection paints every feature exactly onto its geometry, and side views get a real nose, eye sockets and lips.
  - MediaPipe's depth scale matches the meshes (fitted slope 0.99–1.07). A fitted plane keeps the mesh's own head orientation.
  - A harmonic correction from the oval boundary inward makes the new face meet the mesh seamlessly, with no mask-like rim.
- **Jaw outline:** generated faces are often a little shorter than the image's.
  - The geometry keeps the mesh's jaw, since pulling the neck forward opens a crease.
  - The texture's jaw and chin contour is warped onto the mesh's own jaw edge (`align_outline`, a thin-plate-spline warp that holds the eyes, nose and mouth fixed).
  - Generated scarves and collars often ride up right under the chin, where the image shows a band of neck. Face parsing (SegFormer `jonathandinu/face-parsing`, from the Hugging Face cache) finds the image's neck-to-cloth edge, and the warp squeezes the neck band into the slit under the chin. Otherwise the band paints a stubble-like stripe on the collar.
- **Smoothing:** the face and hands are subdivided first (the generator's 5 mm edges are too coarse for lips), then smoothed with Taubin passes. Their vertices get a higher importance weight, so the decimator spends more of the triangle budget on them.
- **Limits:** a hand that rests on a pouch or the thigh comes out of the generator fused to it, and smoothing can't separate the two. Use **Pose: T-pose** for clean hands.

### Bald base, head and hand detail, hair layer (`tools3d/parts.py`)

In a full-body image the face is ~100 px wide and a hand ~60 px, and Hunyuan3D spends ~30 voxels on a face. The 3D tab therefore builds the head and hands at close-up resolution, and the hair as its own mesh. The sculpt above is now the fallback when a part can't be rebuilt.

- **Bald base:** after the T-pose, a minimal klein edit removes the hair and everything worn on the head and keeps the rest of the image pixel-aligned. Body, views and texture are built from it, which gives a clean scalp and ears. If the edit moves the character or loses the face, the job keeps the hair.
- **Head:** klein repaints the head crop and the back-of-head crop as sharp 768 px close-ups. Each is registered back onto its crop (SIFT + RANSAC, or ECC when the crop is too blurry for features) and checked for colour changes. Hunyuan3D builds the head from them (10 steps, ~15× the face resolution). It is placed in the body's model space through the pixel mapping, with depth from the body's own face.
- **Hands:** each hand is rebuilt from MediaPipe's 21 joints as a smooth signed-distance field: palm slab, tapered fingers blended into the palm but never into each other, fingers fanned apart when closer than their thickness, and anatomical finger lengths. The hand is moved onto the mesh's own hand (where the arm ends) and its thickness is fitted to the source silhouette. A hand that doesn't fit, such as a gauntlet, is skipped. A rebuilt hand gets a clean skin albedo (the face's skin tone, or its own colour for gloves).
- **Merge:** the body surface away from the parts and the parts' surfaces are resampled into one watertight mesh with screened Poisson (depth 10). Transition bands mix both sample sets.
- **Hair layer:** the hair (and any headwear) is whatever differs between the haired and bald images around the head. klein redraws the haired head as a close-up, plus the same hair added to the bald back view. Hunyuan3D builds that head, and its surface outside the bald body becomes the hair mesh, closed over small gaps and held off the scalp. Its texture comes from the haired front, the back close-up and two side repaints (a side whose repaint turns the face borrows the other side, mirrored). The GLB carries `body` and `hair` as separate meshes.
- **Prompts never name things the character might not have:** naming "goggles" or "hair" makes klein add them. Hair styling (sculpted locks, no loose strands) is only asked for when face parsing finds visible hair.
- **Skin:** `skin_clean` evens out skin the source image doesn't paint itself (the sides of the face, the scalp, the neck) toward the face's own skin colour. Face parsing on the source decides what is skin, so a tan scarf isn't touched.

### How the texture bake avoids the usual artifacts (`tools3d/texture.py`)

- **Faces and hands:** the source image alone paints the face oval and the hands wherever it sees them. `sculpt.ownership` builds that mask from the landmarks, and `bake(own=...)` keeps every other view out of it. Averaging the face with side repaints used to double the eyes and mouth. `face_prior` also boosts the source inside the face found by YuNet (`third_party/models/face_detection_yunet_2023mar.onnx`).
- **Turned repaints:** klein sometimes ignores the render's pose and turns the character to face the camera, which paints a second face on the cheek or the back of the head. `repaint_check` compares the repaint's face with the render's. A failed repaint is retried once with a stronger prompt. If it still fails, the misplaced face is cut out, and the rest of its head (hair, ears) counts at half weight. If the mirror-side view passed, its mirrored repaint also paints this side's head (`mirrored_head`), lined up on the projected head centre.
- **Head trust on its own:** klein often redraws an arm somewhere else while the head still lines up. `repaint_layers` judges the head by the silhouette fit in a window around it, separately from the whole-body fit.
- **Hand skin:** hand texels the source doesn't see (palms, finger sides) can pick up the pouch, sleeve or thigh beside them. `skin_fix` replaces colours far outside the source hand's skin distribution with the nearest source-painted hand colour.
- **Eyes in the sockets:** on a sculpted mesh this holds by construction. Without face landmarks, `Model.align_face` falls back to running the face detector on a lit render of the bare mesh and warping the source's 5 landmarks onto it.
- **Fit:** the mesh is matched to the cut-out with a separate horizontal stretch, because Hunyuan meshes come out about 5% wider or narrower than the image.
- **Bleeding between surfaces:** near an occlusion edge (hand over thigh, arm over torso) a view gets little say, and another view or the fill covers that strip. Cut-out edges are cleaned so white background is never sampled.
- **klein drift:** repaints come back 4–14 px off the render they were painted from. `align_to_render` undoes the large-scale drift with smoothed optical flow (per-pixel flow tears fingers). Repaints whose outline still doesn't match the mesh get little weight.
- **Two-band blending:** broad colour is averaged over every view that sees a texel, even at shallow angles, so skin tones don't patch. Fine detail comes only from well-facing views.
- **Back view check:** a generated back view must match the mirrored front outline (≥ 0.82; good ones score 0.91–0.93). Otherwise it is regenerated, up to twice. klein sometimes drops T-pose arms in back views.

The Hunyuan3D license doesn't cover the EU, UK or South Korea.

## Holdable props

Every rigged human carries two hand sockets, `RightHand_Socket` and `LeftHand_Socket`, as children of the hand bones (`rig.hand_sockets`). A socket sits where a handle lies in the closed fist: under the proximal finger bones, a little toward the palm (not mid-palm). Its axes:
- +Z along the fingers;
- +Y out of the thumb side of the fist, where a sword's blade points;
- +X = Y × Z.

In the **Prop Editor** you orbit the object, then:
- **Grip:** click to place the grip. The marker snaps to the middle of the handle under the click (halfway between where the view ray enters and leaves the part).
- **Second grip:** place one for two-handed props.
- **Direction:** set where the prop points: click its tip, use **Auto** (the long axis, away from the grip) or **Flip**. Then roll it around that direction.
- **Hold:** **Upright** points along the socket's +Y (swords, axes, tools, torches). **Forward** points along +Z (guns, wands).
- **Size:** set the prop's length in metres.
- **Preview:** one of your rigged humans holds it live, in either hand, playing any clip (or **Bind pose** for no animation). **Focus hand** and **Whole body** frame the camera. Two-handed props bring it in front ("ready pose") and the other arm reaches the second grip with IK. If the second grip is out of that arm's reach, the prop is pulled toward that shoulder by the shortfall, so both hands stay on it.
- **Fingers:** per-hand sliders curl each finger (knuckle, middle and end joint separately). **Auto wrap** closes each joint until it touches the prop (raycasts along the finger segments plus the finger's thickness). **Open**, **Relaxed** and **Fist** are presets.
- **Arms:** **Animated** keeps the clip's arm swing. **Static** locks the holding hand in one carry pose relative to the UpperChest, held through every clip. The carry pose places the hand and the grip places the prop in the hand, so the two edit independently: moving the grip never moves the hand, and moving the carry never changes the grip. **Reset to ready pose** starts it from the default, placed from the character's shoulders and arm length so the elbows bend about 90°.
  - The arm IK (`props.js reachSocket`) treats the elbow as a hinge and picks the elbow swivel that keeps the wrist straightest, with the elbow down and out from the body. The forearm takes 60% of the wrist's twist. The support hand may also turn around the handle to find its most natural wrist.
  - A carry pose dragged further than about 92% of arm length is pulled back toward the shoulders, so the arms never lock straight.
- **Undo / Redo** (Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y): every edit is one step, and a whole slider or gizmo drag counts as one. **Reset** starts over (no grips, default hold and fingers), and **Revert to saved** goes back to the last save; both can be undone.
- **Gizmo:** drag the **grip** (where the prop sits in the hand) or the **carry** pose with a move / rotate / scale gizmo (W / E / R). The gizmo sits on the handle (the primary grip), not on the model's origin, so rotating and scaling pivot around the hand. To scale, grab any scale handle and drag up (bigger) or down (smaller). The size slider under **Size** does the same.

**Save** writes, in the object's folder:
- `prop.glb`: the model under `Prop → Model` (the scale lives on `Model`), with `Grip_Primary` and `Grip_Secondary` nodes beside it. Each grip node has the hand socket's axes.
- `grip.json`: hands, hold, scale, which hand, grip positions and rotations, finger curls per hand, arms (animated/static) and the carry pose, plus the editor's inputs (`tinyimgen/props.py`).

To hold a prop in a game, make the grip coincide with the socket: `prop_root.world = hand_socket.world × inverse(Grip_Primary.local)`.
- **three.js:** `static/props.js` provides `attachProp(character, prop, {hand})`. Each frame after the animation update, call `carryPose` (static arms) or `readyPose` (optional), then `solveSupportHand` (two-handed), then `setFingers` for each hand with the saved curls.
- **Unity / Unreal / Godot:** the socket and grip nodes import as transforms or bones. Parent the prop to the socket with the inverse grip offset.
- The Play page's **Hold** menu uses exactly this path.

The locomotion clips are unarmed: with animated arms a one-handed prop swings with the hand, and two-handed props use the ready-pose overlay. Static arms override the arm swing with the saved carry pose.

## Rig and animations

`tools3d/rig.py` turns any humanoid `game.glb` (Tripo or local) into `rigged.glb`: the same mesh, skinned to a 52-joint skeleton (22 body joints plus 3 per finger), with six looping clips. With the Blender build in `~/Tools/blender-build` (or `TINYIMGEN_BLENDER`) it also writes `rigged.fbx` (~1.5 s).

```bash
.venv3d/bin/python tools3d/rig.py out/3d/<job>          # game.glb -> rigged.glb, rigged.fbx, rig.json
.venv3d/bin/python tools3d/rig.py --job out/3d/<job>    # same, and records it in job.json (what the 3D tab runs)
```

### Skeleton

- **Joint names** follow Unity's and Godot's humanoid profile:
  - Hips, Spine, Chest, UpperChest, Neck, Head
  - Left/Right: Shoulder, UpperArm, LowerArm, Hand
  - Left/Right: UpperLeg, LowerLeg, Foot, Toes

  Godot's `SkeletonProfileHumanoid`, Unity's Humanoid avatar and Unreal's IK Retargeter auto-map them, so other humanoid animations (Mixamo and so on) retarget onto these characters, and these clips retarget onto other rigs.
  - Fingers: Left/Right Thumb, Index, Middle, Ring, Little, each Proximal, Intermediate, Distal (Unity/Godot names). Each finger node's glTF extras carry its `curl_axis` and `curl_max_deg`; `tools3d/anim.py set_fingers` and `props.js setFingers` curl them.
- **Finger fitting:** MediaPipe Hands runs on a close-up render of each hand looking down its flat side, and each landmark's depth is the middle of the finger there. Hands it can't read (gauntlets, fused fists) fall back to a geometric fit from the hand's shape, which curls less convincingly. Finger skin weights are spread along the mesh surface from each bone, so neighbouring fingers don't pull each other.
- **Fitting:** MediaPipe Pose (`third_party/models/pose_landmarker_heavy.task`) runs on a lit front render of the mesh. Each joint's depth is the middle of the limb a ray hits there. The feet (ankle, ball, toe tip) are measured on the mesh. If no humanoid is found, the stage is skipped and the static model is still delivered.
- **Bind pose:** every joint has identity rotation (as VRM does). A clip is then just per-joint rotations, and engines retarget from the joint positions.

### Skinning

Skinning is voxel geodesic, with four influences per vertex:
- The mesh is voxelised solid (~1.1 cm voxels). Each bone's distance grows through the inside of the body only, so a sleeve doesn't bind to the torso across the gap.
- Each vertex then blends its nearest bones by inverse distance, and the weights are smoothed over the surface.

Contacts:
- Generated meshes fuse a hand resting on the thigh, a pouch or an apron into one surface. Near a hand or forearm, anything outside the limb's own radius that a body bone reaches as easily goes to the body.
- The surface is then split along that contact (`split_contacts`), so the arm separates instead of stretching a membrane.
- This only happens in arms-down poses, and `rig.json` / the 3D tab say so. **Pose: T-pose** gives the cleanest rig.

### Clips: motion-captured from generated video (`tools3d/mocap.py`, `tools3d/retarget.py`)

All clips are in place (no root motion), loop seamlessly and run at 30 fps. They are motion capture of a generated
reference video, retargeted onto each character:

| Clip | Speed (1.8 m character) | Reference |
|---|---|---|
| Idle | 0 | 5 s of relaxed standing: breathing, weight shift, glances |
| Walk | ~0.9 m/s | three-quarter tracking shot |
| Run | ~2.1 m/s | three-quarter tracking shot (a jog: ~0.78 s cycle) |
| WalkBackward | ~0.75 m/s | static camera, profile: he crosses the frame backwards |
| StrafeLeft | ~0.3 m/s | front tracking shot, side-step shuffle |
| StrafeRight | ~0.3 m/s | StrafeLeft mirrored |

**Capture** (once per clip; the result is the motion file `tools3d/motions/<Clip>.json`, shared by every character):
1. **Video:** MiniMax H3 Max text-to-video on fal films a man in fitted clothes doing the move. It costs about $0.25 per 8 s clip, and the prompts are in `mocap.CLIPS`.
2. **Cycle:** MediaPipe Pose (local) measures the hips' vertical bob to find the gait cycle. Limb tracks don't work for this: generated video swaps and morphs the legs.
3. **3D pose:** SAM 3D Body (`fal-ai/sam-3/3d-body`, $0.02 per frame) lifts frames to 70 3D keypoints.
   - A sparse pass over the whole video finds the longest stretch of two or more cycles that folds into one cycle within 1.5 cm.
   - A dense pass at every second frame then covers that stretch.
4. **Solve** (`mocap.py solve`):
   - **Floor:** the floor's up direction comes from double-support frames.
   - **Fit:** the stretch is phase-folded and Fourier-fit into one smooth looping cycle.
   - **Contacts:** a foot is planted while it moves against the travel direction; for the run, where the feet leave the floor, contacts come from foot height and speed.
   - **Root motion:** the pelvis's bob, sway and surge are solved by least squares from the planted feet, with gravity during the run's flight phase.
   - **Smoothing:** walk, run and backward walk are symmetrized: each pose is averaged with the mirror of the pose half a cycle later. This halves pose noise and evens out SAM's left/right bias. The pelvis path keeps its first four harmonics only. The un-smoothed pelvis followed the noisiest foot keypoint, which jolted the hips at every touchdown and showed as the run's legs glitching at the loop point.
   - **Timing:** generated video plays at roughly half speed, so each cycle is retimed to a natural cadence.
   - **Direction guard:** if the planted feet are the high ones, the video travels the wrong way, and solve refuses the clip. Asked for a backward walk, both tracking-shot attempts filmed a forward walk; the static-camera take worked.

```bash
.venv3d/bin/python tools3d/mocap.py all Run            # video -> track -> solve (about $1.5-2 per clip)
.venv3d/bin/python tools3d/mocap.py mirror StrafeLeft StrafeRight
```

**Retargeting** (`retarget.py`, every rig):
- **Rotations:** joint rotations come from bone directions in the keypoints.
  - The hip line sets the pelvis, the shoulder line the chest and the ears the head. The bend plane sets elbows and knees, and the knuckle line sets forearm and hand twist.
  - The character keeps its own bone lengths.
  - SAM's view bias is removed by zeroing the cycle mean of head yaw and torso roll; a runner's forward lean is kept.
- **Arms:** arms keep the character's own clearance angle, so bulky hips and pouches don't swallow the hands.
- **Feet:**
  - Each touchdown lands where the reference's did, scaled by leg length.
  - While a foot is planted, it keeps its touchdown heading, and its contact point (heel, sole, then ball with the toes flat) moves exactly with the ground.
  - The ankle is a hinge: the foot points where the shin does (plus 3° of toe-out), and only its up/down pitch comes from the capture. A foot pointing down at toe-off has no reliable heading of its own and used to spin.
  - Leg twist comes from the knee's hinge axis, which stays perpendicular to the shin at any bend. The knee's forward direction lines up with the shin in a deep run heel kick, and the leg flipped.
  - Feet point ahead: the reference person's knee/foot turn-out is removed, and each mesh's own rest toe-out (generated T-poses have the feet turned out 10–20°; `foot_yaw_deg` in `rig.json`) is turned back by the foot bones. Planted feet point straight plus 3°.
  - Two-bone IK reaches it. The lock's correction (position, plus the planted foot's held heading and flat toes) blends smoothly through each swing, so legs don't jerk at touchdown or lift-off.
  - A swinging foot keeps the reference's clearance mid-swing, easing to the floor next to a contact; the floor check hinges the toes at the ball.
- **Fallback:** without the motion files, `anim.py` generates the clips procedurally (the earlier version).

Each glTF animation carries `extras`:
- `speed` (m/s), `direction` (local: +X = left, +Z = forward), `stride` (m per cycle), `duty`;
- `sync`: normalised time at which the left foot's stance begins (0 for the captured clips). Blend clips in step with `time = (phase + sync) × duration`;
- `source` (`mocap:<Clip>` or procedural).

**Speeds:** the Play page has a **Clip speeds** panel: one slider per clip, giving tempo relative to the capture (0.5–3×; the strafes share one).
- Sliders apply live. A faster clip keeps its stride and takes quicker steps, so the feet stay planted.
- **Save + re-rig** stores the rates in `out/clip_rates.json` (the default for every rig from then on, also `GET`/`POST /api/clip_rates`) and re-rigs the model, so the GLB, FBX and the `speed` extras carry them.
- `locomotion.js` exposes `rate(name)` / `setRate(name, r)` for the same thing at runtime.

**Checking:** `tools3d/rig_check.py out/3d/_rig/<name>` prints each clip's foot slide. Captured clips measure 0.000–0.006 m/s on all ten test characters, including at 2.5× rate. `tools3d/clip_strip.py out/3d/<job> Run` renders a filmstrip.

### Using it in a game

- **three.js:** use `static/locomotion.js`.
  ```js
  const loco = new Locomotion(gltf);
  const v = loco.update(dt, { x: left, z: forward }, running);
  ```
  It blends the six clips by local velocity and drives them from one shared gait phase (distance travelled ÷ stride). The feet stay planted at any speed, and diagonals (e.g. Walk + StrafeLeft) stay in step. `/play` is a working example with strafe and free-turn controls.
- **Godot 4:** import `rigged.glb`.
  - The skeleton maps to `SkeletonProfileHumanoid` in the import dock's retarget section.
  - Set each animation's loop mode to Linear on import.
  - Drive an `AnimationTree` 2D blend space with the clips' `speed` values.
- **Unity:** import `rigged.fbx`, then set Rig → Humanoid. The avatar auto-maps the bone names.
  - Tick *Loop Time* on each clip.
  - The clips are in place: move the character from code at the clip `speed` (see `rig.json`), or blend them in a 2D Freeform Directional blend tree.
  - `rigged.glb` also works through the glTFast package.
- **Unreal 5:** import `rigged.fbx` (or the GLB via Interchange) as a skeletal mesh with animations.

**Checking changes:** `.venv3d/bin/python tools3d/rig_check.py out/3d/_rig/<name> …` rigs, prints foot slide per clip and writes `anim_sheet.png`. `out/3d/_rig/` holds the test set:
- six Tripo characters: smith, ranger, guns, scav, soldier, mechanic;
- the local engine's body + hair split: localhair;
- two arms-down meshes with fused hands: smithdown, scavdown;
- a broken mesh: sword.

The FBX was checked against the GLB by re-importing both in Blender: joints agree within 0.1 mm across clips. The GLB passes the Khronos glTF validator with no errors or warnings.
