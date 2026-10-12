"""Headless Blender turntable contact sheet for a GLB.

    blender -b -P tools3d/blender_preview.py -- in.glb out.png [--solid]

Renders front, 3/4, right side, back, left side with an orthographic camera
(EEVEE with the mesh's texture; --solid uses clay shading to judge geometry).
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
solid = "--solid" in argv

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

lo = Vector((1e9, 1e9, 1e9)); hi = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
center = (lo + hi) / 2
size = max(hi - lo)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH" if solid else "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
    e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
if solid:
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.75, 0.72, 0.68)
else:
    world = bpy.data.worlds.new("w"); scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.9, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
scene.render.resolution_x, scene.render.resolution_y = 512, 768
scene.render.film_transparent = False

cam_data = bpy.data.cameras.new("cam"); cam_data.type = "ORTHO"; cam_data.ortho_scale = size * 1.1
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam

# glTF +Z forward is Blender -Y; the character faces -Y, so "front" camera sits at -Y.
angles = {"front": 0, "three_quarter": 35, "right": 90, "back": 180, "left": 270}
paths = []
for name, deg in angles.items():
    a = math.radians(deg)
    d = size * 3
    # rotate the camera around Z, starting in front (-Y); +90 = character's right side
    pos = center + Vector((-math.sin(a) * d, -math.cos(a) * d, 0))
    cam.location = pos
    cam.rotation_euler = (center - pos).to_track_quat("-Z", "Y").to_euler()
    p = f"{out}.{name}.png"
    scene.render.filepath = p
    bpy.ops.render.render(write_still=True)
    paths.append(p)

# stitch into one strip
imgs = [bpy.data.images.load(p) for p in paths]
w, h = imgs[0].size
strip = bpy.data.images.new("strip", w * len(imgs), h)
px = [0.0] * (w * len(imgs) * h * 4)
for i, im in enumerate(imgs):
    src_px = list(im.pixels)
    for y in range(h):
        row = src_px[y * w * 4:(y + 1) * w * 4]
        start = (y * w * len(imgs) + i * w) * 4
        px[start:start + w * 4] = row
strip.pixels = px
strip.filepath_raw = out; strip.file_format = "PNG"; strip.save()
print("wrote", out)
