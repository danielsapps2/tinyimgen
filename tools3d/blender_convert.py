"""Convert an FBX or OBJ to GLB inside Blender (Explorer import):

    blender -b -P tools3d/blender_convert.py -- in.fbx out.glb
"""
import os
import sys

import bpy

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(src)[1].lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=src)
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=src)
else:
    raise SystemExit(f"unsupported {ext}")
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_animations=True)
print("CONVERT_OK", dst)
