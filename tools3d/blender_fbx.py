"""Convert a rigged GLB (rig.py) to FBX for engines that prefer it (Unity's Humanoid import, Unreal). Runs inside
Blender:

    blender -b -P tools3d/blender_fbx.py -- out/3d/<job>/rigged.glb out/3d/<job>/rigged.fbx

Every clip becomes its own FBX take, named like the GLB animation. Leaf bones are not added and the scale is 1 unit =
1 m, so Unity reads the bones' names and lengths as rig.py wrote them.
"""
import sys

import bpy

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30       # rig.py bakes at 30 fps: keys land on whole frames, nothing is resampled
bpy.ops.import_scene.gltf(filepath=src, bone_heuristic="TEMPERANCE", import_shading="SMOOTH")
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
# glTF import parks each clip on its own NLA track; the FBX exporter bakes each action as a take
for a in bpy.data.actions:
    a.use_fake_user = True
bpy.ops.export_scene.fbx(filepath=dst, use_selection=False, object_types={"ARMATURE", "MESH"}, add_leaf_bones=False,
                         bake_anim=True, bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
                         bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0, path_mode="COPY",
                         embed_textures=True, apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS",
                         armature_nodetype="NULL", primary_bone_axis="Y", secondary_bone_axis="X")
print("FBX_OK", dst, [a.name for a in bpy.data.actions])
