"""Bake reference views onto a mesh: decimate, UV-unwrap, project with occlusion, blend, fill, export GLB.

    .venv3d/bin/python tools3d/texture.py OUTDIR [--mesh shape_mv.glb] [--faces 30000] [--tex 2048]

Pass 1 uses OUTDIR/views/{front,back,left,right}.png (RGBA cutouts; "left" = camera on the
character's left, Hunyuan3D convention). tools3d/refine.py builds pass 2 on top of this module.
Mesh convention (Hunyuan3D / glTF): +Y up, character faces +Z.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinyimgen.device import DEVICE as dev, empty_cache  # noqa: E402,F401  (before torch: MPS fallback switch)

import cv2
import numpy as np
import pymeshlab
import torch
import trimesh
import xatlas
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

UP = np.array([0, 1, 0], np.float32)
YUNET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "models",
                     "face_detection_yunet_2023mar.onnx")


def camera(azimuth_deg):
    """Orthographic camera orbiting the character. 0 = front, +90 = character's left, 180 = back."""
    a = math.radians(azimuth_deg)
    cam = np.array([math.sin(a), 0, math.cos(a)], np.float32)   # direction from character to camera
    right = np.cross(UP, cam).astype(np.float32)                 # image-right axis
    return cam, right


PASS1 = {"front": 0, "left": 90, "back": 180, "right": 270}
OCCLUSION_JUMP = 0.012   # depth step (fraction of height) that counts as one surface in front of another
SHARPEN = 1.0            # detail-band blend exponent relative to the best view per texel (see bake)
FLOW_SMOOTH = 15.0       # px; scale of the drift correction for klein repaints
LOW_SIGMA = 8.0          # px; below this scale colour comes from the best view, above it from all views


def rasterize(xy, z, faces, H, W):
    """Z-buffered triangle rasterizer in pure torch. xy: (N,2) pixel coords, z: (N,) depth (smaller = nearer).
    Returns per-pixel face id (-1 = empty), covered pixel coords, and their barycentrics."""
    tri = xy[faces]
    zt = z[faces]
    mn = tri.min(1).values.floor()
    ext = (tri.max(1).values.ceil() - mn).max(1).values
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    den = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
    ok = den.abs() > 1e-9
    zbuf = torch.full((H * W,), float("inf"), device=dev)
    fid = torch.full((H * W,), -1, dtype=torch.long, device=dev)

    def bary_of(p, f):
        a, b, c, d = v0[f], v1[f], v2[f], den[f]
        w0 = ((b[..., 1] - c[..., 1]) * (p[..., 0] - c[..., 0]) + (c[..., 0] - b[..., 0]) * (p[..., 1] - c[..., 1])) / d
        w1 = ((c[..., 1] - a[..., 1]) * (p[..., 0] - c[..., 0]) + (a[..., 0] - c[..., 0]) * (p[..., 1] - c[..., 1])) / d
        return w0, w1, 1 - w0 - w1

    def candidates(sel, K):
        off = torch.stack(torch.meshgrid(torch.arange(K, device=dev), torch.arange(K, device=dev), indexing="xy"), -1).reshape(-1, 2)
        step = max(1, int(6e6 // (K * K)))  # bounded scratch so we coexist with the image server
        for i in range(0, len(sel), step):
            f = sel[i:i + step]
            p = mn[f, None, :] + off[None].float() + 0.5
            w0, w1, w2 = bary_of(p, f[:, None])
            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
            inside &= (p[..., 0] >= 0) & (p[..., 0] < W) & (p[..., 1] >= 0) & (p[..., 1] < H)
            depth = w0 * zt[f, None, 0] + w1 * zt[f, None, 1] + w2 * zt[f, None, 2]
            idx = p[..., 1].long().clamp(0, H - 1) * W + p[..., 0].long().clamp(0, W - 1)
            yield f, inside, depth, idx

    passes, done = [], torch.zeros(len(faces), dtype=torch.bool, device=dev)
    for K in (4, 16, 64, 256, 1024, 4096):
        sel = torch.nonzero(ok & ~done & (ext < K)).squeeze(1)
        done |= ok & (ext < K)
        if len(sel):
            passes.append((sel, K))
    for sel, K in passes:
        for f, inside, depth, idx in candidates(sel, K):
            zbuf.scatter_reduce_(0, idx[inside], depth[inside], "amin")
    for sel, K in passes:
        for f, inside, depth, idx in candidates(sel, K):
            win = inside & (depth <= zbuf[idx])
            fid[idx[win]] = f[:, None].expand_as(idx)[win]
    fid = fid.view(H, W)
    ys, xs = torch.nonzero(fid >= 0, as_tuple=True)
    p = torch.stack([xs.float() + 0.5, ys.float() + 0.5], -1)
    bary = torch.stack(bary_of(p, fid[ys, xs]), -1)
    return fid, (ys, xs), bary


class Model:
    """Decimated, UV-unwrapped mesh plus per-texel surface points; cached so every pass shares one UV layout."""

    def __init__(self, out, mesh_name="shape_mv.glb", faces=30000, tex=2048, front=None, replaced=None,
                 cache_name=None, raw=False, occluder=None):
        """front: the source cut-out (float RGBA). When given, the face and hands are re-sculpted from it before
        decimation (sculpt.py); later passes can load the cached result without it. replaced: parts the mesh already
        got from close-ups (parts.py), which sculpt leaves alone and the UV layout gives more texels. raw: no hole
        closing or smoothing (an open shell such as the hair layer). occluder: (V, F) of other geometry that hides
        this mesh from the views (the body behind the hair)."""
        cache = os.path.join(out, cache_name or f"uvmesh_{faces}.npz")
        self.occ = None if occluder is None else tuple(torch.tensor(np.asarray(x), device=dev) for x in
                                                       (np.asarray(occluder[0], np.float32),
                                                        np.asarray(occluder[1], np.int64)))
        self.sculpt_info = {}
        d = np.load(cache) if os.path.exists(cache) else None
        if d is not None and front is not None and "sculpt" not in d:
            d = None   # cached before sculpting existed
        if d is not None:
            V, F, N, uv = d["V"], d["F"], d["N"], d["uv"]
            if "sculpt" in d:
                self.sculpt_info = json.loads(str(d["sculpt"]))
        else:
            mesh = trimesh.load(os.path.join(out, mesh_name), force="mesh")
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(3))  # floaters
            # repair + smooth at full resolution, before decimating: the triangle budget is unchanged. HC Laplacian
            # removes the generator's lumpy banding (faces, cloth) but keeps noses, pockets and laces; Taubin barely
            # changed anything at this density.
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            if not raw:
                ms.meshing_close_holes(maxholesize=200)
                for _ in range(2):
                    ms.apply_coord_hc_laplacian_smoothing()
            weighted = False
            if front is not None:
                from sculpt import sculpt
                m = ms.current_mesh()
                V0, F0 = m.vertex_matrix(), m.face_matrix()
                V0, F0, importance, self.sculpt_info = sculpt(
                    V0, F0, front, lambda V, F: Model.fit_geometry(V, F, front[..., 3] > 0.5), replaced)
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(V0, F0, v_scalar_array=importance))
                weighted = True   # decimation spends more triangles where importance is high (face, hands)
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces, preservenormal=True,
                                                         preserveboundary=True, optimalplacement=True, qualitythr=0.4,
                                                         qualityweight=weighted)
            ms.meshing_repair_non_manifold_edges()  # edge collapses can occasionally pinch the surface
            ms.meshing_repair_non_manifold_vertices()
            m = ms.current_mesh()
            mesh = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)
            vmap, idx, uv = xatlas.parametrize(uv_boosted(mesh.vertices, self.sculpt_info, replaced), mesh.faces)
            V = mesh.vertices[vmap].astype(np.float32)
            F = idx.astype(np.int64)
            N = mesh.vertex_normals[vmap].astype(np.float32)
            np.savez(cache, V=V, F=F, N=N, uv=uv, **({"sculpt": json.dumps(self.sculpt_info)} if weighted else {}))
        self.V, self.F, self.N, self.uv, self.T = V, F, N, uv.astype(np.float32), tex
        self.Vt, self.Ft, self.Nt = (torch.tensor(x, device=dev) for x in (V, F, N))
        self.height = float(V[:, 1].max() - V[:, 1].min())
        uvp = torch.tensor(np.stack([uv[:, 0] * tex, (1 - uv[:, 1]) * tex], -1), device=dev, dtype=torch.float32)
        self.uvp = uvp
        fid, (self.ty, self.tx), bary = rasterize(uvp, torch.zeros(len(uv), device=dev), self.Ft, tex, tex)
        fv = self.Ft[fid[self.ty, self.tx]]
        self.P = (bary[..., None] * self.Vt[fv]).sum(1)
        self.Nrm = torch.nn.functional.normalize((bary[..., None] * self.Nt[fv]).sum(1), dim=-1)

    @staticmethod
    def fit_geometry(V, F, mask, azimuth=0):
        """Fit (as in the bake) for a mesh that isn't a Model yet; used by sculpt and parts."""
        g = Model.__new__(Model)
        g.V = V.astype(np.float32)
        g.Vt = torch.tensor(g.V, device=dev)
        g.Ft = torch.tensor(F.astype(np.int64), device=dev)
        return g.fit(*camera(azimuth), mask)[0]

    def pixels(self, X, cam, right, fit):
        """fit = (scale, tx, ty[, ax]); ax stretches horizontally (generated meshes often come out wider or narrower
        than the image they were made from)."""
        s, tx, ty = fit[:3]
        sx = s * (fit[3] if len(fit) > 3 else 1.0)
        cam_t, right_t = torch.tensor(cam, device=dev), torch.tensor(right, device=dev)
        return torch.stack([tx + sx * (X @ right_t), ty - s * X[:, 1]], -1), -(X @ cam_t)

    def silhouette(self, cam, right, fit, H, W):
        xy, _ = self.pixels(self.Vt, cam, right, fit)
        fid, _, _ = rasterize(xy, torch.zeros(len(self.V), device=dev), self.Ft, H, W)
        return fid >= 0

    def fit(self, cam, right, mask):
        """Scale/offset so the mesh silhouette lands on the cutout: match height, then search horizontal stretch and
        offset (coarse to fine)."""
        H, W = mask.shape
        u, v = self.V @ right, self.V[:, 1]
        ys, xs = np.nonzero(mask)
        s = (ys.max() - ys.min()) / (v.max() - v.min())
        ty = ys.min() + s * v.max()
        m = torch.tensor(mask, device=dev)
        uc = (u.min() + u.max()) / 2

        def score(ax, dx):
            tx = (xs.min() + xs.max()) / 2 - s * ax * uc + dx
            sil = self.silhouette(cam, right, (s, tx, ty, ax), H, W)
            return (sil & m).sum().item() / max(1, (sil | m).sum().item()), (s, tx, ty, ax)

        best = max(score(1.0, dx) for dx in range(-24, 25, 4))
        for axs, dxs in ((np.linspace(0.84, 1.16, 17), (-4, 0, 4)), (np.linspace(-0.02, 0.02, 5), range(-3, 4))):
            ax0, dx0 = best[1][3], best[1][1] - ((xs.min() + xs.max()) / 2 - s * best[1][3] * uc)
            best = max([best] + [score(ax0 + da if len(axs) == 5 else da, dx0 + dd) for da in axs for dd in dxs])
        return tuple(float(x) for x in best[1]), best[0]

    def render(self, tex8, cam, right, fit, H, W, bg=(0.82, 0.82, 0.82)):
        """Unlit render of the textured mesh with exactly the projection used for baking."""
        xy, depth = self.pixels(self.Vt, cam, right, fit)
        fid, (ys, xs), bary = rasterize(xy, depth, self.Ft, H, W)
        uvs = (bary[..., None] * torch.tensor(self.uv, device=dev)[self.Ft[fid[ys, xs]]]).sum(1)
        img = torch.tensor(tex8, device=dev).float().permute(2, 0, 1)[None] / 255
        grid = torch.stack([uvs[:, 0] * 2 - 1, (1 - uvs[:, 1]) * 2 - 1], -1)[None, None]
        col = torch.nn.functional.grid_sample(img, grid, mode="bilinear", align_corners=False)[0, :, 0].T
        out = torch.tensor(bg, device=dev).expand(H, W, 3).clone()
        out[ys, xs] = col
        return (out.cpu().numpy() * 255).round().astype(np.uint8), (fid >= 0).cpu().numpy()

    def shaded(self, cam, right, fit, H, W, light=(0.0, 0.3, 1.0), bg=0.85):
        """Untextured, lit grey render of the geometry (uint8 RGB), for finding facial features on the mesh."""
        xy, depth = self.pixels(self.Vt, cam, right, fit)
        fid, (ys, xs), bary = rasterize(xy, depth, self.Ft, H, W)
        n = torch.nn.functional.normalize((bary[..., None] * self.Nt[self.Ft[fid[ys, xs]]]).sum(1), dim=-1)
        L = torch.nn.functional.normalize(torch.tensor(light, device=dev), dim=0)
        g = torch.full((H, W), bg, device=dev)
        g[ys, xs] = 0.25 + 0.7 * (n @ L).clamp(min=0)
        return np.repeat((g.cpu().numpy() * 255).astype(np.uint8)[..., None], 3, -1)

    def align_face(self, rgba, cam, right, fit):
        """Warp the face of a source image so its eyes, nose and mouth land on the mesh's own (found by running the
        face detector on a lit render of the bare geometry). The body-wide fit leaves the face several pixels off,
        because generated meshes don't keep the image's head proportions exactly. Returns (rgba, info)."""
        H, W = rgba.shape[:2]
        img = (np.clip(rgba[..., :3] * rgba[..., 3:] + 0.85 * (1 - rgba[..., 3:]), 0, 1) * 255).astype(np.uint8)
        fi = detect_face(img)
        if fi is None:
            return rgba, {"face_align": "no face in image"}
        geo = [detect_face(self.shaded(cam, right, fit, H, W, light)) for light in ((0, 0.1, 1), (0, 0.55, 0.85))]
        geo = [g for g in geo if g is not None]
        if not geo:
            return rgba, {"face_align": "no face found on the mesh"}
        pi = fi[4:14].reshape(5, 2)
        pg = np.mean([g[4:14].reshape(5, 2) for g in geo], 0)
        A, _ = cv2.estimateAffinePartial2D(pg, pi)   # mesh-space pixel -> where that feature is in the image
        if A is None:
            return rgba, {"face_align": "landmarks inconsistent"}
        scale = float(np.hypot(A[0, 0], A[1, 0]))
        shift = float(np.abs(pi - pg).mean())
        if not 0.8 < scale < 1.25 or shift > 0.4 * fi[2]:
            return rgba, {"face_align": f"implausible (scale {scale:.2f}, shift {shift:.0f}px)"}
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        tx = A[0, 0] * gx + A[0, 1] * gy + A[0, 2]
        ty = A[1, 0] * gx + A[1, 1] * gy + A[1, 2]
        # apply on the eye-to-chin area only, fading out before the hairline and ears so they keep the global fit
        x, y, w, h = np.mean([g[:4] for g in geo], 0)
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(x + w / 2), int(y + h * 0.58)), (int(w * 0.5), int(h * 0.5)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), w * 0.15)
        mx, my = (gx + m * (tx - gx)).astype(np.float32), (gy + m * (ty - gy)).astype(np.float32)
        out = cv2.remap(rgba, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out, {"face_align": "ok", "face_shift_px": round(shift, 1), "face_scale": round(scale, 3)}

    def front_layer(self, rgba, cam, right, fit, power, trust):
        """Project the source (front) image. Returns (layer, ownership, info). On a sculpted mesh the eyes, nose and
        mouth already sit under the painted ones, so only the jaw outline is warped onto the mesh's (align_outline);
        otherwise the face is moved onto the features found on the bare geometry (align_face). The ownership marks
        texels (face, hands) that only this image may paint; pass it to bake()."""
        import sculpt
        if self.sculpt_info.get("head_replaced"):
            info = {"face_align": "head rebuilt from the close-up (aligned by construction)"}
        elif self.sculpt_info.get("face_landmarks"):
            rgba, info = sculpt.align_outline(rgba, self.V, self.F, fit)
        else:
            rgba, info = self.align_face(rgba, cam, right, fit)
        prior, box = face_prior(rgba)
        info["face_box"] = box
        if prior is not None:
            # the whole face keeps the source's say at occlusion edges (the jaw over the neck): other views are kept
            # out of it (ownership), so nothing else would paint that rim
            prior = (prior[0], np.maximum(prior[1], sculpt.face_area(rgba)))
        own = self.ownership(sculpt.ownership(rgba), cam, right, fit)
        self.skin_label = self.front_skin_labels(rgba, cam, right, fit)
        return self.project(rgba, cam, right, fit, power, trust, prior), own, info

    def front_skin_labels(self, rgba, cam, right, fit):
        """Per texel: 1 where the source image shows skin there (face parsing: skin, nose, ears), 2 for the neck, 0 where
        it shows something else (a scarf, a collar, hair), -1 where the source doesn't see it. skin_clean only evens
        out skin, never a tan scarf that happens to be close to skin in colour."""
        import landmarks
        rgb = (np.clip(rgba[..., :3] * rgba[..., 3:] + 0.85 * (1 - rgba[..., 3:]), 0, 1) * 255).astype(np.uint8)
        lab = landmarks.face_parse(rgb, detect_face(rgb))
        out = torch.full((len(self.P),), -1, dtype=torch.int8, device=dev)
        if lab is None:
            return out
        H, W = lab.shape
        xy, depth = self.pixels(self.Vt, cam, right, fit)
        vfid, (vy, vx), vb = rasterize(xy, depth, self.Ft, H, W)
        zmap = torch.full((H, W), float("inf"), device=dev)
        zmap[vy, vx] = (vb * depth[self.Ft[vfid[vy, vx]]]).sum(-1)
        pxy, pz = self.pixels(self.P, cam, right, fit)
        xi, yi = pxy[:, 0].long().clamp(0, W - 1), pxy[:, 1].long().clamp(0, H - 1)
        visible = (pz <= zmap[yi, xi] + 0.004 * self.height) & ((self.Nrm @ torch.tensor(cam, device=dev)) > 0.2)
        L = torch.tensor(lab, device=dev)[yi, xi]
        parsed = visible & (L > 0)
        skin = (L == 1) | (L == 2) | (L == 8) | (L == 9)
        out[parsed] = skin[parsed].to(torch.int8)
        out[parsed & (L == 17)] = 2          # neck: skin, but checked more strictly (skin_clean)
        return out

    def head_weight(self):
        """Per-texel weight in [0, 1] of the head sphere found by sculpt; None if unknown."""
        head = self.sculpt_info.get("head")
        if not head:
            return None
        d = torch.linalg.norm(self.P - torch.tensor(head["c"], device=dev, dtype=torch.float32), dim=1)
        return 1 - ((d - head["r"]) / (0.25 * head["r"])).clamp(0, 1)

    def without_head(self, layer):
        """A layer with no say on the head (for repaints that drew the face somewhere else); None if the head
        isn't known, so the caller drops the view instead."""
        h = self.head_weight()
        if h is None:
            return None
        col, w, low, wl = (t.to(dev) for t in layer)
        return col, w * (1 - h), low, wl * (1 - h)

    def head_iou(self, mask, sil, right, fit):
        """Silhouette IoU of a repaint against the render inside a window around the head: klein often redraws arms
        (low whole-body IoU) while the head still lines up. None if the head isn't known."""
        head = self.sculpt_info.get("head")
        if not head:
            return None
        s, tx, ty = fit[:3]
        c = np.asarray(head["c"], np.float32)
        x, y, r = tx + s * float(c @ right), ty - s * c[1], 1.3 * s * head["r"]
        H, W = mask.shape
        x0, x1, y0, y1 = (int(np.clip(v, 0, m)) for v, m in ((x - r, W), (x + r, W), (y - r, H), (y + r, H)))
        a, b = mask[y0:y1, x0:x1], sil[y0:y1, x0:x1]
        return float((a & b).sum() / max(1, (a | b).sum()))

    def head_only(self, layer, trust):
        h = self.head_weight()
        col, w, low, wl = (t.to(dev) for t in layer)
        return col, w * h * trust, low, wl * h * trust

    def repaint_layers(self, name, rgba, ok, sil, cam, right, fit, power, stats):
        """Layers from one aligned klein repaint. The head is judged on its own (face check + silhouette fit in a
        window around it): klein often redraws an arm somewhere else while the head still lines up, or turns the
        face while the clothing is fine. Returns {layer name: layer}; '<name>_head' only if the head is usable."""
        m = rgba[..., 3] > 0.5
        iou = float((m & sil).sum() / max(1, (m | sil).sum()))
        trust = float(np.clip((iou - 0.75) / 0.17, 0, 1)) ** 2   # repaints that don't match the mesh get little say
        stats[f"{name}_iou"] = round(iou, 3)
        layer = self.project(rgba, cam, right, fit, power, 1.0)
        head_scale = 1.0
        if not ok:
            # the face klein drew in the wrong place is cut out; the hair and ears around it are still good evidence
            fp = detect_face((np.clip(rgba[..., :3] * rgba[..., 3:] + 0.82 * (1 - rgba[..., 3:]), 0, 1) * 255)
                             .astype(np.uint8), 0.5)
            if fp is not None:
                x, y, w, h = fp[:4]
                rgba = rgba.copy()
                rgba[max(0, int(y - 0.25 * h)):int(y + 1.2 * h), max(0, int(x - 0.3 * w)):int(x + 1.3 * w), 3] = 0
                head_layer = self.project(rgba, cam, right, fit, power, 1.0)
                head_scale = 0.5
            else:
                head_layer = None
        else:
            head_layer = layer
        scaled = lambda lay, f: (lay[0], lay[1] * f, lay[2], lay[3] * f)  # noqa: E731
        if self.head_weight() is None:   # head unknown: one trust for the whole view, none if the face turned
            return {name: scaled(layer, trust if ok else 0.0)}
        head_iou = self.head_iou(m, sil, right, fit)
        head_trust = float(np.clip((head_iou - 0.75) / 0.17, 0, 1)) ** 2 * head_scale if head_layer else 0.0
        stats[f"{name}_head_iou"] = round(head_iou, 3)
        stats[f"{name}_head_trust"] = round(head_trust, 3)
        out = {name: scaled(self.without_head(layer), trust)}
        if head_trust > 0:
            out[name + "_head" if head_scale == 1 else name + "_hair"] = self.head_only(head_layer, head_trust)
        return out

    def mirrored_head(self, rgba_other, other_right, other_fit, render, cam, right, fit, power, azimuth, trust=1.0):
        """Head-only layer from the mirror image of the opposite view's repaint (characters are close to symmetric),
        for a view whose own repaint turned the face or doesn't line up. Returns (layer or None, reason)."""
        h = self.head_weight()
        if h is None:
            return None, "head unknown"
        W = rgba_other.shape[1]
        # each orbit view centres the whole silhouette, so the heads sit at different x: line them up first
        c = np.asarray(self.sculpt_info["head"]["c"], np.float32)
        x_other = W - (other_fit[1] + other_fit[0] * float(c @ other_right))
        dx = fit[1] + fit[0] * float(c @ right) - x_other
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        rgba = cv2.warpAffine(np.ascontiguousarray(rgba_other[:, ::-1]), M, (W, rgba_other.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ok, why = repaint_check(render, rgba, azimuth)
        if not ok:
            return None, "mirror: " + why
        rgba, _ = align_to_render(rgba, render)
        col, w, low, wl = self.project(rgba, cam, right, fit, power, trust)
        return (col, w * h, low, wl * h), "mirror: " + why

    def ownership(self, own_maps, cam, right, fit):
        """Per-texel share of the image-space (face, hands) ownership maps (sculpt.ownership) for texels this view
        sees well enough."""
        face_map, hand_map = own_maps
        H, W = face_map.shape
        xy, depth = self.pixels(self.Vt, cam, right, fit)
        vfid, (vy, vx), vb = rasterize(xy, depth, self.Ft, H, W)
        zmap = torch.full((H, W), float("inf"), device=dev)
        zmap[vy, vx] = (vb * depth[self.Ft[vfid[vy, vx]]]).sum(-1)
        pxy, pz = self.pixels(self.P, cam, right, fit)
        xi, yi = pxy[:, 0].long().clamp(0, W - 1), pxy[:, 1].long().clamp(0, H - 1)
        visible = pz <= zmap[yi, xi] + 0.004 * self.height
        cosang = self.Nrm @ torch.tensor(cam, device=dev)
        face = torch.tensor(face_map, device=dev, dtype=torch.float32)[yi, xi]
        hand = torch.tensor(hand_map, device=dev, dtype=torch.float32)[yi, xi]
        # the face all the way to the silhouette: a profile repaint's eye or brow on the side of the cheek doubles the
        # face. Hands only where the source sees them well: their sides need the other views.
        own = torch.maximum(face * (cosang / 0.05).clamp(0, 1), hand * ((cosang - 0.15) / 0.15).clamp(0, 1))
        return own * visible.float()

    def project(self, rgba, cam, right, fit, power, trust, prior=None):
        """Colour + blend weight per texel from one view (occlusion-tested against the view's depth buffer).
        prior: optional HxW map multiplying the weight (face priority, flow confidence)."""
        H, W = rgba.shape[:2]
        rgba = clean_rgba(rgba)
        mask = rgba[..., 3] >= 0.9  # only solid pixels: a half-transparent cut-out (failed matting) is not evidence
        xy, depth = self.pixels(self.Vt, cam, right, fit)
        vfid, (vy, vx), vb = rasterize(xy, depth, self.Ft, H, W)
        zmap = torch.full((H, W), float("inf"), device=dev)
        zmap[vy, vx] = (vb * depth[self.Ft[vfid[vy, vx]]]).sum(-1)
        if self.occ is not None:
            oxy, odepth = self.pixels(self.occ[0], cam, right, fit)
            ofid, (oy, ox), ob = rasterize(oxy, odepth, self.occ[1], H, W)
            zmap[oy, ox] = torch.minimum(zmap[oy, ox], (ob * odepth[self.occ[1][ofid[oy, ox]]]).sum(-1))
        # occlusion edges (hand over thigh, arm over torso, the silhouette): small misregistration there paints one
        # surface's colour onto the other, so this view gets little say within a few pixels of them
        zf = torch.where(torch.isfinite(zmap), zmap, torch.full_like(zmap, 1e3))
        zp = torch.nn.functional.pad(zf[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        jump = torch.zeros_like(zf, dtype=torch.bool)
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            jump |= (zf - zp[dy:dy + H, dx:dx + W]).abs() > OCCLUSION_JUMP * self.height
        dz = cv2.distanceTransform((~jump).cpu().numpy().astype(np.uint8), cv2.DIST_L2, 5)
        pxy, pz = self.pixels(self.P, cam, right, fit)
        px, py = pxy[:, 0], pxy[:, 1]
        xi, yi = px.long().clamp(0, W - 1), py.long().clamp(0, H - 1)
        inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        visible = inb & (pz <= zmap[yi, xi] + 0.004 * self.height)
        edge = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        occl, occl_low = np.clip((dz - 1.0) / 6.0, 0, 1), np.clip(dz / 2.0, 0, 1)  # stay off occlusion edges
        if prior is not None:  # ...except inside the face core: a protruding nose is not a different surface
            occl, occl_low = np.maximum(occl, prior[1]), np.maximum(occl_low, prior[1])
        edge_map = np.clip(edge / 6.0, 0, 1) * occl                                 # and off the cutout edge
        low_map = np.clip(edge / 2.0, 0, 1) * occl_low
        edge_w = torch.tensor(edge_map, device=dev, dtype=torch.float32)[yi, xi]
        low_w = torch.tensor(low_map, device=dev, dtype=torch.float32)[yi, xi]
        cosang = self.Nrm @ torch.tensor(cam, device=dev)
        if prior is not None:
            # prior = (boost map, core map). On the face's edges (temples, jaw sides) the boost fades out as the
            # surface turns away, since a stretched front view there leaves a hard seam against the side views. The
            # core (eyes, nose, mouth) keeps it at any angle: under the nose, side views would paint smudges.
            pmap, core = (torch.tensor(p, device=dev, dtype=torch.float32)[yi, xi] for p in prior)
            fade = torch.maximum(core, ((cosang - 0.45) / 0.35).clamp(0, 1))
            boost = 1 + (pmap - 1) * fade
            edge_w, low_w = edge_w * boost, low_w * boost
        # ignore grazing views, except on the face: other views are kept off it (ownership), and a stretched source is
        # still the right colour for the underside of the chin or the side of the nose
        cut = 0.25 if prior is None else 0.25 - 0.22 * core
        facing = torch.where(cosang > cut, cosang, torch.zeros_like(cosang)) ** power
        w = facing * edge_w * visible.float() * trust
        # broad colour also accepts shallow angles: it only has to get the tone right, not the detail
        wl = cosang.clamp(min=0) ** 3 * (cosang > 0.08) * low_w * visible.float() * trust
        grid = torch.stack([px / W * 2 - 1, py / H * 2 - 1], -1)[None, None]
        def sample(a):
            t = torch.tensor(np.ascontiguousarray(a), device=dev).permute(2, 0, 1)[None]
            return torch.nn.functional.grid_sample(t, grid, mode="bilinear", align_corners=False)[0, :, 0].T
        col = sample(rgba[..., :3])
        low = sample(cv2.GaussianBlur(rgba[..., :3], (0, 0), LOW_SIGMA))
        return col, w, low, wl

    def skin_fix(self, Cn, own, stats):
        """Hands: texels the source image doesn't own (palms, finger sides) take colour from side views and the
        unseen-fill, which often bring in the pouch, sleeve or thigh beside them. Where such a texel's colour is far
        from the hand's own skin (as the source paints it), use the nearest source-painted hand colour instead."""
        hands = self.sculpt_info.get("hands")
        if not hands or own is None:
            return Cn
        from sculpt import hand_region
        Pn = self.P.cpu().numpy()
        reg = hand_region(Pn, hands)
        on = own.cpu().numpy()
        ref = (reg > 0.5) & (on > 0.7)
        fix = (reg > 0.05) & (on < 0.7)
        if ref.sum() < 50 or not fix.any():
            return Cn
        mu = Cn[ref].mean(0)
        cov = np.cov(Cn[ref].T) + np.eye(3) * 1e-4
        dc = Cn[fix] - mu
        m2 = np.einsum("ni,ij,nj->n", dc, np.linalg.inv(cov), dc)
        bad = np.clip((m2 - 4.0) / 5.0, 0, 1) * reg[fix] * (1 - on[fix] / 0.7)   # beyond ~2-3 sigma of the skin
        _, nn = cKDTree(Pn[ref]).query(Pn[fix], k=8, workers=-1)
        Cn = Cn.copy()
        Cn[fix] = Cn[fix] * (1 - bad[:, None]) + Cn[ref][nn].mean(1) * bad[:, None]
        # what's left there was stitched from several views and the unseen-fill: blur it over the hand surface (in 3D,
        # within the hand only), so it reads as one skin rather than a patchwork
        hand = np.nonzero(reg > 0.05)[0]
        _, nb = cKDTree(Pn[hand]).query(Pn[fix], k=16, workers=-1)
        a = (reg[fix] * (1 - on[fix] / 0.7))[:, None]
        Cn[fix] = Cn[fix] * (1 - a) + Cn[hand][nb].mean(1) * a
        stats["hand_texels_recoloured"] = int((bad > 0.5).sum())
        return Cn

    def skin_clean(self, Cn, own, stats):
        """Even out skin the source image doesn't paint itself. Side repaints bring shadows and blotches onto the
        sides of the face, the (bald) scalp and the neck, and a rebuilt hand takes the image's hand colours slightly
        off its fingers. Skin model: the source's own face (robust). Texels of skin hue there are pulled toward it,
        keeping a fraction of their variation; beards, eyes and lips differ in hue and are left alone. Rebuilt hands
        get the source hand's own colour as a clean albedo."""
        head = self.sculpt_info.get("head")
        if own is None or not head:
            return Cn
        Pn, on = self.P.cpu().numpy(), own.cpu().numpy()
        lab = cv2.cvtColor(Cn.astype(np.float32)[:, None], cv2.COLOR_RGB2Lab)[:, 0]
        c, r = np.asarray(head["c"], np.float32), float(head["r"])
        fh = r / 1.15
        d = np.linalg.norm(Pn - c, axis=1)
        ref = (on > 0.9) & (d < r)
        out, mu = lab.copy(), None
        if ref.sum() > 200:
            med = np.median(lab[ref], 0)
            mad = np.median(np.abs(lab[ref] - med), 0) * 1.4826 + 1e-3
            core = ref & (np.abs(lab - med) < [2.5, 2.0, 2.0] * mad).all(1)
            mu, sd = lab[core].mean(0), lab[core].std(0) + [1.0, 0.5, 0.5]
            chin = c[1] - 0.6 * fh
            horiz = np.linalg.norm((Pn - c)[:, [0, 2]], axis=1)
            # skin the source shows (parsed), or, where it doesn't see the surface, the (bald) head above the chin
            sl = getattr(self, "skin_label", None)
            sl = sl.cpu().numpy() if sl is not None else np.full(len(Pn), -1, np.int8)
            near_head = ((d < 1.15 * r) & (Pn[:, 1] > chin - 0.1 * fh)) | ((Pn[:, 1] > chin - 0.9 * fh) & (horiz < 0.65 * fh))
            region = near_head & ((sl >= 1) | ((sl == -1) & (d < 1.15 * r) & (Pn[:, 1] > chin - 0.1 * fh)))
            chroma = np.sqrt((((lab[:, 1:] - mu[1:]) / sd[1:]) ** 2).sum(1))
            # how much the source owns the surroundings (in 3D): the evening fades in over a band around the face
            # instead of stopping at the edge of the owned texels
            sel = np.nonzero(region)[0]
            on_s = on.copy()
            if len(sel):
                dd, nn = cKDTree(Pn).query(Pn[sel], k=48, distance_upper_bound=0.12 * fh, workers=-1)
                ok = np.isfinite(dd)
                on_s[sel] = (np.where(ok, on[np.minimum(nn, len(on) - 1)], 0).sum(1) / np.maximum(ok.sum(1), 1))
            # parsing calls the top of a scarf "neck" often enough: neck texels must also be close to the face's own
            # skin colour
            limit = np.where(sl == 2, 2.5, 5.0)
            w = region * np.clip((limit - chroma) / 1.5, 0, 1) * (lab[:, 0] > 15) * np.clip(1 - np.maximum(on, on_s) / 0.9, 0, 1)
            even = mu + np.array([0.3, 0.5, 0.5], np.float32) * (lab - mu)
            out = lab * (1 - w[:, None]) + even * w[:, None]
            stats["skin_evened_pct"] = round(float((w > 0.5).mean() * 100), 2)
        hands = self.sculpt_info.get("hands")
        if hands and self.sculpt_info.get("hands_replaced"):
            from sculpt import hand_region
            hw = hand_region(Pn, hands)
            src = (hw > 0.5) & (on > 0.7)
            if src.sum() > 50:
                col = np.median(lab[src], 0)
                # a bare hand has the face's skin (its pixels in the image are small and shadowed); a glove keeps
                # its own colour
                if mu is not None and np.linalg.norm(col[1:] - mu[1:]) < 12:
                    col = mu.copy()
                tone = col + np.array([0.25, 0.4, 0.4], np.float32) * (out - col)
                k = np.clip(hw * 1.5, 0, 1)[:, None]
                out = out * (1 - k) + tone * k
                stats["hand_skin_lab"] = [round(float(v), 1) for v in col]
        return np.clip(cv2.cvtColor(out.astype(np.float32)[:, None], cv2.COLOR_Lab2RGB)[:, 0], 0, 1)

    def paint_cap(self, tex8, layers, strength=1.0):
        """Overwrite texels with the weighted colour of `layers` (project() output) wherever they see something: the
        hair cap, so the scalp under the hair mesh is hair-coloured and gaps don't show bare skin."""
        col = torch.zeros(len(self.P), 3, device=dev)
        ws = torch.zeros(len(self.P), device=dev)
        for c, w, low, wl in (tuple(x.to(dev) for x in v) for v in layers):
            col += low * wl[:, None]
            ws += wl
        a = (ws / 0.05).clamp(0, 1) * strength
        sel = (a > 0.01).cpu().numpy()
        if not sel.any():
            return tex8, 0
        C = (col / ws.clamp(min=1e-6)[:, None]).cpu().numpy()
        an = a.cpu().numpy()[:, None]
        ty, tx = self.ty.cpu().numpy()[sel], self.tx.cpu().numpy()[sel]
        out = tex8.copy()
        out[ty, tx] = np.clip(out[ty, tx] * (1 - an[sel]) + C[sel] * 255 * an[sel], 0, 255).astype(np.uint8)
        return out, int(sel.sum())

    def bake(self, layers, stats, match_to="front", own=None):
        """layers: {name: (colour, weight, broad colour, broad weight)} from project(). Colour-match to the
        reference layer, blend in two bands, fill unseen, pad charts. own: per-texel share (Model.ownership) that
        only the reference layer may paint."""
        layers = {k: tuple(t.to(dev) for t in v) for k, v in layers.items()}  # callers may park layers in RAM
        if own is not None and match_to in layers:
            own = own.to(dev)
            keep = 1 - own
            layers = {k: v if k == match_to else (v[0], v[1] * keep, v[2], v[3] * keep) for k, v in layers.items()}
        if match_to in layers:
            fc, fw = layers[match_to][:2]
            for name, (col, w, low, wl) in list(layers.items()):
                if name == match_to:
                    continue
                ov = (fw > 0.02) & (w > 0.02)
                if ov.sum() > 500:
                    gain = (fc[ov] * col[ov]).sum(0) / (col[ov] ** 2).sum(0).clamp(min=1e-6)
                    stats[f"{name}_gain"] = [round(g, 3) for g in gain.tolist()]
                    if ((gain > 0.8) & (gain < 1.25)).all():  # bigger "gains" mean misregistration, not exposure
                        layers[name] = ((col * gain).clamp(0, 1), w, (low * gain).clamp(0, 1), wl)
        # Two bands. Broad colour: a weighted average of every view that sees the texel, even at shallow angles, so
        # tones don't patch and little is left to the unseen-fill. Detail: only well-facing views, with weights
        # raised to SHARPEN relative to the best one (1 = plain average; higher switches views harder, which kills
        # ghosting but leaves patchwork on small parts like hands). Faces avoid ghosting through face_prior instead.
        wmax = torch.stack([v[1] for v in layers.values()]).max(0).values
        hi = torch.zeros(len(self.P), 3, device=dev)
        hsum = torch.zeros(len(self.P), device=dev)
        lo = torch.zeros(len(self.P), 3, device=dev)
        lsum = torch.zeros(len(self.P), device=dev)
        for col, w, low, wl in layers.values():
            ws = (w / wmax.clamp(min=1e-12)) ** SHARPEN * (w > 1e-6)
            hi += (col - low) * ws[:, None]
            hsum += ws
            lo += low * wl[:, None]
            lsum += wl
        seen = lsum > 1e-4
        color = torch.zeros_like(lo)
        color[seen] = lo[seen] / lsum[seen, None]
        detail = hsum > 1e-6
        color[detail] += hi[detail] / hsum[detail, None]
        color = color.clamp(0, 1)
        stats["texels_seen_pct"] = round(100 * seen.float().mean().item(), 1)
        stats["texels_detail_pct"] = round(100 * detail.float().mean().item(), 1)
        Pn, Cn, Sn = self.P.cpu().numpy(), color.cpu().numpy(), seen.cpu().numpy()
        if (~Sn).any():  # fill from the nearest seen surface points in 3D, never across UV charts
            d, nn = cKDTree(Pn[Sn]).query(Pn[~Sn], k=16, workers=-1)
            wk = 1.0 / (d + 1e-3 * self.height)
            Cn[~Sn] = (Cn[Sn][nn] * wk[..., None]).sum(1) / wk.sum(1, keepdims=True)
        Cn = self.skin_fix(Cn, own, stats)
        Cn = self.skin_clean(Cn, own, stats)
        T = self.T
        tex8 = np.zeros((T, T, 3), np.uint8)
        covered = np.zeros((T, T), np.uint8)
        ty, tx = self.ty.cpu().numpy(), self.tx.cpu().numpy()
        tex8[ty, tx] = (Cn * 255).round().astype(np.uint8)
        covered[ty, tx] = 1
        for _ in range(8):  # pad charts so bilinear filtering / mipmaps don't pull in black
            dil = cv2.dilate(tex8, np.ones((3, 3), np.uint8))
            grown = cv2.dilate(covered, np.ones((3, 3), np.uint8))
            new = (grown == 1) & (covered == 0)
            tex8[new] = dil[new]
            covered = grown
        return tex8

    def export(self, tex8, path, height=None):
        """height: scale to this many metres with the origin at the feet (game-ready); None keeps model units."""
        V = self.V
        if height:
            lo, hi = V.min(0), V.max(0)
            V = (V - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2], np.float32)) * (height / (hi[1] - lo[1]))
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex8), metallicFactor=0.0,
                                                  roughnessFactor=0.85)
        trimesh.Trimesh(V, self.F, visual=trimesh.visual.TextureVisuals(uv=self.uv, material=mat),
                        process=False).export(path)

    def orbit_fit(self, right, H, W, margin=0.06):
        """Projection that frames the whole character in an H x W image."""
        s = H * (1 - 2 * margin) / self.height
        u = self.V @ right
        return s, W / 2 - s * (u.min() + u.max()) / 2, H * margin + s * float(self.V[:, 1].max())


UV_BOOST_HEAD = 2.0   # linear texel-density factor for parts rebuilt from close-ups (area x4)
UV_BOOST_HAND = 1.6


def uv_boosted(V, sculpt_info, replaced):
    """Vertex positions to unwrap with: the head and hands rebuilt from close-ups are scaled up about their centres
    (smooth falloff), so xatlas, which sizes charts by surface area, gives them more of the texture."""
    if not replaced:
        return V
    V = np.asarray(V, np.float64)
    g = np.ones(len(V))
    regions = []
    if replaced.get("head") and sculpt_info.get("head"):
        regions.append((np.asarray(sculpt_info["head"]["c"]), sculpt_info["head"]["r"], UV_BOOST_HEAD))
    for h in (sculpt_info.get("hands") or []) if replaced.get("hands") else []:
        P = np.asarray(h["P"])
        regions.append((P.mean(0), 1.4 * h["palm"], UV_BOOST_HAND))
    out = V.copy()
    for c, r, f in regions:
        d = np.linalg.norm(V - c, axis=1)
        k = 1 + (f - 1) * np.clip((1.3 * r - d) / (0.5 * r), 0, 1)
        sel = k > g
        out[sel] = c + (V[sel] - c) * k[sel, None]
        g = np.maximum(g, k)
    return out


def render_scene(items, cam, right, fit, H, W, bg=(0.82, 0.82, 0.82)):
    """Unlit render of several textured Models together (body + hair): items = [(model, tex8)]."""
    V = torch.cat([m.Vt for m, _ in items])
    offs = np.cumsum([0] + [len(m.V) for m, _ in items])
    F = torch.cat([m.Ft + int(o) for (m, _), o in zip(items, offs)])
    fo = np.cumsum([0] + [len(m.F) for m, _ in items])
    xy, depth = items[0][0].pixels(V, cam, right, fit)
    fid, (ys, xs), bary = rasterize(xy, depth, F, H, W)
    out = torch.tensor(bg, device=dev).expand(H, W, 3).clone()
    f = fid[ys, xs]
    for k, (m, tex8) in enumerate(items):
        sel = (f >= int(fo[k])) & (f < int(fo[k + 1]))
        if not sel.any():
            continue
        lf = f[sel] - int(fo[k])
        uvs = (bary[sel][..., None] * torch.tensor(m.uv, device=dev)[m.Ft[lf]]).sum(1)
        img = torch.tensor(tex8, device=dev).float().permute(2, 0, 1)[None] / 255
        grid = torch.stack([uvs[:, 0] * 2 - 1, (1 - uvs[:, 1]) * 2 - 1], -1)[None, None]
        out[ys[sel], xs[sel]] = torch.nn.functional.grid_sample(img, grid, mode="bilinear", align_corners=False)[0, :, 0].T
    return (out.cpu().numpy() * 255).round().astype(np.uint8)


def export_scene(items, path, height=None):
    """One GLB with a mesh (and texture) per item [(name, model, tex8)]; the first item sets the game-ready
    transform (scaled to `height` metres, origin at its feet)."""
    V0 = items[0][1].V
    lo, hi = V0.min(0), V0.max(0)
    scene = trimesh.Scene()
    for name, m, tex8 in items:
        V = m.V
        if height:
            V = (V - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2], np.float32)) * (height / (hi[1] - lo[1]))
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex8), metallicFactor=0.0,
                                                  roughnessFactor=0.85, doubleSided=name == "hair")
        scene.add_geometry(trimesh.Trimesh(V, m.F, visual=trimesh.visual.TextureVisuals(uv=m.uv, material=mat),
                                           process=False), geom_name=name, node_name=name)
    scene.export(path)


def clean_rgba(rgba, solid=0.9):
    """Cut-outs keep the background colour (white) under their soft edges. Push interior colours outward so
    lookups near the silhouette never sample background."""
    inside = rgba[..., 3] >= solid
    if not inside.any():
        return rgba
    _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    out = rgba.copy()
    out[..., :3] = rgba[iy, ix, :3]
    return out


def flat(rgba, bg=0.82):
    """RGBA float -> uint8 grey image on the render background, for optical flow."""
    a = rgba[..., 3:]
    rgb = rgba[..., :3] * a + bg * (1 - a)
    return cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)


def align_to_render(rgba, render_rgb, smooth=FLOW_SMOOTH):
    """Warp a klein repaint back onto the mesh render it was painted from. klein redraws the character a few pixels
    off (limbs drift, the head shifts). Only the large-scale drift is undone: optical flow, kept where the forward and
    backward estimates agree, averaged at `smooth` px. Per-pixel flow tears small parts such as fingers.
    Returns the warped RGBA and the median drift in pixels."""
    a = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    b = flat(rgba)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fwd = dis.calc(a, b, None)   # render pixel x  ->  repaint pixel x + fwd(x)
    bwd = dis.calc(b, a, None)
    H, W = a.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    back = cv2.remap(bwd, gx + fwd[..., 0], gy + fwd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    c = (np.linalg.norm(fwd + back, axis=-1) < 2.0).astype(np.float32) * (rgba[..., 3] > 0.5)
    flow = cv2.GaussianBlur(fwd * c[..., None], (0, 0), smooth) / np.maximum(
        cv2.GaussianBlur(c, (0, 0), smooth), 1e-3)[..., None]
    warped = cv2.remap(rgba, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)
    return warped, float(np.median(np.linalg.norm(fwd, axis=-1)[rgba[..., 3] > 0.5]))


def detect_face(rgb, threshold=0.6):
    """Most confident face (YuNet): [x, y, w, h, 5 landmark (x, y) pairs: eyes, nose tip, mouth corners, score]."""
    if not os.path.exists(YUNET):
        return None
    H, W = rgb.shape[:2]
    det = cv2.FaceDetectorYN.create(YUNET, "", (W, H), threshold)
    _, faces = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return None if faces is None else faces[np.argmax(faces[:, -1])]


def repaint_check(render_rgb, rgba, azimuth=None):
    """klein sometimes ignores the render's pose and turns the character to face the camera (seen on three-quarter
    back and profile views). Even at low trust such a repaint paints a second face on the cheek or the back of the
    head. Returns (ok, reason): not ok when the repaint shows a face the render doesn't, shows it elsewhere, or shows
    it more frontally (eye spacing relative to face width; ~0.4 frontal, ~0.25 three-quarter, ~0.1 profile)."""
    fr = detect_face(render_rgb, 0.5)
    fp = detect_face((np.clip(rgba[..., :3] * rgba[..., 3:] + 0.82 * (1 - rgba[..., 3:]), 0, 1) * 255)
                     .astype(np.uint8), 0.75)
    if fp is None:
        return True, "no face"
    if azimuth is not None and 135 <= azimuth % 360 <= 225:
        return False, "face in a back view"
    if fr is None:
        return False, "face where the render has none"
    eyes_r, eyes_p = fr[4:8].reshape(2, 2).mean(0), fp[4:8].reshape(2, 2).mean(0)
    off = float(np.linalg.norm(eyes_r - eyes_p))
    if off > 0.35 * fr[2] or not 0.5 < fp[2] / fr[2] < 2.0:
        return False, f"face moved {off:.0f}px"
    spread_r = np.linalg.norm(fr[4:6] - fr[6:8]) / fr[2]
    spread_p = np.linalg.norm(fp[4:6] - fp[6:8]) / fp[2]
    if spread_p > spread_r + 0.1:
        return False, f"face turned toward the camera ({spread_r:.2f} -> {spread_p:.2f})"
    return True, f"face matches ({spread_r:.2f} -> {spread_p:.2f})"


def face_prior(rgba, boost=30.0):
    """Weights so the source image alone paints the detected face, for Model.project(prior=...).
    Returns ((boost map, core map) or None, face box)."""
    H, W = rgba.shape[:2]
    face = detect_face((np.clip(rgba[..., :3] * rgba[..., 3:] + 0.8 * (1 - rgba[..., 3:]), 0, 1) * 255)
                       .astype(np.uint8))
    if face is None:
        return None, None
    x, y, w, h = face[:4]
    cx, cy = x + w / 2, y + h / 2
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(w * 0.55), int(h * 0.6)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(2.0, w * 0.12))
    core = np.zeros((H, W), np.float32)   # eyes to chin
    cv2.ellipse(core, (int(cx), int(y + h * 0.58)), (int(w * 0.36), int(h * 0.42)), 0, 0, 360, 1.0, -1)
    core = cv2.GaussianBlur(core, (0, 0), max(2.0, w * 0.08))
    return (1 + boost * m, core), [float(v) for v in (x, y, w, h)]


def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mesh", default="shape_mv.glb")
    ap.add_argument("--faces", type=int, default=30000)
    ap.add_argument("--tex", type=int, default=2048)
    ap.add_argument("--power", type=float, default=6.0, help="view-angle sharpness for blending")
    ap.add_argument("--front-bias", type=float, default=1.5, help="trust the front (source) image more")
    a = ap.parse_args()
    t0 = time.time()
    front_path = os.path.join(a.out, "views", "front.png")
    model = Model(a.out, a.mesh, a.faces, a.tex, front=load_rgba(front_path) if os.path.exists(front_path) else None)
    stats, layers, fits, own = {"faces": len(model.F), **model.sculpt_info}, {}, {}, None
    stats.pop("hands", None)
    for name, az in PASS1.items():
        path = os.path.join(a.out, "views", f"{name}.png")
        if not os.path.exists(path):
            continue
        rgba = load_rgba(path)
        cam, right = camera(az)
        fit, iou = model.fit(cam, right, rgba[..., 3] > 0.5)
        fits[name] = [float(x) for x in fit]
        stats[f"{name}_silhouette_iou"] = round(iou, 3)
        # generated side/back views fit the mesh less well than the source image: trust them by silhouette fit
        trust = a.front_bias if name == "front" else min(1.0, iou / 0.85) ** 2
        if name == "front":
            layers[name], own, info = model.front_layer(rgba, cam, right, fit, a.power, trust)
            stats.update(info)
        else:
            layers[name] = model.project(rgba, cam, right, fit, a.power, trust)
    tex8 = model.bake(layers, stats, own=own)
    Image.fromarray(tex8).save(os.path.join(a.out, "albedo_pass1.png"))
    model.export(tex8, os.path.join(a.out, "textured_pass1.glb"))
    stats["seconds_total"] = round(time.time() - t0, 1)
    stats["fits"] = fits
    with open(os.path.join(a.out, "texture_pass1.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "fits"}))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
