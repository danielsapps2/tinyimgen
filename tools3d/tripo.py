"""Image -> textured 3D through Tripo H3.1 on fal, plus the local post-process that makes it a game asset.

Tripo returns a ~2M-triangle mesh with one 2K texture. Here it is turned to face +Z (glTF front), scaled to the
requested height with the origin at the feet, and decimated to the triangle budget. The game mesh gets fresh UVs, and
Tripo's texture is transferred onto them from the nearest point of the full-detail surface: decimating with the original
UVs spends the budget holding Tripo's many texture seams and leaves fingers jagged.

The fal key comes from FAL_KEY or ~/.config/tinyimgen/fal_key.
"""
import os
import time

import numpy as np

ENDPOINT = "tripo3d/h3.1/image-to-3d"
ENDPOINT_MULTI = "tripo3d/h3.1/multiview-to-3d"
KEY_FILE = os.path.expanduser("~/.config/tinyimgen/fal_key")
# quality preset -> (geometry_quality, texture_quality). "standard" is what won the 2026-10 comparison.
QUALITY = {"draft": ("standard", "standard"), "standard": ("detailed", "standard"), "high": ("detailed", "detailed")}


def fal_key():
    key = os.environ.get("FAL_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
    return key or None


def generate(images, quality, out_path, progress=lambda msg: None):
    """images: {"front": path[, "left", "back", "right"]}. Multiview is used only with all four views (the endpoint
    takes them by position: front, left, back, right). Downloads the GLB to out_path. Returns info for job stats."""
    import fal_client
    import httpx
    key = fal_key()
    if not key:
        raise RuntimeError(f"No fal API key: set FAL_KEY or write it to {KEY_FILE}")
    client = fal_client.SyncClient(key=key, default_timeout=600)
    order = ("front", "left", "back", "right")
    multi = all(images.get(k) for k in order)
    progress("uploading")
    urls = [client.upload_file(images[k]) for k in (order if multi else ("front",))]
    geo, tex = QUALITY.get(quality, QUALITY["standard"])
    args = {"texture": True, "pbr": False, "geometry_quality": geo, "texture_quality": tex}
    if multi:
        endpoint, args["image_urls"] = ENDPOINT_MULTI, urls
    else:
        endpoint, args["image_url"] = ENDPOINT, urls[0]
    t0 = time.time()
    handle = client.submit(endpoint, args)
    for ev in handle.iter_events(with_logs=False, interval=2.0):
        pos = getattr(ev, "position", None)
        progress(f"Tripo: queued ({pos} ahead)" if pos else "Tripo: generating")
    res = handle.get()
    progress("downloading")
    url = res["model_mesh"]["url"]
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    return {"endpoint": endpoint, "request_id": handle.request_id, "task_id": res.get("task_id"),
            "seconds": round(time.time() - t0, 1), "geometry_quality": geo, "texture_quality": tex,
            "views": list(order if multi else ("front",))}


# --- mesh I/O -------------------------------------------------------------------------------------------------------

def load(path):
    """(V, F, UV, texture RGB uint8) of a GLB with one textured mesh (Tripo's output)."""
    import trimesh
    sc = trimesh.load(path)
    geoms = [g for g in (sc.dump() if isinstance(sc, trimesh.Scene) else [sc]) if len(g.faces)]
    if len(geoms) > 1:   # concatenate keeps one texture only when they share it; Tripo emits a single mesh
        geoms = [trimesh.util.concatenate(geoms)]
    g = geoms[0]
    mat = g.visual.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    tex = np.asarray(img.convert("RGB"))
    return (np.asarray(g.vertices, np.float64), np.asarray(g.faces, np.int64), np.asarray(g.visual.uv, np.float64), tex)


def export(V, F, UV, tex, path, normals=None):
    import trimesh
    from PIL import Image
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex), metallicFactor=0.0,
                                              roughnessFactor=0.85)
    m = trimesh.Trimesh(V, F, visual=trimesh.visual.TextureVisuals(uv=UV, material=mat), process=False)
    if normals is not None:
        m.vertex_normals = normals   # cached normals are written to the GLB (else viewers shade it flat)
    m.export(path)


def weld(V, F):
    """Merge the per-UV-chart duplicate vertices: one connected surface for decimation and smooth normals.
    Returns (V, F, map from the input vertices to the welded ones)."""
    _, inv = np.unique(np.round(V, 7), axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((inv.max() + 1, 3))
    Vw[inv] = V
    return Vw, inv[F], inv


def smooth_normals(V, F):
    """Vertex normals that are continuous across UV seams (computed on the welded surface)."""
    import trimesh
    Vw, Fw, inv = weld(V, F)
    return trimesh.Trimesh(Vw, Fw, process=False).vertex_normals[inv]


# --- orientation and scale ------------------------------------------------------------------------------------------

def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def front_azimuth(V, F, UV, tex):
    """Azimuth (deg) of the character's front, from a face found on textured renders of the head. Tripo's default
    output faces +X (90); None when no face is found (helmets, non-humans, props)."""
    from texture import detect_face
    best = None
    top, h = V[:, 1].max(), np.ptp(V[:, 1])
    head = V[V[:, 1] > top - 0.15 * h]
    c = np.r_[(head[:, 0].min() + head[:, 0].max()) / 2, top - 0.08 * h, (head[:, 2].min() + head[:, 2].max()) / 2]
    for az in (0, 90, 180, 270):
        img = render(V, F, UV, tex, c, 0.1 * h, az, size=320)
        f = detect_face(img, 0.6)
        if f is not None and (best is None or f[-1] > best[1]):
            best = (az, float(f[-1]))
    return None if best is None else best[0]


def normalise(V, front_az, height):
    """Turn the front to +Z, scale to `height` with the feet at y=0, centred on x/z."""
    V = V @ rot_y(-front_az).T
    lo, hi = V.min(0), V.max(0)
    V = (V - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) * (height / (hi[1] - lo[1]))
    return V


# --- game mesh ------------------------------------------------------------------------------------------------------

def game_mesh(V, F, UV, tex, faces, tex_size=2048, gpu=None):
    """Decimate to `faces` triangles with fresh UVs and Tripo's texture transferred onto them. gpu: context manager
    held around the GPU part only (the decimation is CPU work and shouldn't block the image server).
    Returns (V, F, UV, normals, texture)."""
    import contextlib
    import pymeshlab
    import trimesh
    import xatlas
    Vw, Fw, _ = weld(V, F)
    # a fast pre-pass to 10x the budget (~2 s for Tripo's ~2M triangles), then the careful quadric pass for the rest:
    # same result as pymeshlab all the way (~30 s), which alone leaves spikes and jagged fingers at the final budget
    try:
        import fast_simplification
        if len(Fw) > 20 * faces:
            Vw, Fw = fast_simplification.simplify(Vw.astype(np.float32), Fw.astype(np.int32), target_count=10 * int(faces))
            Vw = Vw.astype(np.float64)
    except ImportError:
        pass
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(Vw, Fw.astype(np.int32)))
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(faces), qualitythr=0.4, preserveboundary=True,
                                                optimalplacement=True, preservenormal=True)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    m = ms.current_mesh()
    Vd, Fd = m.vertex_matrix(), m.face_matrix()
    N = trimesh.Trimesh(Vd, Fd, process=False).vertex_normals
    vmap, Fu, UVu = xatlas.parametrize(Vd, Fd)
    Vu, Nu = Vd[vmap], N[vmap]
    tex2 = transfer(V, F, UV, tex, Vu, Fu, UVu, tex_size, gpu or contextlib.nullcontext)
    return Vu, Fu, UVu, Nu, tex2


def transfer(Vs, Fs, UVs, tex_s, Vt, Ft, UVt, size, gpu):
    """Bake the source mesh's texture onto the target's UV layout: each target texel takes the colour at the nearest
    point of the source surface (nearest triangle by centroid, then the clamped projection onto it)."""
    import cv2
    import torch
    from scipy.spatial import cKDTree
    from texture import dev, empty_cache, rasterize
    with gpu(), torch.no_grad():
        uvp = torch.tensor(np.c_[UVt[:, 0] * size, (1 - UVt[:, 1]) * size], device=dev, dtype=torch.float32)
        Ftt = torch.tensor(Ft.astype(np.int64), device=dev)
        fid, (ty, tx), bary = rasterize(uvp, torch.zeros(len(UVt), device=dev), Ftt, size, size)
        P = (bary[..., None] * torch.tensor(Vt, device=dev, dtype=torch.float32)[Ftt[fid[ty, tx]]]).sum(1).cpu().numpy()
        ty, tx = ty.cpu().numpy(), tx.cpu().numpy()
        del uvp, Ftt, fid, bary
        empty_cache()
    tri = Vs[Fs]
    _, k = cKDTree(tri.mean(1)).query(P, k=4, workers=-1)
    # the nearest of the 4 nearest-centroid triangles, by distance to the clamped projection
    best_d, best_uv = np.full(len(P), np.inf), np.zeros((len(P), 2))
    for j in range(k.shape[1]):
        a, b, c = (tri[k[:, j], i] for i in range(3))
        w = _closest_bary(P, a, b, c)
        q = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c
        d = np.linalg.norm(P - q, axis=1)
        uvk = UVs[Fs[k[:, j]]]
        uv = (w[..., None] * uvk).sum(1)
        better = d < best_d
        best_d[better], best_uv[better] = d[better], uv[better]
    H, W = tex_s.shape[:2]
    x = np.clip(best_uv[:, 0] * W - 0.5, 0, W - 1.001)
    y = np.clip((1 - best_uv[:, 1]) * H - 0.5, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    t = tex_s.astype(np.float32)
    col = ((t[y0, x0] * (1 - fx) + t[y0, x0 + 1] * fx) * (1 - fy) + (t[y0 + 1, x0] * (1 - fx) + t[y0 + 1, x0 + 1] * fx) * fy)
    out = np.zeros((size, size, 3), np.uint8)
    out[ty, tx] = np.clip(col + 0.5, 0, 255).astype(np.uint8)
    covered = np.zeros((size, size), np.uint8)
    covered[ty, tx] = 1
    for _ in range(8):   # pad charts so bilinear filtering and mipmaps don't pull in black
        dil = cv2.dilate(out, np.ones((3, 3), np.uint8))
        grown = cv2.dilate(covered, np.ones((3, 3), np.uint8))
        new = (grown == 1) & (covered == 0)
        out[new] = dil[new]
        covered = grown
    return out


def _closest_bary(p, a, b, c):
    """Barycentric weights (N, 3) of the closest point on triangle abc to p (Ericson, Real-Time Collision Detection)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-20, 1e-20, va + vb + vc)
    v, w = vb / den, vc / den
    out = np.stack([1 - v - w, v, w], 1)

    def put(mask, wa, wb, wc):
        out[mask] = np.stack([wa, wb, wc], 1)[mask]
    z = np.zeros(len(p))
    safe = lambda x: np.where(np.abs(x) < 1e-20, 1e-20, x)  # noqa: E731
    t_ab = d1 / safe(d1 - d3)
    t_ac = d2 / safe(d2 - d6)
    t_bc = (d4 - d3) / safe((d4 - d3) + (d5 - d6))
    # regions, applied from the least to the most specific so the right one wins
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t_ab, t_ab, z)                      # edge ab
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t_ac, z, t_ac)                      # edge ac
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), z, 1 - t_bc, t_bc)        # edge bc
    put((d1 <= 0) & (d2 <= 0), z + 1, z, z)                                        # vertex a
    put((d3 >= 0) & (d4 <= d3), z, z + 1, z)                                       # vertex b
    put((d6 >= 0) & (d5 <= d6), z, z, z + 1)                                       # vertex c
    return out


# --- preview --------------------------------------------------------------------------------------------------------

def render(V, F, UV, tex, center, half, az, size=512, el=0.0, bg=0.12, W=None):
    """Lit textured orthographic render (uint8 RGB), size x size (or W wide), looking at `center` from azimuth `az`."""
    import torch
    from texture import dev, rasterize
    W = W or size
    a, e = np.radians(az), np.radians(el)
    cam = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    right = np.cross([0, 1, 0], cam)
    right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    with torch.no_grad():
        Vt = torch.tensor(V - center, device=dev, dtype=torch.float32)
        Ft = torch.tensor(F.astype(np.int64), device=dev)
        S = size / (2 * half)
        r_, u_, c_ = (torch.tensor(x, device=dev, dtype=torch.float32) for x in (right, up, cam))
        xy = torch.stack([W / 2 + S * (Vt @ r_), size / 2 - S * (Vt @ u_)], -1)
        fid, (ys, xs), bary = rasterize(xy, -(Vt @ c_), Ft, size, W)
        out = torch.full((size, W, 3), bg, device=dev)
        if len(ys):
            f = fid[ys, xs]
            fn = torch.nn.functional.normalize(torch.cross(Vt[Ft[:, 1]] - Vt[Ft[:, 0]], Vt[Ft[:, 2]] - Vt[Ft[:, 0]], dim=1),
                                               dim=-1)
            # smooth shading: area-weighted vertex normals from face normals
            vn = torch.zeros_like(Vt).index_add_(0, Ft.reshape(-1), fn.repeat_interleave(3, 0))
            vn = torch.nn.functional.normalize(vn, dim=-1)
            n = torch.nn.functional.normalize((bary[..., None] * vn[Ft[f]]).sum(1), dim=-1)
            L = torch.nn.functional.normalize(c_ + 0.6 * torch.tensor([0., 1, 0], device=dev) + 0.4 * r_, dim=0)
            sh = 0.3 + 0.62 * (n @ L).abs()
            uvs = (bary[..., None] * torch.tensor(UV, device=dev, dtype=torch.float32)[Ft[f]]).sum(1)
            img = torch.tensor(tex, device=dev).float().permute(2, 0, 1)[None] / 255
            grid = torch.stack([uvs[:, 0] * 2 - 1, (1 - uvs[:, 1]) * 2 - 1], -1)[None, None]
            col = torch.nn.functional.grid_sample(img, grid, mode="bilinear", align_corners=False)[0, :, 0].T
            out[ys, xs] = (col * sh[:, None]).clamp(0, 1)
        return (out.cpu().numpy() * 255).astype(np.uint8)
