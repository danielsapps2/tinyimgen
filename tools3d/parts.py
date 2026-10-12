"""Part detail pass: rebuild the head and hands from close-ups.

In a full-body image the face is ~100 px wide and a hand ~60 px, and Hunyuan3D spends ~30 voxels on a face, so
those are the parts that look worst. For each part found in the source image:

1. crop it, and have klein repaint the crop as a sharp close-up (same framing); register the close-up back onto the
   crop (SIFT + RANSAC similarity), so close-up pixels map exactly onto source pixels;
2. build the part's shape at close-up resolution (Hunyuan3D on the close-ups);
3. place it in the body's model space through that pixel mapping (x, y) and the body's own depth (z);
4. merge: sample the body surface away from the parts and the parts' surfaces inside them, and rebuild one
   watertight mesh with screened Poisson, whose octree adapts to the denser part samples.

Parts that can't be found or don't register are skipped; the rest of the pipeline runs as before.
"""
import os

import cv2
import numpy as np
from PIL import Image

import landmarks
from texture import detect_face, empty_cache

CLOSE = 768            # close-up size in px (Hunyuan sees 518 px; the bake's 3x upscale is finer than the atlas)
GREY = 0.85            # background the crops are composited on (klein's "plain light grey")

HEAD_PROMPT = ("Image 1 is a blurry low-resolution crop of a 3D game character's head and shoulders. Redraw image 1 as "
               "a sharp, highly detailed, high-resolution close-up of exactly the same character: the same face, "
               "facial features, expression, hair, headwear, clothing, colours and materials. Keep exactly the same "
               "framing, pose, head angle, size and position as image 1, so the two images line up, and add nothing "
               "that image 1 does not show. Soft even lighting, plain light grey background.")
BACK_PROMPT = ("Image 1 is a blurry low-resolution crop of a 3D game character's head and shoulders seen from "
               "behind. Redraw image 1 as a sharp, highly detailed, high-resolution close-up of exactly the same view: "
               "the back of the head, the same hair, headwear, clothing, colours and materials. Do not show a face. "
               "Keep exactly the same framing, size and position as image 1. Soft even lighting, plain light grey "
               "background.")
HAND_PROMPT = ("Image 1 is a blurry low-resolution crop of a 3D game character's hand. Redraw image 1 as a sharp, "
               "highly detailed, high-resolution close-up of exactly the same hand: anatomically correct with five "
               "fingers in exactly the same pose, the same skin, sleeve, colours and materials, adding nothing that image 1 "
               "does not show. Keep exactly the same framing, size and position as image 1. Soft even lighting, plain "
               "light grey background.")


def flat_rgb(rgba, bg=GREY):
    return (np.clip(rgba[..., :3] * rgba[..., 3:] + bg * (1 - rgba[..., 3:]), 0, 1) * 255).astype(np.uint8)


def head_box(rgba):
    """Square (x0, y0, size) around the head and the top of the shoulders, or None without a face. Sized from the
    face, and grown upward to the top of the silhouette above it (hair, hats)."""
    rgb = flat_rgb(rgba)
    f = detect_face(rgb)
    if f is None:
        return None
    x, y, w, h = f[:4]
    cx, cy = x + w / 2, y + h / 2
    S = 2.6 * max(w, h)
    top = cy - 0.5 * S + 0.1 * S
    m = rgba[..., 3] > 0.5
    cols = m[:, int(max(0, cx - 0.4 * S)):int(cx + 0.4 * S)]
    rows = np.nonzero(cols[:int(max(1, cy)), :].any(1))[0]
    if len(rows):
        top = min(top, rows.min() - 0.04 * S)    # tall hair or a hat
    bottom = max(top + S, cy + 1.25 * h)         # down to the top of the shoulders
    S = bottom - top
    return float(cx - S / 2), float(top), float(S)


def hand_boxes(rgba):
    """[(x0, y0, size, landmarks)] per detected hand: the hand plus the wrist and a bit of sleeve."""
    out = []
    for h in landmarks.hands(flat_rgb(rgba)):
        P = h[:, :2]
        c = (P.min(0) + P.max(0)) / 2
        S = max(1.6 * np.ptp(P, 0).max(), 48.0)
        out.append((float(c[0] - S / 2), float(c[1] - S / 2), float(S), h))
    return out


def crop(img, box, size=CLOSE, flags=cv2.INTER_LANCZOS4):
    """Square crop (x0, y0, S) resampled to size x size; outside the image is background."""
    x0, y0, S = box[:3]
    k = size / S
    M = np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
    bv = (GREY,) * img.shape[2] if img.dtype != np.uint8 else (int(GREY * 255),) * img.shape[2]
    if img.ndim == 3 and img.shape[2] == 4:
        bv = (0,) * 4
    return cv2.warpAffine(img, M, (size, size), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)


def register(ref_rgb, img_rgb):
    """Similarity transform (2x3) taking img pixels onto ref pixels: SIFT matches with RANSAC, or, when the crop is
    too blurry for features (hands), intensity alignment (ECC). Returns (M, info); M is None when the two don't line
    up (klein reframed or redrew the part)."""
    M, info = _register_sift(ref_rgb, img_rgb)
    if M is None:
        M2, info2 = _register_ecc(ref_rgb, img_rgb)
        info = {**info, **info2}
        M = M2
    return M, info


def _register_ecc(ref_rgb, img_rgb, size=256):
    k = size / ref_rgb.shape[1]
    g = [cv2.GaussianBlur(cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA)
                          .astype(np.float32) / 255, (0, 0), 1.5) for x in (ref_rgb, img_rgb)]
    M = np.eye(2, 3, dtype=np.float32)
    try:
        cc, M = cv2.findTransformECC(g[0], g[1], M, cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        return None, {"register_ecc": "did not converge"}
    # ECC gives ref -> img; invert to img -> ref and back to full resolution
    A = cv2.invertAffineTransform(M)
    A[:, 2] /= k
    scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    shear = float(abs(A[0, 0] - A[1, 1]) + abs(A[0, 1] + A[1, 0]))
    c = ref_rgb.shape[1] / 2
    shift = float(np.hypot(*(A[:, :2] @ [c, c] + A[:, 2] - c)))
    info = {"register_ecc_cc": round(float(cc), 3), "register_scale": round(scale, 3), "register_shift_px": round(shift, 1)}
    if cc < 0.9 or not 0.8 < scale < 1.25 or shear > 0.1 * scale or shift > 0.15 * ref_rgb.shape[1]:
        info["register"] = "implausible (ecc)"
        return None, info
    info["register"] = "ok (ecc)"
    return A, info


def _register_sift(ref_rgb, img_rgb):
    sift = cv2.SIFT_create(4000)
    g1, g2 = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (ref_rgb, img_rgb))
    # the reference crop is an upscaled blur: blur the close-up to match before looking for features
    g2b = cv2.GaussianBlur(g2, (0, 0), 2.0)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2b, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, {"register": "too few features"}
    m = cv2.BFMatcher().knnMatch(d2, d1, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.8 * b.distance]
    if len(good) < 12:
        return None, {"register": f"{len(good)} matches"}
    src = np.float32([k2[a.queryIdx].pt for a in good])
    dst = np.float32([k1[a.trainIdx].pt for a in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=5000)
    if M is None:
        return None, {"register": "no consistent transform"}
    n = int(inl.sum())
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    c = ref_rgb.shape[1] / 2
    shift = float(np.hypot(*(M[:, :2] @ [c, c] + M[:, 2] - c)))
    info = {"register_inliers": n, "register_scale": round(scale, 3), "register_shift_px": round(shift, 1)}
    if n < 15 or not 0.8 < scale < 1.25 or shift > 0.15 * ref_rgb.shape[1]:
        info["register"] = "implausible"
        return None, info
    info["register"] = "ok"
    return M, info


def closeup(klein_fn, rgb_crop, prompt, path, seeds=(11, 29), extra_refs=(), check=True):
    """klein repaint of a crop, registered back onto it. klein_fn(prompt, [ref paths], seed) -> generated RGB path;
    the crop is image 1, extra_refs follow. check=False skips the colour check (for edits meant to change the
    part, such as adding hair). Returns (aligned close-up RGB or None, info)."""
    Image.fromarray(rgb_crop).save(path)
    best = None
    for seed in seeds:
        out = np.asarray(Image.open(klein_fn(prompt, [path, *extra_refs], seed)).convert("RGB"))
        if out.shape[:2] != rgb_crop.shape[:2]:
            out = cv2.resize(out, rgb_crop.shape[1::-1], interpolation=cv2.INTER_AREA)
        M, info = register(rgb_crop, out)
        info["seed"] = seed
        if M is not None:
            al = cv2.warpAffine(out, M, rgb_crop.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            # klein sometimes changes the part while keeping the framing (gloves on a bare hand): compare colours at
            # the crop's own blur
            a = cv2.GaussianBlur(al.astype(np.float32), (0, 0), 3) - rgb_crop.astype(np.float32)
            info["residual"] = round(float(np.abs(cv2.GaussianBlur(a, (0, 0), 3)).mean()), 2)
            if info["residual"] < MAX_RESIDUAL or not check:
                best = (al, info)
                break
            info["register"] = "changed the part"
        best = (None, info)
    return best


MAX_RESIDUAL = 24.0    # mean |blurred close-up - crop| (0-255); faithful close-ups measure 11-18, a glove added 34


# --- shape, placement, merge --------------------------------------------------------------------------------------

def part_shape(views, octree=380, steps=10, seed=1234):
    """Hunyuan3D on close-up cut-outs {"front": PIL RGBA[, "back": ...]}. Returns (V, F, stats). 10 steps instead of
    the body's 5: at close-up scale the turbo model's 5-step surface is lumpy around the eyes and lips, and HC
    smoothing removes the voxel-level noise without softening the features."""
    from shape import generate_shape
    mesh, stats = generate_shape(views, octree=octree, steps=steps, seed=seed)
    V, F = clean_mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64), smooth=3)
    return V, F, stats


PLACE_SPREAD = 0.08    # x face width: a part whose face disagrees with the body's by more is regenerated


def shape_and_place(views, gpu, cmask, box, bfit, bV, bF, H, W, region, cut_y, fw, stats, name, back=None):
    """part_shape + place, with a second shape-model seed when the first result doesn't sit on the body's face
    (Hunyuan now and then deforms a head, e.g. hair standing off the back as a separate sheet). Keeps the better.
    Returns (V placed or None, F, place info)."""
    best = None
    for seed in (1234, 77):
        with gpu():
            pV, pF, sst = part_shape(views, seed=seed)
        with gpu():
            V, info = place(pV, pF, cmask, box, bfit, bV, bF, H, W, region, cut_y=cut_y, back=back)
        info["seed"] = seed
        stats[f"{name}_shape"] = {k: sst[k] for k in ("seconds", "vertices")}
        if V is not None and (best is None or info["z_spread_px"] < best[2]["z_spread_px"]):
            best = (V, pF, info)
        if V is not None and info["z_spread_px"] < PLACE_SPREAD * fw:
            break
    if best is None:
        stats[f"{name}_place"] = info
        return None, None, info
    stats[f"{name}_place"] = best[2]
    return best


def clean_mesh(V, F, min_diag_pct=3, smooth=0):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(min_diag_pct))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    for _ in range(smooth):
        ms.apply_coord_hc_laplacian_smoothing()
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()


def _decimated(V, F, faces):
    """A coarse copy for silhouette fitting (vertex clustering: much faster than edge collapse, and a silhouette
    doesn't need more)."""
    if len(F) <= faces:
        return V, F
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F))
    cell = float(np.ptp(V, 0).max()) * np.sqrt(2.0 / faces)
    ms.meshing_decimation_clustering(threshold=pymeshlab.PureValue(cell))
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()


def _zback(V, F, back_fit, H, W):
    """Depth map of the surface seen from behind (model Z, per back-view pixel): the mesh turned 180 degrees about
    the vertical axis, seen by the front View (whose fit then matches the back view's)."""
    from sculpt import View
    return -View(V * np.array([-1.0, 1.0, -1.0]), F, back_fit, H, W).zfront


def place(Vp, Fp, close_mask, box, body_fit, body_V, body_F, H, W, depth_region, cut_y=None, back=None):
    """Move a part mesh into the body's model space. x, y: part fit on its close-up (px) -> source px (crop box) ->
    body model (inverse body fit). Scale is shared, so z gets the same factor; the z offset lines the part's front
    surface up with the body's over `depth_region` (HxW bool, source px). cut_y: close-up row below which neither
    the mask nor the mesh is fitted (the crop cuts the shoulders off, and the generator then makes them any width).
    Returns (V in body space, info)."""
    from sculpt import View
    from texture import Model
    Vf, Ff = _decimated(Vp, Fp, 60000)   # the silhouette fit rasterizes ~100 times: a coarse copy is plenty
    sp, txp, typ, axp = Model.fit_geometry(Vf, Ff, close_mask)
    if cut_y is not None:
        for _ in range(2):   # refit on what lies above the cut, then cut again with that fit
            keep = (typ - sp * Vf[:, 1]) < cut_y
            Fk = Ff[keep[Ff].all(1)]
            m = close_mask.copy()
            m[int(cut_y):] = False
            if len(Fk) < 100 or m.sum() < 100:
                break
            used, Fc = np.unique(Fk, return_inverse=True)
            sp, txp, typ, axp = Model.fit_geometry(Vf[used], Fc.reshape(-1, 3), m)
    x0, y0, S = box[:3]
    k = S / close_mask.shape[1]
    sb, txb, tyb, axb = body_fit
    xs = x0 + (txp + sp * axp * Vp[:, 0]) * k
    ys = y0 + (typ - sp * Vp[:, 1]) * k
    zs = sp * np.sqrt(axp) * k / (sb * np.sqrt(axb))      # model units of the part -> of the body
    V = np.c_[(xs - txb) / (sb * axb), (tyb - ys) / sb, Vp[:, 2] * zs]
    zb = View(body_V, body_F, body_fit, H, W).zfront
    zp = View(V, Fp, body_fit, H, W).zfront
    ok = depth_region & np.isfinite(zb) & np.isfinite(zp)
    info = {"part_fit": [round(float(v), 4) for v in (sp, txp, typ, axp)], "depth_px": int(ok.sum())}
    if ok.sum() < 50:
        info["place"] = "no depth overlap"
        return None, info
    # depth: scale and offset against the body's own face. Hunyuan's depth scale varies from run to run, and an
    # offset alone left a haired head 15 px out across the face (its hair then stood off the scalp)
    a_, b_ = zp[ok], zb[ok]
    if back is not None:
        # the back of the head too: with the face alone, a head that came out deeper than the body's sits right at
        # the face and stands off at the back (hair floating behind the scalp)
        back_fit, back_region = back
        zbb, zpb = _zback(body_V, body_F, back_fit, H, W), _zback(V, Fp, back_fit, H, W)
        okb = back_region & np.isfinite(zbb) & np.isfinite(zpb)
        if okb.sum() > 50:
            n = min(ok.sum(), okb.sum())
            pick = lambda m, k: np.random.default_rng(0).choice(np.nonzero(m.ravel())[0], k, replace=False)  # noqa: E731
            fi, bi = pick(ok, n), pick(okb, n)       # equal say for the face and the back of the head
            a_ = np.r_[zp.ravel()[fi], zpb.ravel()[bi]]
            b_ = np.r_[zb.ravel()[fi], zbb.ravel()[bi]]
            info["back_px"] = int(okb.sum())
    keep = np.ones(len(a_), bool)
    k, off = 1.0, float(np.median(b_ - a_))
    for _ in range(3):
        if keep.sum() < 50:
            break
        A = np.c_[a_[keep], np.ones(keep.sum())]
        (k, off), *_ = np.linalg.lstsq(A, b_[keep], rcond=None)
        k = float(np.clip(k, 0.75, 1.33))
        off = float(np.median(b_[keep] - k * a_[keep]))
        res = np.abs(b_ - (k * a_ + off))
        keep = res < max(2.5 * np.median(res), 1e-6)
    spread_offset = float(np.percentile(np.abs((b_ - a_) - np.median(b_ - a_)), 75)) * sb
    spread = float(np.percentile(np.abs(b_ - (k * a_ + off)), 75)) * sb
    if spread > 0.9 * spread_offset:      # scaling doesn't really help: keep the plain offset
        k, off = 1.0, float(np.median(b_ - a_))
    V[:, 2] = k * V[:, 2] + off
    # the part can come out turned a few degrees against the body (yaw/pitch): rigidly lock its face onto the body's
    # face (ICP over the front-visible face region), about the face centre so the front image stays put
    zp = k * zp + off
    if back is not None:
        def all_spread(V_):
            zf_, zk_ = View(V_, Fp, body_fit, H, W).zfront, _zback(V_, Fp, back[0], H, W)
            zbk = _zback(body_V, body_F, back[0], H, W)
            m1 = depth_region & np.isfinite(zb) & np.isfinite(zf_)
            m2 = back[1] & np.isfinite(zbk) & np.isfinite(zk_)
            r = np.r_[np.abs(zb[m1] - zf_[m1]), np.abs(zbk[m2] - zk_[m2])]
            return float(np.percentile(r, 75)) * sb if len(r) else np.inf
        R, rot_deg = _lock_face(V, Fp, body_V, body_F, body_fit, H, W, depth_region, back=back)
        if R is not None:
            V2 = R(V)
            before, after = all_spread(V), all_spread(V2)
            info["lock"] = {"deg": round(rot_deg, 1), "spread_px": [round(before, 1), round(after, 1)]}
            if after < 0.85 * before:
                V = V2
                zp = View(V, Fp, body_fit, H, W).zfront
                ok = depth_region & np.isfinite(zb) & np.isfinite(zp)
    face_spread = float(np.percentile(np.abs(zb[ok] - zp[ok]), 75)) * sb
    info.update(place="ok", z_scale=round(k, 3), z_offset=round(off * sb, 1), z_spread_px=round(face_spread, 1),
                z_spread_all_px=round(float(np.percentile(np.abs(b_ - (k * a_ + off)), 75)) * sb, 1))
    return V, info


def _lock_face(V, F, bV, bF, fit, H, W, region, iters=12, max_deg=12.0, back=None):
    """Rigid rotation (about the face centre) taking the part's front-visible face points (and, with back = (fit,
    region), its back-visible back-of-head points) onto the body's surface.
    Returns (callable applying it, angle in degrees) or (None, 0) when there is too little face or it would turn
    the part implausibly far."""
    import trimesh
    from scipy.spatial import cKDTree
    from sculpt import View
    view = View(V, F, fit, H, W)
    zf = view.sample(view.zfront)
    vis = np.isfinite(zf) & (V[:, 2] >= zf - 0.01 * np.ptp(V[:, 1]))
    px = view.px
    xi, yi = np.clip(px[:, 0].astype(int), 0, W - 1), np.clip(px[:, 1].astype(int), 0, H - 1)
    sel = vis & region[yi, xi]
    if sel.sum() < 200:
        return None, 0.0
    c = V[sel].mean(0)
    if back is not None:
        bfit, breg = back
        Vm = V * np.array([-1.0, 1.0, -1.0])
        bview = View(Vm, F, bfit, H, W)
        zb_ = bview.sample(bview.zfront)
        bvis = np.isfinite(zb_) & (Vm[:, 2] >= zb_ - 0.01 * np.ptp(V[:, 1]))
        bx, by = np.clip(bview.px[:, 0].astype(int), 0, W - 1), np.clip(bview.px[:, 1].astype(int), 0, H - 1)
        bsel = bvis & breg[by, bx]
        if bsel.sum() > 200:   # equal weight for the back
            idx = np.nonzero(bsel)[0]
            idx = np.random.default_rng(0).choice(idx, min(len(idx), sel.sum()), replace=False)
            sel = sel.copy()
            sel[idx] = True
    P = V[sel]
    near = np.linalg.norm(bV - c, axis=1) < 2.0 * np.ptp(P, 0).max()
    tree = cKDTree(bV[near])
    Q = bV[near]
    Rt = np.eye(3)
    for _ in range(iters):
        X = (P - c) @ Rt.T + c
        d, nn = tree.query(X, workers=-1)
        keep = d < max(2.5 * np.median(d), 1e-6)
        M, _, _ = trimesh.registration.procrustes(X[keep] - c, Q[nn[keep]] - c, reflection=False, scale=False,
                                                  translation=False, return_cost=True)
        Rt = M[:3, :3] @ Rt
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1))))
    if ang > max_deg or ang < 0.3:
        return None, ang
    return (lambda X: (X - c) @ Rt.T + c), ang


def back_anchor(back_rgba, bV, bF, bb):
    """(body back-view fit, back-of-head region in back-view pixels) for place(), or None."""
    from texture import Model
    if back_rgba is None or bb is None:
        return None
    H, W = back_rgba.shape[:2]
    fit = Model.fit_geometry(*_decimated(bV, bF, 80000), back_rgba[..., 3] > 0.5, 180)
    x0, y0, S = bb[:3]
    m = np.zeros((H, W), bool)
    m[int(max(0, y0 + 0.1 * S)):int(min(H, y0 + 0.45 * S)), int(max(0, x0 + 0.3 * S)):int(min(W, x0 + 0.7 * S))] = True
    return fit, m


def merge(body_V, body_F, parts_list, depth=10, seed=0):
    """One watertight mesh from the body and replacement parts. parts_list: [(V, F, w_body, w_part)] where w_body is
    a per-body-vertex weight in [0, 1] of being replaced by this part and w_part a per-part-vertex weight of being
    used. Points are kept at random by those weights, so a transition band mixes both surfaces and screened Poisson
    blends them. Returns (V, F, info)."""
    import pymeshlab
    import trimesh
    rng = np.random.default_rng(seed)
    nb = trimesh.Trimesh(body_V, body_F, process=False).vertex_normals
    drop = np.zeros(len(body_V))
    for _, _, wb, _ in parts_list:
        drop = np.maximum(drop, wb)
    keep = rng.random(len(body_V)) >= drop
    P, N = [body_V[keep]], [nb[keep]]
    for Vp, Fp, _, wp in parts_list:
        npn = trimesh.Trimesh(Vp, Fp, process=False).vertex_normals
        k = rng.random(len(Vp)) < wp
        P.append(Vp[k]); N.append(npn[k])
    P, N = np.concatenate(P), np.concatenate(N)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=P, v_normals_matrix=N))
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, samplespernode=1.5, pointweight=4, preclean=True)
    # Poisson closes every surface it is given into a volume, sometimes with thin sheets where points are missing:
    # drop vertices far from any input point
    m = ms.current_mesh()
    from scipy.spatial import cKDTree
    V, F = m.vertex_matrix(), m.face_matrix()
    tree = cKDTree(P)
    d, nn = tree.query(V, workers=-1)
    spacing = tree.query(P, k=5, workers=-1)[0][:, 4]       # local sample spacing: the parts are sampled far denser than the body
    # only clearly spurious sheets: a surface Poisson spans across a gap the parts left (a sleeve's cuff around a new
    # wrist) is wanted, and is within a few samples of real points
    far = d > 10 * spacing[nn] + 1e-6
    if far.any():
        F = F[~far[F].any(1)]
    V, F = clean_mesh(V, F, min_diag_pct=4)   # leftover bits of a replaced mitten float free: drop them
    import pymeshlab as pml
    ms = pml.MeshSet()
    ms.add_mesh(pml.Mesh(V, F))
    ms.meshing_close_holes(maxholesize=3000)
    V, F = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
    return V, F, {"merge_points": int(len(P)), "merge_far_verts": int(far.sum()), "merge_faces": int(len(F))}


# --- hands ----------------------------------------------------------------------------------------------------------
# A hand built from MediaPipe's 21 joints as a smooth signed-distance field: a palm slab and tapered finger capsules.
# The generator makes hands as mittens and a hand is ~60 px in the source, too little for any image-to-3D model;
# MediaPipe is trained for exactly this (3D joints from one image, depth included). Radii are fractions of the palm
# width (knuckle 5 to 17), scaled as a whole to fit the source silhouette (slim hands, gloves).

FINGERS = {"thumb": ((1, 2, 3, 4), (0.17, 0.15, 0.13, 0.12)),
           "index": ((5, 6, 7, 8), (0.12, 0.11, 0.10, 0.09)),
           "middle": ((9, 10, 11, 12), (0.125, 0.115, 0.105, 0.095)),
           "ring": ((13, 14, 15, 16), (0.115, 0.105, 0.095, 0.085)),
           "pinky": ((17, 18, 19, 20), (0.10, 0.09, 0.08, 0.075))}
PALM_TRIS = ((0, 5, 9), (0, 9, 13), (0, 13, 17), (0, 1, 5))


def _round_cone(p, a, b, r1, r2):
    """Signed distance to a capsule tapering from radius r1 at a to r2 at b (Quilez)."""
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    x = pa * l2 - y[:, None] * ba
    x2 = (x * x).sum(1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d = np.where(np.sign(z) * a2 * z2 > k, np.sqrt(x2 + z2) * il2 - r2,
                 np.where(np.sign(y) * a2 * y2 < k, np.sqrt(x2 + y2) * il2 - r1,
                          (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1))
    return d


def _tri_dist(p, a, b, c):
    """Unsigned distance from points p (N, 3) to triangle abc."""
    ab, ac, ap = b - a, c - a, p - a
    n = np.cross(ab, ac)
    n2 = n @ n
    # barycentric of the projection; outside the triangle fall back to the nearest edge
    u = (np.cross(ap, ac) @ n) / n2
    v = (np.cross(ab, ap) @ n) / n2
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    d_plane = np.abs(ap @ n) / np.sqrt(n2)
    def seg(q0, q1):
        e = q1 - q0
        t = np.clip(((p - q0) @ e) / (e @ e), 0, 1)
        return np.linalg.norm(p - (q0 + t[:, None] * e), axis=1)
    d_edge = np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return np.where(inside, d_plane, d_edge)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def hand_sdf(X, P, scale=1.0, r_arm=None):
    """Signed distance at points X (N, 3) for joints P (21, 3), model units. Each finger blends into the palm only,
    never into its neighbours, so the gaps between fingers stay open."""
    w = np.linalg.norm(P[5] - P[17])
    k = 0.07 * w
    palm = np.full(len(X), np.inf)
    for (a, b, c), t in zip(PALM_TRIS, (0.19, 0.19, 0.18, 0.14)):
        palm = np.minimum(palm, _tri_dist(X, P[a], P[b], P[c]) - t * w * scale)
    # wrist and the start of the forearm, so the hand overlaps the sleeve or arm it is merged onto
    u = (P[9] - P[0]) / np.linalg.norm(P[9] - P[0])
    L = np.linalg.norm(P[9] - P[0])
    # (r_arm: the body's own forearm radius behind the wrist, so a bare arm meets the hand without a step)
    palm = _smin(palm, _round_cone(X, P[0] - 0.6 * L * u, P[0] + 0.15 * L * u, r_arm or 0.36 * w * scale,
                                   0.33 * w * scale), k)
    d = palm
    for name, (idx, rad) in FINGERS.items():
        f = np.full(len(X), 1e3)   # finite: the smooth minimum of inf is NaN
        for j in range(len(idx) - 1):
            a, b = P[idx[j]], P[idx[j + 1]]
            ra, rb = rad[j] * w * scale, rad[j + 1] * w * scale
            if j == len(idx) - 2:                           # the tip landmark is on the skin: end the cap there
                e = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                b = b - e * rb
            f = _smin(f, _round_cone(X, a, b, ra, rb), 0.02 * w)
        if name == "thumb":                                 # the thumb grows out of the wrist
            f = _smin(f, _round_cone(X, P[0], P[1], 0.2 * w * scale, rad[0] * w * scale), 0.02 * w)
        d = np.minimum(d, _smin(palm, f, k))
    return d


FINGER_LEN = {"thumb": 0.75, "index": 0.88, "middle": 0.97, "ring": 0.9, "pinky": 0.72}   # x palm length
BONE_SPLIT = (0.45, 0.30, 0.25)   # proximal, middle, distal share of a finger's length


def proportion_fingers(P):
    """Keep MediaPipe's finger directions but give each finger an anatomical length (x palm length, wrist to middle
    knuckle) and bone split: on a hand ~60 px across, its lengths are unreliable and come out long and spidery."""
    P = P.copy()
    L = np.linalg.norm(P[9] - P[0])
    for name, (idx, _) in FINGERS.items():
        base = P[idx[0]]
        total = FINGER_LEN[name] * L
        for j, share in zip(range(1, 4), BONE_SPLIT):
            d = P[idx[j]] - P[idx[j - 1]]
            n = np.linalg.norm(d)
            d = d / n if n > 1e-9 else (P[idx[0]] - P[0]) / max(np.linalg.norm(P[idx[0]] - P[0]), 1e-9)
            P[idx[j]] = base + d * share * total
            base = P[idx[j]]
    return P


def spread_fingers(P, scale=1.0, gap=0.12):
    """Move the joints of neighbouring fingers apart (in the palm's plane) wherever they are closer than their
    thickness plus `gap` (x palm width): fingers held together in the image would otherwise fuse into a mitten."""
    P = P.copy()
    w = np.linalg.norm(P[5] - P[17])
    n = np.cross(P[5] - P[0], P[17] - P[0])
    n /= max(np.linalg.norm(n), 1e-9)
    order = [FINGERS[k] for k in ("index", "middle", "ring", "pinky")]
    for _ in range(40):
        moved = False
        for (ia, ra), (ib, rb) in zip(order, order[1:]):
            for j in (1, 2, 3):                              # PIP, DIP, tip (the knuckles stay where they are)
                v = P[ia[j]] - P[ib[j]]
                v -= (v @ n) * n
                dist = np.linalg.norm(v)
                need = (ra[j] + rb[j]) * w * scale + gap * w
                if dist < need:
                    e = v / max(dist, 1e-9) if dist > 1e-9 else np.cross(n, P[ia[0]] - P[0]) / w
                    push = 0.5 * (need - dist) * e
                    for jj in range(j, 4):                   # carry the rest of each finger along
                        P[ia[jj]] += push
                        P[ib[jj]] -= push
                    moved = True
        if not moved:
            break
    return P


MAX_GRID = 160


def hand_mesh(P, scale=1.0, res=48, r_arm=None):
    """Marching cubes of hand_sdf; ~res voxels across the palm width. Returns (V, F)."""
    from skimage.measure import marching_cubes
    w = np.linalg.norm(P[5] - P[17])
    L = np.linalg.norm(P[9] - P[0])
    u = (P[9] - P[0]) / L
    pts = np.vstack([P, P[0] - 0.7 * L * u])
    lo, hi = pts.min(0) - 0.5 * w * scale, pts.max(0) + 0.5 * w * scale
    vox = max(w / res, float((hi - lo).max()) / MAX_GRID)   # a sprawling joint estimate must not explode the grid
    n = np.ceil((hi - lo) / vox).astype(int) + 1
    g = np.stack(np.meshgrid(*[lo[i] + vox * np.arange(n[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    d = hand_sdf(g, P, scale, r_arm).reshape(n)
    # a light blur of the distance field rounds off the palm slab's facets and the steps between finger segments
    from scipy.ndimage import gaussian_filter
    d = gaussian_filter(d, res / 32)
    V, F, _, _ = marching_cubes(d, 0.0, spacing=(vox, vox, vox))
    return V + lo, F


def _smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _hand_shift(L, src, sil, reach=40):
    """(dx, dy) px that best lays the image's hand and wrist (src mask) over the mesh's silhouette (sil), searched in
    a window around the hand."""
    H, W = src.shape
    P = L[:, :2]
    S = np.ptp(P, 0).max()
    x0, y0 = np.maximum((P.min(0) - 0.5 * S - reach).astype(int), 0)
    x1, y1 = np.minimum((P.max(0) + 0.5 * S + reach).astype(int), [W, H])
    a, b = src[y0:y1, x0:x1], sil[y0:y1, x0:x1]
    ys, xs = np.nonzero(a)
    if len(xs) < 20:
        return np.zeros(2)

    def iou(dx, dy):
        X, Y = xs + dx, ys + dy
        ok = (X >= 0) & (X < b.shape[1]) & (Y >= 0) & (Y < b.shape[0])
        inter = b[Y[ok], X[ok]].sum()
        return inter / (len(xs) + b.sum() - inter)
    best = max((iou(dx, dy), dx, dy) for dx in range(-reach, reach + 1, 3) for dy in range(-reach, reach + 1, 3))
    best = max((iou(best[1] + i, best[2] + j), best[1] + i, best[2] + j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2))
    if best[0] - iou(0, 0) < 0.02:      # no real improvement: keep the image's position
        return np.zeros(2)
    return np.array([best[1], best[2]], float)


def hand_part(L, body_V, body_F, body_fit, front_rgba):
    """Procedural hand for one MediaPipe detection L (21 x 3, source px), placed in the body's model space.
    Returns (V, F, w_body, w_part, info); V is None when the hand doesn't fit the source silhouette (a gauntlet, a
    claw, a mis-detection), and the body keeps its own hand."""
    import torch
    from sculpt import View
    from texture import dev, rasterize
    H, W = front_rgba.shape[:2]
    s, tx, ty, ax = body_fit
    view = View(body_V, body_F, body_fit, H, W)
    # the generator puts the hand a few pixels from where the image has it, and the arm ends at the mesh's hand:
    # move the joints onto the mesh's hand (the texture is shifted to match, see enhance)
    shift = _hand_shift(L, front_rgba[..., 3] > 0.5, np.isfinite(view.zfront))
    L = L.copy()
    L[:, :2] += shift
    zf = view.sample(view.zfront, L[:, :2])
    P = np.c_[(L[:, 0] - tx) / (s * ax), (ty - L[:, 1]) / s, -(L[:, 2] - np.median(L[:, 2])) / s]
    # MediaPipe's relative depth can sprawl for a hand seen edge-on: keep it within a hand's length of the wrist
    Lxy = max(np.linalg.norm(P[9, :2] - P[0, :2]), np.ptp(P[:, :2], 0).max() / 2.5)
    P[:, 2] = np.clip(P[:, 2], -1.2 * Lxy, 1.2 * Lxy)
    w = np.linalg.norm(P[5] - P[17])
    Lp = np.linalg.norm(P[9] - P[0])
    if not np.isfinite(zf).any() or w < 1e-6 or Lp < 1e-6:
        return None, None, None, None, {"hand": "not on the mesh"}
    # the front surface seen at the joints is the near side of the hand: centre the joints half a palm-thickness in
    P[:, 2] += np.nanmedian(zf) - np.nanmedian(P[:, 2]) - 0.19 * w
    P = proportion_fingers(P)
    u = (P[9] - P[0]) / Lp
    # the hand proper in the image (past the wrist), where its silhouette is compared with the source's
    gx, gy = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    u2 = (L[9, :2] - L[0, :2]) / max(np.linalg.norm(L[9, :2] - L[0, :2]), 1e-6)
    t2 = ((gx - L[0, 0]) * u2[0] + (gy - L[0, 1]) * u2[1]) / max(np.linalg.norm(L[9, :2] - L[0, :2]), 1e-6)
    x0, y0 = L[:, :2].min(0) - 0.6 * np.ptp(L[:, :2], 0).max()
    x1, y1 = L[:, :2].max(0) + 0.6 * np.ptp(L[:, :2], 0).max()
    win = (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1) & (t2 > 0.1)
    src = np.roll(front_rgba[..., 3] > 0.5, (int(round(shift[1])), int(round(shift[0]))), (0, 1))
    best = None
    for scale in (0.8, 0.9, 1.0, 1.12):
        V, F = hand_mesh(spread_fingers(P, scale), scale, res=20)
        xy = torch.tensor(np.c_[tx + s * ax * V[:, 0], ty - s * V[:, 1]], device=dev, dtype=torch.float32)
        fid, _, _ = rasterize(xy, torch.tensor(-V[:, 2], device=dev, dtype=torch.float32),
                              torch.tensor(F.astype(np.int64), device=dev), H, W)
        sil = (fid >= 0).cpu().numpy()
        iou = float((sil & src & win).sum() / max(1, ((sil | src) & win).sum()))
        if best is None or iou > best[0]:
            best = (iou, scale)
    iou, scale = best
    info = {"hand_iou": round(iou, 3), "hand_scale": scale, "palm_px": round(float(w * s), 1),
            "depth_spread_px": round(float(np.ptp(P[:, 2]) * s), 1),
            "shift_px": [round(float(v), 1) for v in shift]}
    if iou < 0.55:
        info["hand"] = "silhouette doesn't fit"
        return None, None, None, None, info
    P = spread_fingers(P, scale)
    # the body's forearm radius just behind the wrist (median distance of its surface from the hand's axis)
    tb = (body_V - P[0]) @ u / Lp
    radial = np.linalg.norm((body_V - P[0]) - np.outer(tb * Lp, u), axis=1)
    sel = (tb > -0.9) & (tb < -0.4) & (radial < 1.2 * w)
    r_arm = float(np.clip(np.median(radial[sel]), 0.3 * w, 0.8 * w)) if sel.sum() > 30 else None
    info["arm_radius_w"] = None if r_arm is None else round(r_arm / w, 2)
    V, F = hand_mesh(P, scale, r_arm=r_arm)
    from scipy.spatial import cKDTree
    t_body = (body_V - P[0]) @ u / Lp
    d, _ = cKDTree(V).query(body_V, workers=-1)
    from sculpt import _dist_to_skeleton
    # the mesh's own hand: near the new surface, or anywhere past the wrist within a hand's length of the bones (the
    # generator's mitten can point somewhere else and is fatter than the hand)
    near = np.maximum(1 - _smooth01((d - 0.15 * w) / (0.25 * w)),
                      1 - _smooth01((_dist_to_skeleton(body_V, P) - 1.2 * Lp) / (0.3 * Lp)))
    w_body = _smooth01((t_body + 0.15) / 0.2) * near
    w_part = _smooth01(((V - P[0]) @ u / Lp + 0.25) / 0.2)
    info["hand"] = "ok"
    info["P"] = P
    return V, F, w_body, w_part, info


def shift_region(rgba, box, dx, dy):
    """Move the image content inside a square box by (dx, dy) px, fading out over half a box around it."""
    H, W = rgba.shape[:2]
    x0, y0, S = box[:3]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    cx, cy = x0 + S / 2, y0 + S / 2
    d = np.maximum(np.abs(gx - cx - dx / 2), np.abs(gy - cy - dy / 2)) - S / 2
    w = 1 - _smooth01(d / (0.5 * S))
    return cv2.remap(rgba, gx - dx * w, gy - dy * w, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def enhance(rgba, pieces, factor=3):
    """The view image upscaled by `factor`, with registered close-ups pasted over their crops (feathered at the
    crop's edges). pieces: [{"box", "close" (RGB or None), "shift" (dx, dy) or None}]; shifts move a region first
    (a hand moved onto the mesh's hand). Alpha stays the view's own cut-out, upscaled. Lets the bake take the
    close-ups' detail through the unchanged projection code: only the fit is multiplied by `factor`."""
    for p in pieces:
        if p.get("shift") is not None and np.any(p["shift"]):
            rgba = shift_region(rgba, p["box"], *p["shift"])
    pieces = [(p["box"] if p.get("shift") is None else (p["box"][0] + p["shift"][0], p["box"][1] + p["shift"][1],
                                                         p["box"][2]), p["close"]) for p in pieces
              if p.get("close") is not None]
    H, W = rgba.shape[:2]
    big = cv2.resize(rgba, (W * factor, H * factor), interpolation=cv2.INTER_CUBIC)
    big[..., 3] = cv2.resize(rgba[..., 3], (W * factor, H * factor), interpolation=cv2.INTER_LINEAR)
    big = np.clip(big, 0, 1)
    for box, close in pieces:
        x0, y0, S = (v * factor for v in box[:3])
        n = close.shape[0]
        k = S / n
        M = np.float32([[k, 0, x0], [0, k, y0]])
        warped = cv2.warpAffine(close.astype(np.float32) / 255, M, (W * factor, H * factor), flags=cv2.INTER_AREA
                                if k < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if close.shape[2] == 4:   # an RGBA close-up also extends the cut-out (hair beyond the bald silhouette)
            warped, wa = warped[..., :3], warped[..., 3]
        # feather: 0 at the crop border, 1 from 8% inside
        ramp = np.minimum.reduce([np.arange(n), np.arange(n)[::-1]]).astype(np.float32) / (0.08 * n)
        ramp = np.clip(ramp, 0, 1)
        m = np.minimum(ramp[None, :], ramp[:, None])
        m = m * m * (3 - 2 * m)
        wm = cv2.warpAffine(m, M, (W * factor, H * factor), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)[..., None]
        big[..., :3] = big[..., :3] * (1 - wm) + warped * wm
        if close.shape[2] == 4:   # inside the box the close-up's own cut-out decides (hair added, headwear gone)
            big[..., 3] = big[..., 3] * (1 - wm[..., 0]) + wa * wm[..., 0]
    return big


# --- pipeline entry -------------------------------------------------------------------------------------------------

def body_mesh(path):
    """The generator mesh as Model prepares it (repaired, HC-smoothed), at full resolution."""
    import pymeshlab
    import trimesh
    m = trimesh.load(path, force="mesh")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(3))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=200)
    for _ in range(2):
        ms.apply_coord_hc_laplacian_smoothing()
    c = ms.current_mesh()
    return c.vertex_matrix(), c.face_matrix()


def back_box(front_rgba, back_rgba, box):
    """Where a front-view box lands in the back view: same height fraction, mirrored about the silhouettes."""
    def bb(a):
        ys, xs = np.nonzero(a[..., 3] > 0.5)
        return xs.min(), ys.min(), xs.max(), ys.max()
    fx0, fy0, fx1, fy1 = bb(front_rgba)
    bx0, by0, bx1, by1 = bb(back_rgba)
    k = (by1 - by0) / max(1, fy1 - fy0)
    x0, y0, S = box[:3]
    return float(bx1 - (x0 + S - fx0) * k), float(by0 + (y0 - fy0) * k), float(S * k)


def run(job_dir, views, klein_fn, gpu_lock, progress=lambda msg: None, stats=None):
    """The part pass for a job. views: {"front": float RGBA[, "back": ..., "haired": ...]} (cut-outs; "haired" is the
    front before the bald edit, for the hair layer). klein_fn(prompt, ref_path,
    seed) -> generated image path. Writes <job>/shape_parts.glb and <job>/parts/*.
    Returns (mesh name to bake, replaced dict for Model, {view: [pieces]} for enhance, hair) where hair is None or
    (V, F, {view: [pieces]}) for the hair mesh."""
    import json
    import trimesh
    from PIL import Image as PILImage
    from shape import cutout
    from texture import Model
    import contextlib
    import torch
    stats = {} if stats is None else stats

    @contextlib.contextmanager
    def gpu():
        # hand the cached VRAM back before klein runs again: the image server needs ~5 GB of the 6 GB card
        with gpu_lock():
            try:
                yield
            finally:
                import gc
                gc.collect()
                empty_cache()
    pd = os.path.join(job_dir, "parts")
    os.makedirs(pd, exist_ok=True)
    front, back, haired = views["front"], views.get("back"), views.get("haired")
    bald = " The character's head is completely bald: no hair at all." if haired is not None else ""
    H, W = front.shape[:2]
    frgb = flat_rgb(front)
    pieces = {"front": [], "back": []}
    replaced = {}
    plist = []

    progress("finding head and hands")
    bV, bF = body_mesh(os.path.join(job_dir, "shape.glb"))
    with gpu():
        bfit = Model.fit_geometry(*_decimated(bV, bF, 80000), front[..., 3] > 0.5)

    # head: close-ups (front, back), shape from them, placed and weighted
    hb = head_box(front)
    face = detect_face(frgb)
    if hb is not None and face is not None:
        progress("klein: head close-up")
        hc, info = closeup(klein_fn, crop(frgb, hb), HEAD_PROMPT + bald, os.path.join(pd, "head_crop.png"))
        stats["head_close"] = info
        hbc = None
        if hc is not None:
            PILImage.fromarray(hc).save(os.path.join(pd, "head_close.png"))
            pieces["front"].append({"box": hb, "close": hc})
            if back is not None:
                progress("klein: back of the head")
                bb = back_box(front, back, hb)
                hbc, info = closeup(klein_fn, crop(flat_rgb(back), bb), BACK_PROMPT + bald,
                                   os.path.join(pd, "head_back_crop.png"))
                stats["head_back_close"] = info
                if hbc is not None:
                    PILImage.fromarray(hbc).save(os.path.join(pd, "head_back_close.png"))
                    pieces["back"].append({"box": bb, "close": hbc})
            progress("Hunyuan3D: head")
            hv = {"front": cutout(PILImage.fromarray(hc))}
            if hbc is not None:
                hv["back"] = cutout(PILImage.fromarray(hbc))
            fx, fy, fw, fh = face[:4]
            region = np.zeros((H, W), bool)
            region[int(fy + 0.2 * fh):int(fy + 0.9 * fh), int(fx + 0.15 * fw):int(fx + 0.85 * fw)] = True
            cmask = np.asarray(hv["front"])[..., 3] > 127
            with gpu():
                anchor = back_anchor(back, bV, bF, back_box(front, back, hb)) if back is not None else None
            V, pF, info = shape_and_place(hv, gpu, cmask, hb, bfit, bV, bF, H, W, region,
                                          (fy + 1.35 * fh - hb[1]) * cmask.shape[0] / hb[2], fw, stats, "head",
                                          back=anchor)
            # a head that doesn't sit on the body's own (the depth disagrees across the face) is not used
            if V is not None and info["z_spread_px"] < 0.12 * fw:
                s, tx, ty, ax = bfit
                x0, y0, S = hb

                def wpix(X):
                    px, py = tx + s * ax * X[:, 0], ty - s * X[:, 1]
                    wy = np.clip((fy + 1.55 * fh - py) / (0.4 * fh), 0, 1)
                    wx = np.clip(np.minimum(px - x0, x0 + S - px) / (0.06 * S), 0, 1)
                    return _smooth01(wy * wx)
                from scipy.spatial import cKDTree
                dn, _ = cKDTree(V).query(bV, workers=-1)
                near = np.clip(1 - (dn * s - 0.06 * S) / (0.06 * S), 0, 1)   # what isn't near the head stays
                plist.append((V, pF, wpix(bV) * near, wpix(V)))
                replaced["head"] = True
                trimesh.Trimesh(V, pF, process=False).export(os.path.join(pd, "head.glb"))
            else:
                stats["head"] = "placement rejected"

    # hands: procedural from the joints; close-ups only for texture
    hands_P = []
    for i, (x0, y0, S, L) in enumerate(hand_boxes(front)):
        with gpu():
            hV, hF, wb, wp, info = hand_part(L, bV, bF, bfit, front)
        piece = {"box": (x0, y0, S), "close": None, "shift": None}
        if hV is not None:
            plist.append((hV, hF, wb, wp))
            hands_P.append(info.pop("P"))
            piece["shift"] = tuple(info["shift_px"])
        stats[f"hand{i}"] = info
        pieces["front"].append(piece)
        if hV is not None:    # a rebuilt hand gets a clean skin albedo (Model.skin_clean): no close-up needed
            continue
        progress(f"klein: hand close-up ({i + 1})")
        hcu, info = closeup(klein_fn, crop(frgb, (x0, y0, S)), HAND_PROMPT, os.path.join(pd, f"hand{i}_crop.png"))
        stats[f"hand{i}_close"] = info
        if hcu is not None:
            PILImage.fromarray(hcu).save(os.path.join(pd, f"hand{i}_close.png"))
            piece["close"] = hcu
    if hands_P:
        replaced["hands"] = [P.tolist() for P in hands_P]

    mesh_name = "shape.glb"
    if plist:
        progress("merging")
        bV, bF, info = merge(bV, bF, plist)
        stats.update(info)
        mesh_name = "shape_parts.glb"
        trimesh.Trimesh(bV, bF, process=False).export(os.path.join(job_dir, mesh_name))
    else:
        stats["parts"] = "nothing replaced"
    hair = None
    if haired is not None and face is not None:
        hair = hair_layer(job_dir, haired, front, back, (bV, bF), bfit, face, klein_fn, gpu, progress, stats)
    with open(os.path.join(pd, "parts.json"), "w") as f:
        json.dump({"replaced": replaced, "stats": stats}, f, indent=1, default=str)
    return mesh_name, (replaced or None), pieces, hair


# --- hair layer -----------------------------------------------------------------------------------------------------
# The body is built from a bald edit of the image (make3d BALD): a clean scalp, ears and face. The hair (and headwear,
# which the edit removes with it) is built here from the haired image and kept as its own mesh, one solid piece.

# The hair close-up redraws exactly what is there. Naming things in a prompt makes klein add them ("hat, goggles"
# gave a bare-headed character goggles; "draw the hair" gave hair under a hat), so headwear is never named and the
# hair sentence is only added when face parsing finds visible hair (HAIR_STYLE). Sculpted locks, because single
# strands and wisps come out of the shape model as paper-thin flaps.
HAIR_PROMPT = ("Image 1 is a blurry low-resolution crop of a 3D game character's head and shoulders. Redraw image 1 "
               "as a sharp, highly detailed, high-resolution close-up of exactly the same character, changing nothing: "
               "the same face, the same clothing, colours and materials, and everything on the head exactly as it is "
               "in image 1. Keep exactly the same framing, pose, head angle, size and position as image 1. Soft even "
               "lighting, plain light grey background.")
HAIR_STYLE = " Draw the hair as clean, smooth, sculpted locks, without loose strands or flyaway wisps."
HAIR_BACK_PROMPT = ("Image 1 is a crop of a 3D game character's bald head and shoulders seen from directly behind. "
                    "Image 2 shows the same character from the front. Give the head in image 1 exactly what image 2 "
                    "has on its head, seen from behind, in the same shape, length, colour and volume, adding nothing "
                    "that image 2 does not show. Do not show a face. Keep the "
                    "framing, size and position of image 1, and keep the shoulders and clothing unchanged. Soft even "
                    "lighting, plain light grey background.")
HAIR_EPS = 0.02        # x face height: how far outside the bald body a surface must be to count as hair
HAIR_MIN = 0.012       # x face height: hair is held at least this far off the scalp (no scalp showing through)
HAIR_CLOSE = 8         # vertex rings of morphological closing over the hair region (fills thin-hair gaps)


def hair_mask(haired_rgb, bald_rgb, near_box, reach=(0.35, 0.15, 0.9)):
    """Pixels that changed between the aligned haired and bald images (the hair and any headwear): connected regions
    that reach into `near_box` (the head), inside a window around it (reach: sideways, up, down, x box size; the
    bald edit also redraws small details elsewhere, a scarf fold or a hand)."""
    a, b = (cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2.0) for x in (haired_rgb, bald_rgb))
    d = np.abs(a - b).sum(-1)
    m = (d > 70).astype(np.uint8)
    H, W = m.shape
    x0, y0, S = (int(v) for v in near_box[:3])
    win = np.zeros_like(m)
    win[max(0, int(y0 - reach[1] * S)):min(H, int(y0 + S + reach[2] * S)),
        max(0, int(x0 - reach[0] * S)):min(W, int(x0 + S + reach[0] * S))] = 1
    m = cv2.morphologyEx(m * win, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    inner = lab[max(0, y0):min(H, y0 + S), max(0, x0):min(W, x0 + S)]
    keep = [k for k in np.unique(inner) if k > 0 and st[k, cv2.CC_STAT_AREA] > 0.002 * S * S]
    return np.isin(lab, keep)


def _square_box(mask, base, margin=0.06):
    """Square (x0, y0, S) covering the base box and the mask's bounding box."""
    ys, xs = np.nonzero(mask)
    bx0, by0, S = base
    x0, y0, x1, y1 = bx0, by0, bx0 + S, by0 + S
    if len(xs):
        x0, y0, x1, y1 = min(x0, xs.min()), min(y0, ys.min()), max(x1, xs.max()), max(y1, ys.max())
    S = min(max(x1 - x0, y1 - y0) * (1 + 2 * margin), 2.0 * base[2])
    return float((x0 + x1) / 2 - S / 2), float(min((y0 + y1) / 2 - S / 2, y0)), float(S)


def hair_layer(job_dir, haired, bald, back, body, bfit, face, klein_fn, gpu, progress, stats):
    """Build the hair mesh. haired, bald, back: float RGBA cut-outs (front with hair, front bald, back bald).
    body: (V, F) of the merged bald body (model space). Returns (V, F, texture pieces) or None."""
    import trimesh
    from PIL import Image as PILImage
    from scipy.spatial import cKDTree
    from shape import cutout
    from texture import Model
    pd = os.path.join(job_dir, "parts")
    H, W = haired.shape[:2]
    hrgb, brgb = flat_rgb(haired), flat_rgb(bald)
    base = head_box(haired) or head_box(bald)
    if base is None:
        return None
    mask_f = hair_mask(hrgb, brgb, base)
    # the bald edit also redraws small facial details (brows, beard edge, eyes): face parsing on the haired image
    # keeps the face itself out of the hair, and inside the face outline only parsed hair, hat or glasses count
    labels = landmarks.face_parse(hrgb, detect_face(hrgb))
    if labels is not None:
        face_parts = np.isin(labels, (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        L = landmarks.face_mesh(hrgb, detect_face(hrgb))
        oval = np.zeros(labels.shape, np.uint8)
        if L is not None:
            cv2.fillPoly(oval, [np.round(L[landmarks.face_oval(), :2]).astype(np.int32)], 1)
        headwear = np.isin(labels, (3, 13, 14))
        # facial hair the bald edit took off (a beard): skin in the bald image, clearly darker in the haired one
        bl = landmarks.face_parse(brgb, detect_face(brgb))
        lum = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2Lab)[..., 0].astype(np.float32)  # noqa: E731
        removed = (bl == 1) & (lum(brgb) - lum(hrgb) > 40) if bl is not None else np.zeros_like(mask_f)
        mask_f &= (~face_parts | removed) & (~oval.astype(bool) | headwear | removed)
    stats["hair_px"] = int(mask_f.sum())
    if mask_f.sum() < 200:
        stats["hair"] = "no hair to add"
        return None
    box = _square_box(mask_f, base)
    progress("klein: hair close-up")
    labels = landmarks.face_parse(hrgb, detect_face(hrgb))
    hair_px = int((labels == landmarks.HAIR).sum()) if labels is not None else 0
    stats["hair_visible_px"] = hair_px
    style = HAIR_STYLE if hair_px > 0.15 * mask_f.sum() else ""
    hc, info = closeup(klein_fn, crop(hrgb, box), HAIR_PROMPT + style, os.path.join(pd, "hair_crop.png"))
    stats["hair_close"] = info
    if hc is None:
        stats["hair"] = "close-up failed"
        return None
    PILImage.fromarray(hc).save(os.path.join(pd, "hair_close.png"))
    hcut = cutout(PILImage.fromarray(hc))
    views = {"front": hcut}
    bb, hbc = None, None
    if back is not None:
        progress("klein: hair from behind")
        bb = back_box(bald, back, box)
        hbc, info = closeup(klein_fn, crop(flat_rgb(back), bb), HAIR_BACK_PROMPT, os.path.join(pd, "hair_back_crop.png"),
                            extra_refs=[os.path.join(pd, "hair_close.png")], check=False)
        stats["hair_back_close"] = info
        if hbc is not None:
            PILImage.fromarray(hbc).save(os.path.join(pd, "hair_back_close.png"))
            views["back"] = cutout(PILImage.fromarray(hbc))
    progress("Hunyuan3D: hair")
    fx, fy, fw, fh = face[:4]
    region = np.zeros((H, W), bool)
    region[int(fy + 0.2 * fh):int(fy + 0.9 * fh), int(fx + 0.15 * fw):int(fx + 0.85 * fw)] = True
    cmask = np.asarray(hcut)[..., 3] > 127
    bV, bF = body
    with gpu():
        anchor = back_anchor(back, bV, bF, bb) if bb is not None else None
    V, pF, info = shape_and_place(views, gpu, cmask, box, bfit, bV, bF, H, W, region,
                                  (fy + 1.35 * fh - box[1]) * cmask.shape[0] / box[2], fw, stats, "hair",
                                  back=anchor)
    if V is None:
        stats["hair"] = "placement failed"
        return None
    # hair = the haired head's surface lying outside the bald body, where the images say there is hair
    s, tx, ty, ax = bfit
    nb = trimesh.Trimesh(bV, bF, process=False).vertex_normals
    d, nn = cKDTree(bV).query(V, workers=-1)
    sd = ((V - bV[nn]) * nb[nn]).sum(1)
    eps = HAIR_EPS * fh / s
    px = np.c_[tx + s * ax * V[:, 0], ty - s * V[:, 1]]
    grow = cv2.dilate(mask_f.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    xi, yi = np.clip(px[:, 0].astype(int), 0, W - 1), np.clip(px[:, 1].astype(int), 0, H - 1)
    # a point counts as hair in a view only where that view sees it (the face projects onto the back view's hair too)
    from sculpt import View
    tol = 0.03 * fh / s
    with gpu():
        zf = View(V, pF, bfit, H, W).zfront
    vis_f = np.isfinite(zf[yi, xi]) & (V[:, 2] >= zf[yi, xi] - tol)
    seen = grow[yi, xi] & vis_f
    if hbc is not None:
        # from behind: the haired back close-up against the bald back crop (around the head), in back-view pixels
        n = hbc.shape[0]
        bmask = hair_mask(hbc, crop(flat_rgb(back), bb), (0.25 * n, 0.0, 0.5 * n))
        with gpu():
            bfit_b = Model.fit_geometry(*_decimated(bV, bF, 80000), back[..., 3] > 0.5, 180)
            zbk = _zback(V, pF, bfit_b, H, W)
        from texture import camera
        _, right_b = camera(180)
        sb, txb, tyb, axb = bfit_b
        gx, gy = txb + sb * axb * (V @ right_b), tyb - sb * V[:, 1]       # back-view pixels
        gxi, gyi = np.clip(gx.astype(int), 0, W - 1), np.clip(gy.astype(int), 0, H - 1)
        vis_b = np.isfinite(zbk[gyi, gxi]) & (V[:, 2] <= zbk[gyi, gxi] + tol)
        k = n / bb[2]
        bx, by = (gx - bb[0]) * k, (gy - bb[1]) * k
        bm = cv2.dilate(bmask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        inb = (bx >= 0) & (bx < n) & (by >= 0) & (by < n)
        seen |= inb & vis_b & bm[np.clip(by.astype(int), 0, n - 1), np.clip(bx.astype(int), 0, n - 1)]
        stats["hair_back_fit"] = [round(float(v), 3) for v in bfit_b]
    # thin hair (pulled back, short) lies within a few mm of the scalp: take anything clearly off it where the images
    # show hair, close the gaps over the surface, then hold every hair vertex at least HAIR_MIN off the scalp
    # above the chin, whatever the images mark as hair is hair, even where the haired head dips inside the bald one
    above_chin = px[:, 1] < fy + fh
    is_hair = ((sd > 0.25 * eps) | above_chin) & seen
    from sculpt import umbrella
    Ab = umbrella(pF, len(V))
    Ab.data[:] = 1.0
    for _ in range(HAIR_CLOSE):
        is_hair = (Ab @ is_hair.astype(np.float32)) > 0
    for _ in range(HAIR_CLOSE):
        is_hair = (Ab @ (~is_hair).astype(np.float32)) == 0
    if os.environ.get("HAIR_DEBUG"):
        np.savez(os.environ["HAIR_DEBUG"], V=V, F=pF, is_hair=is_hair, seen=seen, sd=sd * s, above=above_chin)
    lift = np.clip(HAIR_MIN * fh / s - sd, 0, None) * is_hair
    V = V + nb[nn] * lift[:, None]
    stats["hair_lifted_pct"] = round(float((lift > 0).sum() / max(1, is_hair.sum()) * 100), 1)
    Fh = pF[is_hair[pF].all(1)]
    if len(Fh) < 500:
        stats["hair"] = f"too little hair geometry ({len(Fh)} faces)"
        return None
    m = trimesh.Trimesh(V, Fh, process=True)
    parts_ = m.split(only_watertight=False)
    big = max(len(p.faces) for p in parts_)
    m = trimesh.util.concatenate([p for p in parts_ if len(p.faces) > 0.02 * big])
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=60)   # pinholes show the scalp; the shell's own open rim is far longer
    ms.apply_coord_taubin_smoothing(stepsmoothnum=6)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=HAIR_FACES, preserveboundary=True, preservenormal=True,
                                                qualitythr=0.4)
    hV, hF = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
    trimesh.Trimesh(hV, hF, process=False).export(os.path.join(job_dir, "hair.glb"))
    stats.update(hair="ok", hair_faces=len(hF))
    hca = np.concatenate([hc, np.asarray(hcut)[..., 3:]], -1)
    pieces = {"front": [{"box": box, "close": hca}], "cap_front": mask_f}
    if hbc is not None:
        pieces["back"] = [{"box": bb, "close": np.concatenate([hbc, np.asarray(views["back"])[..., 3:]], -1)}]
        pieces["cap_back"] = (bb, hbc, bmask)
    return hV, hF, pieces


HAIR_FACES = 10000


def uncrop(img, box, H, W):
    """Inverse of crop(): a close-up placed back into an H x W image (zeros outside the box)."""
    x0, y0, S = box[:3]
    k = S / img.shape[0]
    M = np.float32([[k, 0, x0], [0, k, y0]])
    return cv2.warpAffine(img.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)
