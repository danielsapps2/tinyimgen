"""Image-guided repair of the face and hands on the full-resolution generator mesh (before decimation).

Hunyuan3D leaves faces round and soft (no lip line, shallow eyes, the nose a few pixels off where the image has it)
and hands as lumpy mittens. With the source image's dense landmarks (MediaPipe, see landmarks.py):

* face: the front surface inside the face oval is moved along the view axis onto the depth of the image's own 478
  landmarks. x and y stay put, so the front projection paints every feature exactly onto its geometry, and side views
  get a real nose, eye sockets and lips. Only depth comes from MediaPipe; its scale matches the meshes (fitted slope
  1.04 on a test character), and it is anchored to the mesh by the median offset over the inner face.
* hands: extra Taubin smoothing around the hand landmarks evens out the lumps without shrinking the fingers away.

Both regions are subdivided first (the generator's ~5 mm edges are too coarse for lips) and get a high importance
weight, so the decimator spends more of the triangle budget on them.
"""
import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import trimesh
from scipy.interpolate import RBFInterpolator

import landmarks
from texture import detect_face, dev, rasterize

FACE_IMPORTANCE = 10.0   # decimation error multiplier inside the face
HAND_IMPORTANCE = 6.0
HAND_SMOOTH = 30         # Taubin iterations on the hands: evens out knuckle lumps and marching-cubes ribbing
HAND_LAMBDA = 0.5
TPS_SMOOTHING = 10.0     # thin-plate spline regularisation for the landmark depth surface
FIX_W = 0.9              # only the well-seen face core is held at its target; the biharmonic fill shapes the rest
ROI = 1.0                # face sculpt: the displacement fades out smoothly within this many face heights
FACING = 0.25           # face sculpt fades out where the surface turns away from the front view (normal z below this)
DETAIL_SIGMA = 0.0       # face: take landmark relief finer than this (x face width) only; 0 = whole depth (seamless)


def umbrella(F, n):
    """Row-normalised uniform adjacency (mean of neighbours)."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1)) @ A


def taubin(V, A, w, iters, lam=0.5, mu=-0.53):
    """Taubin smoothing with a per-vertex strength w in [0, 1] (0 = untouched)."""
    w = w[:, None]
    for _ in range(iters):
        V = V + lam * w * (A @ V - V)
        V = V + mu * w * (A @ V - V)
    return V


def hc_laplacian(V, A, w, iters, alpha=0.0, beta=0.5):
    """HC Laplacian smoothing (Vollmer et al. 1999) with a per-vertex strength w in [0, 1]: a real low-pass that
    removes finger-scale lumps, with the shrinking of plain Laplacian smoothing pushed back each step."""
    O = V.copy()
    w = w[:, None]
    for _ in range(iters):
        P = A @ V
        B = P - (alpha * O + (1 - alpha) * V)
        V = V + w * (P - (beta * B + (1 - beta) * (A @ B)) - V)
    return V


class View:
    """Front orthographic projection of a mesh with the bake's fit (scale, tx, ty, ax)."""

    def __init__(self, V, F, fit, H, W):
        self.fit, self.H, self.W = fit, H, W
        self.px = self.project(V)
        Vt = torch.tensor(V, device=dev, dtype=torch.float32)
        Ft = torch.tensor(F, device=dev)
        xy = torch.tensor(self.px, device=dev, dtype=torch.float32)
        depth = -Vt[:, 2]
        fid, (ys, xs), b = rasterize(xy, depth, Ft, H, W)
        z = torch.full((H, W), float("nan"), device=dev)
        z[ys, xs] = -(b * depth[Ft[fid[ys, xs]]]).sum(-1)
        self.zfront = z.cpu().numpy()          # model-space Z of the nearest surface per pixel (+Z = toward camera)

    def project(self, V):
        s, tx, ty, ax = self.fit
        return np.stack([tx + s * ax * V[:, 0], ty - s * V[:, 1]], -1)

    def sample(self, img, pts=None):
        """Bilinear lookup at the vertices' (or pts') pixel positions; NaN where any neighbour is undefined."""
        p = self.px if pts is None else pts
        x = np.clip(p[:, 0] - 0.5, 0, self.W - 1.001)   # pixel centres sit at +0.5 (rasterize)
        y = np.clip(p[:, 1] - 0.5, 0, self.H - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
                + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)

    def visible(self, V, tol):
        zf = self.sample(self.zfront)
        return np.isfinite(zf) & (V[:, 2] >= zf - tol)


def _subdivide(V, F, sel_v, rounds=1):
    """Midpoint-subdivide faces touching selected vertices; returns V, F and the selection carried over."""
    import pymeshlab
    for _ in range(rounds):
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(V, F, v_scalar_array=sel_v.astype(np.float64)))
        ms.compute_selection_by_condition_per_face(condselect="(q0 > 0.5) || (q1 > 0.5) || (q2 > 0.5)")
        ms.meshing_surface_subdivision_midpoint(iterations=1, threshold=pymeshlab.PercentageValue(0.05), selected=True)
        m = ms.current_mesh()
        V, F = m.vertex_matrix(), m.face_matrix()
        sel_v = m.vertex_scalar_array() > 0.25   # new midpoints interpolate the quality
    return V, F, sel_v


def sculpt(V, F, rgba, fit_fn, replaced=None):
    """V, F: repaired full-res mesh. rgba: front source cut-out (float HxWx4). fit_fn(V, F) -> fit tuple.
    replaced: parts already rebuilt from close-ups (parts.py): {"head": True, "hands": [21 x 3 joints, model space]}.
    Those are left as they are (only their regions and importance are recorded).
    Returns V, F, per-vertex importance, info dict."""
    replaced = replaced or {}
    H, W = rgba.shape[:2]
    rgb = (np.clip(rgba[..., :3] * rgba[..., 3:] + 0.85 * (1 - rgba[..., 3:]), 0, 1) * 255).astype(np.uint8)
    info = {}
    fit = fit_fn(V, F)
    s = fit[0]
    view = View(V, F, fit, H, W)
    box = detect_face(rgb)
    L = landmarks.face_mesh(rgb, box)
    hands = landmarks.hands(rgb)
    info["face_landmarks"] = L is not None
    info["hands_found"] = len(hands)

    # regions in image space, tied to the front surface in depth
    face_reg = np.zeros(len(V), bool)
    hand_skel = []
    if L is not None:
        # head sphere (for masking repaints that drew the face in the wrong place): around the face, set back by half
        # its height, wide enough for the hair but not the shoulders
        fh = np.ptp(L[:, 1]) / s
        zl = view.sample(view.zfront, L[:, :2])
        P3 = np.c_[(L[:, 0] - fit[1]) / (s * fit[3]), (fit[2] - L[:, 1]) / s, zl]
        hc = np.nanmean(P3, 0) - [0, 0, 0.5 * fh]
        info["head"] = {"c": np.round(hc, 5).tolist(), "r": round(float(1.15 * fh), 5)}
        c = L[:, :2].mean(0)
        r = 0.75 * np.ptp(L[:, 1])
        zc = np.nanmedian(view.sample(view.zfront, L[:, :2]))
        d2 = np.hypot(*(view.px - c).T)
        face_reg = (d2 < r * 1.2) & (V[:, 2] > zc - 1.2 * r / s)
    if replaced.get("hands"):
        info["hands_replaced"] = True
        hands = []
        for P in replaced["hands"]:
            P = np.asarray(P)
            hand_skel.append((P, max(np.linalg.norm(P[5] - P[17]), 0.8 * np.linalg.norm(P[0] - P[9]))))
    for h in hands:
        # hands often hang edge-on (palm toward the thigh), so measure in 3D and fall back on the palm's length
        palm = max(np.linalg.norm(h[5] - h[17]), 0.8 * np.linalg.norm(h[0] - h[9])) / s
        # depth: the hand's median front surface, half a hand-thickness in, plus MediaPipe's relative depth (a single
        # landmark's pixel can miss the hand and hit whatever is behind it)
        zs = view.sample(view.zfront, h[:, :2])
        rel = np.clip(-(h[:, 2] - np.median(h[:, 2])) / s, -palm, palm)
        P = np.c_[(h[:, 0] - fit[1]) / (s * fit[3]), (fit[2] - h[:, 1]) / s, np.nanmedian(zs) - 0.35 * palm + rel]
        hand_skel.append((P, palm))
    info["hands"] = [{"P": np.round(P, 5).tolist(), "palm": round(float(palm), 5)} for P, palm in hand_skel]
    hand_reg = np.zeros(len(V), bool)
    for P, palm in hand_skel:
        hand_reg |= _dist_to_skeleton(V, P) < 0.9 * palm

    if replaced.get("head"):
        face_reg[:] = False
    if replaced.get("hands"):
        hand_reg[:] = False
    if (face_reg | hand_reg).any():
        V, F, sel = _subdivide(V, F, face_reg | hand_reg)
    info["verts_after_subdivide"] = len(V)
    A = umbrella(F, len(V))
    view = View(V, F, fit, H, W)
    importance = np.ones(len(V))

    if L is not None and replaced.get("head"):
        # rebuilt from the close-ups: just spend triangles on the whole head
        hc, hr = np.asarray(info["head"]["c"]), info["head"]["r"]
        wh = np.clip((1.1 * hr - np.linalg.norm(V - hc, axis=1)) / (0.3 * hr), 0, 1)
        importance = np.maximum(importance, 1 + (FACE_IMPORTANCE - 1) * wh)
        info["head_replaced"] = True
    elif L is not None:
        idx, dst = jaw_targets(L, view, s)   # sculpt the features, but inside the mesh's own jaw outline
        if idx is not None:
            L = L.copy()
            info["jaw_shift_px"] = round(float(np.linalg.norm(dst - L[idx, :2], axis=1).mean()), 1)
            L[idx, :2] = dst
        V, w_face = _face_depth(V, F, A, view, L, s, info)
        importance = np.maximum(importance, 1 + (FACE_IMPORTANCE - 1) * np.clip(w_face * 3, 0, 1))
        # light Taubin over the whole head front evens out the triangulated landmark surface and the generator's
        # banding around it; strong enough to clean, too weak to soften the nose or lips
        c = L[:, :2].mean(0)
        r = 0.75 * np.ptp(L[:, 1])
        d2 = np.hypot(*(view.px - c).T) / r
        w = np.clip(1.6 - d2, 0, 1) * (V[:, 2] > np.nanmedian(view.sample(view.zfront, L[:, :2])) - 1.5 * r / s)
        V = taubin(V, A, w, 4)
    for P, palm in hand_skel:
        d = _dist_to_skeleton(V, P)
        w = np.clip((1.1 * palm - d) / (0.4 * palm), 0, 1)
        if not replaced.get("hands"):
            V = taubin(V, A, w, HAND_SMOOTH, lam=HAND_LAMBDA, mu=-HAND_LAMBDA - 0.02)
        importance = np.maximum(importance, 1 + (HAND_IMPORTANCE - 1) * w)
    return V, F, importance, info


def hand_region(X, hands):
    """Per-point weight in [0, 1] for being part of a hand (not the sleeve): near the hand's bones and past the wrist."""
    w = np.zeros(len(X))
    for h in hands:
        P, palm = np.asarray(h["P"]), h["palm"]
        near = np.nonzero(np.all(np.abs(X - P.mean(0)) < np.ptp(P, 0) / 2 + palm, axis=1))[0]   # bounding box first
        Xn = X[near]
        d = _dist_to_skeleton(Xn, P)
        u = P[9] - P[0]
        t = (Xn - P[0]) @ u / max(u @ u, 1e-12)
        w[near] = np.maximum(w[near], np.clip((0.55 * palm - d) / (0.2 * palm), 0, 1) * np.clip((t - 0.02) / 0.15, 0, 1))
    return w


def _dist_to_skeleton(V, P):
    """Distance from each vertex to the hand's bone segments (MediaPipe 21-point layout)."""
    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
             (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20), (0, 9), (0, 13)]
    d = np.full(len(V), np.inf)
    for i, j in bones:
        a, b = P[i], P[j]
        ab = b - a
        t = np.clip(((V - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(V - (a + t[:, None] * ab), axis=1))
    return d


def _face_depth(V, F, A, view, L, s, info):
    H, W = view.H, view.W
    oval = landmarks.face_oval()
    poly = L[oval, :2].astype(np.float32)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inside, [np.round(poly).astype(np.int32)], 1)
    fw = np.ptp(L[:, 0])
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    wmap = np.clip(dist / (0.14 * fw), 0, 1)
    wmap = wmap * wmap * (3 - 2 * wmap)                       # smoothstep feather toward the oval
    # target depth: a thin-plate spline through the landmarks (the 478-point triangulation itself shows facets)
    x0, y0 = np.floor(L[:, :2].min(0)).astype(int) - 2
    x1, y1 = np.ceil(L[:, :2].max(0)).astype(int) + 3
    gy, gx = np.mgrid[y0:y1, x0:x1]
    tps = RBFInterpolator(L[:, :2], -L[:, 2] / s, kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)
    zt = np.full((H, W), np.nan)
    zt[y0:y1, x0:x1] = tps(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    zt[cv2.dilate(inside, np.ones((5, 5), np.uint8)) == 0] = np.nan
    ok = np.isfinite(zt)
    # keep the mesh's own head orientation: MediaPipe's depth often carries a few degrees of yaw or pitch the mesh
    # doesn't have (a 25 px left-right ramp on the test character), so a plane is fitted first, never the features
    inner = (wmap > 0.6) & np.isfinite(view.zfront) & ok
    iy, ix = np.nonzero(inner)
    G = np.c_[ix, iy, np.ones(len(ix))]
    plane, *_ = np.linalg.lstsq(G, view.zfront[inner] - zt[inner], rcond=None)
    off = float(plane[2] + plane[0] * ix.mean() + plane[1] * iy.mean())
    zt = zt + plane[0] * np.arange(W)[None, :] + plane[1] * np.arange(H)[:, None] + plane[2]
    slope = float(np.polyfit(zt[inner], view.zfront[inner], 1)[0])
    if DETAIL_SIGMA:
        # detail transfer: the mesh keeps its own broad face shape (it already matches the silhouette seen from the
        # side) and takes only the landmark surface's relief (nose, sockets, lips, chin). Replacing the whole depth
        # leaves a slope mismatch at the oval that shows as a mask rim and opens the jaw below the ear.
        sig = DETAIL_SIGMA * fw
        zt = _lowpass(view.zfront, inside, sig) + (zt - _lowpass(zt, inside, sig))
    else:
        zt = _seamless(zt, view.zfront, inside, dist, fw, L)
    vis = view.visible(V, 1.5 / s)
    w = view.sample(wmap) * vis
    if FACING:
        # only where the surface faces the camera: on the jaw's underside and the side of the cheek a front-view depth
        # edit moves the visible rim past the surface tucked under it, and the jaw tears open
        nz = trimesh.Trimesh(V, F, process=False).vertex_normals[:, 2]
        w = w * np.clip((nz - FACING) / 0.25, 0, 1)
    tz = view.sample(zt)
    has = (w > 0) & np.isfinite(tz)
    D = np.zeros(len(V))
    D[has] = tz[has] - V[has, 2]
    D[has] = np.clip(D[has], -25 / s, 25 / s)                 # MediaPipe glitch guard: features move < ~25 px
    # spread the displacement into the hidden surface around it (nostrils, under the chin, the cheek behind the hair)
    # so the front layer doesn't fold through it or end in a step: a biharmonic (C1-smooth) fill, with the well-seen
    # face held fixed and everything further than about a face height away held still
    fixed = has & (w > FIX_W)
    target = D * w
    fh = np.ptp(L[:, 1]) / s
    c3 = np.r_[(L[:, :2].mean(0) - [view.fit[1], view.fit[2]]) * [1 / (s * view.fit[3]), -1 / s],
               np.nanmedian(view.sample(view.zfront, L[:, :2]))]
    roi = (np.linalg.norm(V - c3, axis=1) < ROI * fh) & (V[:, 2] > c3[2] - 0.7 * fh)
    X = _biharmonic_fill(A, fixed, target, roi)
    V = V.copy()
    V[:, 2] += X
    # even out the vertices on the new surface: smooth, then put the visible face back onto the target depth
    wf = np.where(has, w, 0)
    for _ in range(4):
        V = taubin(V, A, np.clip(wf * 2, 0, 1), 2)
        tz = view.sample(zt, view.project(V))
        back = has & np.isfinite(tz)
        V[back, 2] += wf[back] * (np.clip(tz[back] - V[back, 2], -25 / s, 25 / s))
    info.update(face_depth_offset_px=round(off * s, 1), face_depth_slope=round(slope, 3),
                face_tilt=[round(float(plane[0] * W), 3), round(float(plane[1] * H), 3)],
                face_move_px=round(float(np.abs(X[fixed]).mean() * s), 2),
                face_move_max_px=round(float(np.abs(X).max() * s), 1), face_verts=int(fixed.sum()))
    w_all = np.zeros(len(V))
    w_all[has] = w[has]
    return V, (A @ (A @ w_all)) if fixed.any() else w_all


def _biharmonic_fill(A, fixed, val, roi):
    """Per-vertex scalar field equal to val on `fixed`, 0 outside `roi`, and as smooth as possible (minimal squared
    umbrella Laplacian) in between."""
    n = A.shape[0]
    Lp = (A - sp.identity(n)).tocsr()
    Q = (Lp.T @ Lp).tocsr()
    known = fixed | ~roi
    x = np.where(fixed, val, 0.0)
    free, kn = np.nonzero(~known)[0], np.nonzero(known)[0]
    if not len(free):
        return x
    Qf = Q[free]
    M = Qf[:, free] + sp.identity(len(free)) * 1e-9   # floaters with no fixed neighbour stay put instead of singular
    x[free] = spla.spsolve(M.tocsc(), -(Qf[:, kn] @ x[kn]))
    return x


def _lowpass(z, inside, sigma):
    """Gaussian low-pass of z over the face oval only, normalised: averaging in the neck behind the jaw would pull the
    jaw's rim back and tear it."""
    m = (inside > 0) & np.isfinite(z)
    zz = np.where(m, z, 0.0)
    num = cv2.GaussianBlur(zz, (0, 0), sigma)
    den = cv2.GaussianBlur(m.astype(np.float64), (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)


def _seamless(zt, zfront, inside, dist, fw, L):
    """Bend the landmark surface so it meets the mesh at the face oval (a harmonic correction, like seamless
    cloning), instead of leaving a mask-like rim where the two disagree. The jaw and chin keep the mesh's outline too:
    pulling the neck forward to the image's chin opens a crease, so the texture is warped onto the mesh's jaw
    instead (align_outline)."""
    ring = (inside > 0) & (dist <= max(2.0, 0.05 * fw)) & np.isfinite(zfront) & np.isfinite(zt)
    R = np.where(ring, zfront - zt, 0.0)
    dom = (inside > 0) & ~ring
    ys, xs = np.nonzero(dom | ring)
    idx = -np.ones(inside.shape, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    fixed = ring[ys, xs]
    rows.append(np.nonzero(fixed)[0]); cols.append(np.nonzero(fixed)[0]); vals.append(np.ones(fixed.sum()))
    b[fixed] = R[ys[fixed], xs[fixed]]
    free = np.nonzero(~fixed)[0]
    deg = np.zeros(len(free))
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        j = idx[ys[free] + dy, xs[free] + dx]
        okj = j >= 0
        rows.append(free[okj]); cols.append(j[okj]); vals.append(-np.ones(okj.sum()))
        deg += okj
    rows.append(free); cols.append(free); vals.append(deg)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    C = np.zeros_like(zt)
    C[ys, xs] = spla.spsolve(M.tocsc(), b)
    return zt + C


def jaw_targets(L, view, s, outward=0.0, with_rise=False):
    """Where the image's jaw/chin landmarks (face oval below the nose) belong on the mesh: the nearest point along the
    ray from the nose where the front depth drops from face to neck, pushed `outward` px (the hit is where the drop
    starts, so the silhouette itself is ~2 px further out). Returns (landmark indices, target px) or (None, None)."""
    zf = np.where(np.isfinite(view.zfront), view.zfront, np.nanmin(view.zfront) - 1.0)
    fw = np.ptp(L[:, 0])
    centre = L[1, :2]
    jump = 6.0 / s                      # a depth drop of 6 px (model units) within 3 px = face edge
    idx, dst, rise = [], [], []
    for i in landmarks.face_oval():
        p = L[i, :2]
        if p[1] < L[1, 1]:              # only below the nose tip: above, hair frames the face, not a depth edge
            continue
        d = (p - centre) / max(np.linalg.norm(p - centre), 1e-6)
        ts = np.arange(-0.15 * fw, 0.15 * fw, 0.5)
        q = p[None] + ts[:, None] * d[None]
        z = view.sample(zf, q)
        z3 = view.sample(zf, q + 3 * d[None])
        hits = np.nonzero(z - z3 > jump)[0]
        if len(hits):
            k = hits[np.argmin(np.abs(ts[hits]))]
            idx.append(i)
            dst.append(q[k] + outward * d)
            # a surface coming back almost to the jaw's depth just past the drop: collar or scarf under a chin slit
            back = np.nonzero((ts > ts[k] + 2) & (ts < ts[k] + 0.25 * fw) & (z > z[k] - 2 * jump))[0]
            rise.append(q[back[0]] if len(back) else (np.nan, np.nan))
    if len(idx) < 5:
        return (None, None, None) if with_rise else (None, None)
    idx, dst, rise = np.array(idx), np.array(dst), np.array(rise)
    off = dst - L[idx, :2]
    # contour outliers (an ear or the collar caught instead of the jaw): median over neighbours along the contour
    off = np.array([np.median(off[max(0, k - 2):k + 3], 0) for k in range(len(off))])
    return (idx, L[idx, :2] + off, rise) if with_rise else (idx, L[idx, :2] + off)


def align_outline(rgba, V, F, fit):
    """Warp the source's jaw and chin outline onto the mesh's own (where the front depth drops from the face to the
    neck), keeping eyes, nose and mouth fixed: the sculpt already put those under the painted features. The generator
    often makes the face a little shorter than the image, which otherwise paints the chin onto the neck.
    Returns (rgba, info)."""
    H, W = rgba.shape[:2]
    rgb = (np.clip(rgba[..., :3] * rgba[..., 3:] + 0.85 * (1 - rgba[..., 3:]), 0, 1) * 255).astype(np.uint8)
    L = landmarks.face_mesh(rgb, detect_face(rgb))
    if L is None:
        return rgba, {"outline": "no face landmarks"}
    view = View(V, F, fit, H, W)
    idx, dst, rise = jaw_targets(L, view, fit[0], outward=2.0, with_rise=True)   # jaw edge onto the silhouette
    if idx is None:
        return rgba, {"outline": "jaw edge not found on the mesh"}
    src = L[idx, :2]
    off = dst - src
    centre, fw, oval = L[1, :2], np.ptp(L[:, 0]), landmarks.face_oval()
    # where the mesh's collar or scarf rises right under the chin, the image usually shows a band of neck there:
    # squeeze that band into the slit, so the clothing gets the image's clothing (face parsing finds its top edge)
    extra_dst, extra_src = [], []
    labels = landmarks.face_parse(rgb, detect_face(rgb)) if np.isfinite(rise).any() else None
    if labels is not None:
        for p, q in zip(src, rise):
            if not np.isfinite(q).all():
                continue
            d = (p - centre) / max(np.linalg.norm(p - centre), 1e-6)
            ts = np.arange(1.0, 0.45 * fw, 1.0)
            r = p[None] + ts[:, None] * d[None]
            lab = labels[np.clip(r[:, 1].astype(int), 0, H - 1), np.clip(r[:, 0].astype(int), 0, W - 1)]
            hit = np.nonzero(lab == landmarks.CLOTH)[0]
            if len(hit) and (lab[:hit[0]] == landmarks.NECK).mean() > 0.5:
                extra_dst.append(q + 1.0 * d)
                extra_src.append(r[hit[0]])
    inner = np.setdiff1d(np.arange(len(L)), oval)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    frame = centre[None] + 1.3 * fw * np.c_[np.cos(ang), np.sin(ang)]
    ed, es = np.array(extra_dst).reshape(-1, 2), np.array(extra_src).reshape(-1, 2)
    ctrl = np.concatenate([dst, ed, L[inner, :2], frame])
    disp = np.concatenate([src - dst, es - ed, np.zeros((len(inner) + len(frame), 2))])
    tps = RBFInterpolator(ctrl, disp, kernel="thin_plate_spline", smoothing=1.0)
    x0, y0 = np.maximum(np.floor(centre - 1.3 * fw).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(centre + 1.3 * fw).astype(int), [W, H])
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dd = tps(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape + (2,)).astype(np.float32)
    mx, my = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx[y0:y1, x0:x1] += dd[..., 0]
    my[y0:y1, x0:x1] += dd[..., 1]
    out = cv2.remap(rgba, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, {"outline": "ok", "outline_shift_px": round(float(np.linalg.norm(off, axis=1).mean()), 1),
                 "outline_points": len(src), "neck_band_points": len(ed),
                 "neck_band_px": round(float(np.linalg.norm(es - ed, axis=1).mean()), 1) if len(ed) else 0}


def face_area(rgba):
    """HxW map: inside the face oval, eroded a couple of pixels so the surface behind the jaw stays outside."""
    rgb = (np.clip(rgba[..., :3] * rgba[..., 3:] + 0.85 * (1 - rgba[..., 3:]), 0, 1) * 255).astype(np.uint8)
    L = landmarks.face_mesh(rgb, detect_face(rgb))
    m = np.zeros(rgba.shape[:2], np.uint8)
    if L is None:
        return m.astype(np.float32)
    cv2.fillPoly(m, [np.round(L[landmarks.face_oval(), :2]).astype(np.int32)], 1)
    return cv2.erode(m, np.ones((5, 5), np.uint8)).astype(np.float32)


def ownership(rgba, feather=None):
    """(face, hands) HxW maps in [0, 1]: where the source image alone should paint. Other views place facial features
    and fingers a few pixels differently, and averaging them in doubles the eyes and mouth."""
    H, W = rgba.shape[:2]
    rgb = (np.clip(rgba[..., :3] * rgba[..., 3:] + 0.85 * (1 - rgba[..., 3:]), 0, 1) * 255).astype(np.uint8)
    m = np.zeros((H, W), np.float32)
    box = detect_face(rgb)
    L = landmarks.face_mesh(rgb, box)
    if L is not None:
        cv2.fillPoly(m, [np.round(L[landmarks.face_oval(), :2]).astype(np.int32)], 1.0)
        fw = np.ptp(L[:, 0])
    elif box is not None:
        x, y, w, h = box[:4]
        cv2.ellipse(m, (int(x + w / 2), int(y + h / 2)), (int(w * 0.5), int(h * 0.6)), 0, 0, 360, 1.0, -1)
        fw = w
    else:
        fw = 0.08 * H
    face, m = m, np.zeros((H, W), np.float32)
    for h in landmarks.hands(rgb):
        hull = cv2.convexHull(np.round(h[:, :2]).astype(np.int32))
        hm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(hm, [hull], 1)
        r = max(3, int(0.12 * np.linalg.norm(h[0, :2] - h[12, :2])))   # fingers are thicker than their joints
        m = np.maximum(m, cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))))
    sigma = feather or max(1.5, fw * 0.04)
    solid = rgba[..., 3] > 0.5
    return tuple(cv2.GaussianBlur(x * solid, (0, 0), sigma) for x in (face, m))
