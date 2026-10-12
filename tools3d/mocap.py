"""Motion capture from generated reference video: the source of anim.py's clips.

For each clip: a text-to-video model (MiniMax H3 Max on fal) films a person doing the motion; SAM 3D Body (fal) lifts
each frame of one or two gait cycles to 70 3D keypoints; solve() turns them into a looping, floor-aligned motion and
saves it to tools3d/motions/<Clip>.json. anim.py retargets that file onto every rigged character, so the video and
pose steps run once per clip, not per character.

    .venv3d/bin/python tools3d/mocap.py video Run      # ~$0.25: out/mocap/Run/video.mp4
    .venv3d/bin/python tools3d/mocap.py track Run      # ~$0.6-1: MediaPipe finds the cycle, SAM 3D Body lifts it
    .venv3d/bin/python tools3d/mocap.py solve Run      # free: tools3d/motions/Run.json
    .venv3d/bin/python tools3d/mocap.py all Run        # all three

Motion file (all in the character's frame: +Y up, +Z facing, +X its left; metres of the reference person):
- "names": the 70 MHR keypoint names; "frames": (F, 70, 3) keypoints over one cycle, the pelvis centre's ground
  point at the origin, sampled at F evenly spaced phases (phase 0 = the left foot's stance begins, for gaits)
- "period" (s), "speed" (m/s, travel speed), "direction" (unit, local), "leg" (m, the reference's leg length),
  "contacts": per foot, per frame, 1 when planted
"""
import json
import os
import sys

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "out", "mocap")
MOTIONS = os.path.join(HERE, "motions")
VIDEO_MODEL = "minimax/h3-max/text-to-video"
POSE_MODEL = "fal-ai/sam-3/3d-body"

_FILM = ("Motion reference footage. A fit adult man in a plain fitted grey t-shirt, black shorts and white sneakers "
         "{action} in a large empty light grey studio with a plain floor. {camera} His whole body from head to feet "
         "stays in frame at all times. {style} Even soft lighting, plain seamless background, no other people, no "
         "objects, one continuous shot with no cuts.")
_TRACK = ("The camera tracks smoothly alongside him at hip height at exactly his speed, keeping him centered, seen "
          "from a three-quarter {side} angle.")
CLIPS = {
    "Run": dict(kind="gait", symmetric=True, seconds=8, cycle_s=(0.64, 0.78), travel=(0, 1), flight=True, action="runs at a steady, moderate running pace, moving from left to right",
                camera=_TRACK.format(side="front-side"),
                style=("Natural athletic running form: arms bent about 90 degrees at the elbows swinging forward and "
                       "back opposite the legs, hands loosely closed, torso and shoulders rotating slightly and the "
                       "body bobbing up and down with every stride, knees driving forward, heels kicking back, a brief "
                       "flight phase between steps.")),
    "Walk": dict(kind="gait", symmetric=True, seconds=8, cycle_s=(1.0, 1.2), travel=(0, 1), action="walks at a relaxed, steady walking pace, moving from left to right",
                 camera=_TRACK.format(side="front-side"),
                 style=("Natural relaxed walking: heel strike then roll onto the toes, arms hanging loosely and "
                        "swinging gently opposite the legs, slight elbow bend, hips swaying and shifting weight onto "
                        "each stance leg, shoulders counter-rotating, the body rising and falling a little with "
                        "every step, head level.")),
    "WalkBackward": dict(kind="gait", symmetric=True, seconds=8, cycle_s=(1.05, 1.4), travel=(0, -1),
                         action=("walks backwards, reversing step by step: he faces the right edge of the frame the whole "
                                 "time, and moves backwards from the right side of the frame to the left side, each foot "
                                 "reaching back behind him and landing toes first before the heel, his back leading the "
                                 "way"),
                         camera=("The camera is static at hip height in a wide shot, seen exactly from his side in "
                                 "profile; he crosses the frame backwards."),
                         style=("Natural backward walking: each foot reaches back and lands toe first then rolls onto "
                                "the heel, knees slightly bent, torso upright leaning very slightly forward, arms "
                                "swinging gently opposite the legs, weight shifting from foot to foot.")),
    "StrafeLeft": dict(kind="gait", seconds=8, cycle_s=(0.8, 1.2), travel=(1, 0),
                       action=("side-steps to his own left in a continuous steady sideways shuffle, always facing "
                               "the camera, moving toward the right side of the frame"),
                       camera=("The camera tracks smoothly sideways at hip height at exactly his speed, keeping him "
                               "centered, seen from the front."),
                       style=("Natural sideways strafing: his left foot steps out to the side, then his right foot "
                              "closes in toward it, feet never crossing, knees slightly bent, hips staying low and "
                              "shifting weight from side to side, torso upright, arms relaxed and slightly away from "
                              "the body, head facing forward.")),
    "StrafeRight": dict(kind="gait", seconds=8, cycle_s=(0.8, 1.2), travel=(-1, 0),
                        action=("side-steps to his own right in a continuous steady sideways shuffle, always facing "
                                "the camera, moving toward the left side of the frame"),
                        camera=("The camera tracks smoothly sideways at hip height at exactly his speed, keeping him "
                                "centered, seen from the front."),
                        style=("Natural sideways strafing: his right foot steps out to the side, then his left foot "
                               "closes in toward it, feet never crossing, knees slightly bent, hips staying low and "
                               "shifting weight from side to side, torso upright, arms relaxed and slightly away from "
                               "the body, head facing forward.")),
    "Idle": dict(kind="idle", seconds=8, action="stands in place in a relaxed idle stance, feet about shoulder width apart",
                 camera="The camera is static at hip height, seen from a three-quarter front angle.",
                 style=("Natural idle: calm breathing that lifts the chest, a slow weight shift from one leg to the "
                        "other, arms hanging relaxed at the sides with a slight bend, small natural head movements "
                        "looking around, he does not walk or step.")),
}


def clip_dir(clip):
    d = os.path.join(OUT, clip)
    os.makedirs(d, exist_ok=True)
    return d


def client():
    import fal_client
    from tripo import KEY_FILE, fal_key
    key = fal_key()
    if not key:
        raise RuntimeError(f"No fal API key: set FAL_KEY or write it to {KEY_FILE}")
    return fal_client.SyncClient(key=key, default_timeout=900)


def download(url, path):
    import httpx
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


# --- 1. reference video ---------------------------------------------------------------------------------------------

def video(clip, seed=7):
    c = CLIPS[clip]
    prompt = _FILM.format(action=c["action"], camera=c["camera"], style=c["style"])
    res = client().subscribe(VIDEO_MODEL, {"prompt": prompt, "duration": c["seconds"], "aspect_ratio": "16:9",
                                           "resolution": "768P", "prompt_expansion_mode": "disabled", "seed": seed})
    d = clip_dir(clip)
    download(res["video"]["url"], os.path.join(d, "video.mp4"))
    with open(os.path.join(d, "video.json"), "w") as f:
        json.dump({"model": VIDEO_MODEL, "prompt": prompt, "seed": seed, "url": res["video"]["url"]}, f, indent=1)
    return os.path.join(d, "video.mp4")


def frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[..., ::-1].copy())
    return out, fps


# --- 2. cycle length and 3D pose ---

def hip_bob_period(imgs, fps):
    """Gait cycle length (frames) from the hips' vertical bob in 2D (MediaPipe Pose, local and free): the pelvis
    rises and falls once per step, two steps per cycle. Unlike limb tracks this survives left/right label swaps and
    the leg morphing of generated video. Returns (period, autocorrelation at the step lag)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from rig import POSE_TASK
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_TASK), running_mode=vision.RunningMode.VIDEO, num_poses=1))
    y = np.full(len(imgs), np.nan)
    for i, im in enumerate(imgs):
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(im)),
                                 int(i * 1000 / fps))
        if r.pose_landmarks:
            lm = r.pose_landmarks[0]
            y[i] = (lm[23].y + lm[24].y) / 2
    det.close()
    t = np.arange(len(y))
    ok = np.isfinite(y)
    y = np.interp(t, t[ok], y[ok])
    win = int(2 * fps) | 1                                   # remove camera drift (2 s high-pass)
    y = y - np.convolve(np.pad(y, win // 2, mode="edge"), np.ones(win) / win, "valid")
    y = (y - y.mean()) / (y.std() + 1e-9)
    n = len(y)
    lo, hi = int(0.2 * fps), min(n // 3, int(2.0 * fps))     # a step lasts 0.2-2 s (generated video runs slow)
    ac = np.array([np.mean(y[:n - k] * y[k:]) for k in range(hi + 2)])
    dip = int(np.argmax(ac < 0)) if (ac < 0).any() else lo  # the step peak comes after the first dip
    lo = max(lo, dip)
    k = lo + int(np.argmax(ac[lo:hi]))
    kf = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-12)
    return 2 * float(kf), float(ac[k])


# --- 3D pose ---------------------------------------------------------------------------------

def sam(clip, imgs, idx):
    """SAM 3D Body on frames `idx` of the clip's video (cached per frame in out/mocap/<clip>/sam/). Returns calls made."""
    import cv2
    sam_dir = os.path.join(clip_dir(clip), "sam")
    os.makedirs(sam_dir, exist_ok=True)
    c = client()
    todo = []
    for i in idx:
        out = os.path.join(sam_dir, f"{i:04d}.json")
        if os.path.exists(out):
            continue
        p = os.path.join(sam_dir, f"{i:04d}.jpg")
        cv2.imwrite(p, imgs[i][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        todo.append((i, out, c.submit(POSE_MODEL, {"image_url": c.upload_file(p), "export_meshes": False,
                                                    "include_3d_keypoints": False, "include_mhr_params": False})))
    for i, out, h in todo:
        r = h.get()
        people = r["metadata"]["people"]
        if not people:
            continue
        p = max(people, key=lambda q: (q["bbox"][2] - q["bbox"][0]) * (q["bbox"][3] - q["bbox"][1]))
        with open(out, "w") as f:
            json.dump({"frame": i, "names": r["metadata"]["keypoint_names"], "k3d": p["keypoints_3d"],
                       "k2d": p["keypoints_2d"], "cam_t": p["pred_cam_t"], "focal": p["focal_length"]}, f)
    return len(todo)


def idle_window(names, fi, Q, fps, lo_s=3.5, hi_s=6.0):
    """For idle (no cycle): the 3.5-6 s stretch whose first and last poses match best, so it loops."""
    order = np.argsort(fi)
    fi, Q = fi[order], Q[order]
    best = None
    for a in range(len(fi)):
        for b in range(a + 1, len(fi)):
            span = (fi[b] - fi[a]) / fps
            if lo_s <= span <= hi_s:
                d = float(np.sqrt(np.mean((Q[a] - Q[b]) ** 2)))
                if best is None or d < best[0]:
                    best = (d, fi[a], fi[b])
    d, a, b = best
    return float(a), float(b), float(b - a), d


def track(clip, sparse=40, dense_step=2):
    """3D pose for the clip: a sparse SAM pass over the whole video, the cleanest window (gaits: >= 2 cycles that
    fold into one; idle: a stretch that loops), then a dense pass over that window."""
    d = clip_dir(clip)
    imgs, fps = frames(os.path.join(d, "video.mp4"))
    edge = int(0.3 * fps)
    step = max(1, (len(imgs) - 2 * edge) // sparse)
    have = len([n for n in os.listdir(os.path.join(d, "sam")) if n.endswith(".json")]) if os.path.isdir(
        os.path.join(d, "sam")) else 0
    calls = 0 if have >= 0.8 * sparse else sam(clip, imgs, range(edge, len(imgs) - edge, step))
    names, fi, K3 = load_sam(clip)
    Q, _ = body_frames(names, K3)
    if CLIPS[clip]["kind"] == "gait":
        P0, _ = hip_bob_period(imgs, fps)
        # walking and running bob twice per cycle, a side shuffle can bob once: take half the period when the
        # samples fold as well at it
        ix = {n: i for i, n in enumerate(names)}
        sel = [ix[n] for n in GAIT_KEYS]
        fold = lambda P: min(fourier_fit((fi / q) % 1, Q[:, sel], 4)[1]  # noqa: E731
                             for q in np.arange(0.94 * P, 1.06 * P, 0.25))
        if fold(P0 / 2) <= 1.15 * fold(P0):
            P0 /= 2
        lo, hi, period, res = clean_window(names, fi, Q, P0)
    else:
        lo, hi, period, res = idle_window(names, fi, Q, fps)
    calls += sam(clip, imgs, range(int(lo), int(hi) + 1, dense_step))
    info = {"fps": fps, "window": [lo, hi], "period_frames": period, "fold_residual_m": res,
            "sam_frames": len(os.listdir(os.path.join(d, "sam"))) // 2, "sam_calls": calls}
    with open(os.path.join(d, "track.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


# --- 3. solve: one clean, floor-aligned, looping cycle ------------------------------------------------------------

FOOT = {"Left": ["left-heel", "left-big-toe-tip", "left-small-toe-tip"],
        "Right": ["right-heel", "right-big-toe-tip", "right-small-toe-tip"]}


def load_sam(clip):
    import glob
    D = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, clip, "sam", "*.json")))]
    names = D[0]["names"]
    return names, np.array([d["frame"] for d in D], float), np.array([d["k3d"] for d in D], float)


def fourier_fit(phase, Y, K):
    """Least-squares periodic fit of Y (N, ...) sampled at `phase` (N,) with K harmonics. Returns a function of
    phase and the RMS residual."""
    def basis(ph):
        ph = np.asarray(ph, float)[:, None] * 2 * np.pi
        k = np.arange(1, K + 1)[None]
        return np.concatenate([np.ones((len(ph), 1)), np.cos(k * ph), np.sin(k * ph)], 1)
    B = basis(phase)
    flat = Y.reshape(len(Y), -1)
    C, *_ = np.linalg.lstsq(B, flat, rcond=None)
    res = float(np.sqrt(np.mean((B @ C - flat) ** 2)))
    return (lambda ph: (basis(np.atleast_1d(ph)) @ C).reshape((-1,) + Y.shape[1:])), res


def floor_normal(names, K3, n0=np.array([0.0, -1.0, 0.0]), max_deg=10.0):
    """The floor's up direction in SAM's camera axes. Start from the camera's vertical (the videos are filmed with a
    level camera at hip height) and refine with double support: whenever both feet are down, the line between their
    lowest points lies in the floor, so up is the direction most perpendicular to all those lines (with a pull
    toward the start so a few frames can't tip it). Small foot triangles are too noisy to give a floor normal."""
    ix = {n: i for i, n in enumerate(names)}
    lows = []
    for side in FOOT:
        P = K3[:, [ix[n] for n in FOOT[side]]]                     # (N, 3, 3)
        j = (P @ n0).argmin(1)                                     # lowest along the start's up
        lows.append(P[np.arange(len(P)), j])
    d = lows[0] - lows[1]
    both = np.abs(d @ n0) < 0.03
    if both.sum() < 6:
        return n0
    D = d[both]
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    A = D.T @ D / len(D) + 0.05 * (np.eye(3) - np.outer(n0, n0))
    w, V = np.linalg.eigh(A)
    n = V[:, 0] * np.sign(V[:, 0] @ n0)
    if np.degrees(np.arccos(np.clip(n @ n0, -1, 1))) > max_deg:
        return n0
    return n


def camera_to_body(names, K3):
    """Rotation (3x3, rows X left, Y up, Z facing) from SAM's camera axes (x right, y down, z forward) to the
    character frame: up from floor_normal, the facing from the mean horizontal hip normal."""
    ix = {n: i for i, n in enumerate(names)}
    up = floor_normal(names, K3)
    across = K3[:, ix["left-hip"]] - K3[:, ix["right-hip"]]
    fwd = np.cross(across, up)
    fwd = fwd - (fwd @ up)[:, None] * up
    fwd = fwd.mean(0)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    return np.stack([left, up, fwd]), up


GAIT_KEYS = ["left-ankle", "right-ankle", "left-wrist", "right-wrist", "left-knee", "right-knee", "left-elbow",
             "right-elbow"]


def body_frames(names, K3):
    """Pelvis-relative keypoints in the character's axes (see camera_to_body), plus that rotation."""
    ix = {n: i for i, n in enumerate(names)}
    Rc, _ = camera_to_body(names, K3)
    pel = (K3[:, ix["left-hip"]] + K3[:, ix["right-hip"]]) / 2
    return (K3 - pel[:, None]) @ Rc.T, Rc


def clean_window(names, fi, Q, P0, tol=0.015, K=3):
    """Generated video drifts and sometimes morphs the legs (a left/right swap that doesn't come back), so the whole
    clip doesn't fold into one cycle. Find the longest run of frames, at least two cycles and 3x the fit's parameter
    count of samples long, that folds at a period within 6% of P0 (from hip_bob_period): returns (first frame, last
    frame, period in frames, fold residual in m). Without such a window (a slow cycle in a short video), the whole
    clip."""
    ix = {n: i for i, n in enumerate(names)}
    sel = [ix[n] for n in GAIT_KEYS]
    order = np.argsort(fi)
    fi, Y = fi[order], Q[order][:, sel]
    best = None
    Ps = np.arange(0.94 * P0, 1.06 * P0, 0.25)
    for a in range(len(fi)):
        for b in range(len(fi) - 1, a + 3 * (2 * K + 1) - 2, -1):
            span = fi[b] - fi[a]
            if span < 2 * P0:
                break
            m = slice(a, b + 1)
            r, P = min((fourier_fit((fi[m] / P) % 1, Y[m], K)[1], P) for P in Ps)
            score = (r <= tol, span if r <= tol else -r)
            if best is None or score > best[0]:
                best = (score, (float(fi[a]), float(fi[b]), float(P), float(r)))
            if r <= tol:
                break                                      # longest acceptable window from this start
    if best is None:
        r, P = min((fourier_fit((fi / P) % 1, Y, K)[1], P) for P in Ps)
        return float(fi[0]), float(fi[-1]), float(P), float(r)
    return best[1]


def foot_points(names, M):
    """(F, 2 feet, 3 points, 3) heel / big toe / small toe of each foot."""
    ix = {n: i for i, n in enumerate(names)}
    return np.stack([M[:, [ix[n] for n in FOOT[side]]] for side in ("Left", "Right")], 1)


def root_motion(fp, contact, T, D_fixed=None, smooth=0.15):
    """Pelvis path over one cycle from the feet: p (F, 3, pelvis in a ground frame, y = height above the floor) and
    the cycle's travel D (x, z), by least squares:
    - a planted foot's lowest point doesn't move (pelvis motion cancels the foot's motion relative to the pelvis),
    - planted feet touch the floor,
    - the path is smooth, and with no foot down it falls with gravity (running's flight phase),
    - p wraps around: p[k + F] = p[k] + D.
    fp: (F, 2, 3, 3) foot points relative to the pelvis; contact: (F, 2) bool; T: cycle length in seconds."""
    F = len(fp)
    n = 3 * F + 2
    rows, rhs, wts = [], [], []

    def P(k, a):
        """Coefficients for p[k][a] (k may be F: wraps with + D)."""
        c = np.zeros(n)
        c[(k % F) * 3 + a] = 1
        extra = 0.0
        if k >= F and a != 1:
            c[3 * F + (0 if a == 0 else 1)] += 1
        return c, extra
    for f in range(2):
        low = fp[:, f, :, 1].argmin(1)                         # the foot point nearest the floor
        for k in range(F):
            if not contact[k, f]:
                continue
            j = low[k]
            c = np.zeros(n)
            c[k * 3 + 1] = 1
            rows.append(c); rhs.append(-fp[k, f, j, 1]); wts.append(3.0)          # on the floor
            if contact[(k + 1) % F, f]:
                for a in range(3):                               # the planted point stays put
                    c1, _ = P(k + 1, a)
                    c0, _ = P(k, a)
                    rows.append(c1 - c0)
                    rhs.append(-(fp[(k + 1) % F, f, j, a] - fp[k, f, j, a]))
                    wts.append(1.0)
    g_dt2 = 9.81 * (T / F) ** 2
    airborne = ~contact.any(1)
    for k in range(F):
        for a in range(3):
            cm, _ = P(k - 1 + F, a) if k == 0 else P(k - 1, a)
            c0, _ = P(k, a) if k > 0 else P(F, a)
            cp, _ = P(k + 1, a) if k > 0 else P(F + 1, a)
            if k == 0:                                       # wrap: use k = F (= 0 shifted by D) as the centre
                cm, _ = P(F - 1, a)
            c = cm - 2 * c0 + cp
            if a == 1 and airborne[k]:
                rows.append(c); rhs.append(-g_dt2); wts.append(1.0)
            else:
                rows.append(c); rhs.append(0.0); wts.append(smooth)
    for a, col in ((0, 0), (2, 2)):                          # gauge: the horizontal path is centred
        c = np.zeros(n)
        c[col:3 * F:3] = 1.0 / F
        rows.append(c); rhs.append(0.0); wts.append(1.0)
    if D_fixed is not None:
        for i in range(2):
            c = np.zeros(n)
            c[3 * F + i] = 1
            rows.append(c); rhs.append(D_fixed[i]); wts.append(10.0)
    A = np.array(rows) * np.array(wts)[:, None]
    b = np.array(rhs) * np.array(wts)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3 * F].reshape(F, 3), x[3 * F:]


def unit2(v):
    v = np.asarray(v, float)
    return v / max(1e-9, np.linalg.norm(v))


def clean_runs(c, fill=6, min_len=3):
    """Cyclic boolean cleanup per column (one foot): fill gaps of <= fill samples (a foot is in the air for over
    half the cycle, so a short gap is pose noise), drop runs of < min_len samples."""
    c = c.copy()
    F = len(c)
    for f in range(c.shape[1]):
        for val, limit in ((False, fill), (True, min_len - 1)):
            col = c[:, f]
            if col.all() or not col.any():
                continue
            start = int(np.argmax(col != col[0]))               # rotate so the sequence starts at a change
            r = np.roll(col, -start)
            i = 0
            while i < F:
                j = i
                while j < F and r[j] == r[i]:
                    j += 1
                if r[i] == val and j - i <= limit:
                    r[i:j] = not val
                i = j
            c[:, f] = np.roll(r, start)
    return c


def detect_contacts(fp, travel, p=None, D=None):
    """Planted frames per foot, judged on the foot's lowest point (heel or toes: the pivot while the foot rolls).
    First guess (no root yet): that point moves backward relative to the pelvis and is near its lowest. With the root
    solved: within 2.5 cm of the floor and slower than a quarter of the travel speed. Short gaps and blips are
    cleaned up (clean_runs)."""
    F = len(fp)
    j = fp[:, :, :, 1].argmin(2)                                 # (F, 2) lowest point per foot
    low = np.take_along_axis(fp, j[:, :, None, None].repeat(3, 3), 2)[:, :, 0]          # (F, 2, 3)
    nxt = np.take_along_axis(np.roll(fp, -1, 0), j[:, :, None, None].repeat(3, 3), 2)[:, :, 0]
    prv = np.take_along_axis(np.roll(fp, 1, 0), j[:, :, None, None].repeat(3, 3), 2)[:, :, 0]
    vel = (nxt - prv) / 2                                        # same physical point, relative to the pelvis
    if p is None:
        if np.linalg.norm(travel) == 0:
            return np.ones((F, 2), bool)
        back = -(vel[:, :, [0, 2]] @ travel)
        near = low[:, :, 1] <= low[:, :, 1].min(0) + 0.05
        return clean_runs((back > 0.15 * np.abs(back).max()) & near)
    dp = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / 2
    dp[0, [0, 2]] += D / 2                                       # p wraps with + D
    dp[-1, [0, 2]] += D / 2
    world_v = vel + dp[:, None]
    speed = np.linalg.norm(world_v[:, :, [0, 2]], axis=2)
    world_low = p[:, 1:2] + low[:, :, 1]
    step = np.linalg.norm(D) / F
    return clean_runs((world_low < 0.025) & (speed < max(0.25 * step, 0.002)))


def mirror_perm(names):
    """Index permutation that swaps every left-/right- keypoint."""
    ix = {n: i for i, n in enumerate(names)}
    return [ix["right-" + n[5:]] if n.startswith("left-") else ix["left-" + n[6:]] if n.startswith("right-") else i
            for i, n in enumerate(names)]


def symmetrize(names, M):
    """Walking and running are symmetric: the second step is the first one mirrored. Average each pose with the
    mirror of the pose half a cycle later: halves the pose noise and evens out SAM's left/right bias."""
    F = len(M)
    Mm = np.roll(M, -F // 2, 0)[:, mirror_perm(names)].copy()
    Mm[..., 0] *= -1
    return 0.5 * (M + Mm)


def lowpass(Y, K):
    """Keep the first K harmonics of a periodic signal sampled evenly over one cycle (axis 0)."""
    C = np.fft.rfft(Y, axis=0)
    C[K + 1:] = 0
    return np.fft.irfft(C, n=len(Y), axis=0)


def solve(clip, F=64, K=7):
    """One looping cycle of the clip as a motion file (tools3d/motions/<clip>.json); see the module docstring."""
    d = os.path.join(OUT, clip)
    with open(os.path.join(d, "track.json")) as f:
        tr = json.load(f)
    names, fi, K3 = load_sam(clip)
    lo, hi = tr["window"]
    m = (fi >= lo) & (fi <= hi)
    fi, K3 = fi[m], K3[m]
    Q, Rc = body_frames(names, K3)
    ix = {n: i for i, n in enumerate(names)}
    c = CLIPS[clip]
    gait = c["kind"] == "gait"
    P = tr["period_frames"]
    if gait:   # refine the period on the dense samples
        sel = [ix[n] for n in GAIT_KEYS]
        P = min((fourier_fit((fi / q) % 1, Q[:, sel], K)[1], q) for q in np.linspace(0.95 * P, 1.05 * P, 81))[1]
    phase = ((fi - fi.min()) / P) % 1
    fit, res = fourier_fit(phase, Q, K if gait else 8)
    ph = np.arange(F) / F
    M = fit(ph)
    if c.get("symmetric"):
        M = symmetrize(names, M)
    T_video = P / tr["fps"]
    lo_s, hi_s = c.get("cycle_s", (0, 1e9))
    T = float(np.clip(T_video, lo_s, hi_s)) if gait else T_video
    fp = foot_points(names, M)
    travel = np.array(c.get("travel", (0, 1)), float) if gait else np.zeros(2)
    if not gait:
        contact = np.ones((F, 2), bool)
    elif not c.get("flight"):
        # walking and side-stepping: SAM's foot heights are too noisy for a small subject, but a planted foot moves
        # against the travel direction relative to the pelvis and a swinging one moves with it, fast. Planted =
        # moving backward along the travel; there's always a foot down (the lower one if neither qualifies)
        mid = fp.mean(2)[:, :, [0, 2]]                              # (F, 2, 2) foot centre, horizontal
        along = ((np.roll(mid, -1, 0) - np.roll(mid, 1, 0)) / 2) @ unit2(travel)
        contact = clean_runs(along < 0.05 * np.abs(along).max())
        low = fp[:, :, :, 1].min(2)
        # guard: planted feet must be the low ones. If they are the high ones, the person travels the other way
        # (video models often film a forward walk when asked for a backward one)
        if (low[~contact].mean() - low[contact].mean()) < 0:
            raise ValueError(f"{clip}: the video's motion travels opposite to {tuple(travel)}; regenerate the video")
        none = ~contact.any(1)
        contact[none, np.argmin(low[none], 1)] = True
        if c.get("symmetric"):                                   # the right foot is the left one half a cycle on
            both = contact[:, 0] | np.roll(contact[:, 1], -F // 2)
            contact = np.stack([both, np.roll(both, F // 2)], 1)
    else:
        contact = detect_contacts(fp, travel)
    for _ in range(3):
        p, D = root_motion(fp, contact, T, D_fixed=None if gait else (0.0, 0.0))
        if not gait or not c.get("flight"):
            break
        new = detect_contacts(fp, travel, p, D)
        if c.get("symmetric"):
            both = new[:, 0] & np.roll(new[:, 1], -F // 2)
            new = np.stack([both, np.roll(both, F // 2)], 1)
        if (new == contact).all():
            break
        contact = new
    # phase 0 at the left foot's touchdown
    shift = 0
    if gait:
        starts = np.nonzero(contact[:, 0] & ~np.roll(contact[:, 0], 1))[0]
        shift = int(starts[0]) if len(starts) else 0
    M, contact = np.roll(M, -shift, 0), np.roll(contact, -shift, 0)
    p = np.roll(p, -shift, 0)
    if shift:
        p[F - shift:, [0, 2]] += np.r_[D[0], D[1]][[0, 1]]   # samples moved past the seam belong to the next cycle
    p[:, [0, 2]] -= np.arange(F)[:, None] / F * D         # in place: remove the steady travel
    p[:, [0, 2]] -= p[:, [0, 2]].mean(0)
    # the pelvis path follows the noisiest keypoints (the lowest foot point switches heel -> toes): keep its first
    # harmonics only (a gait's bob is the 2nd), and for symmetric gaits make the sway antisymmetric and the bob and
    # surge repeat every step
    p = lowpass(p, 4 if gait else 6)
    if c.get("symmetric"):
        q = np.roll(p, -F // 2, 0)
        p = np.stack([(p[:, 0] - q[:, 0]) / 2, (p[:, 1] + q[:, 1]) / 2, (p[:, 2] + q[:, 2]) / 2], 1)
    leg = float(np.mean([np.linalg.norm(Q[:, ix[s + "-hip"]] - Q[:, ix[s + "-knee"]], axis=1).mean()
                         + np.linalg.norm(Q[:, ix[s + "-knee"]] - Q[:, ix[s + "-ankle"]], axis=1).mean()
                         for s in ("left", "right")]))
    stride = float(np.linalg.norm(D))
    out = {"clip": clip, "kind": c["kind"], "names": names, "period": round(T, 4), "video_period": round(T_video, 4),
           "speed": round(stride / T, 4) if gait else 0.0,
           "direction": ([round(float(D[0] / stride), 4), 0.0, round(float(D[1] / stride), 4)]
                         if gait and stride > 1e-6 else [0.0, 0.0, 0.0]),
           "stride": round(stride, 4), "leg": round(leg, 4), "fold_residual_m": round(res, 4),
           "duty": [round(float(contact[:, i].mean()), 3) for i in range(2)],
           "frames": np.round(M, 4).tolist(), "root": np.round(p, 4).tolist(),
           "contacts": contact.astype(int).T.tolist()}
    os.makedirs(MOTIONS, exist_ok=True)
    with open(os.path.join(MOTIONS, clip + ".json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return {k: v for k, v in out.items() if k not in ("frames", "root", "names")}


def mirror(src, dst):
    """A motion mirrored left-right (e.g. StrafeRight from StrafeLeft): swap the left/right keypoints and contacts,
    negate x."""
    with open(os.path.join(MOTIONS, src + ".json")) as f:
        m = json.load(f)
    names = m["names"]
    ix = {n: i for i, n in enumerate(names)}
    perm = [ix["right-" + n[5:]] if n.startswith("left-") else ix["left-" + n[6:]] if n.startswith("right-") else i
            for i, n in enumerate(names)]
    M = np.array(m["frames"])[:, perm]
    M[..., 0] *= -1
    root = np.array(m["root"])
    root[:, 0] *= -1
    m.update(clip=dst, frames=np.round(M, 4).tolist(), root=np.round(root, 4).tolist(),
             contacts=m["contacts"][::-1], duty=m["duty"][::-1], mirrored_from=src,
             direction=[-m["direction"][0], m["direction"][1], m["direction"][2]])
    with open(os.path.join(MOTIONS, dst + ".json"), "w") as f:
        json.dump(m, f, separators=(",", ":"))
    return {k: v for k, v in m.items() if k not in ("frames", "root", "names", "contacts")}


if __name__ == "__main__":
    cmd, clip = sys.argv[1], sys.argv[2]
    if cmd == "video" or (cmd == "all" and not os.path.exists(os.path.join(clip_dir(clip), "video.mp4"))):
        print(video(clip))
    if cmd in ("track", "all"):
        print(json.dumps(track(clip)))
    if cmd in ("solve", "all"):
        print(json.dumps(solve(clip), indent=1)[:3000])
    if cmd == "mirror":                                          # mocap.py mirror StrafeLeft StrafeRight
        print(json.dumps(mirror(clip, sys.argv[3])))
