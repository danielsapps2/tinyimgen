"""Retarget a captured motion (mocap.py -> tools3d/motions/<Clip>.json) onto a rigged character (rig.py skeleton).

The motion is a cycle of 70 keypoints of the reference person. For the character:
- Joint rotations come from bone directions in the keypoints, with each bone's twist from a second direction: the
  hip line for the pelvis, the shoulder line for the chest, the ears for the head, the knuckle line for the forearm
  and hand, the elbow / knee bend for upper arms and legs. Lengths never come from the reference, so the character
  keeps its own proportions.
- SAM's view bias (a few degrees of head yaw, torso roll) is removed by zeroing their cycle means; real posture
  (a runner's forward lean) is kept.
- Feet: wherever the reference has a foot planted, the character's heel / toe / sole is pinned to the reference's
  planted point scaled by leg length, and two-bone IK reaches it. Planted points move at exactly -speed in the
  in-place clip, so foot slide stays zero for every character.
- Arms keep at least the character's own clearance angle (anim.arm_abduction), so bulky hips and pouches don't
  swallow the hands.
"""
import json
import os

import numpy as np

from anim import FINGERS, FPS, RELAXED, Pose, aim, leg_ik, quat, rx, ry, rz, set_fingers, unit
from rig import INDEX, NAMES, PARENT

MOTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
Z = np.array([0, 0, 1.0])
X_AXIS = np.array([1.0, 0, 0])


def motion_path(name):
    return os.path.join(MOTIONS, name + ".json")


def frame_xy(x, y):
    """Rotation with columns (x, y, z): x exact, y orthogonalised."""
    x = unit(x)
    y = unit(y - (y @ x) * x)
    return np.stack([x, y, np.cross(x, y)], 1)


def slerp(A, B, t):
    """Rotation interpolation (via the relative rotation's axis-angle)."""
    Rr = A.T @ B
    ang = np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))
    if ang < 1e-6:
        return A
    w = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / (2 * np.sin(ang))
    a = t * ang
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return A @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)


def yaw_roll_only(R):
    """The part of R that isn't pitch about X: R = Ry(yaw) Rx(pitch) Rz(roll) -> Ry(yaw) Rz(roll)... applied as a
    correction we only need yaw and roll, so return (yaw, roll)."""
    yaw = np.arctan2(R[0, 2], R[2, 2])
    roll = np.arctan2(R[1, 0], R[1, 1])
    return yaw, roll


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, 0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


class Motion:
    def __init__(self, name):
        with open(motion_path(name)) as f:
            m = json.load(f)
        self.m = m
        self.name = name
        self.ix = {n: i for i, n in enumerate(m["names"])}
        self.M = np.array(m["frames"])
        self.root = np.array(m["root"])
        self.contacts = np.array(m["contacts"]).T.astype(bool)           # (F, 2)
        self.F = len(self.M)
        self.period, self.speed, self.leg = m["period"], m["speed"], m["leg"]
        self.direction = np.array(m["direction"])
        self.kind = m["kind"]
        raw = [self.raw_frames(k) for k in range(self.F)]
        # calibration: chest yaw / roll relative to the pelvis and the head's world offset average to zero; a
        # foot's pitch averages to flat while planted
        rel = mean_rotation([r["pelvis"].T @ r["chest"] for r in raw])
        self.chest_fix = yaw_roll_only(rel)
        self.head_fix = mean_rotation([r["head"] for r in raw]).T
        # the heel keypoint sits higher than the toe tips, so a flat foot reads toes-down by foot_bias: measured on
        # frames where the reference foot is flat on the floor (heel and toes both down)
        self.foot_bias = {}
        for f, side in enumerate(("Left", "Right")):
            pre = side.lower()
            b = []
            for k in range(self.F):
                heel = self.root[k] + self.M[k, self.ix[pre + "-heel"]]
                toe = self.root[k] + (self.M[k, self.ix[pre + "-big-toe-tip"]] + self.M[k, self.ix[pre + "-small-toe-tip"]]) / 2
                if self.contacts[k, f] and heel[1] < 0.03 and toe[1] < 0.03:
                    d = toe - heel
                    b.append(np.arctan2(-d[1], np.hypot(d[0], d[2])))
            self.foot_bias[side] = float(np.median(b)) if b else 0.0
        # the reference person walks with knees and feet turned out; the characters shouldn't copy that: mean
        # sideways angle of the knee (bent frames) relative to the pelvis, removed in pose_from_motion
        self.knee_yaw = {}
        for side in ("Left", "Right"):
            pre = side.lower()
            yaws = []
            for k, r in enumerate(raw):
                hp, kn, an = (self.M[k, self.ix[pre + j]] for j in ("-hip", "-knee", "-ankle"))
                u, w = unit(kn - hp), an - kn
                perp = w - (w @ u) * u
                if np.linalg.norm(perp) > 0.3 * np.linalg.norm(w):     # clearly bent: the bend shows the knee's way
                    loc = r["pelvis"].T @ unit(-perp)
                    yaws.append(np.arctan2(loc[0], loc[2]))
            self.knee_yaw[side] = float(np.median(yaws)) if yaws else 0.0
        # reference standing pelvis height: leg + planted ankle height
        ank = [self.root[k, 1] + self.M[k, self.ix["left-ankle"], 1] for k in range(self.F) if self.contacts[k, 0]]
        self.ankle_h = float(np.median(ank)) if ank else 0.08
        self.stand = self.leg + self.ankle_h

    def kp(self, k, name):
        return self.M[k, self.ix[name]]

    def raw_frames(self, k):
        g = lambda n: self.kp(k, n)  # noqa: E731
        mid_hip = (g("left-hip") + g("right-hip")) / 2
        mid_sh = (g("left-shoulder") + g("right-shoulder")) / 2
        spine = mid_sh - mid_hip
        out = {"pelvis": frame_xy(g("left-hip") - g("right-hip"), spine),
               "chest": frame_xy(g("left-shoulder") - g("right-shoulder"), spine),
               "head": frame_xy(g("left-ear") - g("right-ear"), (g("left-ear") + g("right-ear")) / 2 - g("neck"))}
        for side, s in (("Left", "left"), ("Right", "right")):
            heel = g(s + "-heel")
            toe = (g(s + "-big-toe-tip") + g(s + "-small-toe-tip")) / 2
            lateral = g(s + "-small-toe-tip") - g(s + "-big-toe-tip")
            x = lateral if side == "Left" else -lateral                 # +X is the character's left
            out["foot" + side] = foot_frame(toe - heel, x)
        return out


TOE_OUT = np.radians(3.0)        # the characters' natural toe-out (per side, outward)


def axis_angle(u, a):
    """Rotation by angle a about unit axis u."""
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def foot_frame(toe_dir, left):
    """Foot rotation (identity = flat, toes +Z) from the heel-to-toe direction and the foot's +X side."""
    fz = unit(toe_dir)
    y = unit(np.cross(fz, unit(left - (left @ fz) * fz)))
    return np.stack([np.cross(y, fz), y, fz], 1)


def bend_pole(a, b, c, fallback, inward=False):
    """Direction the joint at b bends toward (for a -> b -> c), falling back as the limb straightens. For knees the
    pole points the way the knee does (forward), so it is the opposite of the lower bone's sideways component.
    Knees and elbows never bend backward: a measured direction that disagrees with the fallback (a straight or
    slightly over-straightened limb reads as bent the wrong way) fades to the fallback instead of flipping the
    joint."""
    u = unit(b - a)
    w = c - b
    perp = w - (w @ u) * u
    n = np.linalg.norm(perp)
    fb = unit(fallback - (fallback @ u) * u)
    if n < 1e-9:
        return fb
    pole = unit(-perp if inward else perp)
    bend = min(1.0, n / (0.35 * np.linalg.norm(w) + 1e-9))
    agree = np.clip((pole @ fb) / 0.5, 0.0, 1.0)            # full trust within 60 degrees of the fallback
    wgt = bend * agree * agree * (3 - 2 * agree)
    return unit(wgt * pole + (1 - wgt) * fb)


MHR_FINGER = {"Thumb": "thumb", "Index": "index", "Middle": "middle", "Ring": "ring", "Little": "pinky"}
BASELINE = {"Thumb": (30.0, 8.0, 5.0)}           # the thumb's bends at rest (flat hand): only bending beyond counts


def finger_curls(g):
    """Finger bends (degrees per joint) from the reference's hand keypoints: the angle between consecutive
    segments wrist -> knuckle -> middle -> end joint -> tip, never less than a relaxed hand."""
    from rig import CURL_MAX
    out = {}
    for side, s in (("Left", "left"), ("Right", "right")):
        out[side] = {}
        w = g(s + "-wrist")
        for f in FINGERS:
            m = MHR_FINGER[f]
            pts = [w] + [g(f"{s}-{m}-{j}") for j in ("third-joint", "second-joint", "first-joint", "tip")]
            seg = [unit(pts[k + 1] - pts[k]) for k in range(4)]
            ang = []
            for k in range(3):
                a = np.degrees(np.arccos(np.clip(seg[k] @ seg[k + 1], -1, 1)))
                a -= BASELINE.get(f, (5.0, 5.0, 5.0))[k]
                ang.append(float(np.clip(max(a, RELAXED[f][k]), 0, CURL_MAX[f][k])))
            out[side][f] = tuple(ang)
    return out


def pose_from_motion(sk, mo, k_float, drop=0.0):
    """Pose of the character at fractional sample index k_float of the motion (no foot locking yet). Returns the
    pose, the knee poles, and the reference keypoints used."""
    F = mo.F
    k0 = int(np.floor(k_float)) % F
    k1 = (k0 + 1) % F
    t = k_float - np.floor(k_float)
    M = (1 - t) * mo.M[k0] + t * mo.M[k1]
    root = (1 - t) * mo.root[k0] + t * mo.root[k1]
    g = lambda n: M[mo.ix[n]]  # noqa: E731
    pose = Pose(sk)
    mid_hip = (g("left-hip") + g("right-hip")) / 2
    mid_sh = (g("left-shoulder") + g("right-shoulder")) / 2
    spine = mid_sh - mid_hip
    R_pel = frame_xy(g("left-hip") - g("right-hip"), spine)
    R_ch = frame_xy(g("left-shoulder") - g("right-shoulder"), spine)
    yaw, roll = mo.chest_fix
    # remove the chest's mean yaw / roll relative to the pelvis (SAM bias), keep its motion and the lean
    rel = R_pel.T @ R_ch
    rel = ry(-yaw) @ rel @ rz(-roll)
    R_ch = R_pel @ rel
    R_head = frame_xy(g("left-ear") - g("right-ear"), (g("left-ear") + g("right-ear")) / 2 - g("neck")) @ mo.head_fix
    pose.set("Hips", R_pel)
    pose.set("Spine", slerp(R_pel, R_ch, 1 / 3))
    pose.set("Chest", slerp(R_pel, R_ch, 2 / 3))
    pose.set("UpperChest", R_ch)
    pose.set("Neck", slerp(R_ch, R_head, 0.5))
    pose.set("Head", R_head)
    for side, s, sgn in (("Left", "left", 1), ("Right", "right", -1)):
        pose.set(side + "Shoulder", R_ch)
        sh, el, wr = g(s + "-shoulder"), g(s + "-elbow"), g(s + "-wrist")
        knuckle = g(s + "-middle-third-joint")
        thumb_side = g(s + "-index-third-joint") - g(s + "-pinky-third-joint")
        fwd = R_ch @ Z
        S, E, W = sk.g(side + "UpperArm"), sk.g(side + "LowerArm"), sk.g(side + "Hand")
        tip = sk.tips[side + "HandTip"]
        Ru = aim(E - S, Z, el - sh, bend_pole(sh, el, wr, fwd))
        Rl = aim(W - E, Z, wr - el, thumb_side)
        Rh = aim(tip - W, Z, knuckle - wr, thumb_side)
        # keep the character's own clearance: rotate the arm out about the chest's forward axis if it hangs inside
        u = R_ch.T @ unit(el - sh)
        if u[1] < -0.3:
            out = np.arctan2(sgn * u[0], -u[1])
            need = sk.abduction[side] - out
            if need > 0:
                C = R_ch @ rz(sgn * need) @ R_ch.T
                Ru, Rl, Rh = C @ Ru, C @ Rl, C @ Rh
        pose.set(side + "UpperArm", Ru)
        pose.set(side + "LowerArm", Rl)
        pose.set(side + "Hand", Rh)
    set_fingers(pose, finger_curls(g))
    knees = {}
    for side, s, sgn in (("Left", "left", 1), ("Right", "right", -1)):
        hp, kn, an = g(s + "-hip"), g(s + "-knee"), g(s + "-ankle")
        pole = bend_pole(hp, kn, an, R_pel @ Z, inward=True)
        # turn the knee from the reference's toe-out to the character's (twist about the hip-ankle line)
        # (a turn about the downward leg axis by +a is a yaw of -a)
        pole = axis_angle(unit(an - hp), (mo.knee_yaw[side] - sgn * TOE_OUT) * np.sign(-(an - hp)[1] or 1)) @ pole
        knees[side] = pole
        Rk = sk.g(side + "LowerLeg") - sk.g(side + "UpperLeg")
        Ra = sk.g(side + "Foot") - sk.g(side + "LowerLeg")
        # twist from the knee's hinge axis (stays perpendicular to the shin at any bend; see anim.leg_ik)
        hinge = unit(np.cross(unit(an - hp), pole))
        pose.set(side + "UpperLeg", aim(Rk, -X_AXIS, kn - hp, hinge))
        R_shin = aim(Ra, -X_AXIS, an - kn, hinge)
        pose.set(side + "LowerLeg", R_shin)
        # the ankle is a hinge: the foot points the way the shin (knee) does, plus the natural toe-out, and only
        # its up / down pitch comes from the reference (a foot pointing down at toe-off has no reliable heading)
        toe = (g(s + "-big-toe-tip") + g(s + "-small-toe-tip")) / 2
        d = R_shin.T @ (toe - g(s + "-heel"))
        pitch_down = np.arctan2(-d[1], d[2]) - mo.foot_bias[side]
        W = R_shin @ ry(sgn * TOE_OUT) @ rx(pitch_down)                  # the mesh foot's world frame
        Rf = W @ ry(sk.foot_yaw[side]).T                                 # its bone: undo the mesh's rest turn
        pose.set(side + "Foot", Rf)
        pose.set(side + "Toes", Rf)
    # pelvis: the reference's in-place root, scaled by leg length; the Hips joint sits above the hip joints' midpoint
    s = sk.leg / mo.leg
    mid_rest = (sk.g("LeftUpperLeg") + sk.g("RightUpperLeg")) / 2
    ankle = (sk.g("LeftFoot")[1] + sk.g("RightFoot")[1]) / 2
    target = np.array([mid_rest[0] + root[0] * s, (root[1] - mo.ankle_h) * s + ankle - drop, mid_rest[2] + root[2] * s])
    pose.hips = target + R_pel @ (sk.g("Hips") - mid_rest)
    return pose, knees, M, root


def pivot_kind(mo, M, root, side):
    """Which part of the reference's planted foot is on the floor: 0 heel, 1 whole sole, 2 toes."""
    pre = side.lower()
    heel = (M[mo.ix[pre + "-heel"]] + root)[1]
    toe = ((M[mo.ix[pre + "-big-toe-tip"]] + M[mo.ix[pre + "-small-toe-tip"]]) / 2 + root)[1]
    return 0 if heel < toe - 0.01 else 2 if toe < heel - 0.01 else 1


def ref_pivot(mo, M, root, side, kind):
    pre = side.lower()
    heel = M[mo.ix[pre + "-heel"]] + root
    toe = (M[mo.ix[pre + "-big-toe-tip"]] + M[mo.ix[pre + "-small-toe-tip"]]) / 2 + root
    return (heel, (heel + toe) / 2, toe)[kind]


def my_pivot(sk, side, kind):
    """The character's heel / sole middle / ball in its rest (flat) foot, on the floor."""
    heel, ball = sk.foot[side]["heel"], sk.foot[side]["ball"]
    return (heel, (heel + ball) / 2, ball)[kind]


def planted_foot(R_bone, yaw0, foot_yaw, flat_deg=2.0):
    """A planted foot keeps its touchdown heading (yaw0, of the mesh foot) and doesn't roll; only its heel-to-toe
    pitch follows the reference. Returns (foot bone rotation, toes bone rotation, pivot kind 0 heel / 1 sole / 2 ball).
    With the heel up the toes stay flat on the floor and the foot pivots on the ball."""
    Yr = ry(foot_yaw)
    W = R_bone @ Yr                                                     # the mesh foot's frame
    pitch_up = np.arctan2(W[1, 2], W[2, 2])
    flat = ry(yaw0) @ Yr.T
    if abs(pitch_up) < np.radians(flat_deg):
        return flat, flat, 1
    pitch_up -= np.sign(pitch_up) * np.radians(flat_deg)              # continuous through the flat band
    Rf = ry(yaw0) @ rx(-pitch_up) @ Yr.T
    return (Rf, Rf, 0) if pitch_up > 0 else (Rf, flat, 2)


def mesh_heading(R_bone, foot_yaw):
    W = R_bone @ ry(foot_yaw)
    return np.arctan2(W[0, 2], W[2, 2])


def foot_points_world(sk, pose, side):
    """Heel, ball and toe tip on the sole, in the pose (the toe tip hinges at the ball with the toes bone)."""
    f = sk.foot[side]
    R, A = pose.get(side + "Foot"), pose.pos(side + "Foot")
    heel, ball = (A + R @ (q - f["ankle"]) for q in (f["heel"], f["ball"]))
    return [heel, ball, ball + pose.get(side + "Toes") @ (f["toe"] - f["ball"])]


def floor_guard(sk, pose, side, knee, clearance=0.0):
    """A swinging foot must not sink into the floor (the character's feet and legs aren't the reference's): lift it
    by IK when its lowest point is below `clearance`."""
    low = min(q[1] for q in foot_points_world(sk, pose, side))
    if low < clearance:
        ankle = pose.pos(side + "Foot") + np.array([0, clearance - low, 0])
        leg_ik(pose, side, ankle, pose.get(side + "Foot"), pose.get(side + "Toes"), knee)


def bake_motion(sk, name, clip_name=None, rate=1.0):
    """Clip dict (as anim.bake) for the motion `name` on skeleton `sk`. rate > 1 plays it faster: a shorter cycle
    with the same stride (speed and cadence scale together, the feet stay planted)."""
    mo = Motion(name)
    n = max(2, int(round(mo.period / rate * FPS)))
    s = sk.leg / mo.leg
    blend = 3                                                          # frames to ease the foot lock in and out

    step = mo.speed * s * mo.period / n * mo.direction                 # ground travel per frame (in place: -step)
    mid_rest = (sk.g("LeftUpperLeg") + sk.g("RightUpperLeg")) / 2
    contact = np.array([mo.contacts[int(round(i / n * mo.F)) % mo.F] for i in range(n)])          # (n, 2)

    def solve(drop):
        base = [pose_from_motion(sk, mo, i / n * mo.F, drop) for i in range(n)]
        poses = [b[0] for b in base]
        fk = {f: [b[0].pos(side + "Foot") for b in base] for f, side in enumerate(("Left", "Right"))}
        worst = 0.0
        for f, side in enumerate(("Left", "Right")):
            c = contact[:, f]
            targets, rot_fix = {}, {}
            if c.any():
                # rolling contact over each planted interval, starting at its touchdown
                starts = [i for i in range(n) if c[i] and not c[i - 1]] or [0]
                for st in starts:
                    pose, _, M, root = base[st]
                    R0 = pose.get(side + "Foot")
                    yaw0 = (1 if side == "Left" else -1) * TOE_OUT                # planted feet point ahead
                    _, _, kind = planted_foot(R0, yaw0, sk.foot_yaw[side])
                    ref = ref_pivot(mo, M, root, side, pivot_kind(mo, M, root, side))
                    anchor = np.array([mid_rest[0] + ref[0] * s, 0.0, mid_rest[2] + ref[2] * s])
                    i = st
                    while True:
                        pose, knees, M, root = base[i]
                        R_foot, R_toes, new = planted_foot(pose.get(side + "Foot"), yaw0, sk.foot_yaw[side])
                        if i != st:
                            anchor = anchor - step
                            if new != kind:                            # roll onto another part of the foot
                                prev = poses[(i - 1) % n]
                                pa = prev.pos(side + "Foot") + prev.get(side + "Foot") @ (
                                    my_pivot(sk, side, new) - sk.foot[side]["ankle"])
                                anchor = np.array([pa[0], 0.0, pa[2]]) - step
                                kind = new
                        targets[i] = anchor + R_foot @ (sk.foot[side]["ankle"] - my_pivot(sk, side, kind))
                        Rc = pose.get(side + "Foot")
                        rot_fix[i] = (R_foot @ Rc.T, R_toes @ Rc.T)
                        worst = max(worst, leg_ik(pose, side, targets[i], R_foot, R_toes, knees[side]))
                        i = (i + 1) % n
                        if not c[i] or i == st:
                            break
            # swing: the lock's correction (target - captured foot position) blends smoothly from lift-off to the
            # next touchdown, so the leg never jumps between the captured pose and the planted spot
            if targets:
                off = {i: targets[i] - fk[f][i] for i in targets}
                for i in range(n):
                    if c[i]:
                        continue
                    a = i
                    while not c[a % n]:
                        a -= 1
                    b = i
                    while not c[b % n]:
                        b += 1
                    t = (i - a) / (b - a)
                    t = t * t * (3 - 2 * t)
                    pose, knees, M, root = base[i]
                    ankle = fk[f][i] + (1 - t) * off[a % n] + t * off[b % n]
                    # the planted foot's rotation correction (heading held, toes flat) eases out and back in too
                    Rc = pose.get(side + "Foot")
                    Rf = slerp(rot_fix[a % n][0], rot_fix[b % n][0], t) @ Rc
                    Rt = slerp(rot_fix[a % n][1], rot_fix[b % n][1], t) @ Rc
                    leg_ik(pose, side, ankle, Rf, Rt, knees[side])
            # swinging feet keep the reference's clearance mid-swing (no skimming), easing to the floor next to a
            # contact (no drop at touchdown)
            for i in range(n):
                if c[i]:
                    continue
                pose, knees, M, root = base[i]
                d = min([abs(j) for j in range(-n, n) if c[(i + j) % n]] or [n])
                ramp = np.clip((d - 1) / 3, 0, 1)
                ramp = ramp * ramp * (3 - 2 * ramp)
                pre = side.lower()
                ref_low = min((M[mo.ix[pre + q]] + root)[1] for q in ("-heel", "-big-toe-tip", "-small-toe-tip"))
                floor_guard(sk, pose, side, knees[side], 0.002 + ramp * max(0.0, ref_low * s))
        poses.append(poses[0])                                         # closing frame = first: seamless loop
        return poses, worst
    drop = 0.0
    for _ in range(4):
        poses, worst = solve(drop)
        if worst < 1e-3:
            break
        drop += worst * 1.1
    rot = np.zeros((len(poses), len(NAMES), 4))
    hips = np.zeros((len(poses), 3))
    for fi, p in enumerate(poses):
        for j in range(len(NAMES)):
            Rl = p.R[j] if PARENT[j] is None else p.R[PARENT[j]].T @ p.R[j]
            rot[fi, j] = quat(Rl)
        hips[fi] = p.hips
    for fi in range(1, len(poses)):
        flip = (rot[fi] * rot[fi - 1]).sum(1) < 0
        rot[fi, flip] *= -1
    speed = mo.speed * mo.period * s / (n / FPS)                       # this character's stride over this cycle
    extras = {"speed": round(float(speed), 4), "loop": True, "source": "mocap:" + name, "rate": round(float(rate), 3)}
    if mo.kind == "gait":
        extras.update(direction=[round(float(x), 4) for x in mo.direction], stride=round(float(speed * n / FPS), 4),
                      duty=round(float(mo.contacts[:, 0].mean()), 3), sync=0.0)
    return {"name": clip_name or name, "fps": FPS, "rot": rot, "hips": hips, "extras": extras, "poses": poses,
            "drop": drop}
