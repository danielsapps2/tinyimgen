"""Clip library for rig.py's humanoid skeleton: Idle, Walk, Run, WalkBackward, StrafeLeft, StrafeRight.

Each clip is motion-captured from generated reference video (mocap.py -> tools3d/motions/<Clip>.json) and retargeted
onto the character by retarget.py: real full-body motion (torso rotation and bob, weight shift, arm carriage, knee
drive), with the feet pinned wherever the reference's feet were planted, so foot slide stays ~0 at the clip's speed.

The procedural generator below is the fallback when a motion file is missing:
- Feet follow planted trajectories on the ground (heel strike -> flat -> roll onto the ball -> swing), and the legs
  reach them with two-bone IK. In the in-place clip a planted foot moves backward at exactly the clip's speed, so a
  game that moves the character at `speed` m/s (stored in each animation's extras) gets no foot sliding.
- The pelvis bobs, sways over the stance foot and twists with the stride; the spine counter-rotates; the head stays
  level and facing the travel direction.
- Arms hang outward by the angle that clears the character's own hips, belts and pouches (measured on the mesh), and
  swing opposite the legs.

Both kinds start at the left foot's touchdown (phase 0; the animation extras carry `sync` for the rest), so an engine
can blend them in sync by normalised time (e.g. Walk + StrafeLeft for a diagonal).

Conventions: +Z forward, +Y up, +X is the character's left. Every joint's bind pose has identity rotation, so a pose
is a global rotation per joint; bake() turns it into glTF local rotations (x, y, z, w) plus the Hips translation.
"""
import numpy as np

from rig import INDEX, NAMES, PARENT

FPS = 30


# --- small math -----------------------------------------------------------------------------------------------------

def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def unit(v):
    return v / max(1e-12, np.linalg.norm(v))


def frame(d, p):
    d = unit(d)
    p = unit(p - (p @ d) * d)
    return np.stack([d, p, np.cross(d, p)], 1)


def aim(rest_dir, rest_pole, d, pole):
    """Global rotation taking a bone from its rest direction to `d`, with its bend side (pole) turned to `pole`."""
    return frame(d, pole) @ frame(rest_dir, rest_pole).T


def quat(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k]))
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def hermite(p0, p1, m0, m1, s):
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1


# --- skeleton measurements ------------------------------------------------------------------------------------------

class Skeleton:
    def __init__(self, J, tips, abduction=None, foot_yaw=None):
        self.J = np.asarray(J, np.float64)
        self.tips = {k: np.asarray(v, np.float64) for k, v in tips.items()}
        g = self.g
        self.H = tips["HeadTop"][1]
        self.thigh = {s: np.linalg.norm(g(s + "LowerLeg") - g(s + "UpperLeg")) for s in ("Left", "Right")}
        self.shin = {s: np.linalg.norm(g(s + "Foot") - g(s + "LowerLeg")) for s in ("Left", "Right")}
        self.leg = float(np.mean([self.thigh[s] + self.shin[s] for s in ("Left", "Right")]))
        # the mesh's own feet may point out at rest (generated T-poses often do): foot_yaw (radians, + toward +X)
        # turns the sole's heel / ball / toe points about the ankle to where the mesh foot really is
        self.foot_yaw = foot_yaw or {"Left": 0.0, "Right": 0.0}
        self.foot = {}
        for s in ("Left", "Right"):
            ank, ball, tip = g(s + "Foot"), g(s + "Toes"), self.tips[s + "ToeTip"]
            flen = (tip[2] - ank[2]) / 0.75
            Ry = ry(self.foot_yaw[s])
            on_sole = lambda dz: ank + Ry @ np.array([0.0, -ank[1], dz])  # noqa: E731
            self.foot[s] = {"ankle": ank, "heel": on_sole(-0.25 * flen), "ball": on_sole(ball[2] - ank[2]),
                            "toe": on_sole(tip[2] - ank[2]), "len": flen}
        # how far the arms must hang out from the body (radians, per side); default when the mesh wasn't measured
        self.abduction = abduction or {"Left": np.radians(10), "Right": np.radians(10)}
        from rig import curl_axes, NAMES as _ALL
        self.curl = curl_axes(self.J, self.tips) if len(self.J) == len(_ALL) else {}

    def g(self, name):
        return self.J[INDEX[name]]


def foot_rest_yaw(V, joints, weights, limit_deg=35.0):
    """Per side, which way the mesh's foot points at rest (radians from +Z, + toward +X): the long horizontal axis
    of the vertices the Foot / Toes bones carry. Generated T-poses often have the feet turned out 10-20 degrees."""
    dom = joints[np.arange(len(V)), np.argmax(weights, 1)]
    out = {}
    for side in ("Left", "Right"):
        P = V[np.isin(dom, [INDEX[side + "Foot"], INDEX[side + "Toes"]])][:, [0, 2]]
        if len(P) < 30:
            out[side] = 0.0
            continue
        P = P - P.mean(0)
        w, U = np.linalg.eigh(P.T @ P)
        d = U[:, -1] * np.sign(U[1, -1] or 1.0)                           # the long axis, pointing forward (+Z)
        out[side] = float(np.clip(np.arctan2(d[0], d[1]), -np.radians(limit_deg), np.radians(limit_deg)))
    return out


def arm_abduction(V, joints, weights, J, tips):
    """Per side, the angle from vertical at which a hanging arm clears the body (hips, belt, pouches, coat) under it.
    V: vertices, joints/weights: skin from rig.skin_weights."""
    arm_bones = {INDEX[s + b] for s in ("Left", "Right") for b in ("UpperArm", "LowerArm", "Hand")}
    arm_bones |= {i for i, n in enumerate(NAMES) if any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little"))}
    dom = joints[np.arange(len(V)), np.argmax(weights, 1)]
    body = V[~np.isin(dom, list(arm_bones))]
    H = tips["HeadTop"][1]
    cx = J[INDEX["Hips"]][0]
    out = {}
    for side, sgn in (("Left", 1), ("Right", -1)):
        S = J[INDEX[side + "UpperArm"]]
        reach = (np.linalg.norm(J[INDEX[side + "LowerArm"]] - S) + np.linalg.norm(J[INDEX[side + "Hand"]]
                 - J[INDEX[side + "LowerArm"]]) + np.linalg.norm(tips[side + "HandTip"] - J[INDEX[side + "Hand"]]))
        r_arm = 0.028 * H
        need = np.radians(4)
        near = body[np.abs(body[:, 2] - S[2]) < 0.09 * H]
        for dy in np.linspace(0.14 * H, reach, 24):
            sl = near[np.abs(near[:, 1] - (S[1] - dy)) < 0.012 * H]
            if not len(sl):
                continue
            hw = np.percentile((sl[:, 0] - cx) * sgn, 99)
            need = max(need, np.arctan2(hw + r_arm - (S[0] - cx) * sgn, dy))
        out[side] = float(np.clip(need + np.radians(2), np.radians(5), np.radians(40)))
    return out


# --- clip definitions -----------------------------------------------------------------------------------------------
# Lengths are in leg lengths (hip joint to ankle), angles in degrees, times in seconds. elbow: (bend with the arm
# swung back, bend with it forward); arm: swing amplitude, arm_fwd: its forward offset.

WALK = dict(motion="Walk", name="Walk", kind="gait", move=(0, 0, 1), period=1.08, stride=1.55, duty=0.62, track=0.75, lift=0.11,
            lift_shape=0.85, bob=0.018, sway=0.022, yaw=6, roll=4, lean=3, arm=20, arm_fwd=4, elbow=(10, 32),
            strike=18, toeoff=28, s1=0.10, s2=0.26, reach=0.975, drop=0.01, toe_out=6, head=0)
RUN = dict(motion="Run", name="Run", kind="gait", move=(0, 0, 1), period=0.70, stride=2.45, duty=0.36, track=0.55, lift=0.30,
           lift_shape=0.7, bob=-0.03, sway=0.012, yaw=9, roll=3, lean=9, arm=30, arm_fwd=2, elbow=(95, 72),
           strike=8, toeoff=34, s1=0.06, s2=0.45, reach=0.94, drop=0.045, toe_out=4, head=-4, cross=0.25)
WALK_BACK = dict(WALK, motion="WalkBackward", name="WalkBackward", reverse=True, period=1.15, stride=1.15, lean=1, arm=12, lift=0.10,
                 yaw=4)
STRAFE_L = dict(motion="StrafeLeft", name="StrafeLeft", kind="gait", move=(1, 0, 0), period=0.80, stride=0.9, duty=0.58, track=1.2,
                lift=0.09, lift_shape=1.0, bob=0.012, sway=0.0, yaw=0, roll=3, lean=2, arm=4, arm_fwd=4,
                elbow=(12, 18), strike=0, toeoff=12, s1=0.05, s2=0.2, reach=0.97, drop=0.025, toe_out=8, head=0,
                min_gap=0.06, max_gap=0.62)
STRAFE_R = dict(STRAFE_L, motion="StrafeRight", name="StrafeRight", move=(-1, 0, 0))
IDLE = dict(motion="Idle", name="Idle", kind="idle", period=4.0, drop=0.015, sway=0.012, breathe=1.6, look=5, elbow=12)
LIBRARY = [IDLE, WALK, RUN, WALK_BACK, STRAFE_L, STRAFE_R]


# --- pose evaluation ------------------------------------------------------------------------------------------------

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
PHALANX = ("Proximal", "Intermediate", "Distal")
RELAXED = {"Thumb": (5.0, 10.0, 10.0), "Index": (12.0, 18.0, 10.0), "Middle": (15.0, 22.0, 12.0),
           "Ring": (18.0, 25.0, 14.0), "Little": (20.0, 28.0, 16.0)}


def axis_angle(u, a):
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def set_fingers(pose, curls=None):
    """Finger rotations under each hand: curls {side: {finger: (proximal, intermediate, distal) degrees}}; missing
    fingers get a relaxed, slightly curled hand. Call after the hands are set (rotations chain from the hand)."""
    sk = pose.sk
    if not sk.curl:
        return
    for side in ("Left", "Right"):
        for f in FINGERS:
            angles = ((curls or {}).get(side) or {}).get(f, RELAXED[f])
            R = pose.get(side + "Hand")
            for ph, a in zip(PHALANX, angles):
                name = side + f + ph
                R = R @ axis_angle(sk.curl[name], np.radians(a))
                pose.set(name, R)


class Pose:
    """Global rotation per joint plus the Hips position."""

    def __init__(self, sk):
        self.sk = sk
        self.R = [np.eye(3) for _ in NAMES]
        self.hips = sk.g("Hips").copy()

    def set(self, name, R):
        self.R[INDEX[name]] = R

    def get(self, name):
        return self.R[INDEX[name]]

    def pos(self, name):
        """FK position of a joint."""
        chain = []
        j = INDEX[name]
        while j is not None:
            chain.append(j)
            j = PARENT[j]
        p = self.hips.copy()
        for a, b in zip(chain[::-1][:-1], chain[::-1][1:]):
            p = p + self.R[a] @ (self.sk.J[b] - self.sk.J[a])
        return p


def trunk(pose, yaw, pitch, roll, lean, breathe=0.0, head_yaw=0.0, head_pitch=0.0):
    """Pelvis rotation plus a spine that counter-rotates the twist, distributes the lean, keeps the head level."""
    d = np.radians
    pose.set("Hips", ry(yaw) @ rx(pitch + d(lean) * 0.35) @ rz(roll))
    pose.set("Spine", ry(0.5 * yaw) @ rx(d(lean) * 0.6) @ rz(0.5 * roll))
    pose.set("Chest", ry(-0.1 * yaw) @ rx(d(lean) * 0.85 + breathe * 0.5) @ rz(0.1 * roll))
    pose.set("UpperChest", ry(-0.6 * yaw) @ rx(d(lean) + breathe) @ rz(-0.2 * roll))
    pose.set("Neck", ry(-0.3 * yaw + 0.5 * head_yaw) @ rx(d(lean) * 0.4 + 0.5 * head_pitch))
    pose.set("Head", ry(head_yaw) @ rx(head_pitch))
    for s in ("Left", "Right"):
        pose.set(s + "Shoulder", pose.get("UpperChest"))


def arm(pose, side, swing, elbow, abduct, cross=0.0, wrist=0.0):
    """Hanging arm in the chest's frame: `swing` forward (+) / back (-), elbow bend, outward abduction (radians)."""
    sk = pose.sk
    sgn = 1 if side == "Left" else -1
    C = pose.get("UpperChest")
    u = unit(np.array([sgn * np.sin(abduct) * np.cos(swing) - sgn * cross * max(0, np.sin(swing)),
                       -np.cos(abduct) * np.cos(swing), np.sin(swing)]))
    fwd = np.array([0, 0, 1.0])
    n = unit(fwd - (fwd @ u) * u)
    f = np.cos(elbow) * u + np.sin(elbow) * n
    nf = unit(fwd - (fwd @ f) * f) if abs(f @ fwd) < 0.99 else unit(np.cross(f, [sgn, 0, 0]))
    S, E, W = sk.g(side + "UpperArm"), sk.g(side + "LowerArm"), sk.g(side + "Hand")
    Z = np.array([0, 0, 1.0])
    pose.set(side + "UpperArm", aim(E - S, Z, C @ u, C @ n))
    pose.set(side + "LowerArm", aim(W - E, Z, C @ f, C @ nf))
    pose.set(side + "Hand", pose.get(side + "LowerArm") @ rx(-wrist))


def leg_ik(pose, side, ankle, foot_R, toes_R, knee_dir):
    sk = pose.sk
    Hp = pose.pos(side + "UpperLeg")
    a, b = sk.thigh[side], sk.shin[side]
    d = ankle - Hp
    L = np.clip(np.linalg.norm(d), abs(a - b) + 1e-4, (a + b) * 0.9995)
    u = unit(d)
    cos_h = np.clip((a * a + L * L - b * b) / (2 * a * L), -1, 1)
    n = unit(knee_dir - (knee_dir @ u) * u)
    K = Hp + a * (cos_h * u + np.sqrt(1 - cos_h ** 2) * n)
    A = Hp + L * u
    rh, rk, ra = sk.g(side + "UpperLeg"), sk.g(side + "LowerLeg"), sk.g(side + "Foot")
    # twist from the knee's hinge axis (normal of the hip-knee-ankle plane, -X at rest): unlike the knee's forward
    # direction it stays perpendicular to the shin at any bend, so a deep knee bend can't flip the shin and foot
    hinge = unit(np.cross(u, n))
    X = np.array([-1.0, 0, 0])
    pose.set(side + "UpperLeg", aim(rk - rh, X, K - Hp, hinge))
    pose.set(side + "LowerLeg", aim(ra - rk, X, A - K, hinge))
    pose.set(side + "Foot", foot_R)
    pose.set(side + "Toes", toes_R)
    return float(np.linalg.norm(d) - L)   # > 0: the foot target was out of reach (would slide)


def foot_state(sk, side, ground, pitch, lift, yaw):
    """Ankle position and foot rotation for a foot whose flat-pose ankle sits over `ground` (x, z), pitched about the
    heel (pitch > 0, toes up) or the ball (pitch < 0, heel up), lifted by `lift`."""
    f = sk.foot[side]
    R_yaw = ry(yaw)
    R = R_yaw @ rx(-pitch)
    flat_ankle = np.array([ground[0], f["ankle"][1] + lift, ground[1]])
    piv_rest = f["heel"] if pitch > 0 else f["ball"]
    piv = flat_ankle + R_yaw @ (piv_rest - f["ankle"])
    return piv + R @ (f["ankle"] - piv_rest), R


def gait_pose(sk, c, phase, base_drop, speed_dir, A):
    """Pose at `phase` in [0, 1) of a gait clip. A: half the distance a planted foot travels."""
    d = np.radians
    pose = Pose(sk)
    beta = c["duty"]
    m = speed_dir
    side_axis = np.array([1.0, 0, 0])
    phL = phase % 1.0
    # pelvis
    mid = 2 * np.pi * (phL - beta / 2)
    bob = c["bob"] * sk.leg * np.cos(2 * mid)
    sway = c["sway"] * sk.leg * np.cos(mid)
    fwd_motion = abs(m[2]) > 0.5
    yaw = -d(c["yaw"]) * np.cos(2 * np.pi * phL) * (np.sign(m[2]) if fwd_motion else 0)
    roll = d(c["roll"]) * np.cos(mid)
    pose.hips = sk.g("Hips") + np.array([0, -base_drop + bob, 0]) + sway * side_axis
    trunk(pose, yaw, 0.0, roll, c["lean"], head_pitch=d(c.get("head", 0)))
    # legs
    reach_err = 0.0
    for side, off, sgn in (("Left", 0.0, 1), ("Right", 0.5, -1)):
        ph = (phL + off) % 1.0
        f = sk.foot[side]
        cx = sk.g("Hips")[0]
        half = c.get("half_width")
        neutral = np.array([cx + (sgn * half if half else (f["ankle"][0] - cx) * c["track"]), f["ankle"][2]])
        mv = np.array([m[0], m[2]])
        u, s = foot_travel(c, A, ph)
        if s is None:                                   # stance: planted point travels back at the clip speed
            lift = 0.0
            if ph < c["s1"]:
                pitch = d(c["strike"]) * (1 - smooth(ph / c["s1"]))
            elif ph > beta - c["s2"]:
                pitch = -d(c["toeoff"]) * smooth((ph - (beta - c["s2"])) / c["s2"])
            else:
                pitch = 0.0
            toe_flat = 1.0
        else:                                           # swing: Hermite path matching the ground speed at both ends
            lift = c["lift"] * sk.leg * np.sin(np.pi * s ** c["lift_shape"]) ** 1.0
            pitch = -d(c["toeoff"]) + (d(c["strike"]) + d(c["toeoff"])) * smooth((s - 0.1) / 0.85)
            toe_flat = 1 - smooth(s / 0.25)
        ground = neutral + mv * u
        yaw_f = sgn * d(c["toe_out"]) - sk.foot_yaw[side]
        ankle, R = foot_state(sk, side, ground, pitch, lift, yaw_f)
        toes = ry(yaw_f) if pitch < 0 and toe_flat > 0.999 else (
            ry(yaw_f) @ rx(-pitch * (1 - toe_flat)) if pitch < 0 else R)
        knee = unit(ry(0.5 * yaw + yaw_f * 0.6) @ np.array([0, 0, 1.0]) + np.array([sgn * 0.08, 0, 0]))
        reach_err = max(reach_err, leg_ik(pose, side, ankle, R, toes, knee))
    # arms: opposite the same-side leg, a little behind it in phase
    for side, off in (("Left", 0.0), ("Right", 0.5)):
        ph = (phL + off - 0.04) % 1.0
        sw = -np.cos(2 * np.pi * ph)                     # +1 = forward (when the opposite leg is forward)
        swing = d(c["arm"]) * sw + d(c["arm_fwd"])
        e0, e1 = c["elbow"]
        elbow = d(e0 + (e1 - e0) * (sw + 1) / 2)
        arm(pose, side, swing, elbow, sk.abduction[side], cross=c.get("cross", 0.0), wrist=d(8))
    set_fingers(pose)
    return pose, reach_err


def idle_pose(sk, c, t):
    d = np.radians
    pose = Pose(sk)
    w = 2 * np.pi * t / c["period"]
    sway = c["sway"] * sk.leg * np.sin(w)
    breathe = d(c["breathe"]) * (0.5 - 0.5 * np.cos(2 * w))
    pose.hips = sk.g("Hips") + np.array([sway, -c["drop"] * sk.leg - 0.004 * sk.leg * np.sin(2 * w), 0])
    trunk(pose, d(1.5) * np.sin(w), 0.0, d(1.2) * np.sin(w), 1.0, breathe=-breathe,
          head_yaw=d(c["look"]) * np.sin(w + 0.6), head_pitch=d(1.5) * np.sin(2 * w))
    for side, sgn in (("Left", 1), ("Right", -1)):
        f = sk.foot[side]
        yf = sgn * d(5) - sk.foot_yaw[side]
        ankle, R = foot_state(sk, side, np.array([f["ankle"][0], f["ankle"][2]]), 0.0, 0.0, yf)
        knee = unit(np.array([sgn * 0.12, 0, 1.0]))
        leg_ik(pose, side, ankle, R, ry(yf), knee)
        arm(pose, side, d(2.5) * np.sin(w + (0 if sgn > 0 else np.pi)) + d(3), d(c["elbow"]) + d(3) * np.sin(2 * w),
            sk.abduction[side] + d(1.5) * np.sin(2 * w), wrist=d(10))
    set_fingers(pose)
    return pose


def foot_travel(c, A, ph):
    """Position of a foot along the travel axis (planted at +A on heel strike, -A at toe-off) and its lift phase:
    (u, None) in stance, (u, s) in swing with s in [0, 1)."""
    beta = c["duty"]
    if ph < beta:
        return A * (1 - 2 * ph / beta), None
    s = (ph - beta) / (1 - beta)
    v = -2 * A / beta * (1 - beta)
    return hermite(-A, A, v, v, s), s


def gait_setup(sk, c):
    """Planted-foot travel A and the pelvis drop that keeps both legs within reach over the whole cycle. Side steps
    move the feet along the line joining them, so their stride is capped to keep the feet at least `min_gap` (of the
    height) and at most `max_gap` (of the leg length) apart."""
    beta = c["duty"]
    m = np.array(c["move"], np.float64)
    D = c["stride"] * sk.leg
    if c.get("min_gap"):
        # the feet's gap is width + lead * (u_L - u_R), linear in A: choose the stance width and stride so it swings
        # between min_gap (feet closing) and max_gap (feet apart)
        lead = 1 if m[0] > 0 else -1          # moving left (+X): the left foot leads
        phs = np.linspace(0, 1, 400, endpoint=False)
        dl = np.array([lead * (foot_travel(c, 1.0, p)[0] - foot_travel(c, 1.0, (p + 0.5) % 1)[0]) for p in phs])
        gmin, gmax = c["min_gap"] * sk.H, c["max_gap"] * sk.leg
        A = min((gmax - gmin) / (dl.max() - dl.min()), D * beta / 2)
        c["half_width"] = 0.5 * (gmin - A * dl.min())
        D = 2 * A / beta
    A = D * beta / 2
    drop = c["drop"] * sk.leg
    for _ in range(3):
        need = drop
        for ph in np.linspace(0, 1, 60, endpoint=False):
            pose, err = gait_pose(sk, c, ph, drop, m, A)
            if err > 1e-5:
                need = max(need, drop + err * 1.05)
        if need <= drop + 1e-6:
            break
        drop = need
    return A, drop, D


def bake(sk, c, rate=1.0):
    """Clip dict for rig.write_glb: local rotations (frames, joints, xyzw), Hips translation, extras. Clips with a
    captured motion (tools3d/motions/<name>.json, made by mocap.py) are retargeted from it; the rest, or all of them
    when the motion files are missing, are generated procedurally."""
    import os
    import retarget
    if c.get("motion") and os.path.exists(retarget.motion_path(c["motion"])):
        return retarget.bake_motion(sk, c["motion"], c["name"], rate)
    c = dict(c, period=c["period"] / rate)                                # faster: same stride, shorter cycle
    if c["kind"] == "idle":
        n = int(round(c["period"] * FPS))
        poses = [idle_pose(sk, c, i / FPS) for i in range(n + 1)]
        extras = {"speed": 0.0, "loop": True}
        fps = FPS
    else:
        c = dict(c)                    # gait_setup may add the side-step stance width
        A, drop, D = gait_setup(sk, c)
        n = int(round(c["period"] * FPS))
        m = np.array(c["move"], np.float64)
        poses = []
        for i in range(n + 1):
            ph = i / n
            if c.get("reverse"):
                ph = (1 - ph) % 1.0
            poses.append(gait_pose(sk, c, ph, drop, m, A)[0])
        fps = FPS                      # whole frames: the loop is n / FPS seconds and the speed matches it exactly
        speed = D / (n / FPS)
        direction = (-m if c.get("reverse") else m).tolist()
        # sync: the clip's normalised time at which the left foot's stance begins (phase 0 of the other clips), so
        # an engine can blend clips by setting time = (phase + sync) * duration. Reversed clips meet the ground
        # at their forward toe-off.
        extras = {"speed": round(float(speed), 4), "direction": direction, "loop": True,
                  "stride": round(float(D), 4), "duty": c["duty"],
                  "sync": round(1 - c["duty"], 4) if c.get("reverse") else 0.0}
    rot = np.zeros((len(poses), len(NAMES), 4))
    hips = np.zeros((len(poses), 3))
    for f, p in enumerate(poses):
        for j in range(len(NAMES)):
            Rl = p.R[j] if PARENT[j] is None else p.R[PARENT[j]].T @ p.R[j]
            rot[f, j] = quat(Rl)
        hips[f] = p.hips
    for f in range(1, len(poses)):    # keep quaternions on one hemisphere so LINEAR interpolation takes the short way
        flip = (rot[f] * rot[f - 1]).sum(1) < 0
        rot[f, flip] *= -1
    extras["rate"] = round(float(rate), 3)
    return {"name": c["name"], "fps": fps, "rot": rot, "hips": hips, "extras": extras, "poses": poses}
