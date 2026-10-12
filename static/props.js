// Holdable props for tinyimgen characters (three.js). Drop-in for a game:
//
//   import { attachProp, readyPose, solveSupportHand } from "./props.js";
//   const hero = (await loader.loadAsync("rigged.glb")).scene;      // a rigged human (has *Hand_Socket nodes)
//   const sword = (await loader.loadAsync("prop.glb")).scene;       // a prop (has Grip_Primary / Grip_Secondary)
//   const held = attachProp(hero, sword, { hand: "Right" });
//   // every frame, after mixer.update(dt):
//   readyPose(hero, held, "upright");                               // optional: carry it in front (no armed clips)
//   if (held.secondary) solveSupportHand(hero, held);               // two-handed: the other hand on Grip_Secondary
//
//   setFingers(hero, "Right", grip.fingers.primary);               // fingers around the grip (0 open .. 1 fist)
//   carryPose(hero, held, grip.carry);                              // a saved static hold, if the prop has one
//
// Convention (tinyimgen/props.py, tools3d/rig.py hand_sockets): a hand socket and a grip node share axes
// (+Z along the fingers, +Y out of the thumb side of the fist), so holding = making the grip coincide with the
// socket: prop_root.world = socket.world * inverse(grip.local).
import * as THREE from "three";

const _v = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
const _q = [new THREE.Quaternion(), new THREE.Quaternion(), new THREE.Quaternion()];

function socketOf(character, hand) {
  let s = character.getObjectByName(hand + "Hand_Socket");
  if (!s) {   // rigs from before sockets: the hand bone itself
    const bone = character.getObjectByName(hand + "Hand");
    if (!bone) throw new Error(`no ${hand}Hand bone`);
    s = new THREE.Object3D(); s.name = hand + "Hand_Socket"; bone.add(s);
  }
  return s;
}

// Parent `prop` (a prop.glb scene, or any object with a "Prop" root and Grip_* children) to the character's hand.
export function attachProp(character, prop, { hand = "Right" } = {}) {
  const root = prop.getObjectByName("Prop") || prop;
  const grip = root.getObjectByName("Grip_Primary");
  if (!grip) throw new Error("prop has no Grip_Primary node");
  const socket = socketOf(character, hand);
  const holder = new THREE.Group(); holder.name = "PropHolder";
  socket.add(holder);
  holder.add(root);
  root.position.set(0, 0, 0); root.quaternion.identity(); root.scale.set(1, 1, 1);
  updateHold({ holder, grip });
  const other = hand === "Right" ? "Left" : "Right";
  return { holder, root, grip, socket, hand, secondary: root.getObjectByName("Grip_Secondary") || null, other };
}

// Re-apply the grip offset (after editing the grip node's transform).
export function updateHold(held) {
  held.grip.updateMatrix();
  const inv = held.grip.matrix.clone().invert();
  inv.decompose(held.holder.position, held.holder.quaternion, held.holder.scale);
}

export function detachProp(held) { held.holder.removeFromParent(); }

function setWorldQuaternion(obj, q) {
  obj.parent.getWorldQuaternion(_q[0]);
  obj.quaternion.copy(_q[0].invert().multiply(q));
  obj.updateMatrixWorld(true);
}

function aim(bone, child, target) {
  bone.getWorldPosition(_v[0]); child.getWorldPosition(_v[1]);
  const cur = _v[1].sub(_v[0]).normalize();
  const want = _v[2].copy(target).sub(_v[0]).normalize();
  const delta = _q[1].setFromUnitVectors(cur, want);
  bone.getWorldQuaternion(_q[2]);
  setWorldQuaternion(bone, delta.multiply(_q[2]));
}

// ---------- arm IK ----------
// Bind pose has identity joint rotations, so a bone's local position is its bind direction in character space.
// The elbow is a hinge: in the bind (T-) pose it flexes the forearm toward the front (+Z).
const UP = new THREE.Vector3(0, 1, 0), FWD = new THREE.Vector3(0, 0, 1);
function armRig(character, side) {
  const key = "_arm" + side;
  if (character.userData[key]) return character.userData[key];
  const upper = character.getObjectByName(side + "UpperArm"), lower = character.getObjectByName(side + "LowerArm");
  const hand = character.getObjectByName(side + "Hand");
  if (!upper || !lower || !hand) return null;
  const socket = socketOf(character, side), mid = character.getObjectByName(side + "MiddleProximal");
  const u0 = lower.position.clone().normalize(), l0 = hand.position.clone().normalize();
  const h0 = mid ? mid.position.clone().normalize() : new THREE.Vector3(0, 0, 1).applyQuaternion(socket.quaternion);
  const hinge0 = new THREE.Vector3().crossVectors(u0, FWD).normalize();
  const B0 = basisQ(u0, hinge0);
  const rig = { upper, lower, hand, socket, u0, l0, h0, B0, bend0: l0.angleTo(h0), prev: null };
  character.userData[key] = rig;
  return rig;
}
function basisQ(a, b) {   // rotation whose X, Y axes are a, b (b orthogonalised against a)
  const x = a.clone().normalize(), y = b.clone().sub(x.clone().multiplyScalar(b.dot(x))).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, new THREE.Vector3().crossVectors(x, y)));
}
function twistAngle(q, axis) {   // q's rotation about `axis` (swing-twist), in -PI..PI
  const t = 2 * Math.atan2(q.x * axis.x + q.y * axis.y + q.z * axis.z, q.w);
  return Math.atan2(Math.sin(t), Math.cos(t));
}

// Two-bone IK: put `side`'s hand socket at `pos` with rotation `quat` (world). The arm picks the elbow swivel that
// keeps the wrist straightest, with the elbow down and out from the body, and the forearm takes part of the
// wrist's twist (FOREARM_TWIST), as the radius does. rollAxis (socket space, e.g. the handle through the fist, +Y)
// lets the hand also turn around that axis (a support hand on a round grip) to find the most natural wrist.
const FOREARM_TWIST = 0.6;
export function reachSocket(character, side, pos, quat, { pole = null, rollAxis = null, rollRange = 1.6 } = {}) {
  const rig = armRig(character, side);
  if (!rig) return false;
  const { upper, lower, hand, socket, u0, l0, h0, B0 } = rig;
  character.updateMatrixWorld(true);
  const Rc = character.getWorldQuaternion(new THREE.Quaternion());
  const scale = hand.getWorldScale(new THREE.Vector3()).x;
  const S = upper.getWorldPosition(new THREE.Vector3());
  const a = lower.position.length() * upper.getWorldScale(new THREE.Vector3()).x, b = hand.position.length() * scale;
  const out = new THREE.Vector3(side === "Left" ? 1 : -1, 0, 0).applyQuaternion(Rc);
  const up = UP.clone().applyQuaternion(Rc);
  const P = (pole ? pole.clone() : new THREE.Vector3(0, -1, -0.3)).applyQuaternion(Rc).addScaledVector(out, 0.75).normalize();
  const sockInv = socket.quaternion.clone().invert(), sockOff = socket.position.clone().multiplyScalar(scale);
  const rollQ = new THREE.Quaternion();

  // the arm for one (hand roll, elbow swivel); returns its cost and pose
  const tmp = { hq: new THREE.Quaternion(), W: new THREE.Vector3(), u: new THREE.Vector3(), e1: new THREE.Vector3(), e2: new THREE.Vector3() };
  function frameFor(roll) {
    const hq = rollAxis ? quat.clone().multiply(rollQ.setFromAxisAngle(rollAxis, roll)) : quat.clone();
    const handQ = hq.multiply(sockInv);
    const W = pos.clone().sub(sockOff.clone().applyQuaternion(handQ));
    const d = W.clone().sub(S);
    const dist = THREE.MathUtils.clamp(d.length(), Math.abs(a - b) + 1e-4, (a + b) * 0.999);
    const u = d.normalize();
    const cosA = THREE.MathUtils.clamp((a * a + dist * dist - b * b) / (2 * a * dist), -1, 1);
    const e1 = P.clone().sub(u.clone().multiplyScalar(P.dot(u)));
    if (e1.lengthSq() < 1e-8) e1.copy(out).sub(u.clone().multiplyScalar(out.dot(u)));
    e1.normalize();
    const e2 = new THREE.Vector3().crossVectors(u, e1);
    W.copy(S).addScaledVector(u, dist);
    return { handQ, W, u, e1, e2, C: S.clone().addScaledVector(u, a * cosA), r: a * Math.sqrt(1 - cosA * cosA), roll };
  }
  function pose(f, phi) {
    const n = f.e1.clone().multiplyScalar(Math.cos(phi)).addScaledVector(f.e2, Math.sin(phi));
    const E = f.C.clone().addScaledVector(n, f.r);
    const dU = E.clone().sub(S).normalize(), dF = f.W.clone().sub(E).normalize();
    const h = new THREE.Vector3().crossVectors(n, f.u).normalize();               // the elbow's hinge axis
    const upperQ = basisQ(dU, h).multiply(B0.clone().invert());                   // bind (u0, hinge0) -> (dU, h)
    const lowerQ = new THREE.Quaternion().setFromUnitVectors(l0.clone().applyQuaternion(upperQ), dF).multiply(upperQ);
    const tw = twistAngle(lowerQ.clone().invert().multiply(f.handQ), l0);
    lowerQ.multiply(new THREE.Quaternion().setFromAxisAngle(l0, FOREARM_TWIST * tw));
    const bend = Math.max(0, dF.angleTo(h0.clone().applyQuaternion(f.handQ)) - rig.bend0);
    const rel = E.clone().sub(S);
    const elbowOut = rel.dot(out), elbowUp = rel.dot(up);
    let cost = 1.0 * bend * bend + 0.5 * Math.max(0, Math.abs(tw) - 0.6) ** 2 + 0.6 * (1 - n.dot(P))
      + ((Math.max(0, 0.05 * (a + b) - elbowOut)) / (0.2 * (a + b))) ** 2 * 4
      + ((Math.max(0, elbowUp + 0.1 * (a + b))) / (0.2 * (a + b))) ** 2 * 2
      + 0.15 * f.roll * f.roll;
    if (rig.prev) cost += 0.05 * (1 - Math.cos(phi - rig.prev.phi)) + 0.05 * (1 - Math.cos(f.roll - rig.prev.roll));
    return { cost, phi, upperQ, lowerQ, f };
  }
  // coarse grid, then a shrinking local search (continuous, so animated poses don't step)
  const rolls = rollAxis ? [-1, -0.66, -0.33, 0, 0.33, 0.66, 1].map(t => t * rollRange) : [0];
  const frames = new Map();
  const getF = roll => { const k = roll.toFixed(4); if (!frames.has(k)) frames.set(k, frameFor(roll)); return frames.get(k); };
  let best = null;
  for (const r of rolls) for (let i = 0; i < 24; i++) {
    const c = pose(getF(r), i / 24 * Math.PI * 2);
    if (!best || c.cost < best.cost) best = c;
  }
  if (rig.prev) { const c = pose(getF(THREE.MathUtils.clamp(rig.prev.roll, -rollRange, rollRange) * (rollAxis ? 1 : 0)), rig.prev.phi); if (c.cost < best.cost) best = c; }
  let sp = Math.PI / 12, sr = rollAxis ? rollRange / 3 : 0;
  for (let it = 0; it < 8; it++) {
    for (const [dp, dr] of [[sp, 0], [-sp, 0], [0, sr], [0, -sr]]) {
      if (!dp && !dr) continue;
      const r = THREE.MathUtils.clamp(best.f.roll + dr, -rollRange, rollRange);
      const c = pose(dr ? frameFor(r) : best.f, best.phi + dp);
      if (c.cost < best.cost) best = c;
    }
    sp /= 2; sr /= 2;
  }
  rig.prev = { phi: best.phi, roll: best.f.roll };
  setWorldQuaternion(upper, best.upperQ);
  setWorldQuaternion(lower, best.lowerQ);
  setWorldQuaternion(hand, best.f.handQ);
  return true;
}

// How far each arm involved in holding the prop must reach if the holding hand's socket goes to (pos, quat); the
// prop is pulled toward the shoulders until every arm is within `cap` of its full length (so elbows stay bent).
function comfortable(character, held, pos, quat, cap) {
  character.updateMatrixWorld(true);
  held.grip.updateMatrix();
  const arms = [];
  for (const [side, node] of [[held.hand, null], [held.other, held.secondary]]) {
    if (side === held.other && !node) continue;
    const rig = armRig(character, side);
    if (!rig) continue;
    const s = rig.hand.getWorldScale(new THREE.Vector3()).x;
    const L = (rig.lower.position.length() + rig.hand.position.length()) * s;
    arms.push({ rig, node, L, S: rig.upper.getWorldPosition(new THREE.Vector3()), s });
  }
  const p = pos.clone();
  for (let it = 0; it < 6; it++) {
    const shift = new THREE.Vector3();
    const T = new THREE.Matrix4().compose(p, quat, new THREE.Vector3(1, 1, 1));
    for (const A of arms) {
      let sp = p, sq = quat;
      if (A.node) {   // the support grip, where it lands with the prop held this way
        A.node.updateMatrix();
        const m = T.clone().multiply(held.grip.matrix.clone().invert()).multiply(A.node.matrix);
        sp = new THREE.Vector3(); sq = new THREE.Quaternion(); m.decompose(sp, sq, new THREE.Vector3());
      }
      const handQ = sq.clone().multiply(A.rig.socket.quaternion.clone().invert());
      const W = sp.clone().sub(A.rig.socket.position.clone().multiplyScalar(A.s).applyQuaternion(handQ));
      const d = W.distanceTo(A.S) - cap * A.L;
      if (d > 0) shift.add(A.S.clone().sub(W).normalize().multiplyScalar(d));
    }
    if (shift.lengthSq() < 1e-8) break;
    p.add(shift.multiplyScalar(arms.length > 1 ? 0.7 : 1));
  }
  return p;
}

// Two-handed props: the other hand's socket on Grip_Secondary, palm around the grip. The support hand may turn
// around the handle (socket +Y, the axis through the fist) to keep its wrist natural. When the second grip is
// still out of that arm's reach (a pose dragged too far, a long prop), the prop is pulled toward that shoulder by
// the shortfall (the holding hand follows) so both hands stay on it.
export function solveSupportHand(character, held, opts = {}) {
  if (!held.secondary) return false;
  const side = held.other;
  const upper = character.getObjectByName(side + "UpperArm"), lower = character.getObjectByName(side + "LowerArm");
  const hand = character.getObjectByName(side + "Hand"), sock = socketOf(character, side);
  if (upper && lower && hand) {
    character.updateMatrixWorld(true);
    const gQ = held.secondary.getWorldQuaternion(new THREE.Quaternion());
    const gP = held.secondary.getWorldPosition(new THREE.Vector3());
    const wrist = gP.clone().sub(sock.position.clone().applyQuaternion(gQ.clone().multiply(sock.quaternion.clone().invert())));
    const S = upper.getWorldPosition(new THREE.Vector3());
    const reach = 0.97 * (lower.position.length() + hand.position.length()) * upper.getWorldScale(_v[3]).x;
    const short = S.distanceTo(wrist) - reach;
    if (short > 0) {   // pull the prop (through the holding hand) toward the support shoulder
      const shift = S.clone().sub(wrist).normalize().multiplyScalar(short);
      const p0 = held.socket.getWorldPosition(new THREE.Vector3()).add(shift);
      reachSocket(character, held.hand, p0, held.socket.getWorldQuaternion(new THREE.Quaternion()));
    }
  }
  held.root.updateMatrixWorld(true);
  return reachSocket(character, side, held.secondary.getWorldPosition(new THREE.Vector3()),
                     held.secondary.getWorldQuaternion(new THREE.Quaternion()),
                     { rollAxis: opts.roll === false ? null : new THREE.Vector3(0, 1, 0), ...opts });
}

// A "ready" carry for previews and games without armed animations: the holding hand brings the prop in front of
// the body (upright props: point up and forward at belly height; forward props: aimed ahead at chest height), placed
// from the shoulders and arm length so the elbows stay bent. Call after mixer.update(dt), before solveSupportHand.
export function readyPose(character, held, hold = "upright") {
  const rig = armRig(character, held.hand), other = character.getObjectByName(held.other + "UpperArm");
  if (!rig || !other) return false;
  character.updateMatrixWorld(true);
  const rootQ = character.getWorldQuaternion(new THREE.Quaternion());
  const inv = character.matrixWorld.clone().invert();
  const sh = rig.upper.getWorldPosition(new THREE.Vector3()).applyMatrix4(inv);
  const so = other.getWorldPosition(new THREE.Vector3()).applyMatrix4(inv);
  const s = rig.hand.getWorldScale(new THREE.Vector3()).x / character.getWorldScale(new THREE.Vector3()).x;
  const L = (rig.lower.position.length() + rig.hand.position.length()) * s;
  const sgn = held.hand === "Right" ? -1 : 1;                 // +X is the character's left
  let local, X, Y, Z;
  if (hold === "forward") {   // aimed ahead: fingers (and the prop) forward, thumb up
    local = sh.clone().add(new THREE.Vector3(-sgn * 0.3 * L, -0.25 * L, 0.8 * L));
    Z = new THREE.Vector3(-sgn * 0.12, 0, 1).normalize(); Y = UP.clone();
  } else {                    // blade up and forward, the fist turned in across the body
    local = sh.clone().add(new THREE.Vector3(-sgn * 0.2 * L, -0.62 * L, 0.55 * L));
    Y = new THREE.Vector3(0, 0.8, 0.6).normalize(); Z = new THREE.Vector3(-sgn * 0.55, -0.5, 0.67).normalize();
  }
  X = new THREE.Vector3().crossVectors(Y, Z).normalize(); Z = new THREE.Vector3().crossVectors(X, Y).normalize();
  const qLocal = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(X, Y, Z));
  if (held.secondary) {   // two hands: centre the pair of grips in front of the body, so both arms reach easily
    held.grip.updateMatrix(); held.secondary.updateMatrix();
    const rel = held.secondary.position.clone().sub(held.grip.position)
      .applyQuaternion(held.grip.quaternion.clone().invert()).applyQuaternion(qLocal);   // grip 1 -> 2, body space
    const mid = sh.clone().add(so).multiplyScalar(0.5);
    local = mid.add(new THREE.Vector3(0, -0.62 * L, 0.55 * L)).addScaledVector(rel, -0.5);
  }
  const q = qLocal.clone().premultiply(rootQ);
  const p = comfortable(character, held, local.applyMatrix4(character.matrixWorld), q, 0.85);
  return reachSocket(character, held.hand, p, q);
}

// ---------- fingers ----------
export const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"];
const PHALANX = ["Proximal", "Intermediate", "Distal"];
const _axis = new THREE.Vector3();

// Curl one hand's fingers: curls = {Thumb: c, Index: c, ...}, c = 0..1 for the whole finger (0 straight, 1 a full
// fist) or [knuckle, middle, end] per joint (what autoWrap returns). Needs the curl axes tools3d/rig.py writes on
// the finger joints. Call after mixer.update: it overrides the clip's fingers.
export function setFingers(character, side, curls) {
  if (!curls) return;
  for (const f of FINGERS) {
    const c = curls[f];
    if (c == null) continue;
    PHALANX.forEach((ph, k) => {
      const bone = character.getObjectByName(side + f + ph);
      const ax = bone?.userData?.curl_axis;
      if (!ax) return;
      const v = Array.isArray(c) ? c[k] : c;
      bone.quaternion.setFromAxisAngle(_axis.fromArray(ax), THREE.MathUtils.degToRad(v * bone.userData.curl_max_deg));
    });
  }
}
export const curlValue = c => Array.isArray(c) ? (c[0] + c[1] + c[2]) / 3 : c;   // one number for a slider

// Close each finger around the prop the way a hand does, joint by joint: the knuckle bends until the finger's
// middle or end segment meets the prop, then the middle joint until the end segment does, then the end joint until
// the fingertip does. A segment "meets" the prop when its bone, or a finger's thickness in the direction it's
// closing, hits the prop's surface. The thumb's base stays put (it rests beside the handle). A finger that never
// meets the prop takes its neighbours' curl. Returns {finger: [knuckle, middle, end]} (0..1 of each joint's range).
export function autoWrap(character, side, propRoot, { margin = 0.03 } = {}) {
  const meshes = []; propRoot.traverse(o => { if (o.isMesh) meshes.push(o); });
  const ray = new THREE.Raycaster(); const out = {};
  const bones = f => PHALANX.map(ph => character.getObjectByName(side + f + ph));
  const chain = P => {
    const list = P.map(b => b.getWorldPosition(new THREE.Vector3()));
    const tip = P[2].userData.tip ? new THREE.Vector3().fromArray(P[2].userData.tip) : null;
    list.push(tip ? P[2].localToWorld(tip) : list[2].clone().add(list[2].clone().sub(list[1]).multiplyScalar(0.8)));
    return list;
  };
  // does any segment from `from` on (with the samples given) meet the prop?
  const meets = (P, from, firstSamples) => {
    character.updateMatrixWorld(true);
    const pts = chain(P);
    const thick = 0.3 * pts[0].distanceTo(pts[1]);
    for (let k = from; k < 3; k++) {
      const seg = pts[k + 1].clone().sub(pts[k]); const len = seg.length(); seg.normalize();
      const samples = k === from ? firstSamples : [0.3, 0.65, 1.0];
      const t0 = samples[0];
      ray.set(pts[k].clone().addScaledVector(seg, len * t0), seg); ray.far = len * (1 - t0);
      if (ray.intersectObjects(meshes, false).length) return true;
      const ax = new THREE.Vector3().fromArray(P[k].userData.curl_axis || [1, 0, 0]).applyQuaternion(P[k].getWorldQuaternion(new THREE.Quaternion())).normalize();
      const move = new THREE.Vector3().crossVectors(ax, seg).normalize();
      for (const t of samples) {
        ray.set(pts[k].clone().addScaledVector(seg, len * t), move); ray.far = thick;
        if (ray.intersectObjects(meshes, false).length) return true;
      }
    }
    return false;
  };
  const missed = [];
  for (const f of FINGERS) {
    const P = bones(f); if (P.some(b => !b)) continue;
    const c = f === "Thumb" ? [0.15, 0, 0] : [0, 0, 0];
    let touched = false;
    for (let k = f === "Thumb" ? 1 : 0; k < 3; k++) {
      // joint k: the segments it moves, judged from their far part for its own segment (whose base may already
      // rest on the handle) and the end joint by its fingertip
      const first = k === 2 ? [0.85, 1.0] : [0.6, 0.85, 1.0];
      const at = v => { c[k] = v; setFingers(character, side, { [f]: c }); return meets(P, k, first); };
      if (!at(1)) { c[k] = 1; continue; }
      let lo = 0, hi = 1;
      if (at(0)) { c[k] = 0; touched = true; continue; }
      for (let i = 0; i < 9; i++) { const m = (lo + hi) / 2; if (at(m)) hi = m; else lo = m; }
      c[k] = Math.max(0, lo - margin); touched = true;
    }
    if (!touched) missed.push(f);
    out[f] = c.map(v => +v.toFixed(3));
  }
  const got = FINGERS.filter(f => f !== "Thumb" && out[f] && !missed.includes(f));
  if (got.length) for (const f of missed) if (f !== "Thumb") out[f] = [0, 1, 2].map(k => +(got.reduce((a, g) => a + out[g][k], 0) / got.length).toFixed(3));
  setFingers(character, side, out);
  return out;
}

// ---------- static carry ----------
// carry = {position: [x, y, z] in units of the character's height, rotation: [x, y, z, w], of: "grip"}: where the
// holding hand's grip (Grip_Primary, i.e. the hand socket) sits relative to the character's UpperChest bone, so the
// hand rides with the torso (bob, twist, lean). Carry and grip are independent: the carry places the hand, the grip
// places the prop in the hand. (Older grips without `of` stored the Prop root's place instead; still read.)
export const CARRY_BONE = "UpperChest";
export function carryPose(character, held, carry, height) {
  if (!carry) return false;
  const h = height || character.userData.height || 1.8;
  const target = new THREE.Matrix4().compose(new THREE.Vector3().fromArray(carry.position).multiplyScalar(h),
    new THREE.Quaternion().fromArray(carry.rotation), new THREE.Vector3(1, 1, 1));
  character.updateMatrixWorld(true);
  const chest = character.getObjectByName(CARRY_BONE) || character;
  target.premultiply(chest.matrixWorld);
  held.grip.updateMatrix();
  const sock = carry.of === "grip" ? target : target.clone().multiply(held.grip.matrix);
  const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
  sock.decompose(pos, quat, scl);
  return reachSocket(character, held.hand, comfortable(character, held, pos, quat, 0.92), quat);
}
