// Locomotion runtime for tinyimgen's rigged characters (three.js). Drop-in for a game:
//
//   import { Locomotion } from "./locomotion.js";
//   const gltf = await new GLTFLoader().loadAsync("rigged.glb");
//   const loco = new Locomotion(gltf);
//   // every frame: input.x = left(+)/right(-), input.z = forward(+)/back(-), |input| <= 1
//   const v = loco.update(dt, input, running);   // local velocity (m/s): v.x left, v.z forward
//   character.position.addScaledVector(worldDirFromLocal(v), dt);
//
// It blends Idle / Walk / Run / WalkBackward / StrafeLeft / StrafeRight by the character's local velocity. The
// locomotion clips are driven by one shared gait phase that advances by distance travelled / stride, so the feet
// stay planted at any speed and diagonal blends (e.g. Walk + StrafeLeft) keep both feet in step. Each clip's speed,
// stride and sync offset come from the GLB's animation extras, written by tools3d/anim.py.
import * as THREE from "three";

const DIRECTIONAL = ["Walk", "WalkBackward", "StrafeLeft", "StrafeRight", "Run"];

export class Locomotion {
  constructor(gltf, { root = gltf.scene, accel = 7, idleBlend = 0.25 } = {}) {
    this.mixer = new THREE.AnimationMixer(root);
    this.accel = accel;
    this.idleBlend = idleBlend;      // fraction of the walk speed below which Idle fades in
    const extras = {};
    for (const a of gltf.parser?.json?.animations || []) extras[a.name] = a.extras || {};
    this.clips = {};
    for (const clip of gltf.animations) {
      const action = this.mixer.clipAction(clip);
      const ex = extras[clip.name] || {};
      action.play();
      action.setEffectiveWeight(0);
      if (DIRECTIONAL.includes(clip.name)) action.timeScale = 0;   // driven by the shared phase instead
      this.clips[clip.name] = { action, duration: clip.duration, speed: ex.speed || 0, stride: ex.stride || 1,
                                sync: ex.sync || 0, baked: ex.rate || 1, live: 1 };
    }
    if (!this.clips.Idle || !this.clips.Walk) throw new Error("not a tinyimgen rigged GLB (needs Idle and Walk clips)");
    this.phase = 0;
    this.velocity = new THREE.Vector2();   // x = left, y = forward (m/s)
    this.weights = {};
  }

  speedOf(name, fallback) { const c = this.clips[name]; return c?.speed ? c.speed * c.live : fallback; }

  // playback rate relative to the captured clip (1 = as captured). The GLB may already bake a rate in; this scales
  // from there. Faster = same stride, quicker steps, so the feet stay planted at any rate.
  rate(name) { const c = this.clips[name]; return c ? c.baked * c.live : 1; }
  setRate(name, r) {
    const c = this.clips[name];
    if (!c) return;
    c.live = r / c.baked;
    if (name === "Idle") c.action.timeScale = c.live;
  }

  // top speed in a local direction (unit dx = left, dz = forward): an ellipse through the clip speeds
  topSpeed(dx, dz, running) {
    const walk = this.speedOf("Walk", 1.2);
    const fwd = running ? this.speedOf("Run", walk * 2.4) : walk;
    const back = this.speedOf("WalkBackward", walk * 0.7);
    const side = this.speedOf(dx >= 0 ? "StrafeLeft" : "StrafeRight", walk * 0.45);
    const along = dz >= 0 ? fwd : back;
    return 1 / Math.hypot(dx / side, dz / along);
  }

  update(dt, input = { x: 0, z: 0 }, running = false) {
    // 1. velocity toward the input direction's top speed, smoothed
    const mag = Math.min(1, Math.hypot(input.x, input.z));
    const target = new THREE.Vector2();
    if (mag > 1e-3) {
      const dx = input.x / Math.hypot(input.x, input.z), dz = input.z / Math.hypot(input.x, input.z);
      target.set(dx, dz).multiplyScalar(this.topSpeed(dx, dz, running) * mag);
    }
    this.velocity.lerp(target, 1 - Math.exp(-this.accel * dt));
    const v = this.velocity, s = v.length();

    // 2. directional weights: forward/back/left/right by the velocity's direction cosines
    const w = { Idle: 0, Walk: 0, Run: 0, WalkBackward: 0, StrafeLeft: 0, StrafeRight: 0 };
    if (s > 1e-4) {
      const dx = v.x / s, dz = v.y / s;
      const f = Math.max(0, dz), b = Math.max(0, -dz), l = Math.max(0, dx), r = Math.max(0, -dx);
      const sum = f + b + l + r;
      const walk = this.speedOf("Walk", 1.2), run = this.speedOf("Run", walk * 2.4);
      const runT = this.clips.Run ? THREE.MathUtils.clamp((s * dz - walk) / (run - walk), 0, 1) : 0;
      w.Walk = (f / sum) * (1 - runT); w.Run = (f / sum) * runT;
      w.WalkBackward = b / sum; w.StrafeLeft = l / sum; w.StrafeRight = r / sum;
      // idle fades in at low speed
      const move = THREE.MathUtils.clamp(s / (this.idleBlend * this.topSpeed(dx, dz, false)), 0, 1);
      for (const k of DIRECTIONAL) w[k] *= move;
      w.Idle = 1 - move;
    } else w.Idle = 1;

    // 3. one gait phase for all directional clips: advance by distance / blended stride
    let stride = 0, tot = 0;
    for (const k of DIRECTIONAL) if (this.clips[k] && w[k] > 0) { stride += w[k] * this.clips[k].stride; tot += w[k]; }
    if (tot > 0) this.phase = (this.phase + s * dt / (stride / tot)) % 1;
    for (const [name, c] of Object.entries(this.clips)) {
      c.action.setEffectiveWeight(w[name] || 0);
      if (DIRECTIONAL.includes(name)) c.action.time = ((this.phase + c.sync) % 1) * c.duration;
    }
    this.weights = w;
    this.mixer.update(dt);
    return { x: v.x, z: v.y, speed: s };
  }
}
