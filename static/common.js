// Shared by the Image Studio and Showcase pages: header status, theme toggle, lightbox.
const $ = s => document.querySelector(s);
const NAMES = { klein: "klein 4B", zimage: "Z-Image", nb_pro: "Nano Banana Pro", nb2: "Nano Banana 2", flux2pro: "FLUX.2 pro", seedream45: "Seedream 4.5" };
const FAL = new Set(["nb_pro", "nb2", "flux2pro", "seedream45"]);   // cloud models (paid per image on fal.ai)

// theme toggle (system → light → dark)
const THEMES = ["system", "light", "dark"];
let themeIdx = 0;
try { themeIdx = Math.max(0, THEMES.indexOf(localStorage.getItem("theme") || "system")); } catch (e) {}
function applyTheme() {
  const m = THEMES[themeIdx];
  if (m === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.dataset.theme = m;
  $("#theme").textContent = "Theme: " + m;
}
$("#theme").onclick = () => {
  themeIdx = (themeIdx + 1) % 3;
  try { localStorage.setItem("theme", THEMES[themeIdx]); } catch (e) {}
  applyTheme();
};
applyTheme();

function lightbox(src, cap) { $("#lbimg").src = src; $("#lbcap").textContent = cap; $("#lb").classList.add("on"); }
$("#lb").onclick = () => $("#lb").classList.remove("on");
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#lb").classList.remove("on"); });

// Polls /api/info until the models are loaded; onInfo gets every response.
async function pollInfo(onInfo) {
  try {
    const i = await (await fetch("/api/info")).json();
    $("#gpu").textContent = `${i.gpu.replace("NVIDIA GeForce ", "")} · ${i.vram_total_gb} GB`;
    if (onInfo) onInfo(i);
    if (i.error) { $("#ready").textContent = "Load failed: " + i.error; return; }
    if (i.ready) { $("#dot").classList.add("ok"); $("#ready").textContent = `Ready · loaded in ${i.load_seconds}s`; return; }
    $("#ready").textContent = `Loading models… ${i.loading_for}s`;
  } catch (e) { $("#ready").textContent = "Server unreachable"; }
  setTimeout(() => pollInfo(onInfo), 2000);
}

// ---------- references: upload + library picker, shared by the Image Studio and 3D ----------
async function uploadFile(file) {
  const data = await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(file); });
  const r = await fetch("/api/upload", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: file.name, data }) });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  const j = await r.json();
  return { ref: j.ref, url: j.url, label: file.name, kind: "upload" };
}

function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

// Modal: pick up to `max` images from generations / uploads / 3D renders, or upload new ones.
function openPicker({ title = "Choose images", max = 1, onPick }) {
  let modal = $("#picker");
  if (!modal) {
    modal = el("div", "modal"); modal.id = "picker";
    modal.innerHTML = `<div class="sheet card"><div class="sheet-h"><h3></h3><div class="row-gap">
        <label class="btn2">Upload…<input type="file" accept="image/*" multiple hidden></label>
        <button class="btn2" data-a="cancel" type="button">Cancel</button><button class="go small" data-a="use" type="button">Use</button></div></div>
      <div class="sheet-tabs seg"><button data-k="generations" class="on" type="button">Generations</button><button data-k="uploads" type="button">Uploads</button><button data-k="renders" type="button">3D renders</button></div>
      <div class="pick-grid"></div><p class="sub pick-empty" hidden>Nothing here yet.</p></div>`;
    document.body.append(modal);
    modal.addEventListener("click", e => { if (e.target === modal) modal.classList.remove("on"); });
  }
  const st = { kind: "generations", lib: null, sel: [] };
  modal.querySelector("h3").textContent = title + (max > 1 ? ` (up to ${max})` : "");
  const useBtn = modal.querySelector('[data-a="use"]');
  const refresh = () => { useBtn.textContent = st.sel.length ? `Use ${st.sel.length}` : "Use"; useBtn.disabled = !st.sel.length; };
  function render() {
    const g = modal.querySelector(".pick-grid"); g.innerHTML = "";
    const items = (st.lib && st.lib[st.kind]) || [];
    modal.querySelector(".pick-empty").hidden = items.length > 0;
    for (const it of items) {
      const b = el("button", "pick"); b.type = "button"; b.title = it.label || "";
      const img = el("img"); img.loading = "lazy"; img.src = it.url; img.alt = it.label || ""; b.append(img);
      if (st.sel.some(s => s.ref === it.ref)) b.classList.add("on");
      b.onclick = () => {
        const i = st.sel.findIndex(s => s.ref === it.ref);
        if (i >= 0) st.sel.splice(i, 1);
        else { if (max === 1) st.sel = []; if (st.sel.length < max) st.sel.push(it); }
        render(); refresh();
      };
      g.append(b);
    }
  }
  const tabs = modal.querySelector(".sheet-tabs");
  tabs.onclick = e => { const b = e.target.closest("button"); if (!b) return; [...tabs.children].forEach(x => x.classList.toggle("on", x === b)); st.kind = b.dataset.k; render(); };
  [...tabs.children].forEach(x => x.classList.toggle("on", x.dataset.k === "generations"));
  modal.querySelector('[data-a="cancel"]').onclick = () => modal.classList.remove("on");
  useBtn.onclick = () => { modal.classList.remove("on"); onPick(st.sel.slice()); };
  modal.querySelector('input[type=file]').onchange = async e => {
    const files = [...e.target.files].slice(0, max); e.target.value = "";
    try { const ups = []; for (const f of files) ups.push(await uploadFile(f)); modal.classList.remove("on"); onPick(ups); }
    catch (err) { alert("Upload failed: " + err.message); }
  };
  modal.classList.add("on"); refresh();
  fetch("/api/library").then(r => r.json()).then(lib => { st.lib = lib; render(); });
}

// A strip of reference thumbnails with remove buttons, an "Add" picker and drag-and-drop upload.
function refTray(root, { max = 4, title = "Choose reference images", onChange } = {}) {
  const tray = { items: [] };
  const change = () => { draw(); if (onChange) onChange(tray.items); };
  function draw() {
    root.innerHTML = "";
    root.classList.add("tray");
    tray.items.forEach((it, i) => {
      const t = el("div", "ref"); const img = el("img"); img.src = it.url; img.alt = it.label || ""; img.title = it.label || "";
      const x = el("button", "x", "×"); x.type = "button"; x.title = "Remove"; x.onclick = () => { tray.items.splice(i, 1); change(); };
      t.append(img, x); root.append(t);
    });
    if (tray.items.length < max) {
      const add = el("button", "ref add", "+ Add"); add.type = "button"; add.title = "Generations, uploads or 3D renders — or drop files here";
      add.onclick = () => openPicker({ title, max: max - tray.items.length, onPick: picks => tray.add(...picks) });
      root.append(add);
    }
  }
  tray.add = (...picks) => { for (const p of picks) if (tray.items.length < max && !tray.items.some(x => x.ref === p.ref)) tray.items.push(p); change(); };
  tray.clear = () => { tray.items = []; change(); };
  root.addEventListener("dragover", e => { e.preventDefault(); root.classList.add("drop"); });
  root.addEventListener("dragleave", () => root.classList.remove("drop"));
  root.addEventListener("drop", async e => {
    e.preventDefault(); root.classList.remove("drop");
    for (const f of [...e.dataTransfer.files].filter(f => f.type.startsWith("image/"))) {
      if (tray.items.length >= max) break;
      try { tray.add(await uploadFile(f)); } catch (err) { alert("Upload failed: " + err.message); }
    }
  });
  draw();
  return tray;
}

function refFromUrlParam(name) {  // ?ref=uploads/x.png or ?src=<gallery file>
  const v = new URLSearchParams(location.search).get(name);
  if (!v) return null;
  return { ref: v, url: v.includes("/") ? "/media/" + v : "/media/gallery/" + v, label: v };
}
