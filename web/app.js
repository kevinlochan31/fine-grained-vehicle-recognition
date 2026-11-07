/* AutoLens // Optical Intelligence - frontend.
   Vanilla JS + Motion (motion.dev, successor of Framer Motion) + a canvas particle system. */
(() => {
  "use strict";
  const M = window.Motion || null;
  const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FINE = window.matchMedia("(pointer: fine)").matches;
  const MOBILE = window.matchMedia("(max-width: 640px)").matches;
  const live = !!M && !RM;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const EASE = [0.22, 1, 0.36, 1];
  const SPRING = { type: "spring", stiffness: 240, damping: 24 };

  /* =============================================================== motion helpers */
  function setFinal(el, kf) {
    const last = (v) => (Array.isArray(v) ? v[v.length - 1] : v);
    const t = [];
    for (const [k, v] of Object.entries(kf)) {
      const val = last(v);
      if (k === "opacity") el.style.opacity = val;
      else if (k === "x" || k === "y") t.push(`translate${k.toUpperCase()}(${typeof val === "number" ? val + "px" : val})`);
      else if (k.startsWith("scale") || k.startsWith("rotate")) t.push(`${k}(${val}${k.startsWith("rotate") ? "deg" : ""})`);
      else el.style[k] = typeof val === "number" && k !== "strokeDashoffset" ? val : val;
    }
    if (t.length) el.style.transform = t.join(" ");
  }
  function anim(target, kf, opts = {}) {
    const els = typeof target === "string" ? $$(target) : target instanceof Element ? [target] : [...target];
    if (!els.length) return Promise.resolve();
    if (!live) { els.forEach((e) => setFinal(e, kf)); return Promise.resolve(); }
    return M.animate(els, kf, opts);
  }
  const stagger = (s, o) => (live ? M.stagger(s, o) : 0);
  function countUp(el, to, fmt, duration = 1.1) {
    if (!live) { el.textContent = fmt(to); return; }
    M.animate(0, to, { duration, ease: EASE, onUpdate: (v) => (el.textContent = fmt(v)) });
  }
  const GLYPHS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789#%&/<>";
  function scramble(el, text, dur = 750) {
    if (!live) { el.textContent = text; return; }
    el._scr = text;          // fallback: animation frames pause in background tabs, so always land the text
    setTimeout(() => { if (el._scr === text) el.textContent = text; }, dur + 150);
    const t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur), n = Math.floor(p * text.length);
      let s = text.slice(0, n);
      for (let i = n; i < text.length; i++) s += text[i] === " " ? " " : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      el.textContent = s;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function onView(el, cb, amount = 0.2) {
    if (!el) return;
    if (!live) { cb(); return; }
    let done = false;
    M.inView(el, () => { if (!done) { done = true; cb(); } }, { amount });
  }

  /* =============================================================== boot sequence */
  const health = fetch("/api/health").then((r) => r.json()).catch(() => null);
  async function boot() {
    const bootEl = $("#boot");
    const finish = () => { bootEl.remove(); document.body.classList.remove("is-booting"); heroIntro(); };
    if (!live) { finish(); return; }
    // safety net: never leave the boot screen up (e.g. a background tab pauses animations)
    setTimeout(() => { if (document.body.contains(bootEl)) finish(); }, 6000);
    const log = $("#boot-log"), num = $("#boot-num"), bar = $("#boot-bar");
    const lines = ["init optical core", "load yolo11s · stanford-cars", "load convnext-t · 196 classes", "calibrate T=0.65 · t=0.71", "link /api"];
    M.animate(0, 100, { duration: 1.05, ease: [0.6, 0, 0.2, 1], onUpdate: (v) => { num.textContent = String(Math.round(v)).padStart(3, "0"); bar.style.transform = `scaleX(${v / 100})`; } });
    for (const [i, l] of lines.entries()) {
      await wait(170);
      const ok = i === lines.length - 1 ? ((await Promise.race([health, wait(800)])) ? "ok" : "pending") : "ok";
      log.innerHTML += `> ${l} ${".".repeat(Math.max(2, 34 - l.length))} <b>${ok}</b>\n`;
    }
    await wait(200);
    await M.animate($("#boot-curtain"), { y: ["100%", "0%"] }, { duration: 0.38, ease: [0.7, 0, 0.3, 1] });
    if (!document.body.contains(bootEl)) return;          // the safety net already finished the boot
    $(".boot__inner").style.visibility = "hidden";
    bootEl.classList.add("is-done");
    document.body.classList.remove("is-booting");
    heroIntro();
    await M.animate(bootEl, { y: ["0%", "-100%"] }, { duration: 0.6, ease: [0.7, 0, 0.3, 1] });
    bootEl.remove();
  }

  /* =============================================================== hero */
  function heroIntro() {
    // letters spring up out of their line masks, tumbling on X with a brief motion blur
    const chars = $$(".hero__title .ch");
    anim(chars, { y: ["115%", "0%"], rotateX: [-80, 0], filter: ["blur(10px)", "blur(0px)"] },
      { type: "spring", stiffness: 150, damping: 17, delay: stagger(0.035) }).then?.(() => chars.forEach((c) => (c.style.filter = "")));
    anim($$("[data-fade]"), { opacity: [0, 1], y: [24, 0] }, { duration: 0.8, delay: stagger(0.12, { startDelay: 0.45 }), ease: EASE });
    setTimeout(() => $$("[data-count]").forEach((el) => {
      const dec = +(el.dataset.dec ?? 1), suf = el.dataset.suffix || "";
      countUp(el, +el.dataset.count, (v) => `${v.toFixed(dec)}${suf}`, 1.4);
    }), 600);
    particles.assemble();
  }

  /* -------------------------------------------------------------- particle car */
  const particles = (() => {
    const cv = $("#particles"), ctx = cv.getContext("2d");
    let W = 0, H = 0, DPR = 1, pts = [], mouse = { x: -9999, y: -9999 }, running = false, visible = true, scrollP = 0, t = 0, assembled = false;
    function carPoints() {
      const o = document.createElement("canvas"); o.width = 1000; o.height = 420;
      const c = o.getContext("2d");
      c.fillStyle = "#fff"; c.strokeStyle = "#fff";
      c.beginPath();
      c.moveTo(40, 300); c.bezierCurveTo(36, 262, 70, 246, 150, 236);
      c.lineTo(340, 206); c.bezierCurveTo(410, 154, 480, 126, 575, 124);
      c.bezierCurveTo(665, 122, 730, 146, 800, 196); c.bezierCurveTo(880, 206, 940, 222, 962, 250);
      c.lineTo(968, 300); c.closePath(); c.fill();
      c.globalCompositeOperation = "destination-out";
      c.beginPath(); c.moveTo(395, 204); c.bezierCurveTo(450, 160, 500, 142, 572, 140); c.lineTo(705, 146); c.lineTo(770, 198); c.closePath(); c.fill();
      [235, 790].forEach((x) => { c.beginPath(); c.arc(x, 300, 70, 0, Math.PI * 2); c.fill(); });
      c.globalCompositeOperation = "source-over";
      c.lineWidth = 7;
      c.beginPath(); c.moveTo(395, 204); c.bezierCurveTo(450, 160, 500, 142, 572, 140); c.lineTo(705, 146); c.lineTo(770, 198); c.closePath(); c.stroke();
      c.beginPath(); c.moveTo(600, 140); c.lineTo(592, 200); c.stroke();
      [235, 790].forEach((x) => {
        c.lineWidth = 16; c.beginPath(); c.arc(x, 300, 52, 0, Math.PI * 2); c.stroke();
        c.lineWidth = 4; c.beginPath(); c.arc(x, 300, 24, 0, Math.PI * 2); c.stroke();
        for (let a = 0; a < 5; a++) { const r = (a / 5) * Math.PI * 2; c.beginPath(); c.moveTo(x + Math.cos(r) * 24, 300 + Math.sin(r) * 24); c.lineTo(x + Math.cos(r) * 44, 300 + Math.sin(r) * 44); c.stroke(); }
      });
      c.lineWidth = 5; c.beginPath(); c.moveTo(70, 262); c.lineTo(130, 252); c.stroke();      // headlight
      c.beginPath(); c.moveTo(905, 226); c.lineTo(950, 240); c.stroke();                     // tail light
      const d = c.getImageData(0, 0, 1000, 420).data, edge = [], fill = [], step = 4;
      const on = (x, y) => x >= 0 && y >= 0 && x < 1000 && y < 420 && d[(y * 1000 + x) * 4 + 3] > 128;
      for (let y = 0; y < 420; y += step) for (let x = 0; x < 1000; x += step) {
        if (!on(x, y)) continue;
        (on(x - step, y) && on(x + step, y) && on(x, y - step) && on(x, y + step) ? fill : edge).push([x, y]);
      }
      return { edge, fill };
    }
    const shape = carPoints();
    function layout() {
      DPR = Math.min(2, window.devicePixelRatio || 1);
      const r = cv.getBoundingClientRect(); W = r.width; H = r.height;
      cv.width = W * DPR; cv.height = H * DPR; ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      const N = MOBILE ? 1100 : W < 1100 ? 1700 : 2300;
      const narrow = W < 900;
      const cw = narrow ? W * 0.98 : Math.min(W * 0.6, 1150), s = cw / 1000;
      const cx = narrow ? W * 0.5 : W * 0.65, cy = narrow ? H * 0.46 : H * 0.6;
      const pick = (arr, n) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, n); };
      const nEdge = Math.min(shape.edge.length, Math.round(N * 0.62));
      const targets = pick(shape.edge, nEdge).concat(pick(shape.fill, N - nEdge));
      const old = pts;
      pts = targets.map(([x, y], i) => {
        const p = old[i] || { x: Math.random() * W, y: Math.random() * H, vx: 0, vy: 0, seed: Math.random() };
        p.tx = cx + (x - 500) * s; p.ty = cy + (y - 210) * s;
        p.c = Math.random() < 0.18 ? 1 : 0; p.sz = Math.random() < 0.1 ? 2.2 : 1.4;
        p.delay = (x / 1000) * 0.9 + Math.random() * 0.25;
        return p;
      });
      floor = { y: cy + 110 * s, cx, s, narrow };
    }
    let floor = { y: 0 };
    function drawFloor() {
      const { y: hy, cx } = floor; if (!hy) return;
      ctx.save(); ctx.globalAlpha = 0.5 * (1 - scrollP);
      if (floor.narrow) ctx.translate(0, H * 0.18);       // keep the horizon below the text on narrow screens
      ctx.strokeStyle = "rgba(198,255,61,0.16)"; ctx.lineWidth = 1;
      const off = (t * 22) % 40;
      for (let i = 0; i < 14; i++) {                                             // horizontal lines (moving)
        const z = i * 40 + off, yy = hy + (z * z) / 900;
        if (yy > H) break;
        ctx.globalAlpha = 0.5 * (1 - scrollP) * Math.min(1, (yy - hy) / 60 + 0.1);
        ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(W, yy); ctx.stroke();
      }
      ctx.globalAlpha = 0.35 * (1 - scrollP);
      for (let i = -12; i <= 12; i++) { ctx.beginPath(); ctx.moveTo(cx + i * 18, hy); ctx.lineTo(cx + i * 260, H); ctx.stroke(); }
      const g = ctx.createLinearGradient(0, hy - 2, 0, hy + 2); g.addColorStop(0, "rgba(56,232,255,0)"); g.addColorStop(0.5, "rgba(56,232,255,0.6)"); g.addColorStop(1, "rgba(56,232,255,0)");
      ctx.globalAlpha = 0.9 * (1 - scrollP); ctx.fillStyle = g; ctx.fillRect(0, hy - 2, W, 4);
      ctx.restore();
    }
    let start = 0;
    function frame(now) {
      if (!running) return;
      requestAnimationFrame(frame);
      if (!visible || document.hidden) return;
      t = now / 1000;
      const el = assembled ? (now - start) / 1000 : -1;
      ctx.clearRect(0, 0, W, H);
      drawFloor();
      ctx.globalCompositeOperation = "lighter";
      const a = floor.narrow ? 0.45 : 0.85;              // behind the headline on narrow screens: dimmer
      const cols = [`rgba(198,255,61,${a})`, `rgba(56,232,255,${a})`];
      const lift = scrollP * 260;
      for (let c = 0; c < 2; c++) {
        ctx.fillStyle = cols[c];
        for (const p of pts) {
          if (p.c !== c) continue;
          const active = el >= p.delay;
          if (active) {
            const wob = Math.sin(t * 1.6 + p.seed * 12) * 1.2;
            const tx = p.tx + (p.seed - 0.5) * scrollP * 900, ty = p.ty + wob - lift * (0.6 + p.seed);
            p.vx += (tx - p.x) * 0.045; p.vy += (ty - p.y) * 0.045;
          } else { p.vx += Math.sin(t + p.seed * 9) * 0.05; p.vy += Math.cos(t + p.seed * 7) * 0.05; }
          const dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
          if (d2 < 14400) { const f = (14400 - d2) / 14400 * 2.6, d = Math.sqrt(d2) || 1; p.vx += (dx / d) * f; p.vy += (dy / d) * f; }
          p.vx *= 0.84; p.vy *= 0.84; p.x += p.vx; p.y += p.vy;
          ctx.fillRect(p.x, p.y, p.sz, p.sz);
        }
      }
      ctx.globalCompositeOperation = "source-over";
    }
    function staticDraw() {                                                        // reduced motion
      ctx.clearRect(0, 0, W, H); drawFloor(); ctx.fillStyle = "rgba(198,255,61,0.7)";
      for (const p of pts) ctx.fillRect(p.tx, p.ty, p.sz, p.sz);
    }
    layout();
    window.addEventListener("resize", () => { layout(); if (!live) staticDraw(); });
    window.addEventListener("pointermove", (e) => { const r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; }, { passive: true });
    window.addEventListener("pointerleave", () => { mouse.x = mouse.y = -9999; });
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(cv);
    if (live) M.scroll((p) => { scrollP = p; }, { target: $("#hero"), offset: ["start start", "end start"] });
    return {
      assemble() {
        if (!live) { staticDraw(); return; }
        if (running) return;
        running = true; assembled = true; start = performance.now(); requestAnimationFrame(frame);
      },
    };
  })();

  /* =============================================================== cursor + magnetic */
  if (FINE && live) {
    const cur = $("#cursor"), dot = $(".cursor__dot", cur), ring = $(".cursor__ring", cur), label = $("#cursor-label");
    let mx = innerWidth / 2, my = innerHeight / 2, rx = mx, ry = my;
    window.addEventListener("pointermove", (e) => { mx = e.clientX; my = e.clientY; dot.style.transform = `translate(${mx}px,${my}px)`; }, { passive: true });
    (function loop() { rx += (mx - rx) * 0.18; ry += (my - ry) * 0.18; ring.style.transform = `translate(${rx}px,${ry}px)`; label.style.transform = `translate(${rx}px,${ry}px) translate(-50%,-50%)`; requestAnimationFrame(loop); })();
    document.addEventListener("pointerover", (e) => {
      const t = e.target.closest("a, button, label.drop, [data-cursor], .bar__hit");
      cur.classList.toggle("is-active", !!t);
      label.textContent = t?.dataset.cursor || (t ? "" : "");
    });
  }
  function magnetize(el) {
    if (!FINE || !live) return;
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      M.animate(el, { x: (e.clientX - r.left - r.width / 2) * 0.3, y: (e.clientY - r.top - r.height / 2) * 0.45 }, { duration: 0.3, ease: "easeOut" });
    });
    el.addEventListener("pointerleave", () => M.animate(el, { x: 0, y: 0 }, { type: "spring", stiffness: 300, damping: 14 }));
  }
  $$(".magnetic").forEach(magnetize);
  function tilt(el) {
    if (!FINE || !live) return;
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      el.style.setProperty("--mx", `${x * 100}%`); el.style.setProperty("--my", `${y * 100}%`);
      el.style.transform = `perspective(900px) rotateX(${(0.5 - y) * 10}deg) rotateY(${(x - 0.5) * 12}deg) translateZ(0)`;
    });
    el.addEventListener("pointerleave", () => { el.style.transition = "transform 500ms cubic-bezier(.22,1,.36,1)"; el.style.transform = ""; setTimeout(() => (el.style.transition = ""), 500); });
  }

  /* =============================================================== nav, progress, status, reveal */
  if (live) {
    const bar = $("#progress");
    M.scroll((p) => { bar.style.transform = `scaleX(${p})`; });
  }
  const navLinks = $$("[data-nav]");
  new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) navLinks.forEach((a) => a.classList.toggle("is-active", a.dataset.nav === e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" }).observe($("#scan"));
  ["#numbers", "#pipeline"].forEach((s) => navLinks.length && new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) navLinks.forEach((a) => a.classList.toggle("is-active", a.dataset.nav === e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" }).observe($(s)));
  health.then((h) => {
    const d = $("#status .dot");
    d.classList.add(h ? "is-on" : "is-off");
    $("#status-text").textContent = h ? `MODEL ONLINE · ${h.backend.toUpperCase()}` : "API OFFLINE";
  });
  $$("[data-reveal]").forEach((el) => {
    if (live) el.style.opacity = 0;
    // panels wipe open like a shutter; headings just rise
    const kf = el.classList.contains("panel")
      ? { opacity: [0, 1], y: [50, 0], clipPath: ["inset(0% 0% 100% 0% round 18px)", "inset(0% 0% 0% 0% round 18px)"] }
      : { opacity: [0, 1], y: [40, 0] };
    onView(el, () => anim(el, kf, { duration: 0.9, ease: EASE }).then?.(() => (el.style.clipPath = "")), 0.15);
  });

  /* =============================================================== tooltip */
  const tip = $("#tooltip");
  function bindTips(root) {
    $$("[data-tip]", root).forEach((el) => {
      const show = () => {
        tip.textContent = el.dataset.tip; tip.classList.add("is-on");
        const r = el.getBoundingClientRect();
        tip.style.left = `${Math.min(innerWidth - tip.offsetWidth - 8, Math.max(8, r.left + r.width / 2 - tip.offsetWidth / 2))}px`;
        tip.style.top = `${Math.max(8, r.top - tip.offsetHeight - 10)}px`;
      };
      const hide = () => tip.classList.remove("is-on");
      el.addEventListener("mouseenter", show); el.addEventListener("mouseleave", hide);
      el.addEventListener("focus", show); el.addEventListener("blur", hide);
    });
  }

  /* =============================================================== marquee */
  const MQ = [];
  fetch("/api/classes").then((r) => r.json()).then((names) => {
    const half = Math.ceil(names.length / 2);
    [["#mq1", names.slice(0, half)], ["#mq2", names.slice(half)]].forEach(([sel, list], i) => {
      const row = $(sel), html = list.map((n) => `<span>${esc(n)}</span>`).join("");
      row.innerHTML = html + html;
      if (live) {
        const a = M.animate(row, { x: i ? ["-50%", "0%"] : ["0%", "-50%"] }, { duration: list.length * 2.6, ease: "linear", repeat: Infinity });
        MQ.push(a);
        row.addEventListener("mouseenter", () => (a.hover = true)); row.addEventListener("mouseleave", () => (a.hover = false));
      }
    });
  }).catch(() => {});

  /* =============================================================== scanner */
  const input = $("#file"), dz = $("#dropzone");
  let lastFile = null, busy = false;
  input.addEventListener("change", () => input.files[0] && analyse(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) analyse(f); });
  dz.tabIndex = 0;
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  window.addEventListener("paste", (e) => { const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/")); if (it) analyse(it.getAsFile()); });
  if (live) M.press(dz, (el) => { M.animate(el, { scale: 0.98 }, { duration: 0.12 }); return () => M.animate(el, { scale: 1 }, SPRING); });
  $("#retry").addEventListener("click", () => (lastFile ? analyse(lastFile) : input.click()));

  fetch("/api/examples").then((r) => r.json()).then((list) => {
    const row = $("#examples");
    row.innerHTML = list.map((src, i) => `<button class="thumb" type="button" role="listitem" data-src="${src}" data-cursor="Scan" aria-label="Scan example image ${i + 1}"><img src="${src}" alt="" loading="lazy" width="160" height="120"/></button>`).join("");
    $$(".thumb", row).forEach((b) => b.addEventListener("click", async () => {
      $$(".thumb").forEach((t) => t.classList.remove("is-active")); b.classList.add("is-active");
      const blob = await (await fetch(b.dataset.src)).blob();
      analyse(new File([blob], b.dataset.src.split("/").pop(), { type: blob.type || "image/jpeg" }));
    }));
  }).catch(() => {});

  const stage = $("#stage"), viewer = $("#viewer"), photo = $("#photo"), overlay = $("#overlay");
  let beamCtl = null, loadTimer = null;
  const STEPS = ["VALIDATING INPUT", "ACQUIRING VEHICLE · YOLO11", "MEASURING QUALITY", "ENHANCING · CLAHE / GAMMA", "CLASSIFYING · 196 CLASSES", "CROSS-CHECKING PIPELINES"];
  function setLoading(on) {
    busy = on;
    stage.classList.toggle("is-loading", on); stage.setAttribute("aria-busy", on);
    input.disabled = on; $$(".thumb").forEach((t) => (t.disabled = on));
    clearInterval(loadTimer); beamCtl?.stop?.(); beamCtl = null;
    if (!on) return;
    let i = 0; $("#loading-text").textContent = STEPS[0] + "…";
    loadTimer = setInterval(() => { i = Math.min(i + 1, STEPS.length - 1); scramble($("#loading-text"), STEPS[i] + "…", 300); }, 520);
    if (live) beamCtl = M.animate($("#beam"), { y: [0, viewer.getBoundingClientRect().height || 320] }, { duration: 1.1, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" });
  }
  function showError(msg) {
    $("#error-text").textContent = msg; $("#error").hidden = false;
    anim($("#error"), { opacity: [0, 1], x: [-14, 10, -6, 3, 0] }, { duration: 0.45 });
  }
  async function analyse(file) {
    if (busy) return;
    lastFile = file; $("#error").hidden = true;
    if (!file.type.startsWith("image/")) { showError("That file is not an image. Please choose a JPG, PNG or WebP photo."); return; }
    if (file.size > 15 * 1024 * 1024) { showError("The image is larger than 15 MB. Please choose a smaller photo."); return; }
    const scanTop = $("#scan").getBoundingClientRect().top;
    if (scanTop > innerHeight * 0.5 || scanTop < -innerHeight) $("#scan").scrollIntoView({ behavior: RM ? "auto" : "smooth" });
    const url = URL.createObjectURL(file);
    photo.src = url; overlay.innerHTML = ""; $$(".tagline, .coords", viewer).forEach((e) => e.remove());
    $("#stage-empty").hidden = true; viewer.hidden = false;
    ["#result", "#compare-section", "#steps-section", "#quality-wrap"].forEach((s) => ($(s).hidden = true));
    $("#latency-hint").textContent = "SCANNING";
    anim(viewer, { opacity: [0, 1], scale: [0.96, 1], filter: ["blur(8px)", "blur(0px)"] }, { duration: 0.4 });
    await photo.decode().catch(() => {});
    setLoading(true);
    const fd = new FormData(); fd.append("file", file);
    const ctrl = new AbortController(), to = setTimeout(() => ctrl.abort(), 45000);
    try {
      // the GPU answers in ~150 ms; keep the scan visible long enough to read (real latency is still shown)
      const minScan = wait(live ? 1200 : 0);
      const res = await fetch("/api/analyze", { method: "POST", body: fd, signal: ctrl.signal });
      const data = await res.json().catch(() => ({}));
      await minScan;
      if (!res.ok) throw new Error(data.detail || `Server error (${res.status}).`);
      photo.src = data.image; await photo.decode().catch(() => {});
      setLoading(false);
      render(data);
    } catch (err) {
      setLoading(false); $("#latency-hint").textContent = "ERROR";
      showError(err.name === "AbortError" ? "The server took too long to respond. Please try again." :
        err instanceof TypeError ? "Could not reach the server. Check your connection and try again." : err.message);
    } finally { clearTimeout(to); URL.revokeObjectURL(url); }
  }

  const STATUS = {
    "confident": { cls: "ok", tone: "", text: "Confident", icon: '<path d="M20 6L9 17l-5-5"/>' },
    "year-uncertain": { cls: "warn", tone: "is-warn", text: "Year uncertain", icon: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>' },
    "model-uncertain": { cls: "serious", tone: "is-warn", text: "Make only", icon: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>' },
    "low-confidence": { cls: "low", tone: "is-low", text: "Low confidence", icon: '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>' },
    "open-set": { cls: "cyan", tone: "is-cyan", text: "Unseen model", icon: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"/>' },
    "unknown-model": { cls: "serious", tone: "is-warn", text: "Unknown model", icon: '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>' },
    "unknown": { cls: "low", tone: "is-low", text: "Outside supported classes", icon: '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>' },
    "no-car": { cls: "low", tone: "is-low", text: "No car", icon: '<circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/>' },
  };

  function renderNoCar(d) {
    const r = d.results.hybrid, st = STATUS["no-car"];
    overlay.innerHTML = "";
    $("#latency-hint").textContent = `${Math.round(r.timings.total_ms)} MS · NO VEHICLE`;
    const lock = $("#lock"); lock.textContent = "NO CAR DETECTED"; lock.classList.add("is-nocar");
    anim(lock, { opacity: [0, 1], scale: [1.6, 1] }, { duration: 0.5, ease: EASE });
    const q = d.quality;
    $("#quality").innerHTML = [["Bright", `${q.brightness}`], ["Contrast", q.contrast], ["Noise σ", q.noise], ["Sharp", Math.round(q.sharpness)], ["Min px", q.resolution]]
      .map(([k, v]) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("");
    $("#quality-wrap").hidden = false;
    const res = $("#result"); res.hidden = false; res.classList.add("is-nocar");
    $("#r-make").textContent = "VEHICLE GATE";
    $("#r-model").textContent = ""; scramble($("#r-model"), "No car detected", 700);
    $("#r-year").hidden = true;
    const badge = $("#r-badge"); badge.className = `badge badge--${st.cls}`; badge.innerHTML = `${icon(st.icon)}<span>${st.text}</span>`;
    $("#r-threshold").textContent = `The vehicle check scored ${pct(r.car_probability)}, below the ${pct(d.thresholds.gate, 0)} needed. Upload a photo that shows a car, SUV, van or pickup.`;
    const g = $("#gauge"), C = 2 * Math.PI * 52;
    g.setAttribute("class", "gauge__fg is-low"); g.style.strokeDasharray = `${C}`;
    $("#gauge-label").textContent = "CAR %";
    anim(g, { strokeDashoffset: [C, C * (1 - r.car_probability)] }, { duration: 0.9, ease: EASE });
    anim($("#gauge-tick"), { rotate: [0, 360 * r.car_probability] }, { duration: 0.9, ease: EASE });
    countUp($("#r-conf"), r.car_probability * 100, (v) => v.toFixed(1), 0.9);
    $("#r-facts").innerHTML = [["Vehicle score (CLIP)", pct(r.car_probability)], ["Latency", `${Math.round(r.timings.total_ms)} ms`]]
      .map(([k, v]) => `<div class="fact"><span>${k}</span><b>${esc(v)}</b></div>`).join("");
    $("#top5-wrap").hidden = true;
    anim([...res.children], { opacity: [0, 1], y: [20, 0] }, { duration: 0.5, delay: stagger(0.1), ease: EASE });
  }
  const icon = (p, w = 14) => `<svg width="${w}" height="${w}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;

  function drawTarget(d, r, st) {
    const [x1, y1, x2, y2] = r.box, W = d.width, H = d.height, w = x2 - x1, h = y2 - y1;
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, c = Math.max(10, Math.min(w, h) * 0.14);
    overlay.setAttribute("viewBox", `0 0 ${W} ${H}`); overlay.setAttribute("preserveAspectRatio", "none");
    overlay.innerHTML = `
      <line class="cross" x1="0" y1="${cy}" x2="${W}" y2="${cy}"/><line class="cross" x1="${cx}" y1="0" x2="${cx}" y2="${H}"/>
      <rect class="box ${st.tone}" x="${x1}" y="${y1}" width="${w}" height="${h}"/>
      <path class="corner" data-dx="-1" data-dy="-1" d="M${x1} ${y1 + c}V${y1}H${x1 + c}"/>
      <path class="corner" data-dx="1" data-dy="-1" d="M${x2 - c} ${y1}H${x2}V${y1 + c}"/>
      <path class="corner" data-dx="1" data-dy="1" d="M${x2} ${y2 - c}V${y2}H${x2 - c}"/>
      <path class="corner" data-dx="-1" data-dy="1" d="M${x1 + c} ${y2}H${x1}V${y2 - c}"/>
      <circle class="centre" cx="${cx}" cy="${cy}" r="${Math.min(w, h) * 0.04 + 4}"/>`;
    const tag = document.createElement("div");
    tag.className = `tagline ${st.tone}`; tag.textContent = `${r.display} · ${pct(r.confidence)}`;
    tag.style.left = `${(x1 / W) * 100}%`; tag.style.top = `${(y1 / H) * 100}%`;
    const co = document.createElement("div");
    co.className = "coords"; co.style.left = `${(x2 / W) * 100}%`; co.style.top = `${(y2 / H) * 100}%`;
    viewer.append(tag, co);
    requestAnimationFrame(() => {
      const vr = viewer.getBoundingClientRect(), tr = tag.getBoundingClientRect(), cr = co.getBoundingClientRect();
      let dy = -tr.height - 4; if (tr.top + dy < vr.top) dy = 4;
      tag.style.marginTop = `${dy}px`; if (tr.right > vr.right) tag.style.marginLeft = `${vr.right - tr.right - 4}px`;
      co.style.marginLeft = `${Math.min(4, vr.right - cr.right - 4) - (cr.right > vr.right ? cr.width : 0)}px`;
      co.style.marginTop = cr.bottom + 4 > vr.bottom ? `${-cr.height - 4}px` : "4px";
    });
    scramble(co, `X${x1} Y${y1}  ${w}×${h}PX`, 600);
    const lines = $$(".cross", overlay), corners = $$(".corner", overlay), box = $(".box", overlay), centre = $(".centre", overlay);
    if (!live) return;
    lines.forEach((l) => (l.style.opacity = 0)); [box, centre, tag, co].forEach((e) => (e.style.opacity = 0));
    M.animate(lines, { opacity: [0, 0.8] }, { duration: 0.35 });
    corners.forEach((p) => M.animate(p, { x: [+p.dataset.dx * w * 0.35, 0], y: [+p.dataset.dy * h * 0.35, 0], opacity: [0, 1] }, { type: "spring", stiffness: 180, damping: 16, delay: 0.08 }));
    M.animate(centre, { opacity: [0, 1], scale: [3, 1] }, { duration: 0.5, delay: 0.25, ease: EASE });
    M.animate(box, { opacity: [0, 1] }, { duration: 0.4, delay: 0.3 });
    M.animate(tag, { opacity: [0, 1], y: [10, 0], scale: [0.9, 1] }, { ...SPRING, delay: 0.55 });
    M.animate(co, { opacity: [0, 1] }, { duration: 0.3, delay: 0.5 });
    const lock = $("#lock");
    lock.style.borderColor = lock.style.color = getComputedStyle(box).stroke;
    M.animate(lock, { opacity: [0, 1, 1, 0], scale: [1.6, 1, 1, 1.05] }, { duration: 1.3, times: [0, 0.18, 0.75, 1], delay: 0.2 });
    lockFX(d, cx, cy, Math.max(w, h), getComputedStyle(box).stroke);
  }
  /* target-lock effects: shockwave rings from the box centre, a spark burst, and a one-frame sensor glitch */
  function lockFX(d, cx, cy, size, colour) {
    if (!live) return;
    for (let i = 0; i < 3; i++) {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", cx); c.setAttribute("cy", cy); c.setAttribute("r", size * 0.75);
      c.setAttribute("class", "shock"); c.style.stroke = colour; overlay.appendChild(c);
      c.style.transformBox = "fill-box"; c.style.transformOrigin = "50% 50%";
      M.animate(c, { scale: [0.02, 1], opacity: [0.9, 0] }, { duration: 1.1, delay: 0.25 + i * 0.16, ease: [0.16, 1, 0.3, 1] }).then(() => c.remove());
    }
    const vr = viewer.getBoundingClientRect(), px = (cx / d.width) * vr.width, py = (cy / d.height) * vr.height;
    for (let i = 0; i < 26; i++) {
      const s = document.createElement("i"); s.className = "spark"; s.style.left = `${px}px`; s.style.top = `${py}px`; s.style.background = colour;
      viewer.appendChild(s);
      const a = Math.random() * Math.PI * 2, dist = 60 + Math.random() * Math.min(vr.width, vr.height) * 0.45;
      M.animate(s, { x: [0, Math.cos(a) * dist], y: [0, Math.sin(a) * dist], scale: [1.4, 0], opacity: [1, 0] },
        { duration: 0.7 + Math.random() * 0.5, delay: 0.28, ease: [0.16, 1, 0.3, 1] }).then(() => s.remove());
    }
    M.animate(photo, { filter: ["none", "contrast(1.5) saturate(1.8) hue-rotate(-12deg)", "none", "contrast(1.3) hue-rotate(10deg)", "none"], x: [0, -5, 4, -2, 0] },
      { duration: 0.32, delay: 0.18 });
  }
  /* confident result: a short burst of confetti from the readout */
  function confetti(host) {
    if (!live) return;
    const cols = ["#c6ff3d", "#38e8ff", "#ffffff", "#a78bfa"];
    for (let i = 0; i < 46; i++) {
      const p = document.createElement("i"); p.className = "confetti";
      p.style.background = cols[i % cols.length]; p.style.left = `${20 + Math.random() * 60}%`;
      host.appendChild(p);
      const up = -(120 + Math.random() * 160), drift = (Math.random() - 0.5) * 320;
      M.animate(p, { x: [0, drift * 0.6, drift], y: [0, up, up + 380], rotate: [0, Math.random() * 720 - 360], opacity: [1, 1, 0] },
        { duration: 1.6 + Math.random() * 0.8, ease: "easeOut", delay: 0.5 + Math.random() * 0.25 }).then(() => p.remove());
    }
  }
  function glitch(el) { if (!live) return; el.classList.remove("is-glitch"); void el.offsetWidth; el.classList.add("is-glitch"); setTimeout(() => el.classList.remove("is-glitch"), 900); }

  function render(d) {
    const lock = $("#lock"); lock.textContent = "TARGET LOCKED"; lock.classList.remove("is-nocar");
    $("#result").classList.remove("is-nocar"); $("#r-year").hidden = false; $("#top5-wrap").hidden = false;
    $("#gauge-label").textContent = "CONF %";
    if (d.no_car) { renderNoCar(d); return; }
    const r = d.results.hybrid, st = STATUS[r.status] || STATUS["low-confidence"];
    const open = r.status === "open-set";
    const unk = r.status === "unknown" || r.status === "unknown-model";
    const outside = open || (unk && !r.possibly_outside);          // CLIP says: not one of the 196 classes
    $("#top5-title").textContent = open ? "OPEN-VOCABULARY CANDIDATES · CLIP"
      : unk ? (outside ? "CLOSEST CATALOGUE MATCHES · UNCONFIRMED" : "CLOSEST TRAINED CLASSES · UNCONFIRMED")
      : "TOP-5 CANDIDATES · HYBRID";
    // What the big title shows: never present a guess the system did not commit to.
    let mk = r.make, md = r.model, yr = open ? "YEAR N/A" : r.year;
    if (r.status === "model-uncertain") { mk = r.best_make; md = "Model uncertain"; yr = "YEAR ?"; }
    else if (r.status === "year-uncertain") { const bm = r.best_makemodel || ""; mk = r.make; md = bm.startsWith(mk) ? bm.slice(mk.length).trim() : r.model; yr = "YEAR ?"; }
    else if (unk) { mk = r.make || "Make unclear"; md = r.status === "unknown" ? "Unknown car" : "Unknown model"; yr = "YEAR N/A"; }
    drawTarget(d, r, st);
    $("#latency-hint").textContent = `${Math.round(r.timings.total_ms)} MS · ${r.det_source.toUpperCase()}`;

    const q = d.quality;
    $("#quality").innerHTML = [["Bright", `${q.brightness}`], ["Contrast", q.contrast], ["Noise σ", q.noise], ["Sharp", Math.round(q.sharpness)], ["Min px", q.resolution]]
      .map(([k, v]) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("");
    $("#quality-wrap").hidden = false;
    anim($$("#quality .metric"), { opacity: [0, 1], y: [10, 0] }, { duration: 0.35, delay: stagger(0.05) });

    $("#r-make").textContent = ""; scramble($("#r-make"), mk.toUpperCase(), 500);
    $("#r-model").textContent = ""; scramble($("#r-model"), md, 900);
    $("#r-year").textContent = yr;
    const badge = $("#r-badge"); badge.className = `badge badge--${st.cls}`; badge.innerHTML = `${icon(st.icon)}<span>${st.text}</span>`;
    const t = d.thresholds.confidence;
    const guess = r.top5.length ? `${r.top5[0].label} (${pct(r.top5[0].p)})` : "";
    $("#r-threshold").textContent = open
      ? `Not one of the 196 training classes, so it was named open-vocabulary by CLIP from ~950 catalogued models. Closest trained class: ${r.closest_trained_class}.`
      : r.status === "unknown-model"
        ? `${outside ? "This car does not match any of the 196 supported classes" : "The classifier is not sure this is one of the 196 supported classes"}, so it reports the make only (${pct(r.confidence)} sure) instead of guessing a model. Best unconfirmed guess: ${guess}.`
      : r.status === "unknown"
        ? `${outside ? "This car does not match any of the 196 supported classes" : "The classifier is not sure enough to name this car"} and even the make is unclear, so no make or model is claimed. Best unconfirmed guess: ${guess}.`
      : r.status === "confident"
        ? `Above the ${pct(t, 0)} acceptance threshold (98.5% precision on validation).`
        : `Below the ${pct(t, 0)} threshold, so it backs off: ${r.display}.`;
    $("#gauge-label").textContent = unk ? "MAKE %" : "CONF %";
    const g = $("#gauge"), C = 2 * Math.PI * 52;
    g.setAttribute("class", `gauge__fg ${st.tone}`); g.style.strokeDasharray = `${C}`;
    anim(g, { strokeDashoffset: [C, C * (1 - r.confidence)] }, { duration: 1.2, ease: EASE, delay: 0.3 });
    anim($("#gauge-tick"), { rotate: [0, 360 * r.confidence] }, { duration: 1.2, ease: EASE, delay: 0.3 });
    setTimeout(() => countUp($("#r-conf"), r.confidence * 100, (v) => v.toFixed(1), 1.2), 300);
    $("#r-facts").innerHTML = [
      open ? ["In training set", "No · open-vocabulary"]
        : unk ? ["Closest trained class", r.closest_trained_class || "-"]
        : ["Make-level", `${r.best_make} · ${pct(r.make_confidence)}`],
      ["Vehicle score", pct(r.car_probability)],
      ["Detector", `${r.det_source} · ${r.det_conf.toFixed(2)}`],
      ["Enhancement", r.enhancement_steps.length ? r.enhancement_steps.join(", ") : "not needed"],
      ["Latency", `${Math.round(r.timings.detection_ms)} + ${Math.round(r.timings.classification_ms)} ms`],
    ].map(([k, v]) => `<div class="fact"><span>${k}</span><b>${esc(v)}</b></div>`).join("");
    $("#r-top5").innerHTML = r.top5.map((c) => `<li><div class="top5__row"><span>${esc(c.label)}</span><span class="mono">${pct(c.p)}</span></div><div class="top5__bar"><i></i></div></li>`).join("");
    const res = $("#result"); res.hidden = false;
    anim([...res.children], { opacity: [0, 1], y: [20, 0] }, { duration: 0.55, delay: stagger(0.12, { startDelay: 0.2 }), ease: EASE });
    $$("#r-top5 .top5__bar i").forEach((el, i) => anim(el, { scaleX: [0, Math.max(0.004, r.top5[i].p)] }, { type: "spring", stiffness: 110, damping: 13, delay: 0.6 + i * 0.08 }));
    anim($$("#r-top5 li"), { opacity: [0, 1], x: [-24, 0] }, { duration: 0.45, delay: stagger(0.07, { startDelay: 0.5 }), ease: EASE });
    anim($$("#r-facts .fact"), { opacity: [0, 1], rotateX: [-85, 0], transformPerspective: [600, 600] }, { type: "spring", stiffness: 200, damping: 18, delay: stagger(0.06, { startDelay: 0.45 }) });
    setTimeout(() => glitch($("#r-model")), 950);
    if (live) setTimeout(() => M.animate($(".gauge"), { scale: [1, 1.08, 1] }, { duration: 0.45, ease: "easeOut" }), 1500);
    if (r.status === "confident") confetti(res);

    const META = {
      classical: ["Classical CV", "Edge/contour ROI → HOG · LBP · colour · geometry → PCA + logistic regression"],
      deep: ["Deep learning", "YOLO11 crop → fine-tuned ConvNeXt-Tiny, single pass"],
      hybrid: ["Hybrid · ours", "CLIP vehicle gate + ConvNeXt/CLIP ensemble + enhanced view + TTA + open-set naming"],
    };
    const order = ["classical", "deep", "hybrid"].filter((m) => d.results[m]);
    $("#compare").innerHTML = order.map((m, i) => {
      const x = d.results[m], same = x.label === r.label;
      return `<article class="fighter fighter--${m}"><span class="fighter__num" aria-hidden="true">0${i + 1}</span>
        <span class="fighter__tag">${META[m][0]}</span>
        <div class="fighter__name">${esc(x.label)}</div>
        <p class="fighter__desc">${META[m][1]}</p>
        <div class="fighter__stats"><span>CONF <b>${pct(x.confidence)}</b></span><span><b>${Math.round(x.timings.total_ms)}</b> MS</span></div>
        ${m === "hybrid" ? '<span class="agree is-yes">' + icon('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/>') + "reference</span>" :
          `<span class="agree ${same ? "is-yes" : "is-no"}">${icon(same ? '<path d="M20 6L9 17l-5-5"/>' : '<path d="M6 6l12 12M18 6L6 18"/>')}${same ? "agrees with hybrid" : "disagrees with hybrid"}</span>`}
      </article>`;
    }).join("");
    $("#compare-section").hidden = false;
    const fighters = $$("#compare .fighter"); fighters.forEach(tilt);
    if (live) fighters.forEach((f) => (f.style.opacity = 0));
    onView($("#compare"), () => anim(fighters, { opacity: [0, 1], y: [60, 0], rotateX: [25, 0] }, { duration: 0.8, delay: stagger(0.1), ease: EASE }), 0.2);

    $("#gallery").innerHTML = d.steps.map((s, i) => `
      <button class="layer" type="button" data-i="${i}" data-cursor="Zoom" aria-label="Enlarge: ${esc(s.title)}">
        <div class="layer__img"><img src="${s.img}" alt="${esc(s.title)}: ${esc(s.caption)}"/></div>
        <div class="layer__body"><span class="layer__n">LAYER ${String(i + 1).padStart(2, "0")}</span><span class="layer__title">${esc(s.title)}</span><span class="layer__cap">${esc(s.caption)}</span></div>
      </button>`).join("");
    $("#steps-section").hidden = false;
    const layers = $$("#gallery .layer");
    layers.forEach((b) => b.addEventListener("click", () => openModal(d.steps[+b.dataset.i], b)));
    if (live) layers.forEach((l) => (l.style.opacity = 0));
    onView($("#gallery"), () => anim(layers, { opacity: [0, 1], scale: [0.85, 1], rotate: [-3, 0] }, { duration: 0.6, delay: stagger(0.07), ease: EASE }), 0.15);
  }

  /* =============================================================== modal */
  const modal = $("#modal"); let back = null;
  function openModal(step, from) {
    back = from; $("#modal-title").textContent = step.title.toUpperCase(); $("#modal-cap").textContent = step.caption;
    $("#modal-img").src = step.img; $("#modal-img").alt = `${step.title}: ${step.caption}`;
    modal.hidden = false;
    anim($(".modal__scrim", modal), { opacity: [0, 1] }, { duration: 0.2 });
    anim($(".modal__panel", modal), { opacity: [0, 1], scale: [0.9, 1], y: [30, 0] }, SPRING);
    $(".icon-btn", modal).focus();
  }
  async function closeModal() {
    if (modal.hidden) return;
    await anim($(".modal__panel", modal), { opacity: 0, scale: 0.95 }, { duration: 0.15 });
    modal.hidden = true; back?.focus();
  }
  $$("[data-close]", modal).forEach((b) => b.addEventListener("click", closeModal));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  modal.addEventListener("keydown", (e) => { if (e.key === "Tab") { e.preventDefault(); $(".icon-btn", modal).focus(); } });

  /* =============================================================== numbers */
  function barChart(rows, { fmt = (v) => pct(v, 2), max = 1 } = {}) {
    return `<div class="bars">${rows.map(([label, v, tipText]) => `
      <div class="bar"><span class="bar__label">${esc(label)}</span>
      <span class="bar__track"><i class="bar__fill" data-v="${v / max}"></i><span class="bar__hit" tabindex="0" data-tip="${esc(tipText || `${label}: ${fmt(v)}`)}"></span></span>
      <span class="bar__value">${fmt(v)}</span></div>`).join("")}</div>`;
  }
  const tableView = (head, rows) => `<details class="table-view"><summary>VIEW AS TABLE</summary><div class="table-wrap"><table>
    <thead><tr>${head.map((h, i) => `<th class="${i ? "num" : ""}">${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>`;
  const card = (title, hint, body, wide) => `<section class="panel chart ${wide ? "wide" : ""}"><div class="panel__bar mono"><span>${hint}</span></div><h3>${title}</h3><div style="margin-top:16px">${body}</div></section>`;

  async function loadNumbers() {
    const charts = $("#charts");
    charts.innerHTML = `<div class="panel"><div class="skeleton"><div class="sk" style="width:40%"></div><div class="sk"></div><div class="sk" style="width:80%"></div></div></div>`;
    let P;
    try { const r = await fetch("/api/performance"); if (!r.ok) throw new Error((await r.json()).detail); P = await r.json(); }
    catch (e) { charts.innerHTML = `<div class="alert"><span class="mono alert__code">ERR</span><div><b>Could not load evaluation results.</b><p>${esc(e.message || e)}</p></div></div>`; return; }
    const comp = Object.entries(P.comparison), hyb = comp[2][1], det = P.detection[0];
    const kp = [
      ["Top-1", hyb.top1 * 100, (v) => `${v.toFixed(1)}%`, "hybrid · 196 classes"],
      ["Top-5", hyb.top5 * 100, (v) => `${v.toFixed(1)}%`, "hybrid"],
      ["mAP@0.5", det.mAP50, (v) => v.toFixed(3), `@.5:.95 = ${det.mAP50_95.toFixed(3)}`],
      ["Mean IoU", det.mean_iou, (v) => v.toFixed(3), "fine-tuned YOLO11s"],
      ["ECE", P.calibration?.test_ece_hybrid_after ?? hyb.ece, (v) => v.toFixed(3), `calibrated · was ${(P.calibration?.test_ece_hybrid_before ?? 0).toFixed(3)}`],
    ];
    $("#kpis").innerHTML = kp.map(([k, , , s], i) => `<div class="big__cell"><span>${k}</span><b class="mono" data-k="${i}">0</b><small>${s}</small></div>`).join("");
    onView($("#kpis"), () => {
      kp.forEach(([, v, f], i) => countUp($(`[data-k="${i}"]`), v, f, 1.6));
      // cells flip over like a departures board
      anim($$("#kpis .big__cell"), { opacity: [0, 1], rotateX: [-95, 0], transformPerspective: [700, 700] }, { type: "spring", stiffness: 140, damping: 15, delay: stagger(0.09) });
    }, 0.3);
    $$("#kpis .big__cell").forEach((c) => tiltSoft(c, 0.6));

    const short = (k) => k.split(" (")[0].replace(/^\w/, (c) => c.toUpperCase());
    const robust = P.robustness || [];
    const robustHtml = robust.length ? `
      <div class="legend" aria-hidden="true"><span><i style="background:var(--series-1)"></i>Deep learning</span><span><i style="background:var(--series-2)"></i>Hybrid</span></div>
      <div class="groups">${robust.map((r) => `<div class="group"><span class="bar__label">${esc(r.condition.replace(/_/g, " "))}</span><div class="group__bars">
        ${[["deep", "Deep", "var(--series-1)"], ["hybrid", "Hybrid", "var(--series-2)"]].map(([k, n, c]) => `<div class="group__bar"><span class="bar__track"><i class="bar__fill" data-v="${r[k + "_top1"]}" style="background:${c}"></i><span class="bar__hit" tabindex="0" data-tip="${esc(`${r.condition}: ${n} ${pct(r[k + "_top1"])} · classical ${pct(r.classical_top1)}`)}"></span></span><span class="bar__value">${pct(r[k + "_top1"])}</span></div>`).join("")}
      </div></div>`).join("")}</div>
      <p class="axisnote">End-to-end top-1 on 1,000 random test images per condition (bars start at 0%). Standard error ≈ ±0.8 points, so only gaps above ~1.5 points are meaningful.</p>
      ${tableView(["Condition", "Classical", "Deep", "Hybrid", "Hybrid IoU"], robust.map((r) => [r.condition, pct(r.classical_top1), pct(r.deep_top1), pct(r.hybrid_top1), r.hybrid_iou.toFixed(3)]))}` : "";
    const O = P.optimization;
    const optHtml = O ? `<div class="table-wrap"><table><thead><tr><th>Variant</th><th class="num">Size</th><th class="num">Test top-1</th><th class="num">Latency</th></tr></thead><tbody>
      ${Object.entries(O.variants).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.size_mb} MB</td><td class="num">${pct(v.test_top1)}</td><td class="num">${v.latency_gpu_ms ? v.latency_gpu_ms.toFixed(1) + " ms GPU" : v.latency_cpu_ms ? v.latency_cpu_ms.toFixed(1) + " ms CPU" : "-"}</td></tr>`).join("")}
      </tbody></table></div><p class="axisnote">Test top-1 on a fixed 1,500-image subset · batch-1 latency on an RTX 5060 Laptop GPU / laptop CPU.</p>` : "";
    const lc = Object.entries(P.low_confidence);
    charts.innerHTML = [
      card("Approach comparison", "TEST TOP-1 · 8,041 IMAGES", barChart(comp.map(([k, v]) => [short(k), v.top1, `${short(k)}: top-1 ${pct(v.top1, 2)} · top-5 ${pct(v.top5, 2)}`])) +
        tableView(["Approach", "Top-1", "Top-5", "Make acc."], comp.map(([k, v]) => [short(k), pct(v.top1, 2), pct(v.top5, 2), pct(v.make_acc, 2)]))),
      card("Detection quality", "mAP@0.5:0.95", barChart(P.detection.map((d) => [d.detector.replace(/_/g, " "), d.mAP50_95, `${d.detector}: mAP@.5 ${d.mAP50.toFixed(3)} · mAP@.5:.95 ${d.mAP50_95.toFixed(3)} · IoU ${d.mean_iou.toFixed(3)}`]), { fmt: (v) => v.toFixed(3) }) +
        tableView(["Detector", "Mean IoU", "mAP@.5", "mAP@.5:.95"], P.detection.map((d) => [d.detector, d.mean_iou.toFixed(3), d.mAP50.toFixed(3), d.mAP50_95.toFixed(3)]))),
      card("Robustness under degradation", "DEEP VS HYBRID · TOP-1", robustHtml, true),
      card("Ablation study", "TEST TOP-1 · COMPONENT REMOVED", barChart(Object.entries(P.ablation).map(([k, v]) => [k, v.top1])) +
        `<p class="axisnote">Classifying the whole image instead of the detected crop costs 4.3 points; our crop is within 0.02 points of ground-truth boxes.</p>`),
      card("Low-confidence handling", `THRESHOLD ${pct(P.hybrid_config.conf_threshold, 0)} · T ${P.hybrid_config.temperature}`,
        `<div class="table-wrap"><table><thead><tr><th>Reported level</th><th class="num">Share</th><th class="num">Accuracy at level</th></tr></thead><tbody>
        ${lc.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${pct(v.fraction)}</td><td class="num">${v.accuracy_at_reported_level == null ? "-" : pct(v.accuracy_at_reported_level)}</td></tr>`).join("")}</tbody></table></div>
        <p class="axisnote">${P.errors.n_errors} errors: ${pct(P.errors.same_make_model_diff_year)} wrong year only, ${pct(P.errors.same_make_diff_model)} wrong model of the right make, ${pct(P.errors.different_make)} different make (often badge-engineered twins).</p>`),
      P.openset ? card("Bad input & unseen cars", "CLIP VEHICLE GATE · OPEN-SET", (() => {
        const g = P.openset.gate_on_stanford_test, o = P.openset.open_set_routing, z = P.openset.zero_shot_catalogue_test;
        const rows = [
          ["Real cars accepted by the gate (8,041 test images)", pct(g.accept_two_signal)],
          ["Non-car probe photos rejected", pct(P.openset.gate_non_car_rejected_final ?? P.openset["gate_non_car_rejected@0.5"])],
          [`Zero-shot make accuracy · ${z.catalogue_size} catalogued models`, pct(z.make)],
          ["Zero-shot make + model family accuracy", pct(z.model_family)],
          ["Simulated unseen models flagged as 'not in training set'", pct(o.test_simulated_unseen_recall)],
          ["Known cars wrongly flagged as unseen", pct(o.test_known_rerouted)],
        ];
        const U = P.unknown;
        if (U) rows.push(
          ["Wrong answers, unseen car in catalogue (old → make-first rule)", `${pct(U.test_catalogued_old.wrong)} → ${pct(U.test_catalogued_new.wrong)}`],
          ["Wrong answers, unseen car not in catalogue (old → make-first rule)", `${pct(U.test_uncatalogued_old.wrong)} → ${pct(U.test_uncatalogued_new.wrong)}`],
        );
        return `<div class="table-wrap"><table><tbody>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v}</td></tr>`).join("")}</tbody></table></div>
          <p class="axisnote">Unseen models are simulated by hiding each test car's true class from the classifier. Anything that is not a car returns "No car detected"; unfamiliar cars get a model name only when CLIP is sure, otherwise "make, model not recognised" or "unknown car".</p>`;
      })()) : "",
      card("Model optimisation", O ? `${O.classifier_params_M} M-PARAM CONVNEXT-T` : "OPTIMISATION", optHtml, true),
    ].join("");
    bindTips(charts);
    $$(".chart", charts).forEach((c) => {
      tiltSoft(c, 0.22);
      if (live) c.style.opacity = 0;
      onView(c, () => {
        anim(c, { opacity: [0, 1], y: [60, 0], rotateX: [18, 0], transformPerspective: [900, 900] }, { duration: 0.8, ease: EASE });
        $$(".bar__fill", c).forEach((f, i) => anim(f, { scaleX: [0, Math.max(0.004, +f.dataset.v)] }, { type: "spring", stiffness: 90, damping: 12, delay: 0.2 + i * 0.035 }));
        anim($$(".bar__value", c), { opacity: [0, 1] }, { duration: 0.4, delay: stagger(0.035, { startDelay: 0.5 }) });
      }, 0.2);
    });
  }

  /* =============================================================== pipeline timeline */
  function buildPipeline() {
    const S = [
      ["classical", "Input validation", "Guards the door", "Rejects empty, non-image, tiny or blank files, normalises greyscale/RGBA and caps huge photos so latency stays bounded."],
      ["classical", "Quality analysis", "Measures the light", "Brightness, contrast, Immerkaer noise σ, Laplacian sharpness and resolution, which decide what enhancement runs."],
      ["deep", "Vehicle detection", "Finds the car", "YOLO11s fine-tuned on Stanford Cars boxes (mAP@0.5 0.989). No box? Enhance the frame and retry."],
      ["hybrid", "Vehicle gate", "Is it even a car?", "CLIP ViT-B/16 scores car prompts against everyday non-car scenes; borderline scores need the detector to agree. Anything else returns 'No car detected'."],
      ["classical", "Adaptive enhancement", "Repairs the image", "Gamma for darkness, CLAHE with a contrast-aware clip limit, bilateral denoising and bicubic upscaling, only when needed."],
      ["deep", "Fine-grained ensemble", "Names the car", "ConvNeXt-Tiny (ImageNet-22k) + CLIP ViT-B/16 fine-tuned with harsh lighting, blur, JPEG, viewpoint and occlusion augmentation."],
      ["classical", "Hand-crafted features", "A transparent baseline", "HOG shape, LBP texture, HSV colour, edge orientation and contour geometry. 14.5% top-1, about 28× chance."],
      ["hybrid", "Calibrated decision", "Decides what to trust", "Averages both networks over the original and enhanced views with flip TTA, temperature-calibrated on validation."],
      ["hybrid", "Open-set naming", "Knows what it doesn't know", "If the car is not one of the 196 classes, CLIP names it zero-shot from a 954-model catalogue (83 makes) and says so."],
      ["hybrid", "Confidence back-off", "Never bluffs", "Low confidence falls back to make + model, then make only, then 'uncertain', each level checked on held-out data."],
    ];
    $("#pipeline-steps").innerHTML = S.map(([k, t, s, p], i) => `<li class="tstep"><div><span class="tstep__kind k-${k}">${String(i + 1).padStart(2, "0")} · ${k}</span></div><h3>${t}<br/><small class="muted" style="font:500 13px var(--font)">${s}</small></h3><p>${p}</p></li>`).join("");
    const steps = $$("#pipeline-steps .tstep"), fill = $("#timeline-fill");
    if (!live) { fill.style.transform = "scaleY(1)"; steps.forEach((s) => s.classList.add("is-on")); return; }
    M.scroll((p) => { fill.style.transform = `scaleY(${p})`; }, { target: $("#timeline"), offset: ["start 70%", "end 55%"] });
    steps.forEach((s, i) => {
      s.style.opacity = 0;
      M.inView(s, () => {
        s.classList.add("is-on");
        M.animate(s, { opacity: [0, 1], x: [i % 2 ? 90 : -40, 0], filter: ["blur(8px)", "blur(0px)"], scale: [0.94, 1] }, { type: "spring", stiffness: 120, damping: 16 });
        const k = $(".tstep__kind", s); scramble(k, k.textContent, 500);
      }, { amount: 0.5 });
      tiltSoft(s, 0.35);
    });
    // a glowing comet rides the end of the timeline fill
    const comet = document.createElement("i"); comet.className = "timeline__comet"; $(".timeline__line").appendChild(comet);
    M.scroll((p) => { comet.style.transform = `translate(-50%, ${p * $(".timeline__line").clientHeight}px)`; comet.style.opacity = p > 0.001 && p < 0.999 ? 1 : 0; },
      { target: $("#timeline"), offset: ["start 70%", "end 55%"] });
  }

  /* =============================================================== extra motion layer */
  // soft 3D tilt + spotlight that follows the pointer (k scales the tilt)
  function tiltSoft(el, k = 0.4) {
    if (!FINE || !live) return;
    el.classList.add("has-spot");
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      el.style.setProperty("--mx", `${x * 100}%`); el.style.setProperty("--my", `${y * 100}%`);
      el.style.transform = `perspective(1000px) rotateX(${(0.5 - y) * 10 * k}deg) rotateY(${(x - 0.5) * 12 * k}deg)`;
    });
    el.addEventListener("pointerleave", () => { el.style.transition = "transform 600ms cubic-bezier(.22,1,.36,1)"; el.style.transform = ""; setTimeout(() => (el.style.transition = ""), 600); });
  }
  // wrap every letter of an element in a span (screen readers keep reading the whole word via aria-label)
  function splitChars(el) {
    const label = el.textContent.trim();
    const walk = (node) => [...node.childNodes].forEach((n) => {
      if (n.nodeType === 3) {
        const frag = document.createDocumentFragment();
        [...n.textContent].forEach((ch) => {
          if (ch === " ") { frag.appendChild(document.createTextNode(" ")); return; }
          const s = document.createElement("span"); s.className = "ch"; s.textContent = ch; s.setAttribute("aria-hidden", "true"); frag.appendChild(s);
        });
        n.replaceWith(frag);
      } else if (n.nodeType === 1) walk(n);
    });
    walk(el);
    el.setAttribute("aria-label", label);
    return $$(".ch", el);
  }
  // word-level split for section headings (keeps <em> accents intact)
  function splitWords(el) {
    const label = el.textContent.trim(), out = document.createDocumentFragment();
    [...el.childNodes].forEach((n) => {
      const pieces = n.nodeType === 3 ? n.textContent.split(/(\s+)/) : [n];
      pieces.forEach((p) => {
        if (typeof p === "string") { if (!p) return; if (/^\s+$/.test(p)) { out.appendChild(document.createTextNode(" ")); return; } }
        const w = document.createElement("span"); w.className = "w"; w.setAttribute("aria-hidden", "true");
        const wi = document.createElement("span"); wi.className = "wi";
        if (typeof p === "string") wi.textContent = p; else wi.appendChild(p.cloneNode(true));
        w.appendChild(wi); out.appendChild(w);
      });
    });
    el.textContent = ""; el.appendChild(out); el.setAttribute("aria-label", label);
    return $$(".wi", el);
  }

  // hero letters: split now (heroIntro animates them), then make them magnetic to the pointer
  const heroChars = $$(".hero__title [data-split]").flatMap((w) => splitChars(w));
  if (!live) heroChars.forEach((c) => (c.style.transform = "none"));
  else heroChars.forEach((c) => (c.style.transform = "translateY(115%)"));
  if (FINE && live) {
    const title = $(".hero__title"); let mx = -1e4, my = -1e4, raf = 0;
    const cur = heroChars.map(() => ({ y: 0, s: 1 }));
    const tick = () => {
      let moving = false;
      heroChars.forEach((c, i) => {
        const r = c.getBoundingClientRect(), dx = mx - (r.left + r.width / 2), dy = my - (r.top + r.height / 2);
        const f = Math.max(0, 1 - Math.hypot(dx, dy) / 220);
        const ty = -f * 26, ts = 1 + f * 0.12, o = cur[i];
        o.y += (ty - o.y) * 0.18; o.s += (ts - o.s) * 0.18;
        if (Math.abs(ty - o.y) > 0.05) moving = true;
        if (c.dataset.ready) c.style.transform = `translateY(${o.y.toFixed(2)}px) scale(${o.s.toFixed(3)}) rotate(${(-dx * f * 0.02).toFixed(2)}deg)`;
      });
      raf = moving ? requestAnimationFrame(tick) : 0;
    };
    title.addEventListener("pointermove", (e) => { mx = e.clientX; my = e.clientY; if (!raf) raf = requestAnimationFrame(tick); });
    title.addEventListener("pointerleave", () => { mx = my = -1e4; if (!raf) raf = requestAnimationFrame(tick); });
    setTimeout(() => heroChars.forEach((c) => (c.dataset.ready = "1")), 4200);   // after the intro has landed
  }

  // hero: pointer spotlight + scroll parallax (copy drifts up and fades, stats lag behind)
  if (live) {
    const hero = $("#hero"), spot = document.createElement("div");
    spot.className = "hero__spot"; spot.setAttribute("aria-hidden", "true"); hero.appendChild(spot);
    if (FINE) hero.addEventListener("pointermove", (e) => { const r = hero.getBoundingClientRect(); spot.style.setProperty("--sx", `${e.clientX - r.left}px`); spot.style.setProperty("--sy", `${e.clientY - r.top}px`); });
    const content = $(".hero__content"), stats = $(".hero__stats");
    M.scroll((p) => {
      content.style.transform = `translateY(${-p * 160}px)`; content.style.opacity = String(1 - p * 1.4);
      stats.style.transform = `translateY(${-p * 60}px)`;
    }, { target: hero, offset: ["start start", "end start"] });
  }

  // marquee reacts to scroll velocity: speeds up and skews, then eases back
  if (live) {
    const mq = $(".marquee"); let vel = 0, sm = 0;
    M.scroll((_, info) => { vel = info?.y?.velocity || 0; });
    (function loop() {
      sm += (vel - sm) * 0.08; vel *= 0.9;
      const k = Math.max(-1, Math.min(1, sm / 2500));
      mq.style.transform = `skewY(${(k * -3).toFixed(2)}deg)`;
      const sp = Math.round((a0 => a0)((1 + Math.abs(k) * 5) * 20)) / 20;              // quantised: avoid churning the animation every frame
      MQ.forEach((a) => { const want = (a.hover ? 0.25 : 1) * sp; if (a._sp !== want) { a._sp = want; a.speed = want; } });
      requestAnimationFrame(loop);
    })();
  }

  // section headings: words rise out of masks; the section number decodes
  $$(".section__head h2").forEach((h) => {
    const words = splitWords(h);
    if (live) words.forEach((w) => (w.style.transform = "translateY(110%)"));
    onView(h, () => anim(words, { y: ["110%", "0%"], rotate: [7, 0] }, { type: "spring", stiffness: 130, damping: 17, delay: stagger(0.07) }), 0.4);
  });
  $$(".section__num").forEach((n) => onView(n, () => scramble(n, n.textContent, 700), 0.6));

  // nav: a pill slides between the active links
  (() => {
    const wrap = $(".nav__links"); if (!wrap) return;
    const pill = document.createElement("span"); pill.className = "nav__pill"; pill.setAttribute("aria-hidden", "true"); wrap.prepend(pill);
    let shown = false;
    const move = () => {
      const a = $(".nav__links a.is-active");
      if (!a) { pill.style.opacity = 0; shown = false; return; }
      const wr = wrap.getBoundingClientRect(), r = a.getBoundingClientRect(), kf = { x: r.left - wr.left, width: `${r.width}px`, opacity: 1 };
      if (!live || !shown) { pill.style.transform = `translateX(${kf.x}px)`; pill.style.width = kf.width; pill.style.opacity = 1; shown = true; return; }
      M.animate(pill, kf, { type: "spring", stiffness: 380, damping: 30 });
    };
    new MutationObserver(move).observe(wrap, { subtree: true, attributes: true, attributeFilter: ["class"] });
    window.addEventListener("resize", move);
  })();

  // buttons: ink ripple from the press point
  document.addEventListener("pointerdown", (e) => {
    const b = e.target.closest(".btn, .thumb"); if (!b || !live) return;
    const r = b.getBoundingClientRect(), s = document.createElement("span"), d = Math.max(r.width, r.height) * 2.2;
    s.className = "ripple"; s.style.width = s.style.height = `${d}px`; s.style.left = `${e.clientX - r.left - d / 2}px`; s.style.top = `${e.clientY - r.top - d / 2}px`;
    b.appendChild(s);
    M.animate(s, { scale: [0, 1], opacity: [0.45, 0] }, { duration: 0.6, ease: "easeOut" }).then(() => s.remove());
  });
  // fine pointer: a ring bursts wherever you click
  if (FINE && live) document.addEventListener("pointerdown", (e) => {
    const ring = document.createElement("i"); ring.className = "clickring"; ring.style.left = `${e.clientX}px`; ring.style.top = `${e.clientY}px`;
    document.body.appendChild(ring);
    M.animate(ring, { scale: [0.2, 2.4], opacity: [0.9, 0] }, { duration: 0.5, ease: "easeOut" }).then(() => ring.remove());
  });

  // footer wordmark: letters rise in a wave, and hop when the pointer passes over them
  (() => {
    const g = $(".footer__giant"); if (!g) return;
    const letters = splitChars(g);
    // fit the wordmark to the footer width on any screen
    const fit = () => {
      g.style.fontSize = "";
      const avail = g.clientWidth * 0.97;
      for (let i = 0; i < 4 && g.scrollWidth > avail; i++) {
        const cs = parseFloat(getComputedStyle(g).fontSize);
        g.style.fontSize = `${Math.floor(cs * (avail / g.scrollWidth) * 0.99)}px`;
      }
    };
    fit(); window.addEventListener("resize", fit); document.fonts?.ready.then(fit);
    if (live) letters.forEach((l) => (l.style.transform = "translateY(100%)"));
    onView(g, () => anim(letters, { y: ["100%", "0%"], rotate: [10, 0], opacity: [0, 1] }, { type: "spring", stiffness: 120, damping: 14, delay: stagger(0.05) }), 0.3);
    if (FINE && live) letters.forEach((l) => l.addEventListener("pointerenter", () => {
      M.animate(l, { y: [0, -40, 0], rotate: [0, -6, 0], color: ["rgba(198,255,61,0)", "rgba(198,255,61,1)", "rgba(198,255,61,0)"] }, { duration: 0.7, ease: [0.22, 1, 0.36, 1] });
    }));
  })();

  /* =============================================================== boot */
  buildPipeline();
  loadNumbers();
  boot();
})();
