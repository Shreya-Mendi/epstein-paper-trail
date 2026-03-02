/**
 * timeline.js — Paper Trail UI: section routing, network graph, timeline, drawer.
 *
 * Sections: hero, problem, network, timeline, findings, model
 * Network: force-directed canvas graph with consequence-tier coloring
 * Timeline: filtered event list with animation
 * Drawer: per-person profile with timeline events
 */

const API = "http://127.0.0.1:8000";

const TIER_COLORS = { 0: "#e63946", 1: "#f4a261", 2: "#e9c46a", 3: "#6c757d" };
const TIER_NAMES  = { 0: "CHARGED / CONVICTED", 1: "SETTLED CIVILLY", 2: "NAMED / INVESTIGATED", 3: "NO CONSEQUENCES" };
const TIER_LABELS = { 0: "Charged/Convicted", 1: "Settled Civilly", 2: "Named/Investigated", 3: "No Consequences" };

let allPeople = [];
let allEvents = [];
let activeNetworkTier = "all";
let activeTimelineTier = "all";
let networkFilter = "all";
let hoveredNode = null;
let nodes = [];
let rafId = null;

// ─────────────────────────────────────────────────────────────────
// SECTION ROUTING
// ─────────────────────────────────────────────────────────────────

function showSection(id) {
  document.querySelectorAll(".section-hero, .section-inner").forEach(s => s.classList.remove("active"));
  const el = document.getElementById(id);
  if (el) el.classList.add("active");

  document.querySelectorAll(".nav-link").forEach(l => {
    l.classList.toggle("active", l.dataset.section === id);
  });

  const fab = document.getElementById("chat-fab");
  if (fab) fab.classList.toggle("hidden", id === "hero");

  if (id === "network" && allPeople.length > 0 && nodes.length === 0) {
    setTimeout(initNetwork, 100);
  }

  window.scrollTo(0, 0);
}

window.showSection = showSection;

document.querySelectorAll(".nav-link").forEach(btn => {
  btn.addEventListener("click", () => showSection(btn.dataset.section));
});

document.getElementById("nav-ai-btn")?.addEventListener("click", openChat);
document.getElementById("chat-fab")?.addEventListener("click", openChat);

// ─────────────────────────────────────────────────────────────────
// DATA LOADING
// ─────────────────────────────────────────────────────────────────

async function fetchPeople() {
  try {
    const r = await fetch(`${API}/people`);
    allPeople = await r.json();
  } catch (e) {
    allPeople = [];
    console.warn("Could not fetch people:", e);
  }
}

async function fetchTimeline() {
  try {
    const r = await fetch(`${API}/timeline`);
    allEvents = await r.json();
  } catch (e) {
    allEvents = [];
  }
}

async function fetchPersonProfile(name) {
  try {
    const r = await fetch(`${API}/person/${encodeURIComponent(name)}`);
    return await r.json();
  } catch (e) {
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────
// HERO STATS
// ─────────────────────────────────────────────────────────────────

function populateHeroStats() {
  const counts = { 0: 0, 1: 0, 2: 0, 3: 0 };
  allPeople.forEach(p => { if (counts[p.tier] !== undefined) counts[p.tier]++; });
  document.getElementById("stat-convicted").textContent = counts[0];
  document.getElementById("stat-settled").textContent   = counts[1];
  document.getElementById("stat-named").textContent     = counts[2];
  document.getElementById("stat-none").textContent      = counts[3];

  // Findings: % with no meaningful consequences (tier 2 + 3)
  const noAction = counts[2] + counts[3];
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const pct = total > 0 ? Math.round((noAction / total) * 100) : 0;
  const el = document.getElementById("finding-pct");
  if (el) el.textContent = pct + "%";
}

// ─────────────────────────────────────────────────────────────────
// NETWORK — force-directed canvas graph
// ─────────────────────────────────────────────────────────────────

function initNetwork() {
  const canvas = document.getElementById("network-canvas");
  if (!canvas) return;
  const wrapper = canvas.parentElement;
  canvas.width  = wrapper.offsetWidth;
  canvas.height = wrapper.offsetHeight;
  const W = canvas.width, H = canvas.height;

  nodes = allPeople.map((p, i) => ({
    id: i,
    name: p.name,
    tier: p.tier,
    color: TIER_COLORS[p.tier] || "#6c757d",
    x: W / 2 + (Math.random() - 0.5) * W * 0.55,
    y: H / 2 + (Math.random() - 0.5) * H * 0.55,
    vx: 0, vy: 0,
    radius: p.tier === 0 ? 14 : p.tier === 1 ? 11 : 9,
    fixed: false,
  }));

  // Pin Epstein at center
  const epstein = nodes.find(n => n.name.toLowerCase().includes("epstein"));
  if (epstein) { epstein.x = W / 2; epstein.y = H / 2; epstein.radius = 18; epstein.fixed = true; }

  startPhysics();
}

function startPhysics() {
  if (rafId) cancelAnimationFrame(rafId);
  let ticks = 0;
  function step() {
    if (ticks < 200) { applyForces(); ticks++; }
    drawNetwork();
    rafId = requestAnimationFrame(step);
  }
  rafId = requestAnimationFrame(step);
}

function applyForces() {
  const canvas = document.getElementById("network-canvas");
  if (!canvas) return;
  const W = canvas.width, H = canvas.height;

  nodes.forEach(n => {
    if (n.fixed || !isVisible(n)) return;
    // Gravity toward center
    n.vx += (W / 2 - n.x) * 0.003;
    n.vy += (H / 2 - n.y) * 0.003;
    // Repulsion
    nodes.forEach(m => {
      if (m === n) return;
      const dx = n.x - m.x, dy = n.y - m.y;
      const d2 = dx * dx + dy * dy || 1;
      const dist = Math.sqrt(d2);
      const minD = n.radius + m.radius + 28;
      if (dist < minD) {
        const f = (minD - dist) / dist * 0.12;
        n.vx += dx * f; n.vy += dy * f;
      }
    });
    n.vx *= 0.85; n.vy *= 0.85;
    n.x = Math.max(n.radius, Math.min(W - n.radius, n.x + n.vx));
    n.y = Math.max(n.radius, Math.min(H - n.radius, n.y + n.vy));
  });
}

function isVisible(n) {
  return networkFilter === "all" || String(n.tier) === networkFilter;
}

function drawNetwork() {
  const canvas = document.getElementById("network-canvas");
  if (!canvas || nodes.length === 0) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const epstein = nodes.find(n => n.name.toLowerCase().includes("epstein"));

  // Edges from Epstein to all
  if (epstein) {
    nodes.forEach(n => {
      if (n === epstein) return;
      const alpha = isVisible(n) ? 0.05 : 0.01;
      ctx.beginPath();
      ctx.moveTo(epstein.x, epstein.y);
      ctx.lineTo(n.x, n.y);
      ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });
  }

  // Nodes
  nodes.forEach(n => {
    const visible = isVisible(n);
    const isHov = n === hoveredNode;
    ctx.save();
    ctx.globalAlpha = visible ? (isHov ? 1 : 0.85) : 0.1;

    if (isHov) { ctx.shadowColor = n.color; ctx.shadowBlur = 22; }

    // Fill
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.fill();

    // Ring
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius + 3, 0, Math.PI * 2);
    ctx.strokeStyle = n.color;
    ctx.globalAlpha = visible ? (isHov ? 0.5 : 0.12) : 0.05;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label
    if (visible && (isHov || n.radius >= 14)) {
      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#e8e8e8";
      ctx.font = `${isHov ? 600 : 500} 11px 'Space Grotesk', sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      const lastName = n.name.split(" ").slice(-1)[0];
      ctx.fillText(lastName, n.x, n.y - n.radius - 5);
    }

    ctx.restore();
  });
}

// Mouse events
const canvasEl = document.getElementById("network-canvas");
const tooltipEl = document.getElementById("network-tooltip");

canvasEl?.addEventListener("mousemove", e => {
  if (!canvasEl) return;
  const rect = canvasEl.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvasEl.width / rect.width);
  const my = (e.clientY - rect.top) * (canvasEl.height / rect.height);

  hoveredNode = null;
  for (const n of nodes) {
    const dx = n.x - mx, dy = n.y - my;
    if (Math.sqrt(dx * dx + dy * dy) < n.radius + 8) { hoveredNode = n; break; }
  }
  if (hoveredNode && tooltipEl) {
    tooltipEl.style.display = "block";
    tooltipEl.style.left = (e.clientX - canvasEl.getBoundingClientRect().left + 14) + "px";
    tooltipEl.style.top  = (e.clientY - canvasEl.getBoundingClientRect().top  - 14) + "px";
    tooltipEl.innerHTML  = `<strong style="color:${hoveredNode.color}">${hoveredNode.name}</strong><br>${TIER_LABELS[hoveredNode.tier]}`;
    canvasEl.style.cursor = "pointer";
  } else {
    if (tooltipEl) tooltipEl.style.display = "none";
    canvasEl.style.cursor = "grab";
  }
});

canvasEl?.addEventListener("click", () => {
  if (hoveredNode) openDrawer(hoveredNode.name);
});

canvasEl?.addEventListener("mouseleave", () => {
  if (tooltipEl) tooltipEl.style.display = "none";
  hoveredNode = null;
});

// Filter chips
document.querySelectorAll(".fchip").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".fchip").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    networkFilter = btn.dataset.tier;
    activeNetworkTier = networkFilter;
    renderPeopleStrip();
  });
});

// Search
document.getElementById("network-search")?.addEventListener("input", e => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll(".person-card").forEach(card => {
    const name = card.querySelector(".person-card-name")?.textContent.toLowerCase() || "";
    card.classList.toggle("hidden", q.length > 0 && !name.includes(q));
  });
});

// ─────────────────────────────────────────────────────────────────
// PEOPLE STRIP
// ─────────────────────────────────────────────────────────────────

function renderPeopleStrip() {
  const strip = document.getElementById("people-strip");
  if (!strip) return;
  const filtered = activeNetworkTier === "all"
    ? allPeople
    : allPeople.filter(p => String(p.tier) === activeNetworkTier);

  strip.innerHTML = "";
  filtered.forEach(person => {
    const card = document.createElement("div");
    card.className = "person-card";
    card.style.setProperty("--tier-color", TIER_COLORS[person.tier] || "#6c757d");
    card.innerHTML = `
      <div class="person-card-tier">${TIER_NAMES[person.tier] || "UNKNOWN"}</div>
      <div class="person-card-name">${person.name}</div>
      <div class="person-card-consequence">${TIER_LABELS[person.tier] || ""}</div>
    `;
    card.addEventListener("click", () => openDrawer(person.name));
    strip.appendChild(card);
  });
}

// ─────────────────────────────────────────────────────────────────
// TIMELINE
// ─────────────────────────────────────────────────────────────────

function renderTimeline(tier = "all") {
  const track = document.getElementById("timeline-track");
  if (!track) return;
  const events = tier === "all" ? allEvents : allEvents.filter(e => String(e.tier) === tier);
  track.innerHTML = "";

  events.forEach((ev, i) => {
    const color = TIER_COLORS[ev.tier] || "#6c757d";
    const div = document.createElement("div");
    div.className = "tl-event";
    div.style.setProperty("--event-color", color);
    div.style.animationDelay = `${i * 50}ms`;
    div.innerHTML = `
      <div class="tl-date">${formatDate(ev.date)}</div>
      <div class="tl-person">${ev.person}</div>
      <div class="tl-text">${ev.event}</div>
      <div class="tl-tier-badge">${TIER_LABELS[ev.tier] || ""}</div>
    `;
    track.appendChild(div);
  });

  if (events.length === 0) {
    track.innerHTML = `<p style="color:var(--text-3);font-family:var(--mono);font-size:12px;padding:20px 0">No events for this tier.</p>`;
  }
}

document.querySelectorAll(".tl-fchip").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tl-fchip").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeTimelineTier = btn.dataset.tier;
    renderTimeline(activeTimelineTier);
  });
});

function formatDate(d) {
  if (!d) return "";
  try {
    const dt = new Date(d + "T12:00:00");
    const opts = { year: "numeric", month: "long" };
    if (!d.endsWith("-01")) opts.day = "numeric";
    return dt.toLocaleDateString("en-US", opts);
  } catch (e) { return d; }
}

// ─────────────────────────────────────────────────────────────────
// PROFILE DRAWER
// ─────────────────────────────────────────────────────────────────

async function openDrawer(name) {
  const overlay = document.getElementById("drawer-overlay");
  const body = document.getElementById("drawer-body");
  if (!overlay || !body) return;

  overlay.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  body.innerHTML = `<div class="loading-dots-wrap"><span></span><span></span><span></span></div>`;

  const data = await fetchPersonProfile(name);
  if (!data) {
    body.innerHTML = `<p style="color:var(--text-3)">Could not load profile.</p>`;
    return;
  }

  const tier = data.tier ?? 3;
  const color = TIER_COLORS[tier] || "#6c757d";
  const tierLabel = TIER_LABELS[tier] || "Unknown";
  const events = data.timeline_events || [];

  // Hex to rgb for rgba usage
  const hexToRgb = hex => {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `${r},${g},${b}`;
  };
  const rgb = hexToRgb(color);

  const eventsHtml = events.length > 0 ? `
    <div class="profile-section-label">RELATED EVENTS</div>
    <div class="profile-events">
      ${events.map(e => `
        <div class="profile-event">
          <div class="profile-event-date">${formatDate(e.date)}</div>
          <div class="profile-event-text">${e.event}</div>
        </div>
      `).join("")}
    </div>
  ` : "";

  const tierDesc = {
    0: "Faced criminal charges or conviction.",
    1: "Settled out of court civilly — avoided criminal liability.",
    2: "Named in documents or investigated. No charges were ever filed.",
    3: "No known legal consequences despite documented associations.",
  }[tier] || "";

  body.innerHTML = `
    <div class="profile-tier-badge" style="color:${color};border-color:${color}">${TIER_NAMES[tier]}</div>
    <div class="profile-name">${data.name}</div>
    <div class="profile-consequence">${tierLabel}</div>

    ${data.bio ? `
      <div class="profile-section-label">BACKGROUND</div>
      <p style="font-size:13px;color:var(--text-2);line-height:1.65;margin-bottom:28px">${data.bio}</p>
    ` : ""}

    ${eventsHtml}

    <div class="profile-section-label">CONSEQUENCE VERDICT</div>
    <div style="background:rgba(${rgb},0.07);border:1px solid rgba(${rgb},0.25);padding:16px 20px;margin-bottom:20px">
      <div style="color:${color};font-family:var(--mono);font-size:10px;letter-spacing:0.12em;margin-bottom:6px">TIER ${tier}</div>
      <strong style="color:${color};font-size:15px">${tierLabel}</strong>
      <p style="color:var(--text-2);font-size:13px;margin-top:8px;line-height:1.6">${tierDesc}</p>
    </div>

    <button class="ask-ai-btn" onclick="window._askAboutPerson('${data.name.replace(/'/g, "\\'")}')">
      Ask AI about ${data.name.split(" ").slice(-1)[0]} →
    </button>
  `;
}

window._askAboutPerson = function(name) {
  closeDrawer();
  openChat();
  setTimeout(() => {
    const input = document.getElementById("chat-input");
    if (input) { input.value = `What do the documents say about ${name}?`; input.focus(); }
  }, 300);
};

document.getElementById("drawer-close")?.addEventListener("click", closeDrawer);
document.getElementById("drawer-overlay")?.addEventListener("click", e => {
  if (e.target === document.getElementById("drawer-overlay")) closeDrawer();
});

function closeDrawer() {
  document.getElementById("drawer-overlay")?.classList.add("hidden");
  document.body.style.overflow = "";
}

// ─────────────────────────────────────────────────────────────────
// CHAT HELPERS
// ─────────────────────────────────────────────────────────────────

function openChat() {
  document.getElementById("chat-overlay")?.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  setTimeout(() => document.getElementById("chat-input")?.focus(), 150);
}
window.openChat = openChat;

function sendPrompt(text) {
  openChat();
  setTimeout(() => {
    const input = document.getElementById("chat-input");
    if (input) {
      input.value = text;
      input.dispatchEvent(new Event("input"));
      document.getElementById("chat-send-btn")?.click();
    }
  }, 200);
}
window.sendPrompt = sendPrompt;

// ─────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────

(async function init() {
  await Promise.all([fetchPeople(), fetchTimeline()]);
  populateHeroStats();
  renderPeopleStrip();
  renderTimeline("all");
  // Network graph initializes lazily on first visit to section
})();
