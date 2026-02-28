/**
 * timeline.js — Horizontal scrollable timeline and people grid for Paper Trail.
 *
 * Features:
 *  - Fetches events from GET /timeline and people from GET /people
 *  - Renders a horizontally scrollable timeline track
 *  - Filter buttons to show only specific consequence tiers
 *  - Clicking a card or person opens a profile panel
 */

const API_BASE = "http://127.0.0.1:8000";

const TIER_LABELS = {
  0: "Charged/Convicted",
  1: "Settled Civilly",
  2: "Named/Investigated Only",
  3: "No Consequences",
};

const TIER_COLORS = {
  0: "#e63946",
  1: "#f4a261",
  2: "#e9c46a",
  3: "#6c757d",
};

let allEvents = [];
let allPeople = [];
let activeFilter = "all";

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

async function fetchTimeline() {
  try {
    const res = await fetch(`${API_BASE}/timeline`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("Failed to fetch timeline:", err);
    return getFallbackEvents();
  }
}

async function fetchPeople() {
  try {
    const res = await fetch(`${API_BASE}/people`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("Failed to fetch people:", err);
    return [];
  }
}

async function fetchPerson(name) {
  try {
    const res = await fetch(`${API_BASE}/person/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("Failed to fetch person:", err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Fallback data (used when backend is not running)
// ---------------------------------------------------------------------------

function getFallbackEvents() {
  return [
    { date: "2008-06-30", person: "Jeffrey Epstein", event: "Epstein pleads guilty to Florida state charges.", tier: 0 },
    { date: "2019-07-06", person: "Jeffrey Epstein", event: "Epstein arrested on federal sex trafficking charges.", tier: 0 },
    { date: "2019-07-12", person: "Alexander Acosta", event: "Acosta resigns as Secretary of Labor.", tier: 2 },
    { date: "2019-08-10", person: "Jeffrey Epstein", event: "Epstein found dead in MCC New York.", tier: 0 },
    { date: "2020-07-02", person: "Ghislaine Maxwell", event: "Maxwell arrested in New Hampshire.", tier: 0 },
    { date: "2021-12-29", person: "Ghislaine Maxwell", event: "Maxwell convicted on five federal counts.", tier: 0 },
    { date: "2022-02-15", person: "Prince Andrew", event: "Prince Andrew settles civil lawsuit.", tier: 1 },
    { date: "2022-06-28", person: "Ghislaine Maxwell", event: "Maxwell sentenced to 20 years.", tier: 0 },
    { date: "2024-01-03", person: "Multiple", event: "Court unseals documents naming associates.", tier: 2 },
  ];
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function formatDate(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr + "T00:00:00Z");
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

function tierBadgeHTML(tier) {
  const color = TIER_COLORS[tier] ?? "#888";
  const label = TIER_LABELS[tier] ?? "Unknown";
  return `<span class="tier-badge" style="background:${color}22;color:${color};border:1px solid ${color}44">${label}</span>`;
}

function renderTimeline(events) {
  const track = document.getElementById("timeline-track");
  if (!track) return;
  track.innerHTML = "";

  const filtered = activeFilter === "all"
    ? events
    : events.filter(e => String(e.tier) === String(activeFilter));

  if (filtered.length === 0) {
    track.innerHTML = `<p style="color:var(--text-muted);padding:20px">No events match this filter.</p>`;
    return;
  }

  filtered.forEach(event => {
    const color = TIER_COLORS[event.tier] ?? "#888";
    const card = document.createElement("div");
    card.className = "timeline-card";
    card.style.setProperty("--tier-color", color);
    card.innerHTML = `
      <div class="card-date">${formatDate(event.date)}</div>
      <div class="card-person">${escapeHTML(event.person)}</div>
      <div class="card-event">${escapeHTML(event.event)}</div>
      ${tierBadgeHTML(event.tier)}
    `;
    card.addEventListener("click", () => openProfilePanel(event.person));
    track.appendChild(card);
  });
}

function renderPeopleGrid(people) {
  const grid = document.getElementById("people-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const filtered = activeFilter === "all"
    ? people
    : people.filter(p => String(p.tier) === String(activeFilter));

  if (filtered.length === 0) {
    grid.innerHTML = `<p style="color:var(--text-muted)">No individuals match this filter.</p>`;
    return;
  }

  filtered.forEach(person => {
    const color = TIER_COLORS[person.tier] ?? "#888";
    const initials = person.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();
    const card = document.createElement("div");
    card.className = "person-card";
    card.style.setProperty("--tier-color", color);
    card.innerHTML = `
      <div class="person-photo-placeholder" style="border-color:${color}">${initials}</div>
      <div class="person-name">${escapeHTML(person.name)}</div>
      ${tierBadgeHTML(person.tier)}
      <button class="person-ask-btn" style="--tier-color:${color}">
        Ask about ${escapeHTML(person.name.split(" ")[0])} →
      </button>
    `;
    card.addEventListener("click", () => openProfilePanel(person.name));
    card.querySelector(".person-ask-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      window.chatModule?.prefillQuery(`What happened to ${person.name}?`);
    });
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Profile panel
// ---------------------------------------------------------------------------

async function openProfilePanel(name) {
  if (!name || name === "Multiple") return;

  const panel = document.getElementById("profile-panel");
  if (!panel) return;
  panel.classList.remove("hidden");

  const content = document.getElementById("profile-panel-content");
  content.innerHTML = `<p style="color:var(--text-muted)">Loading…</p>`;

  const data = await fetchPerson(name);
  if (!data) {
    content.innerHTML = `<p style="color:var(--tier-0)">Could not load profile.</p>`;
    return;
  }

  const color = TIER_COLORS[data.tier] ?? "#888";
  const initials = data.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();
  const eventsHTML = (data.timeline_events || []).map(e => `
    <div class="profile-event">
      <div class="event-date">${formatDate(e.date)}</div>
      <div class="event-text">${escapeHTML(e.event)}</div>
    </div>
  `).join("") || `<p style="color:var(--text-muted);font-size:12px">No events on record.</p>`;

  content.innerHTML = `
    <div class="profile-header">
      <div class="person-photo-placeholder" style="width:56px;height:56px;font-size:22px;border-color:${color}">${initials}</div>
      <div>
        <div class="profile-name">${escapeHTML(data.name)}</div>
        ${tierBadgeHTML(data.tier)}
      </div>
    </div>
    <div>
      <h3 style="font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--text-muted);margin-bottom:12px">Timeline</h3>
      <div class="profile-events">${eventsHTML}</div>
    </div>
    <button onclick="window.chatModule?.prefillQuery('What happened to ${escapeJS(data.name)}?')"
      style="padding:10px 16px;background:${color}22;border:1px solid ${color}44;color:${color};
             font-family:var(--font-mono);font-size:12px;border-radius:4px;cursor:pointer;text-align:left">
      Ask about ${escapeHTML(data.name.split(" ")[0])} →
    </button>
  `;
}

function closeProfilePanel() {
  const panel = document.getElementById("profile-panel");
  if (panel) panel.classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Filter buttons
// ---------------------------------------------------------------------------

function setupFilters() {
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      activeFilter = btn.dataset.tier ?? "all";
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderTimeline(allEvents);
      renderPeopleGrid(allPeople);
    });
  });
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHTML(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escapeJS(str) {
  if (!str) return "";
  return str.replace(/'/g, "\\'");
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function initTimeline() {
  setupFilters();

  [allEvents, allPeople] = await Promise.all([fetchTimeline(), fetchPeople()]);

  renderTimeline(allEvents);
  renderPeopleGrid(allPeople);

  // Profile panel close button
  const closeBtn = document.getElementById("profile-close-btn");
  if (closeBtn) closeBtn.addEventListener("click", closeProfilePanel);

  // Close profile on backdrop click
  const panel = document.getElementById("profile-panel");
  if (panel) {
    panel.addEventListener("click", e => {
      if (e.target === panel) closeProfilePanel();
    });
  }
}

document.addEventListener("DOMContentLoaded", initTimeline);
