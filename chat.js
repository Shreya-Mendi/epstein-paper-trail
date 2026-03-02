/**
 * chat.js — Paper Trail AI chat panel.
 *
 * Sends POST /chat to the FastAPI RAG backend.
 * Renders answers with markdown-like formatting and EFTA document citation chips.
 */

const CHAT_API = "http://127.0.0.1:8000/chat";

let chatReady = false;

// ─────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const sendBtn = document.getElementById("chat-send-btn");
  const input   = document.getElementById("chat-input");
  const closeBtn = document.getElementById("chat-close-btn");
  const overlay  = document.getElementById("chat-overlay");

  if (!sendBtn || !input) return;

  sendBtn.addEventListener("click", handleSend);
  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });

  closeBtn?.addEventListener("click", closeChat);
  overlay?.addEventListener("click", e => {
    if (e.target === overlay) closeChat();
  });

  // Show welcome message
  appendWelcome();
  chatReady = true;
});

function closeChat() {
  document.getElementById("chat-overlay")?.classList.add("hidden");
  document.body.style.overflow = "";
}

function appendWelcome() {
  const msgs = document.getElementById("chat-msgs");
  if (!msgs) return;
  const div = document.createElement("div");
  div.className = "bubble bubble-ai";
  div.innerHTML = `
    <div class="bubble-label">PAPER TRAIL AI</div>
    <div class="bubble-text">
      <p>Ask me anything about the Epstein case. I search <strong>961 DOJ files</strong> and cite the specific document behind every answer.</p>
      <p style="color:var(--text-3);font-size:12px;margin-top:8px">Try: "What do the DOJ files say about Ghislaine Maxwell?" or "Who was charged?"</p>
    </div>
  `;
  msgs.appendChild(div);
}

// ─────────────────────────────────────────────────────────────────
// SEND
// ─────────────────────────────────────────────────────────────────

async function handleSend() {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send-btn");
  if (!input) return;

  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  input.style.height = "auto";

  // Hide prompt chips after first use
  const prompts = document.getElementById("chat-prompts");
  if (prompts) prompts.style.display = "none";

  appendUserBubble(query);
  const thinkingEl = appendThinking();
  if (sendBtn) sendBtn.disabled = true;

  try {
    const res = await fetch(CHAT_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    thinkingEl.remove();
    appendAIBubble(data.answer, data.sources || []);
  } catch (err) {
    thinkingEl.remove();
    appendAIBubble(
      "Could not reach the Paper Trail backend. Make sure it is running:\n\n```\nPYTHONPATH=app/backend uvicorn app.backend.main:app --reload\n```",
      []
    );
  } finally {
    if (sendBtn) sendBtn.disabled = false;
    input.focus();
  }
}

// ─────────────────────────────────────────────────────────────────
// BUBBLE RENDERING
// ─────────────────────────────────────────────────────────────────

function appendUserBubble(text) {
  const msgs = document.getElementById("chat-msgs");
  if (!msgs) return;
  const div = document.createElement("div");
  div.className = "bubble bubble-user";
  div.innerHTML = `
    <div class="bubble-label" style="text-align:right">YOU</div>
    <div class="bubble-text">${escapeHtml(text)}</div>
  `;
  msgs.appendChild(div);
  scrollToBottom();
}

function appendThinking() {
  const msgs = document.getElementById("chat-msgs");
  if (!msgs) return document.createElement("div");
  const div = document.createElement("div");
  div.className = "bubble bubble-ai";
  div.innerHTML = `
    <div class="bubble-label">PAPER TRAIL AI</div>
    <div class="thinking"><span></span><span></span><span></span></div>
  `;
  msgs.appendChild(div);
  scrollToBottom();
  return div;
}

function appendAIBubble(text, sources) {
  const msgs = document.getElementById("chat-msgs");
  if (!msgs) return;
  const div = document.createElement("div");
  div.className = "bubble bubble-ai";

  const sourcesHtml = buildSourcesHtml(sources);

  div.innerHTML = `
    <div class="bubble-label">PAPER TRAIL AI</div>
    <div class="bubble-text">${renderMarkdown(text)}</div>
    ${sourcesHtml ? `<div class="sources-row">${sourcesHtml}</div>` : ""}
  `;
  msgs.appendChild(div);
  scrollToBottom();
}

function buildSourcesHtml(sources) {
  if (!sources || sources.length === 0) return "";
  return sources.map(src => {
    if (src.efta_id) {
      const url = src.url || `https://www.justice.gov/epstein/files/DataSet%20${src.dataset}/${src.efta_id}.pdf`;
      const quoteText = src.quote ? `"${src.quote.slice(0, 72)}…"` : "";
      return `
        <a class="source-chip source-chip-doj" href="${url}" target="_blank" rel="noopener" title="Open DOJ document">
          <span class="source-chip-id">${src.efta_id}</span>
          <span class="source-chip-ds">Dataset ${src.dataset}</span>
          ${quoteText ? `<span class="source-chip-quote">${escapeHtml(quoteText)}</span>` : ""}
        </a>
      `;
    }
    const label = src.source === "wikipedia" ? "Wikipedia"
      : src.source === "court" ? "Court Docs"
      : src.source === "epstein_overview" ? "EpsteinOverview"
      : src.source === "doj_press" ? "DOJ Press"
      : `[${src.index}] ${src.source}`;
    const href = src.url || "#";
    return `<a class="source-chip" href="${href}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
  }).join("");
}

// ─────────────────────────────────────────────────────────────────
// MARKDOWN RENDERER (minimal)
// ─────────────────────────────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^---+$/gm, "<hr>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^\s*[-*]\s+(.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^(?!<[hup])(.+)/, "<p>$1")
    .replace(/([^>])$/, "$1</p>");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scrollToBottom() {
  const msgs = document.getElementById("chat-msgs");
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}
