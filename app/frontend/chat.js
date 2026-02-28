/**
 * chat.js — Slide-in chatbot panel for Paper Trail.
 *
 * Features:
 *  - Slide-in panel from the right side of the screen
 *  - Sends POST /chat requests to the FastAPI backend
 *  - Streams-style progressive rendering of responses
 *  - Source document citations rendered as clickable chips
 *  - Starter prompt suggestions shown on load
 *  - prefillQuery() exposed on window.chatModule for cross-module calls
 */

const CHAT_API = "http://127.0.0.1:8000/chat";

const STARTER_PROMPTS = [
  "Who faced criminal charges?",
  "What happened to Ghislaine Maxwell?",
  "What was the 2008 plea deal?",
  "Which politicians were named in documents?",
  "Who settled civil lawsuits?",
];

let chatOpen = false;
let isLoading = false;

// ---------------------------------------------------------------------------
// DOM refs (populated on DOMContentLoaded)
// ---------------------------------------------------------------------------

let chatPanel, chatMessages, chatInput, chatSendBtn, chatToggleBtn;

// ---------------------------------------------------------------------------
// Panel open / close
// ---------------------------------------------------------------------------

function openChat() {
  chatOpen = true;
  chatPanel.classList.add("open");
  chatToggleBtn.innerHTML = "✕";
  chatInput.focus();
}

function closeChat() {
  chatOpen = false;
  chatPanel.classList.remove("open");
  chatToggleBtn.innerHTML = "💬";
}

function toggleChat() {
  if (chatOpen) closeChat();
  else openChat();
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function appendBubble(role, text, sources = []) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;

  // Escape and linkify newlines
  const escaped = escapeHTML(text).replace(/\n/g, "<br>");
  bubble.innerHTML = escaped;

  if (sources.length > 0) {
    const sourceRow = document.createElement("div");
    sourceRow.className = "chat-sources";
    sources.forEach(src => {
      const chip = document.createElement("a");
      chip.className = "source-chip";
      chip.textContent = `[${src.index}] ${src.source}`;
      if (src.url) {
        chip.href = src.url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
      } else {
        chip.href = "#";
      }
      chip.title = src.url || src.source;
      sourceRow.appendChild(chip);
    });
    bubble.appendChild(sourceRow);
  }

  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

function showTypingIndicator() {
  const indicator = document.createElement("div");
  indicator.className = "chat-typing";
  indicator.id = "chat-typing";
  indicator.innerHTML = "<span></span><span></span><span></span>";
  chatMessages.appendChild(indicator);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
  const indicator = document.getElementById("chat-typing");
  if (indicator) indicator.remove();
}

// ---------------------------------------------------------------------------
// Send a message
// ---------------------------------------------------------------------------

async function sendMessage(query) {
  if (!query.trim() || isLoading) return;

  // Hide starter prompts after first message
  const starters = document.getElementById("starter-prompts");
  if (starters) starters.remove();

  appendBubble("user", query);
  chatInput.value = "";
  chatInput.style.height = "auto";

  isLoading = true;
  chatSendBtn.disabled = true;
  showTypingIndicator();

  try {
    const res = await fetch(CHAT_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    removeTypingIndicator();

    if (!res.ok) {
      const err = await res.text();
      appendBubble("assistant", `Error ${res.status}: ${err}`);
      return;
    }

    const data = await res.json();
    appendBubble("assistant", data.answer || "(No response)", data.sources || []);
  } catch (err) {
    removeTypingIndicator();
    appendBubble(
      "assistant",
      "⚠ Could not reach the Paper Trail backend. Make sure it's running:\n\nuvicorn app.backend.main:app --reload"
    );
    console.error("Chat request failed:", err);
  } finally {
    isLoading = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
  }
}

// ---------------------------------------------------------------------------
// Starter prompts
// ---------------------------------------------------------------------------

function renderStarterPrompts() {
  const container = document.createElement("div");
  container.className = "chat-starter-prompts";
  container.id = "starter-prompts";

  const intro = document.createElement("p");
  intro.style.cssText = "font-size:12px;color:var(--text-muted);margin-bottom:6px";
  intro.textContent = "Try asking:";
  container.appendChild(intro);

  STARTER_PROMPTS.forEach(prompt => {
    const btn = document.createElement("button");
    btn.className = "starter-prompt";
    btn.textContent = `"${prompt}"`;
    btn.addEventListener("click", () => {
      openChat();
      sendMessage(prompt);
    });
    container.appendChild(btn);
  });

  chatMessages.appendChild(container);
}

// ---------------------------------------------------------------------------
// Public API — exposed as window.chatModule
// ---------------------------------------------------------------------------

function prefillQuery(text) {
  if (!chatOpen) openChat();
  chatInput.value = text;
  chatInput.focus();
  chatInput.dispatchEvent(new Event("input"));
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHTML(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  chatPanel = document.getElementById("chat-panel");
  chatMessages = document.getElementById("chat-messages");
  chatInput = document.getElementById("chat-input");
  chatSendBtn = document.getElementById("chat-send-btn");
  chatToggleBtn = document.getElementById("chat-toggle-btn");

  if (!chatPanel || !chatMessages || !chatInput || !chatSendBtn || !chatToggleBtn) {
    console.error("Chat DOM elements not found");
    return;
  }

  // Toggle button
  chatToggleBtn.addEventListener("click", toggleChat);

  // Close button inside panel header
  const closeBtn = document.getElementById("chat-close-btn");
  if (closeBtn) closeBtn.addEventListener("click", closeChat);

  // Send on button click
  chatSendBtn.addEventListener("click", () => sendMessage(chatInput.value));

  // Send on Enter (Shift+Enter = newline)
  chatInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(chatInput.value);
    }
  });

  // Auto-resize textarea
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });

  // Render starter prompts
  renderStarterPrompts();

  // Expose public API
  window.chatModule = { prefillQuery, openChat, closeChat };
});
