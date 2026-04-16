// Simple vanilla-JS client for the RAG backend.
const API = "";  // same origin

const $ = (sel) => document.querySelector(sel);
const chatLog = $("#chat-log");
const chatForm = $("#chat-form");
const questionEl = $("#question");
const sendBtn = $("#send-btn");

const fileInput = $("#file-input");
const dropzone = $("#dropzone");
const uploadBtn = $("#upload-btn");
const uploadLog = $("#upload-log");
const resetBtn = $("#reset-btn");
const chunkCountEl = $("#chunk-count");
const statusEl = $("#status");
const statusText = $("#status-text");

let selectedFiles = [];

// ─────────────────── Health / status ───────────────────
async function refreshHealth() {
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    chunkCountEl.textContent = data.indexed_chunks.toLocaleString();
    statusEl.classList.remove("err");
    statusEl.classList.add("ok");
    statusText.textContent = `ready · ${data.chat_model}`;
  } catch (e) {
    statusEl.classList.remove("ok");
    statusEl.classList.add("err");
    statusText.textContent = "backend unreachable";
  }
}

// ─────────────────── Upload ───────────────────
function updateSelected(files) {
  selectedFiles = Array.from(files).filter((f) =>
    f.name.toLowerCase().endsWith(".pdf")
  );
  uploadBtn.disabled = selectedFiles.length === 0;
  if (selectedFiles.length) {
    uploadBtn.textContent = `Ingest ${selectedFiles.length} file${selectedFiles.length > 1 ? "s" : ""}`;
  } else {
    uploadBtn.textContent = "Ingest selected";
  }
}

fileInput.addEventListener("change", (e) => updateSelected(e.target.files));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  updateSelected(e.dataTransfer.files);
});

function logUpload(msg, cls = "") {
  const entry = document.createElement("div");
  entry.className = `entry ${cls}`;
  entry.textContent = msg;
  uploadLog.prepend(entry);
}

uploadBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  const fd = new FormData();
  for (const f of selectedFiles) fd.append("files", f);

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Ingesting...";
  logUpload(`Uploading ${selectedFiles.length} file(s)...`);

  try {
    const r = await fetch(`${API}/ingest`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    for (const f of data.files) {
      logUpload(
        `${f.filename} — ${f.num_pages} pages, ${f.num_chunks} chunks`,
        "ok"
      );
    }
    selectedFiles = [];
    fileInput.value = "";
    uploadBtn.textContent = "Ingest selected";
    await refreshHealth();
  } catch (err) {
    logUpload(`Failed: ${err.message}`, "err");
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Ingest selected";
  }
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Delete the entire index and all uploaded PDFs?")) return;
  await fetch(`${API}/ingest`, { method: "DELETE" });
  logUpload("Index reset.", "ok");
  await refreshHealth();
});

// ─────────────────── Chat ───────────────────
function addMessage(role, content, extras = {}) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (extras.intent) {
    const tag = document.createElement("span");
    tag.className = `intent-tag ${extras.intent}`;
    tag.textContent = extras.intent.replace("_", " ");
    bubble.appendChild(tag);
  }

  const p = document.createElement("span");
  p.innerHTML = content;
  bubble.appendChild(p);

  if (extras.sources && extras.sources.length) {
    const det = document.createElement("details");
    det.className = "sources";
    const sum = document.createElement("summary");
    sum.textContent = `${extras.sources.length} source passage${extras.sources.length > 1 ? "s" : ""}`;
    det.appendChild(sum);
    for (let i = 0; i < extras.sources.length; i++) {
      const s = extras.sources[i];
      const item = document.createElement("div");
      item.className = "source-item";
      item.innerHTML = `
        <div class="meta">[${i + 1}] ${escapeHtml(s.filename)} · page ${s.page} · score ${s.score.toFixed(3)}</div>
        <div class="text">${escapeHtml(s.text)}</div>
      `;
      det.appendChild(item);
    }
    bubble.appendChild(det);
  }

  msg.appendChild(bubble);
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

function escapeHtml(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatAnswer(text) {
  // Escape, then linkify [n] citations with a soft highlight.
  const safe = escapeHtml(text);
  return safe.replace(/\[(\d+)\]/g, '<strong style="color:var(--accent)">[$1]</strong>');
}

questionEl.addEventListener("input", () => {
  questionEl.style.height = "auto";
  questionEl.style.height = Math.min(questionEl.scrollHeight, 200) + "px";
});
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (!q) return;

  addMessage("user", escapeHtml(q));
  questionEl.value = "";
  questionEl.style.height = "auto";
  sendBtn.disabled = true;

  const placeholder = addMessage(
    "assistant",
    '<span class="typing"></span>'
  );

  try {
    const r = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, include_sources: true }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();

    placeholder.innerHTML = "";
    const tag = document.createElement("span");
    tag.className = `intent-tag ${data.intent}`;
    tag.textContent = data.intent.replace("_", " ");
    placeholder.appendChild(tag);

    const text = document.createElement("span");
    text.innerHTML = formatAnswer(data.answer);
    placeholder.appendChild(text);

    if (data.sources && data.sources.length) {
      const det = document.createElement("details");
      det.className = "sources";
      const sum = document.createElement("summary");
      sum.textContent = `${data.sources.length} source passage${data.sources.length > 1 ? "s" : ""}`;
      det.appendChild(sum);
      for (let i = 0; i < data.sources.length; i++) {
        const s = data.sources[i];
        const item = document.createElement("div");
        item.className = "source-item";
        item.innerHTML = `
          <div class="meta">[${i + 1}] ${escapeHtml(s.filename)} · page ${s.page} · score ${s.score.toFixed(3)}</div>
          <div class="text">${escapeHtml(s.text)}</div>
        `;
        det.appendChild(item);
      }
      placeholder.appendChild(det);
    }
  } catch (err) {
    placeholder.innerHTML =
      `<span style="color:var(--err)">Error: ${escapeHtml(err.message || String(err))}</span>`;
  } finally {
    sendBtn.disabled = false;
    questionEl.focus();
  }
});

// ─────────────────── Init ───────────────────
refreshHealth();
setInterval(refreshHealth, 15000);
