/**
 * RockRAG chat frontend
 * BM25 retrieval, multi-turn conversation, intent classification.
 * Paper-only RAG — no instruments, no data scripts.
 */

let CONFIG = {};
let CHUNKS = [];
let paperSearch = null;   // BM25 index: one document per paper
let PAPER_DOCS = [];      // parallel array to paperSearch entries
let WORKER_URL = "";

// History in Gemini format: [{role:"user"|"model", parts:[{text}]}]
let HISTORY = [];

// ── Mode ──────────────────────────────────────────────────────────────────────
let CURRENT_MODE = "ask"; // "ask" | "literature"

// ── Rate limiter ──────────────────────────────────────────────────────────────
const RL_MS  = 2500;
const RL_KEY = "rl_gemini_last";

// ── Context size cap ──────────────────────────────────────────────────────────
const MAX_CONTEXT_CHARS = 18000;

function rlGet() { return parseInt(sessionStorage.getItem(RL_KEY) || "0", 10); }
function rlSet() { sessionStorage.setItem(RL_KEY, String(Date.now())); }

async function throttle() {
  const wait = RL_MS - (Date.now() - rlGet());
  if (wait > 0) {
    setStatus(`Rate limiting — waiting ${(wait / 1000).toFixed(1)}s…`);
    await new Promise(res => setTimeout(res, wait));
    setStatus("");
  }
  rlSet();
}

function trimContext(chunks, maxChars = MAX_CONTEXT_CHARS) {
  let total = 0;
  const out = [];
  for (const c of chunks) {
    const len = (c.text || "").length;
    if (total + len > maxChars) break;
    out.push(c);
    total += len;
  }
  return out;
}

const MAX_HISTORY_CHARS = 6000;
function pruneHistory(history) {
  let total = history.reduce((sum, m) => sum + (m.parts?.[0]?.text?.length || 0), 0);
  const pruned = [...history];
  while (total > MAX_HISTORY_CHARS && pruned.length > 2) {
    const removed = pruned.splice(0, 2);
    total -= removed.reduce((s, m) => s + (m.parts?.[0]?.text?.length || 0), 0);
  }
  return pruned;
}

function filterByScore(results) {
  if (!results.length) return results;
  const threshold = results[0].score * 0.2;
  return results.filter(r => r.score >= threshold);
}

// ── Password helpers ──────────────────────────────────────────────────────────

function getStoredPassword() {
  return sessionStorage.getItem('rrg-pw') || '';
}

function authHeaders() {
  const pw = getStoredPassword();
  return pw
    ? { 'Content-Type': 'application/json', 'Authorization': `Bearer ${pw}` }
    : { 'Content-Type': 'application/json' };
}

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  initTheme();
  CONFIG     = await fetch("config.json").then(r => r.json());
  WORKER_URL = CONFIG.workerUrl;
  setStatus("Loading catalog…");
  CHUNKS = await fetch("public/chunks.json").then(r => r.json());
  setStatus("");

  // Paper index — one document per paper, text = all chunks concatenated (capped).
  paperSearch = new MiniSearch({
    idField: "paperIdx",
    fields: ["title", "text", "keywords"],
    storeFields: ["title"],
    searchOptions: { boost: { title: 3, keywords: 1.5 }, fuzzy: 0.2 },
  });
  const paperMap = new Map(); // baseId → aggregated doc
  CHUNKS.filter(c => c.type === "paper").forEach(c => {
    const base = c.id.replace(/::chunk\d+$/, "");
    if (!paperMap.has(base)) {
      paperMap.set(base, { ...c, id: base, text: "", _chunkTexts: [] });
    }
    paperMap.get(base)._chunkTexts.push(c.text || "");
  });
  PAPER_DOCS = [...paperMap.values()].map((doc, i) => ({
    ...doc,
    text: doc._chunkTexts.join(" ").slice(0, 3000),
    paperIdx: i,
  }));
  paperSearch.addAll(PAPER_DOCS);

  document.getElementById("inputForm").addEventListener("submit", onSubmit);
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      CURRENT_MODE = btn.dataset.mode;
      document.querySelectorAll(".mode-btn").forEach(b => b.classList.toggle("active", b === btn));
      const placeholders = {
        ask:        "e.g. What does Gassmann's equation predict about fluid substitution in sandstones?",
        literature: "e.g. What papers have been published on seismic attenuation in porous rocks?",
      };
      document.getElementById("queryInput").placeholder = placeholders[CURRENT_MODE] || "";
    });
  });
  document.querySelectorAll(".example-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.getElementById("queryInput").value = btn.dataset.query;
      document.getElementById("queryInput").focus();
    });
  });

  showWelcome();
}

// ── Welcome message ───────────────────────────────────────────────────────────

function showWelcome() {
  const text = "Welcome to RockRAG. I can answer questions about rock physics topics and surface relevant published research. What would you like to explore?";
  const contentEl = addMessage("assistant", "");
  contentEl.textContent = text;
  HISTORY.push({ role: "user",  parts: [{ text: "Hello!" }] });
  HISTORY.push({ role: "model", parts: [{ text: text }] });
}

// ── Retrieval ─────────────────────────────────────────────────────────────────

function paperBm25Search(query, k = 6) {
  const raw = paperSearch.search(query).slice(0, k);
  if (!raw.length) return [];
  const threshold = raw[0].score * 0.2;
  return raw.filter(r => r.score >= threshold).map(r => PAPER_DOCS[r.id]);
}

// ── Worker calls ──────────────────────────────────────────────────────────────

async function workerPost(path, body) {
  const r = await fetch(WORKER_URL + path, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    if (r.status === 429) sessionStorage.setItem(RL_KEY, String(Date.now() + 15000));
    if (r.status === 401) showPasswordGate();
    throw new Error(`Worker ${path} returned ${r.status}`);
  }
  return r.json();
}

async function streamChat(query, context, history = []) {
  await throttle();
  return fetch(WORKER_URL + "/chat", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ query, context, history }),
  });
}

// ── Markdown renderer ─────────────────────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return "";
  let h = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  h = h.replace(/```[\w]*\n([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trimEnd()}</code></pre>`);
  h = h.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  h = h.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  h = h.replace(/^### (.+)$/gm, "<h4>$1</h4>");
  h = h.replace(/^## (.+)$/gm,  "<h3>$1</h3>");
  h = h.replace(/^# (.+)$/gm,   "<h2>$1</h2>");
  h = h.replace(/^[ \t]*[-•*] (.+)$/gm, "<li>$1</li>");
  h = h.replace(/((?:<li>[^\n]*\n?)+)/g, "<ul>$1</ul>");
  h = h.replace(/^\d+\. (.+)$/gm, "<nli>$1</nli>");
  h = h.replace(/((?:<nli>[^\n]*\n?)+)/g, m =>
    "<ol>" + m.replace(/<nli>/g, "<li>").replace(/<\/nli>/g, "</li>") + "</ol>");
  h = h.replace(/\n\n+/g, "</p><p>");
  h = h.replace(/\n/g, "<br>");
  h = `<p>${h}</p>`;
  h = h.replace(/<p>\s*(<(?:pre|ul|ol|h[2-4]))/g, "$1");
  h = h.replace(/(<\/(?:pre|ul|ol|h[2-4])>)\s*<\/p>/g, "$1");
  h = h.replace(/<p>\s*<\/p>/g, "");
  return h;
}

// ── URL helpers ───────────────────────────────────────────────────────────────

function getPaperUrl(chunk) {
  if (chunk.doi) return `https://doi.org/${chunk.doi}`;
  const raw = chunk.id.replace(/^paper::/, "").replace(/::chunk\d+$/, "");
  if (raw.startsWith("10.")) return `https://doi.org/${raw}`;
  const m = (chunk.text || "").match(/https?:\/\/doi\.org\/(10\.[^\s,)\]]+)/);
  return m ? `https://doi.org/${m[1]}` : null;
}

// ── Paper panel ───────────────────────────────────────────────────────────────

function renderPapers(chunks) {
  const paperList  = document.getElementById("paperList");
  const papersHint = document.getElementById("papersHint");
  paperList.innerHTML = "";
  const papers = chunks.filter(c => c.type === "paper");
  if (papersHint) papersHint.style.display = papers.length ? "none" : "";
  const seenPaper = new Set();
  papers.forEach(c => {
    const basePaperId = c.id.replace(/::chunk\d+$/, "");
    if (seenPaper.has(basePaperId)) return;
    seenPaper.add(basePaperId);
    const url = getPaperUrl(c);
    const card = document.createElement(url ? "a" : "div");
    card.className = "paper-card";
    if (url) { card.href = url; card.target = "_blank"; card.rel = "noopener noreferrer"; }
    card.innerHTML = `
      <div class="paper-card-title">${c.title}</div>
      <span class="paper-card-citation">${buildCitation(c)}</span>
    `;
    paperList.appendChild(card);
  });
}

function buildCitation(chunk) {
  const parts = [];
  if (chunk.first_author) parts.push(chunk.first_author);
  if (chunk.journal)      parts.push(abbreviateJournal(chunk.journal));
  if (chunk.year)         parts.push(chunk.year);
  if (parts.length)       return parts.join(" · ");

  const idMatch = chunk.id.match(/paper::pdf::(.+?) - (\d{4}) - /);
  if (idMatch) {
    const rawAuthors = idMatch[1];
    const year = idMatch[2];
    const firstLast = rawAuthors.split(/ and | et al\.?/i)[0].split(",")[0].trim();
    const rest = rawAuthors.includes(" and ") ? rawAuthors.split(" and ").slice(1).join(" and ").trim() : null;
    const authorStr = rest ? `${firstLast} & ${rest.split(",")[0].trim()}` : firstLast;
    return [authorStr, year].join(" · ");
  }

  const yearMatch = (chunk.text || "").match(/\b(20[0-2]\d|199\d)\b/);
  return yearMatch ? yearMatch[1] : "PDF";
}

const JOURNAL_ABBREVS = {
  "journal of geophysical research": "JGR",
  "journal of geophysical research: solid earth": "JGR Solid Earth",
  "journal of geophysical research: oceans": "JGR Oceans",
  "geophysical research letters": "GRL",
  "earth and planetary science letters": "EPSL",
  "geophysics": "Geophysics",
  "the leading edge": "TLE",
  "geophysical prospecting": "Geophys. Prospect.",
  "annual review of earth and planetary sciences": "Annu. Rev. Earth Planet. Sci.",
  "nature communications": "Nat. Commun.",
  "nature geoscience": "Nat. Geosci.",
  "science": "Science",
  "nature": "Nature",
  "journal of the acoustical society of america": "J. Acoust. Soc. Am.",
  "geochemistry, geophysics, geosystems": "G-Cubed",
  "rock mechanics and rock engineering": "Rock Mech. Rock Eng.",
  "international journal of rock mechanics and mining sciences": "Int. J. Rock Mech.",
  "journal of petroleum science and engineering": "J. Pet. Sci. Eng.",
  "fuel": "Fuel",
  "water resources research": "Water Resour. Res.",
};

function abbreviateJournal(journal) {
  return JOURNAL_ABBREVS[journal.toLowerCase().trim()] || journal;
}

// ── Status bar ────────────────────────────────────────────────────────────────

function setStatus(text) {
  const bar = document.getElementById("statusBar");
  document.getElementById("statusText").textContent = text;
  bar.hidden = !text;
}

// ── UI: add message bubble ────────────────────────────────────────────────────

function addMessage(role, content, { html = false } = {}) {
  const list  = document.getElementById("messageList");
  const div   = document.createElement("div");
  div.className = `message ${role}`;
  const label = role === "user" ? "You" : role === "assistant" ? "RockRAG" : "Error";
  div.innerHTML = `<div class="message-label">${label}</div><div class="message-content"></div>`;
  const contentEl = div.querySelector(".message-content");
  if (html)         contentEl.innerHTML  = content;
  else if (content) contentEl.textContent = content;
  list.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return contentEl;
}

// ── Intent classification (client-side heuristic) ─────────────────────────────

function classifyIntent(query) {
  const q = query.trim().toLowerCase();

  const literaturePatterns = [
    /\b(paper|papers|publication|publications|article|articles|study|studies|literature|journal)\b/,
    /what (has been|have been|was|were) (published|written|found|studied)/,
    /what (research|work|studies) (exist|are there|has been done)/,
    /\b(cite|citation|reference|bibliography|findings|results)\b/,
    /summarize.*(literature|research|papers)/,
  ];
  if (literaturePatterns.some(p => p.test(q))) return "LITERATURE";

  const capabilityPatterns = /^(what can you|what do you|what are you|tell me what you|how do you work|what are your capabilities)/;
  if (capabilityPatterns.test(q)) return "CAPABILITY";

  if (q.endsWith("?")) return "QUESTION";
  const questionStarters = /^(what|why|how|when|where|who|which|is |are |can you|could you|explain|tell me about|what's|what is|does |do )/;
  if (questionStarters.test(q)) return "QUESTION";

  return "AMBIGUOUS";
}

// ── Streaming helper ──────────────────────────────────────────────────────────

async function streamChatToElement(query, context, contentEl, history = []) {
  const resp = await streamChat(query, context, history);
  if (!resp.ok) {
    if (resp.status === 429) sessionStorage.setItem(RL_KEY, String(Date.now() + 15000));
    if (resp.status === 401) { showPasswordGate(); throw new Error("Unauthorized — please re-enter your password."); }
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error || `Chat stream failed: ${resp.status}`);
  }

  contentEl.innerHTML = '<span class="thinking-indicator">Thinking<span class="thinking-dots"></span></span>';

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer   = "";
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (raw === "[DONE]") break;
      try {
        const chunk = JSON.parse(raw);
        const parts = chunk?.candidates?.[0]?.content?.parts ?? [];
        const text  = parts.filter(p => !p.thought).map(p => p.text ?? "").join("");
        fullText   += text;
        if (fullText) contentEl.innerHTML = renderMarkdown(fullText);
      } catch { /* partial JSON — skip */ }
    }
  }

  if (!fullText) contentEl.innerHTML = '<span class="error-text">No response received. Please try again.</span>';
  contentEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return fullText;
}

// ── Main submit flow ──────────────────────────────────────────────────────────

async function onSubmit(e) {
  e.preventDefault();
  const query = document.getElementById("queryInput").value.trim();
  if (!query) return;

  const submitBtn = document.getElementById("submitBtn");
  submitBtn.disabled = true;
  document.getElementById("queryInput").value = "";
  addMessage("user", query);

  try {
    await handleNewQuery(query);
  } catch (err) {
    setStatus("");
    addMessage("error", `Error: ${err.message}`);
    console.error(err);
  } finally {
    submitBtn.disabled = false;
  }
}

const CAPABILITY_RESPONSE = `**RockRAG** is your literature assistant for rock physics research. Here's what I can do:

**Answer questions** about rock physics concepts — ask me about Gassmann's equation, effective medium theories, seismic attenuation, pore pressure prediction, AVO analysis, and more.

**Summarize published research** — switch to Literature Review mode and ask "What papers have been published on fluid substitution?" or "What research exists on digital rock physics?" I'll draw from a curated library of indexed papers.

**Topics I cover:**
- Elastic wave velocities (P-wave, S-wave) in rocks and sediments
- Fluid substitution and Gassmann's equations
- Effective medium theories (Hertz-Mindlin, Hashin-Shtrikman, DEM)
- Seismic attenuation and dispersion
- Pressure and temperature effects on rock properties
- Reservoir characterization and seismic inversion
- Digital rock physics and CT imaging
- Anisotropy in rocks (VTI, HTI, orthorhombic)
- Pore pressure prediction and AVO analysis

What would you like to explore?`;

async function handleNewQuery(query) {
  let intent = classifyIntent(query);
  if (CURRENT_MODE === "literature") intent = "LITERATURE";

  if (intent === "CAPABILITY") {
    const contentEl = addMessage("assistant", "");
    contentEl.innerHTML = renderMarkdown(CAPABILITY_RESPONSE);
    HISTORY.push({ role: "user",  parts: [{ text: query }] });
    HISTORY.push({ role: "model", parts: [{ text: CAPABILITY_RESPONSE }] });
    HISTORY = pruneHistory(HISTORY);
    return;
  }

  setStatus("Searching catalog…");
  const paperChunks = paperBm25Search(query, 6);
  const context = paperChunks;

  renderPapers(context);

  setStatus("Generating response…");
  const contentEl = addMessage("assistant", "");
  const historySnapshot = pruneHistory([...HISTORY]);
  const fullText = await streamChatToElement(query, trimContext(context), contentEl, historySnapshot);
  setStatus("");

  HISTORY.push({ role: "user",  parts: [{ text: query    }] });
  HISTORY.push({ role: "model", parts: [{ text: fullText }] });
  HISTORY = pruneHistory(HISTORY);
}

// ── Password gate ─────────────────────────────────────────────────────────────

function showPasswordGate() {
  document.getElementById("passwordGate").hidden = false;
}

document.getElementById("passwordForm").addEventListener("submit", e => {
  e.preventDefault();
  const pw = document.getElementById("passwordInput").value.trim();
  if (!pw) return;
  sessionStorage.setItem('rrg-pw', pw);
  document.getElementById("passwordGate").hidden = true;
  // Clear the input for security
  document.getElementById("passwordInput").value = "";
  document.getElementById("passwordError").hidden = true;
  boot().catch(console.error);
});

// ── Theme ─────────────────────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem('rrg-theme');
  if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
  updateThemeIcon();
  document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
}

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  if (isDark) {
    document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('rrg-theme', 'light');
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
    localStorage.setItem('rrg-theme', 'dark');
  }
  updateThemeIcon();
}

function updateThemeIcon() {
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
  btn.querySelector('.icon-sun').hidden = !isDark;
  btn.querySelector('.icon-moon').hidden = isDark;
}

// ── Entry point ───────────────────────────────────────────────────────────────

// On load, check if we already have a stored password
if (getStoredPassword()) {
  document.getElementById("passwordGate").hidden = true;
  boot().catch(console.error);
}
// Otherwise gate is shown by default (not hidden in HTML)
