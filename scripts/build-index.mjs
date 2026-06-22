/**
 * Rock Physics RAG index builder.
 * Reads catalog/papers.json + catalog/pdf-chunks.json → embeds via Gemini →
 * writes public/embeddings.bin, public/chunks.json, public/embed-hashes.json,
 * public/search-index.json
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import crypto from "crypto";

const __dir = path.dirname(fileURLToPath(import.meta.url));
const ROOT  = path.resolve(__dir, "..");

const CONFIG     = JSON.parse(fs.readFileSync(path.join(ROOT, "rag-config.json"), "utf8"));
const PAPERS     = loadOptional(path.join(ROOT, "catalog", "papers.json"));
const PDF_CHUNKS = loadOptional(path.join(ROOT, "catalog", "pdf-chunks.json"));
const GEMINI_KEY = process.env.GEMINI_API_KEY;

function loadOptional(p) {
  try { return JSON.parse(fs.readFileSync(p, "utf8")); }
  catch { console.warn(`  (optional) ${path.basename(p)} not found — skipping`); return null; }
}

const EMBED_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key=${GEMINI_KEY}`;
const OUT_DIR   = path.join(ROOT, "public");

if (!GEMINI_KEY) { console.error("GEMINI_API_KEY not set"); process.exit(1); }
fs.mkdirSync(OUT_DIR, { recursive: true });

// ── Chunk builder ─────────────────────────────────────────────────────────────

function makeChunks() {
  const chunks = [];
  const pdfCoveredDois = new Set();
  const doiLookup = {};
  if (PAPERS) {
    for (const p of PAPERS.papers) {
      if (p.doi) doiLookup[p.doi.toLowerCase()] = p;
    }
  }
  if (PDF_CHUNKS) {
    for (const c of PDF_CHUNKS.chunks) {
      if (c.doi) pdfCoveredDois.add(c.doi);
      const idTitleMatch = c.id.match(/^paper::pdf::.+ - \d{4} - (.+?)::chunk\d+$/);
      const cleanTitle = idTitleMatch ? idTitleMatch[1] : c.title;
      let meta = { year: c.year, journal: c.journal, first_author: c.first_author, doi: c.doi };
      if (!meta.journal || !meta.year) {
        const doiMatch = c.text.match(/\b(10\.\d{4,}\/[^\s,)\]"]+)/);
        if (doiMatch) {
          const found = doiLookup[doiMatch[1].toLowerCase()];
          if (found) {
            meta = { doi: found.doi || meta.doi, year: found.year || meta.year, journal: found.journal || meta.journal, first_author: found.first_author || meta.first_author };
            if (meta.doi) pdfCoveredDois.add(meta.doi);
          }
        }
      }
      const header = [cleanTitle, meta.first_author, meta.year ? `(${meta.year})` : null, meta.journal ? `— ${meta.journal}` : null].filter(Boolean).join(" ");
      chunks.push({ id: c.id, title: cleanTitle, type: "paper", source: "pdf", location: null, keywords: c.tags || [], doi: meta.doi || null, year: meta.year || null, journal: meta.journal || null, first_author: meta.first_author || null, text: c.text, embedText: `From "${header}":\n\n${c.text}` });
    }
  }
  if (PAPERS) {
    for (const paper of PAPERS.papers) {
      if (!paper.abstract || paper.abstract.length < 80) continue;
      if (paper.doi && pdfCoveredDois.has(paper.doi)) continue;
      const yearStr = paper.year ? ` (${paper.year})` : "";
      const journalStr = paper.journal ? ` — ${paper.journal}` : "";
      const text = `${paper.title}${yearStr}${journalStr}\n\n${paper.abstract}`;
      chunks.push({ id: `paper::${paper.doi || paper.title.slice(0, 40).replace(/\s+/g, "-")}`, title: paper.title, type: "paper", source: "zotero", location: null, keywords: paper.tags || [], year: paper.year || null, journal: paper.journal || null, first_author: paper.first_author || null, text, embedText: `Research paper: ${paper.title}\n\n${paper.abstract}` });
    }
  }
  return chunks;
}

// ── Embed ─────────────────────────────────────────────────────────────────────

async function embedBatch(texts, retries = 6) {
  const requests = texts.map(t => ({
    model: "models/gemini-embedding-2",
    content: { parts: [{ text: t }] },
    taskType: "RETRIEVAL_DOCUMENT",
    outputDimensionality: 768,
  }));

  for (let attempt = 0; attempt < retries; attempt++) {
    const res = await fetch(EMBED_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ requests }),
    });
    if (res.ok) {
      const data = await res.json();
      return data.embeddings.map(e => e.values);
    }
    const errText = await res.text();
    const delay = res.status === 429 ? 30000 * (attempt + 1) : 2000 * (attempt + 1);
    console.warn(`Embed attempt ${attempt + 1} failed (${res.status}): ${errText.slice(0, 120)}`);
    if (attempt < retries - 1) {
      console.warn(`  Waiting ${delay / 1000}s before retry...`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
  throw new Error("Embedding failed after retries");
}

function l2Normalize(vec) {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return norm > 0 ? vec.map(v => v / norm) : vec;
}

// ── MiniSearch index ──────────────────────────────────────────────────────────

function buildSearchIndex(chunks) {
  return {
    version:   "1.0",
    fields:    ["title", "text", "keywords", "type", "location"],
    documents: chunks.map(c => ({
      id:       c.id,
      title:    c.title,
      text:     c.text,
      keywords: (c.keywords || []).join(" "),
      type:     c.type,
      location: c.location || "",
    })),
  };
}

// ── Incremental embedding cache ───────────────────────────────────────────────

const CHECKPOINT_PATH = path.join(ROOT, "catalog", "embed-checkpoint.json");

function contentKey(chunk) {
  return crypto.createHash("sha1").update(chunk.embedText).digest("hex").slice(0, 16);
}

function loadExistingEmbeddings(chunks) {
  const chunksPath = path.join(OUT_DIR, "chunks.json");
  const embedPath  = path.join(OUT_DIR, "embeddings.bin");
  const hashesPath = path.join(OUT_DIR, "embed-hashes.json");
  const cache = new Map();

  // Load finished index from public/
  try {
    const oldChunks = JSON.parse(fs.readFileSync(chunksPath, "utf8"));
    const oldHashes = fs.existsSync(hashesPath)
      ? JSON.parse(fs.readFileSync(hashesPath, "utf8"))
      : null;
    const buf    = fs.readFileSync(embedPath);
    const floats = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
    const dim    = floats.length / oldChunks.length;
    oldChunks.forEach((c, i) => {
      const key = oldHashes ? (oldHashes[c.id] ?? c.id) : c.id;
      cache.set(key, Array.from(floats.subarray(i * dim, (i + 1) * dim)));
    });
    console.log(`  Loaded ${cache.size} cached embeddings from public/ (dim=${dim})`);
  } catch {
    console.log("  No finished index found in public/");
  }

  // Merge in-progress checkpoint (survives crashes between batches)
  try {
    const ckpt = JSON.parse(fs.readFileSync(CHECKPOINT_PATH, "utf8"));
    let added = 0;
    for (const [key, vec] of Object.entries(ckpt)) {
      if (!cache.has(key)) { cache.set(key, vec); added++; }
    }
    if (added > 0) console.log(`  Recovered ${added} embeddings from checkpoint`);
  } catch { /* no checkpoint yet */ }

  return cache;
}

function saveCheckpoint(newVecMap) {
  const obj = Object.fromEntries(newVecMap);
  fs.writeFileSync(CHECKPOINT_PATH, JSON.stringify(obj));
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("Building Rock Physics RAG index...");
  const chunks = makeChunks();
  const paperCount   = PAPERS?.papers?.length ?? 0;
  const pdfCount     = PDF_CHUNKS?.chunks?.length ?? 0;
  console.log(`  ${chunks.length} chunks from ${paperCount} papers (${pdfCount} pdf chunks)`);

  const cache   = loadExistingEmbeddings(chunks);
  const toEmbed = chunks.filter(c => !cache.has(contentKey(c)));
  const reused  = chunks.length - toEmbed.length;
  console.log(`  ${reused} chunks reused from cache, ${toEmbed.length} need embedding`);

  // Token-aware throttle: gemini-embedding-2 free tier is 30k TPM.
  const TPM_LIMIT = 28000;
  let windowStart  = Date.now();
  let windowTokens = 0;

  const newVecMap = new Map();
  if (toEmbed.length > 0) {
    const batchSize = CONFIG.batchSize || 25;
    for (let i = 0; i < toEmbed.length; i += batchSize) {
      const batch = toEmbed.slice(i, i + batchSize);
      const batchNum    = Math.floor(i / batchSize) + 1;
      const totalBatches = Math.ceil(toEmbed.length / batchSize);

      const batchTokens = batch.reduce((s, c) => s + Math.ceil(c.embedText.length / 4), 0);

      if (Date.now() - windowStart >= 60000) {
        windowStart  = Date.now();
        windowTokens = 0;
      }

      if (windowTokens + batchTokens > TPM_LIMIT) {
        const wait = 60000 - (Date.now() - windowStart) + 1000;
        console.log(`  TPM limit approaching (${windowTokens} used) — waiting ${Math.ceil(wait / 1000)}s...`);
        await new Promise(r => setTimeout(r, wait));
        windowStart  = Date.now();
        windowTokens = 0;
      }

      console.log(`  Embedding batch ${batchNum}/${totalBatches} (${batch.length} chunks, ~${batchTokens} tokens)...`);
      const vecs = await embedBatch(batch.map(c => c.embedText));
      batch.forEach((c, j) => newVecMap.set(contentKey(c), l2Normalize(vecs[j])));
      windowTokens += batchTokens;
      saveCheckpoint(newVecMap); // persist after every batch so crashes don't lose work
    }
  }

  const allVecs = chunks.map(c => newVecMap.get(contentKey(c)) ?? cache.get(contentKey(c)));

  const dim = allVecs[0].length;
  const bin = new Float32Array(allVecs.length * dim);
  allVecs.forEach((v, i) => bin.set(v, i * dim));
  fs.writeFileSync(path.join(OUT_DIR, "embeddings.bin"), Buffer.from(bin.buffer));
  console.log(`  embeddings.bin: ${allVecs.length} × ${dim}`);

  const chunksOut = chunks.map(({ embedText, ...rest }) => rest);
  fs.writeFileSync(path.join(OUT_DIR, "chunks.json"), JSON.stringify(chunksOut));
  console.log(`  chunks.json: ${chunksOut.length} entries`);

  const hashMap = Object.fromEntries(chunks.map(c => [c.id, contentKey(c)]));
  fs.writeFileSync(path.join(OUT_DIR, "embed-hashes.json"), JSON.stringify(hashMap));
  console.log(`  embed-hashes.json: ${Object.keys(hashMap).length} entries`);

  const searchIdx = buildSearchIndex(chunks);
  fs.writeFileSync(path.join(OUT_DIR, "search-index.json"), JSON.stringify(searchIdx));
  console.log(`  search-index.json: ${searchIdx.documents.length} documents`);

  // Clean up checkpoint now that the final index is written
  try { fs.unlinkSync(CHECKPOINT_PATH); } catch { /* already gone */ }

  console.log(`Done. (${reused} cached, ${toEmbed.length} newly embedded)`);
}

main().catch(e => { console.error(e); process.exit(1); });
