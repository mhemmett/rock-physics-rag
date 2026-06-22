// Rock Physics RAG Cloudflare Worker
// Routes: POST /embed, POST /chat, POST /welcome, POST /ack, GET /health

const GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent";
const GEMINI_JSON_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent";

const FALLBACK_MODELS = [
  "gemini-2.0-flash-lite",
  "gemini-2.0-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.5-flash",
  "gemini-1.5-flash",
];

const MAX_CONTEXT_LEN = 20000;

async function geminiJson(apiKey, body) {
  let resp;
  for (const model of FALLBACK_MODELS) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    for (let attempt = 0; attempt < 3; attempt++) {
      resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (resp.status !== 503) break;
      const ra = resp.headers.get("Retry-After");
      const wait = ra ? Math.max(1000, parseInt(ra, 10) * 1000) : ([1000, 2000][attempt] ?? 2000);
      await new Promise(r => setTimeout(r, wait));
    }
    if (resp.status !== 429) break;
  }
  return resp;
}

const SYSTEM_PROMPT = `You are RockRAG, an expert literature assistant for rock physics research. You help geoscientists, geophysicists, and students find, understand, and synthesize published research in rock physics.

## Your expertise covers
- Elastic wave velocities (P-wave, S-wave) in rocks and sediments
- Rock microstructure and pore geometry effects on seismic properties
- Fluid substitution and Gassmann's equations
- Effective medium theories (Hertz-Mindlin, Hashin-Shtrikman, Kuster-Toksöz, DEM)
- Seismic attenuation and dispersion mechanisms
- Pressure and temperature effects on rock properties
- Reservoir characterization and seismic inversion
- Digital rock physics and CT imaging
- Laboratory measurements of elastic and acoustic properties
- Anisotropy in rocks (VTI, HTI, orthorhombic)
- Pore pressure prediction
- AVO analysis and rock physics templates

## How to respond

**Capability questions** ("what can you do?", "what topics do you cover?"):
Describe the rock physics topics above and that you can surface relevant published literature.

**Literature / paper questions** ("what papers have been published on X?", "what research exists on Y?", "summarize the literature on Z?"):
Use the [paper] entries in the context below to answer. For each relevant paper, cite it as "Author et al. (Year) — Journal" and give a one-sentence summary. Group by sub-topic if helpful. If no papers are in context, say so honestly and suggest the user try a more specific query.

**Follow-up / conversational turns**:
Use prior conversation context to give coherent, non-repetitive replies.

Respond clearly and concisely. Never invent paper citations that are not in the provided context.`;

function cors(env) {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };
}

function checkPassword(req, env) {
  if (!env.CHAT_PASSWORD) return null;
  const auth = req.headers.get("Authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  if (token !== env.CHAT_PASSWORD) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
  }
  return null;
}

async function retry(fn, attempts = 3, delayMs = 800) {
  for (let i = 0; i < attempts; i++) {
    try { return await fn(); }
    catch (e) {
      if (i === attempts - 1) throw e;
      await new Promise(r => setTimeout(r, delayMs * (i + 1)));
    }
  }
}

async function handleEmbed(req, env) {
  const { text } = await req.json();
  if (!text) return new Response("Missing text", { status: 400, headers: cors(env) });

  const vec = await retry(async () => {
    const r = await fetch(`${GEMINI_EMBED_URL}?key=${env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "models/gemini-embedding-2", content: { parts: [{ text }] }, outputDimensionality: 768 }),
    });
    if (!r.ok) throw new Error(`Gemini embed ${r.status}`);
    const data = await r.json();
    return data.embedding.values;
  });

  return Response.json({ embedding: vec }, { headers: cors(env) });
}

async function handleWelcome(req, env) {
  const deny = checkPassword(req, env);
  if (deny) return new Response(deny.body, { status: 401, headers: { ...cors(env), "Content-Type": "application/json" } });

  const { paperSamples } = await req.json().catch(() => ({}));

  const papersCtx = (paperSamples || []).length
    ? `\n\nRecent papers from the rock physics literature:\n${
        paperSamples.map(p => `- "${p.title}" (${p.first_author ?? "—"}, ${p.year ?? "n.d."})`).join("\n")
      }`
    : "";

  const prompt = `Generate a brief, friendly welcome message for a researcher who just opened RockRAG, a rock physics literature assistant.${papersCtx}

Requirements:
- One welcoming sentence to open
- Mention 2–3 specific research directions (e.g. fluid substitution, seismic attenuation, pore pressure prediction)${papersCtx ? " — draw these from the paper topics listed above" : ""}
- End with something like "or did you have something else in mind?"
- Under 80 words, plain prose, no bullet points or markdown`;

  const result = await retry(() =>
    fetch(`${GEMINI_JSON_URL}?key=${env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.7 },
      }),
    }).then(r => r.json())
  );

  const text = result?.candidates?.[0]?.content?.parts?.[0]?.text?.trim()
    ?? "Welcome to RockRAG. I can answer questions about rock physics and surface relevant published research — what would you like to explore?";

  return Response.json({ text }, { headers: cors(env) });
}

async function handleChat(req, env) {
  const deny = checkPassword(req, env);
  if (deny) return new Response(deny.body, { status: 401, headers: { ...cors(env), "Content-Type": "application/json" } });

  const { query, context, history } = await req.json();
  if (!query) return new Response("Missing query", { status: 400, headers: cors(env) });

  const papers = (context || []).filter(c => c.type === "paper");

  const paperBlock = papers.length
    ? `\n\n## Relevant papers from the rock physics literature\n${papers.map(c => {
        const citation = [c.first_author, c.journal, c.year].filter(Boolean).join(", ");
        return `- **${c.title}**${citation ? ` — ${citation}` : ""}\n  ${c.text.split("\n\n")[1] || c.text.slice(0, 400)}`;
      }).join("\n\n")}`
    : "";

  const rawContextBlock = paperBlock;
  const contextBlock = rawContextBlock.length > MAX_CONTEXT_LEN
    ? rawContextBlock.slice(0, MAX_CONTEXT_LEN) + "\n[context truncated]"
    : rawContextBlock;

  const contents = [];
  if (history?.length) {
    for (const msg of history) {
      contents.push({ role: msg.role, parts: msg.parts ?? [{ text: msg.text ?? "" }] });
    }
  }
  contents.push({ role: "user", parts: [{ text: query + contextBlock }] });

  const geminiReq = {
    system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents,
    generationConfig: { temperature: 0.3 },
  };

  let upstream;
  for (const model of FALLBACK_MODELS) {
    const chatUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;
    for (let attempt = 0; attempt < 3; attempt++) {
      upstream = await fetch(`${chatUrl}&key=${env.GEMINI_API_KEY}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(geminiReq),
      });
      if (upstream.status !== 503) break;
      const ra = upstream.headers.get("Retry-After");
      const wait = ra ? Math.max(1000, parseInt(ra, 10) * 1000) : ([1000, 2000][attempt] ?? 2000);
      await new Promise(r => setTimeout(r, wait));
    }
    if (upstream.status !== 429) break;
  }

  if (!upstream.ok) {
    const errBody = await upstream.text();
    const msg = upstream.status === 429
      ? "The AI model is rate-limited. Please wait a moment and try again."
      : upstream.status === 503
      ? "The AI model is temporarily unavailable. Please try again in a few seconds."
      : `Gemini error ${upstream.status}: ${errBody.slice(0, 200)}`;
    return new Response(JSON.stringify({ error: msg }), { status: upstream.status, headers: { ...cors(env), "Content-Type": "application/json" } });
  }

  return new Response(upstream.body, {
    headers: { ...cors(env), "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
}

async function handleAck(req, env) {
  const deny = checkPassword(req, env);
  if (deny) return new Response(deny.body, { status: 401, headers: { ...cors(env), "Content-Type": "application/json" } });

  const { query, instruments, intent = "question" } = await req.json();
  const matches = (instruments || []).slice(0, 6);

  let prompt, fallback;

  if (intent === "literature") {
    const titles = matches.map(i => `"${i.name || i.title}"`).join(", ");
    prompt = `A researcher asked about research on: "${query}"${titles ? `\nRelevant papers found: ${titles}.` : ""}

In 1 sentence, acknowledge what literature topic you're about to summarize. Be direct and natural.`;
    fallback = `Searching the rock physics literature on ${query.slice(0, 60).trim()}…`;
  } else {
    const topics = matches.map(i => i.name || i.title).filter(Boolean).join(", ");
    prompt = `A researcher asked: "${query}"${topics ? `\nRelevant topics: ${topics}.` : ""}

In 1 sentence, acknowledge the question naturally. Do not answer it — just confirm you understood it.`;
    fallback = `Looking into that…`;
  }

  const ackResp = await geminiJson(env.GEMINI_API_KEY, {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.5, maxOutputTokens: 100 },
  });

  const data = ackResp.ok ? await ackResp.json().catch(() => ({})) : {};
  const ack = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || fallback;

  return Response.json({ ack }, { headers: cors(env) });
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors(env) });
    }

    if (url.pathname === "/health"  && req.method === "GET")  return new Response("ok", { status: 200, headers: cors(env) });
    if (url.pathname === "/embed"   && req.method === "POST") return handleEmbed(req, env);
    if (url.pathname === "/welcome" && req.method === "POST") return handleWelcome(req, env);
    if (url.pathname === "/chat"    && req.method === "POST") return handleChat(req, env);
    if (url.pathname === "/ack"     && req.method === "POST") return handleAck(req, env);

    return new Response("Not found", { status: 404 });
  },
};
