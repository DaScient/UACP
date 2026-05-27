/*!
 * ufo-intel-widget — Plug-and-play UFO/UAP release intelligence Web Component.
 *
 * Drop-in usage:
 *   <script type="module" src="ufo-intel-widget.js"></script>
 *   <ufo-intel-widget
 *     data-source="https://example.org/ufo-mirror/index.json"
 *     refresh-interval="30000"
 *     theme="dark"
 *     llm-provider="openai"
 *     llm-model="gpt-4o-mini"></ufo-intel-widget>
 *
 * Design constraints honored:
 *   - 100% client-side. No telemetry. No external CDN calls unless the user
 *     opts in (Transformers.js embeddings, optional Chart.js — both off by
 *     default). All visualisations below are hand-rolled SVG so the widget
 *     stays zero-dependency.
 *   - Shadow DOM style isolation so the widget cannot collide with host CSS.
 *   - IndexedDB for last-known dataset + multi-turn audit log.
 *   - localStorage for the user's API key (never transmitted anywhere except
 *     directly to the LLM endpoint the user configured).
 *   - Graceful degrade when no LLM key is configured.
 *   - Accessible: ARIA roles, keyboard navigation, prefers-reduced-motion.
 *
 * The schema of each release record is expected to be roughly:
 *   { title, agency, url, blurb, releaseDate, type? }
 * with the source endpoint returning either:
 *   { pdfs: [...], images: [...], videos: [...] }
 * or a flat { items: [...] } / [...]. The loader normalises all three.
 */

(() => {
  if (customElements.get('ufo-intel-widget')) return;

  // ────────────────────────────────────────────────────────────────────────
  //  Constants & tiny utilities
  // ────────────────────────────────────────────────────────────────────────

  const DEFAULT_SOURCE     = '';  // intentionally empty — UI prompts the analyst to configure a source
  const DEFAULT_REFRESH_MS = 30_000;
  const DB_NAME            = 'ufo-intel-widget';
  const DB_VERSION         = 1;
  const LS_PREFIX          = 'ufo-intel-widget:';

  const SLASH_COMMANDS = [
    { cmd: '/summarize',  hint: 'Summarise the latest tranche of releases.' },
    { cmd: '/compare',    hint: '/compare agencies — contrast release patterns.' },
    { cmd: '/report',     hint: '/report — generate a structured intelligence report.' },
    { cmd: '/risk',       hint: '/risk <term> — heuristic risk assessment for a keyword.' },
    { cmd: '/timeline',   hint: 'Describe the release timeline in chronological order.' },
    { cmd: '/find',       hint: '/find <pattern> — surface recurring patterns / clusters.' },
  ];

  const STOPWORDS = new Set((
    'a an and are as at be by for from has have he her his i in is it its of on or our she ' +
    'so such that the their them then there these they this to was we were what when where ' +
    'which who will with would you your about above after against all also am any been before ' +
    'being below between both but did do does doing down during each few further into more ' +
    'most no nor not now off once only other out over own same some than too under until up ' +
    'very via while will not'
  ).split(/\s+/));

  const $esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    { '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]
  ));

  const fmtDate = (d) => {
    try {
      const x = new Date(d);
      if (isNaN(x.getTime())) return String(d || '—');
      return x.toISOString().replace('T',' ').replace(/\.\d+Z$/, 'Z');
    } catch { return String(d || '—'); }
  };


  // Light reversible obfuscation for the BYO API key at rest in localStorage.
  // The spec requires the key to remain on-device; this just prevents casual
  // disclosure via a DevTools glance. It is NOT cryptographic protection —
  // anyone with code execution in the page can recover the key.
  const _NONCE = 'ufo-intel/v1';
  const _obfuscate = (s) => {
    if (!s) return '';
    try {
      const bytes = new TextEncoder().encode(s);
      const out   = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) {
        out[i] = bytes[i] ^ _NONCE.charCodeAt(i % _NONCE.length);
      }
      let bin = '';
      for (let i = 0; i < out.length; i++) bin += String.fromCharCode(out[i]);
      return btoa(bin);
    } catch { return ''; }
  };
  const _deobfuscate = (b) => {
    if (!b) return '';
    try {
      const bin   = atob(b);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i) ^ _NONCE.charCodeAt(i % _NONCE.length);
      }
      return new TextDecoder().decode(bytes);
    } catch { return ''; }
  };

  const debounce = (fn, ms) => {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  };

  // ────────────────────────────────────────────────────────────────────────
  //  IndexedDB helpers (cache + audit log)
  // ────────────────────────────────────────────────────────────────────────

  const idbOpen = () => new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('cache')) db.createObjectStore('cache');
      if (!db.objectStoreNames.contains('audit')) {
        const s = db.createObjectStore('audit', { keyPath: 'id', autoIncrement: true });
        s.createIndex('ts', 'ts');
      }
      if (!db.objectStoreNames.contains('notes')) {
        db.createObjectStore('notes', { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });

  const idbGet = async (store, key) => {
    const db = await idbOpen();
    return new Promise((res, rej) => {
      const tx = db.transaction(store, 'readonly');
      const r  = tx.objectStore(store).get(key);
      r.onsuccess = () => res(r.result);
      r.onerror   = () => rej(r.error);
    });
  };

  const idbPut = async (store, value, key) => {
    const db = await idbOpen();
    return new Promise((res, rej) => {
      const tx = db.transaction(store, 'readwrite');
      const os = tx.objectStore(store);
      const r  = key === undefined ? os.put(value) : os.put(value, key);
      r.onsuccess = () => res(r.result);
      r.onerror   = () => rej(r.error);
    });
  };

  const idbAll = async (store) => {
    const db = await idbOpen();
    return new Promise((res, rej) => {
      const tx = db.transaction(store, 'readonly');
      const r  = tx.objectStore(store).getAll();
      r.onsuccess = () => res(r.result || []);
      r.onerror   = () => rej(r.error);
    });
  };

  const audit = (event, detail) => {
    try {
      idbPut('audit', { ts: Date.now(), event, detail: detail || null });
    } catch (_) { /* silent — auditing must never break the UI */ }
  };

  // ────────────────────────────────────────────────────────────────────────
  //  Data normalisation
  // ────────────────────────────────────────────────────────────────────────

  /** Merge any of the supported shapes into a flat list of records. */
  function normaliseDataset(raw) {
    if (!raw) return [];
    const out = [];
    const push = (arr, type) => {
      if (!Array.isArray(arr)) return;
      for (const r of arr) {
        if (!r || typeof r !== 'object') continue;
        out.push({
          id:          r.id || r.url || `${type}:${r.title || Math.random().toString(36).slice(2)}`,
          title:       r.title || r.name || '(untitled)',
          agency:      r.agency || r.source || 'UNKNOWN',
          url:         r.url || r.link || '#',
          blurb:       r.blurb || r.summary || r.description || '',
          releaseDate: r.releaseDate || r.date || r.published || r.timestamp || null,
          type:        r.type || type,
        });
      }
    };
    if (Array.isArray(raw))          push(raw, 'item');
    else if (Array.isArray(raw.items)) push(raw.items, 'item');
    else {
      push(raw.pdfs,   'pdf');
      push(raw.images, 'image');
      push(raw.videos, 'video');
      // fall-through: also surface anything else array-shaped
      for (const k of Object.keys(raw)) {
        if (['pdfs','images','videos','items'].includes(k)) continue;
        if (Array.isArray(raw[k])) push(raw[k], k);
      }
    }
    // Sort newest first.
    out.sort((a, b) => {
      const ta = a.releaseDate ? Date.parse(a.releaseDate) : 0;
      const tb = b.releaseDate ? Date.parse(b.releaseDate) : 0;
      return (tb || 0) - (ta || 0);
    });
    return out;
  }

  // ────────────────────────────────────────────────────────────────────────
  //  TF-IDF retrieval (default, always-available client-side RAG)
  // ────────────────────────────────────────────────────────────────────────

  function tokenise(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s_-]/gu, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 2 && !STOPWORDS.has(t));
  }

  class TfIdfIndex {
    constructor(records) {
      this.records = records;
      this.docs    = records.map((r) => tokenise(`${r.title} ${r.blurb} ${r.agency}`));
      this.df      = new Map();
      for (const doc of this.docs) {
        const seen = new Set();
        for (const t of doc) {
          if (seen.has(t)) continue;
          seen.add(t);
          this.df.set(t, (this.df.get(t) || 0) + 1);
        }
      }
      this.N = this.docs.length || 1;
      this.vecs = this.docs.map((doc) => this._vec(doc));
    }
    _vec(tokens) {
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      const v = new Map();
      let norm = 0;
      for (const [t, c] of tf) {
        const idf = Math.log(1 + this.N / (1 + (this.df.get(t) || 0)));
        const w   = (c / tokens.length) * idf;
        v.set(t, w);
        norm += w * w;
      }
      v.__norm = Math.sqrt(norm) || 1;
      return v;
    }
    query(q, k = 5) {
      if (!q) return [];
      const qv = this._vec(tokenise(q));
      const scored = this.vecs.map((dv, i) => {
        let dot = 0;
        // iterate smaller map
        const [a, b] = qv.size < dv.size ? [qv, dv] : [dv, qv];
        for (const [t, w] of a) { const w2 = b.get(t); if (w2) dot += w * w2; }
        return { i, score: dot / (qv.__norm * dv.__norm) };
      });
      scored.sort((a, b) => b.score - a.score);
      return scored.slice(0, k).filter((s) => s.score > 0).map((s) => ({
        record: this.records[s.i], score: s.score,
      }));
    }
    /** Top-k keywords by raw document frequency, useful for /find patterns. */
    topTerms(k = 25) {
      return [...this.df.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
    }
  }

  // ────────────────────────────────────────────────────────────────────────
  //  LLM adapters (BYO key, in-browser only, streaming where supported)
  // ────────────────────────────────────────────────────────────────────────

  /** A minimal SSE line parser shared by OpenAI-compatible endpoints. */
  async function* sseLines(response) {
    const reader = response.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (line.startsWith('data:')) yield line.slice(5).trim();
      }
    }
  }

  const LLM = {
    async *openai({ apiKey, model, messages, baseUrl }) {
      const url = (baseUrl || 'https://api.openai.com') + '/v1/chat/completions';
      const safeKey = String(apiKey || '').replace(/[\r\n\t\v\f\0\x00-\x1f\x7f]/g, '').trim();
      const r = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${safeKey}`,
        },
        body: JSON.stringify({ model: model || 'gpt-4o-mini', messages, stream: true }),
      });
      if (!r.ok) throw new Error(`OpenAI ${r.status}: ${await r.text()}`);
      for await (const data of sseLines(r)) {
        if (data === '[DONE]') return;
        try {
          const j = JSON.parse(data);
          const delta = j.choices?.[0]?.delta?.content;
          if (delta) yield delta;
        } catch { /* ignore malformed SSE frames */ }
      }
    },

    async *anthropic({ apiKey, model, messages }) {
      // Anthropic requires the system prompt to be separated out.
      const system = messages.filter((m) => m.role === 'system').map((m) => m.content).join('\n\n');
      const turns  = messages.filter((m) => m.role !== 'system');
      const safeKey = String(apiKey || '').replace(/[\r\n\t\v\f\0\x00-\x1f\x7f]/g, '').trim();
      const r = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': safeKey,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: model || 'claude-3-5-sonnet-latest',
          max_tokens: 1024,
          system: system || undefined,
          messages: turns,
          stream: true,
        }),
      });
      if (!r.ok) throw new Error(`Anthropic ${r.status}: ${await r.text()}`);
      for await (const data of sseLines(r)) {
        try {
          const j = JSON.parse(data);
          if (j.type === 'content_block_delta' && j.delta?.text) yield j.delta.text;
        } catch { /* ignore */ }
      }
    },

    /** OpenAI-compatible local endpoint (Ollama, LM Studio, llama.cpp server). */
    async *local({ apiKey, model, messages, baseUrl }) {
      yield* LLM.openai({ apiKey: apiKey || 'local', model, messages, baseUrl: baseUrl || 'http://localhost:11434' });
    },
  };

  // ────────────────────────────────────────────────────────────────────────
  //  Component template & styles
  // ────────────────────────────────────────────────────────────────────────

  const STYLE = /* css */`
    :host {
      --ufo-bg:        #0a0d12;
      --ufo-bg-2:      #11161d;
      --ufo-bg-3:      #1a2230;
      --ufo-fg:        #d6e0ee;
      --ufo-fg-dim:    #8593a8;
      --ufo-accent:    #5fd0c4;
      --ufo-accent-2:  #ffb86b;
      --ufo-danger:    #ff5d6c;
      --ufo-border:    #1f2a3a;
      --ufo-grid:      rgba(95,208,196,0.06);
      --ufo-font:      ui-monospace, "JetBrains Mono", "SF Mono", Consolas, Menlo, monospace;
      display: block;
      contain: layout style;
      color: var(--ufo-fg);
      font-family: var(--ufo-font);
      font-size: 13px;
      line-height: 1.45;
      background: var(--ufo-bg);
      border: 1px solid var(--ufo-border);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
      min-height: 480px;
    }
    :host([theme="light"]) {
      --ufo-bg: #f4f6fb; --ufo-bg-2:#ffffff; --ufo-bg-3:#e7ecf5;
      --ufo-fg:#1a2230; --ufo-fg-dim:#56627a;
      --ufo-border:#cdd5e3; --ufo-grid: rgba(28,55,90,0.06);
    }
    :host([theme="compact"]) .body { font-size: 12px; }
    @media (prefers-reduced-motion: reduce) {
      * { animation: none !important; transition: none !important; }
    }
    *, *::before, *::after { box-sizing: border-box; }
    a { color: var(--ufo-accent); text-decoration: none; }
    a:hover, a:focus-visible { text-decoration: underline; outline: none; }
    button {
      font: inherit; color: inherit; background: var(--ufo-bg-3);
      border: 1px solid var(--ufo-border); border-radius: 3px;
      padding: 4px 10px; cursor: pointer;
    }
    button:hover, button:focus-visible {
      background: var(--ufo-bg-2); border-color: var(--ufo-accent); outline: none;
    }
    input, select, textarea {
      font: inherit; color: var(--ufo-fg);
      background: var(--ufo-bg); border: 1px solid var(--ufo-border);
      border-radius: 3px; padding: 5px 8px;
    }
    input:focus, select:focus, textarea:focus { outline: 1px solid var(--ufo-accent); }

    /* ─── chrome ─── */
    .chrome {
      display: flex; align-items: center; gap: 12px;
      padding: 8px 12px; background: var(--ufo-bg-2);
      border-bottom: 1px solid var(--ufo-border);
    }
    .brand { font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .brand .dot {
      display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      background: var(--ufo-accent); margin-right: 6px;
      box-shadow: 0 0 6px var(--ufo-accent);
    }
    .chrome .spacer { flex: 1; }
    .chrome .status { color: var(--ufo-fg-dim); font-size: 11px; }
    .chrome .status.stale { color: var(--ufo-accent-2); }
    .chrome .status.error { color: var(--ufo-danger); }

    /* ─── tab strip ─── */
    .tabs {
      display: flex; gap: 0; border-bottom: 1px solid var(--ufo-border);
      background: var(--ufo-bg-2);
    }
    .tab {
      padding: 8px 14px; cursor: pointer; color: var(--ufo-fg-dim);
      border: none; background: transparent; border-bottom: 2px solid transparent;
      letter-spacing: 0.05em; text-transform: uppercase; font-size: 11px;
    }
    .tab[aria-selected="true"] {
      color: var(--ufo-fg); border-bottom-color: var(--ufo-accent);
      background: var(--ufo-bg);
    }
    .tab:hover { color: var(--ufo-fg); }

    /* ─── body ─── */
    .body {
      display: grid; grid-template-columns: 280px 1fr; min-height: 360px;
      background:
        linear-gradient(var(--ufo-grid) 1px, transparent 1px) 0 0/40px 40px,
        linear-gradient(90deg, var(--ufo-grid) 1px, transparent 1px) 0 0/40px 40px,
        var(--ufo-bg);
    }
    .sidebar {
      border-right: 1px solid var(--ufo-border);
      overflow-y: auto; max-height: 70vh;
    }
    .ticker {
      padding: 6px 12px; background: var(--ufo-bg-2);
      border-bottom: 1px solid var(--ufo-border);
      font-size: 11px; color: var(--ufo-fg-dim);
      white-space: nowrap; overflow: hidden;
    }
    .ticker span { margin-right: 24px; }
    .feed { list-style: none; margin: 0; padding: 0; }
    .feed li {
      padding: 8px 12px; border-bottom: 1px solid var(--ufo-border);
      cursor: pointer;
    }
    .feed li:hover, .feed li:focus-visible { background: var(--ufo-bg-2); outline: none; }
    .feed .row1 { display: flex; gap: 8px; align-items: center; }
    .feed .agency {
      font-size: 10px; padding: 1px 6px; border-radius: 2px;
      background: var(--ufo-bg-3); color: var(--ufo-accent);
      letter-spacing: 0.08em; text-transform: uppercase;
    }
    .feed .agency.note  { background: rgba(255,184,107,0.15); color: var(--ufo-accent-2); }
    .feed .agency.alert { background: rgba(255,93,108,0.15); color: var(--ufo-danger); }
    .feed .when  { color: var(--ufo-fg-dim); font-size: 10px; margin-left: auto; }
    .feed .title { display: block; margin-top: 4px; }
    .feed .blurb { color: var(--ufo-fg-dim); font-size: 11px; margin-top: 3px; }

    .main { padding: 14px 16px; overflow-y: auto; max-height: 70vh; }
    .main h2 {
      margin: 0 0 10px; font-size: 12px; font-weight: 700;
      letter-spacing: 0.18em; text-transform: uppercase; color: var(--ufo-fg-dim);
    }
    .stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 14px; }
    .stat {
      background: var(--ufo-bg-2); border: 1px solid var(--ufo-border);
      padding: 8px 10px; border-radius: 4px;
    }
    .stat .v { font-size: 20px; font-weight: 700; color: var(--ufo-accent); }
    .stat .k { font-size: 10px; color: var(--ufo-fg-dim); letter-spacing: 0.08em; text-transform: uppercase; }
    .toolbar {
      display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .toolbar input[type="search"] { min-width: 220px; }

    table.files { width: 100%; border-collapse: collapse; font-size: 12px; }
    table.files th, table.files td {
      text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--ufo-border);
      vertical-align: top;
    }
    table.files th {
      font-size: 10px; color: var(--ufo-fg-dim);
      letter-spacing: 0.08em; text-transform: uppercase;
    }
    table.files tr:hover { background: var(--ufo-bg-2); }

    /* ─── viz ─── */
    .viz { background: var(--ufo-bg-2); border: 1px solid var(--ufo-border);
           border-radius: 4px; padding: 8px; margin-bottom: 12px; }
    .viz h3 { margin: 0 0 6px; font-size: 11px; color: var(--ufo-fg-dim);
              letter-spacing: 0.12em; text-transform: uppercase; }
    svg { display: block; width: 100%; height: auto; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; font-size: 10px;
              color: var(--ufo-fg-dim); margin-top: 4px; }
    .legend span::before { content: ''; display: inline-block; width: 10px; height: 10px;
                          margin-right: 4px; vertical-align: middle; border-radius: 2px; }

    /* ─── modal ─── */
    .modal-backdrop {
      position: absolute; inset: 0; background: rgba(0,0,0,0.6);
      display: flex; align-items: center; justify-content: center; z-index: 50;
    }
    .modal {
      background: var(--ufo-bg-2); border: 1px solid var(--ufo-border);
      border-radius: 6px; padding: 16px; max-width: 560px; width: 90%;
      max-height: 80vh; overflow: auto;
    }
    .modal h3 { margin: 0 0 8px; }
    .modal .meta { color: var(--ufo-fg-dim); font-size: 11px; margin-bottom: 10px; }

    /* ─── assistant ─── */
    .asst-toggle {
      position: absolute; right: 12px; bottom: 12px; z-index: 30;
      background: var(--ufo-accent); color: #001; border: none;
      padding: 8px 14px; border-radius: 4px; font-weight: 700;
      letter-spacing: 0.08em; text-transform: uppercase; font-size: 11px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4); cursor: pointer;
    }
    .asst {
      position: absolute; top: 0; right: 0; bottom: 0; width: 420px; max-width: 95%;
      background: var(--ufo-bg-2); border-left: 1px solid var(--ufo-border);
      display: flex; flex-direction: column; z-index: 40;
      transform: translateX(100%); transition: transform 0.2s ease;
    }
    .asst.open { transform: translateX(0); }
    .asst header {
      padding: 10px 12px; border-bottom: 1px solid var(--ufo-border);
      display: flex; align-items: center; gap: 8px;
    }
    .asst header .title {
      font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; font-size: 11px;
    }
    .asst .messages { flex: 1; overflow-y: auto; padding: 10px 12px; }
    .asst .msg { margin-bottom: 12px; }
    .asst .msg .who { font-size: 10px; color: var(--ufo-fg-dim); letter-spacing: 0.08em;
                      text-transform: uppercase; margin-bottom: 3px; }
    .asst .msg.user .who { color: var(--ufo-accent); }
    .asst .msg.asst .who { color: var(--ufo-accent-2); }
    .asst .msg .content { white-space: pre-wrap; }
    .asst .msg .sources { margin-top: 6px; font-size: 11px; }
    .asst .msg .sources a { display: block; }
    .asst .input {
      border-top: 1px solid var(--ufo-border); padding: 8px;
      display: grid; grid-template-columns: 1fr auto; gap: 6px;
    }
    .asst textarea { resize: none; min-height: 38px; max-height: 120px; }
    .asst .hint { padding: 4px 12px; font-size: 10px; color: var(--ufo-fg-dim); }
    .asst .creds {
      padding: 16px; color: var(--ufo-fg-dim); font-size: 12px; line-height: 1.6;
    }
    .asst .creds code { background: var(--ufo-bg-3); padding: 1px 4px; border-radius: 2px; }

    /* ─── settings ─── */
    .settings { padding: 14px 16px; max-width: 720px; }
    .settings label { display: block; margin: 10px 0 4px; font-size: 11px;
                      color: var(--ufo-fg-dim); letter-spacing: 0.08em; text-transform: uppercase; }
    .settings input[type="text"], .settings input[type="url"],
    .settings input[type="number"], .settings input[type="password"],
    .settings select, .settings textarea { width: 100%; }
    .settings .actions { margin-top: 16px; display: flex; gap: 8px; }

    .audit pre {
      background: var(--ufo-bg-2); border: 1px solid var(--ufo-border);
      padding: 8px; max-height: 50vh; overflow: auto; font-size: 11px;
    }

    .empty { padding: 24px; text-align: center; color: var(--ufo-fg-dim); }
    .sr-only {
      position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
      overflow: hidden; clip: rect(0,0,0,0); border: 0;
    }
  `;

  const HTML = /* html */`
    <div class="chrome" role="banner">
      <div class="brand" aria-label="UFO Intelligence Widget">
        <span class="dot" aria-hidden="true"></span>UFO·INTEL
      </div>
      <div class="status" id="status" aria-live="polite">initialising…</div>
      <div class="spacer"></div>
      <button id="btn-refresh" title="Refresh now (R)">Refresh</button>
      <button id="btn-export"  title="Export filtered data">Export</button>
      <button id="btn-settings" aria-label="Open settings">Settings</button>
    </div>

    <div class="tabs" role="tablist" aria-label="View selector">
      <button class="tab" role="tab" data-view="feed"      aria-selected="true">Feed</button>
      <button class="tab" role="tab" data-view="files"     aria-selected="false">Files</button>
      <button class="tab" role="tab" data-view="timeline"  aria-selected="false">Timeline</button>
      <button class="tab" role="tab" data-view="network"   aria-selected="false">Network</button>
      <button class="tab" role="tab" data-view="analytics" aria-selected="false">Analytics</button>
      <button class="tab" role="tab" data-view="audit"     aria-selected="false">Audit</button>
      <button class="tab" role="tab" data-view="settings"  aria-selected="false">Settings</button>
    </div>

    <div class="ticker" id="ticker" aria-live="polite" aria-label="Live intelligence ticker"></div>

    <div class="body">
      <aside class="sidebar" aria-label="Feed">
        <ul class="feed" id="feed" role="list"></ul>
      </aside>
      <section class="main" id="main" role="main" aria-live="polite"></section>
    </div>

    <button class="asst-toggle" id="btn-asst" aria-controls="asst" aria-expanded="false">
      Assistant
    </button>
    <aside class="asst" id="asst" role="complementary" aria-label="Agentic assistant" aria-hidden="true">
      <header>
        <span class="title">Agentic Assistant</span>
        <span class="spacer" style="flex:1"></span>
        <button id="btn-asst-export" title="Export transcript">Export</button>
        <button id="btn-asst-clear"  title="Clear conversation">Clear</button>
        <button id="btn-asst-close"  title="Close">×</button>
      </header>
      <div class="messages" id="asst-messages" role="log"></div>
      <div class="hint" id="asst-hint"></div>
      <form class="input" id="asst-form">
        <textarea id="asst-input" rows="2" placeholder="Ask the analyst…  (try /summarize, /report, /risk anomaly)"
                  aria-label="Assistant prompt"></textarea>
        <button type="submit" id="asst-send">Send</button>
      </form>
    </aside>
  `;

  // ────────────────────────────────────────────────────────────────────────
  //  Web Component
  // ────────────────────────────────────────────────────────────────────────

  class UfoIntelWidget extends HTMLElement {
    static get observedAttributes() {
      return ['data-source','refresh-interval','theme','llm-provider','llm-model','api-key'];
    }

    constructor() {
      super();
      this._shadow = this.attachShadow({ mode: 'open' });
      this._state  = {
        view:        'feed',
        records:     [],
        notes:       [],
        alerts:      [],
        lastFetchAt: 0,
        lastError:   null,
        stale:       false,
        filter:      { q: '', agency: '', type: '' },
        modalRecord: null,
        chat:        [],     // [{role:'user'|'assistant'|'system', content, sources?}]
        streaming:   false,
      };
      this._timer = null;
      this._index = null;    // TfIdfIndex
    }

    // ── lifecycle ────────────────────────────────────────────────────────
    connectedCallback() {
      const style = document.createElement('style');
      style.textContent = STYLE;
      this._shadow.appendChild(style);
      const tpl = document.createElement('div');
      tpl.innerHTML = HTML;
      while (tpl.firstChild) this._shadow.appendChild(tpl.firstChild);

      this._cacheRefs();
      this._bindEvents();
      this._loadCachedKey();
      this._restoreNotes().then(() => this._boot());
    }

    disconnectedCallback() {
      if (this._timer) clearInterval(this._timer);
    }

    attributeChangedCallback(_name, _old, _val) {
      // Restart the polling loop when relevant attributes change.
      if (this._timer) {
        clearInterval(this._timer);
        this._scheduleRefresh();
      }
    }

    // ── plumbing ─────────────────────────────────────────────────────────
    _cacheRefs() {
      const $ = (id) => this._shadow.getElementById(id);
      this.$ = {
        status:  $('status'),
        ticker:  $('ticker'),
        feed:    $('feed'),
        main:    $('main'),
        asst:    $('asst'),
        asstMsg: $('asst-messages'),
        asstIn:  $('asst-input'),
        asstHint:$('asst-hint'),
        asstForm:$('asst-form'),
      };
      this.$tabs = [...this._shadow.querySelectorAll('.tab')];
    }

    _bindEvents() {
      const sh = this._shadow;
      sh.getElementById('btn-refresh').addEventListener('click', () => this._fetch());
      sh.getElementById('btn-export').addEventListener('click', () => this._exportData());
      sh.getElementById('btn-settings').addEventListener('click', () => this._setView('settings'));
      sh.getElementById('btn-asst').addEventListener('click', () => this._toggleAssistant());
      sh.getElementById('btn-asst-close').addEventListener('click', () => this._toggleAssistant(false));
      sh.getElementById('btn-asst-clear').addEventListener('click', () => {
        this._state.chat = [];
        this._renderAssistant();
        audit('assistant.clear');
      });
      sh.getElementById('btn-asst-export').addEventListener('click', () => this._exportTranscript());
      this.$.asstForm.addEventListener('submit', (e) => { e.preventDefault(); this._submitChat(); });
      this.$.asstIn.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this._submitChat(); }
      });

      this.$tabs.forEach((t) => t.addEventListener('click', () => this._setView(t.dataset.view)));

      // Keyboard shortcuts at host level
      this.addEventListener('keydown', (e) => {
        if (e.target && e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'r' || e.key === 'R') this._fetch();
        if (e.key === '/' && !e.shiftKey) {
          e.preventDefault();
          this._toggleAssistant(true);
          this.$.asstIn.focus();
        }
        if (e.key === 'Escape' && this._state.modalRecord) {
          this._state.modalRecord = null; this._render();
        }
      });
    }

    _attr(name, fallback) {
      const v = this.getAttribute(name);
      return v == null || v === '' ? fallback : v;
    }

    _loadCachedKey() {
      const k = this.getAttribute('api-key') || _deobfuscate(localStorage.getItem(LS_PREFIX + 'apiKey'));
      const p = this.getAttribute('llm-provider') || localStorage.getItem(LS_PREFIX + 'provider') || 'openai';
      const m = this.getAttribute('llm-model')    || localStorage.getItem(LS_PREFIX + 'model')    || '';
      const b = localStorage.getItem(LS_PREFIX + 'baseUrl') || '';
      this._llm = { apiKey: k || '', provider: p, model: m, baseUrl: b };
    }

    _saveLlm() {
      const { apiKey, provider, model, baseUrl } = this._llm;
      if (apiKey)  localStorage.setItem(LS_PREFIX + 'apiKey', _obfuscate(apiKey));
      else         localStorage.removeItem(LS_PREFIX + 'apiKey');
      localStorage.setItem(LS_PREFIX + 'provider', provider);
      localStorage.setItem(LS_PREFIX + 'model', model);
      localStorage.setItem(LS_PREFIX + 'baseUrl', baseUrl || '');
    }

    async _restoreNotes() {
      try { this._state.notes = await idbAll('notes'); } catch { this._state.notes = []; }
    }

    // ── data lifecycle ───────────────────────────────────────────────────
    async _boot() {
      // First, paint stale cache instantly so the analyst is never staring at a blank widget.
      try {
        const cached = await idbGet('cache', 'dataset');
        if (cached && cached.records) {
          this._state.records   = cached.records;
          this._state.lastFetchAt = cached.ts;
          this._state.stale     = true;
          this._reindex();
          this._render();
          this._setStatus(`cached · ${fmtDate(cached.ts)}`, 'stale');
        }
      } catch (_) { /* no cache yet */ }
      this._fetch();
      this._scheduleRefresh();
    }

    _scheduleRefresh() {
      const ms = Math.max(5000, parseInt(this._attr('refresh-interval', String(DEFAULT_REFRESH_MS)), 10) || DEFAULT_REFRESH_MS);
      this._timer = setInterval(() => this._fetch(), ms);
    }

    async _fetch() {
      const sources = this._sources();
      if (!sources.length) {
        this._setStatus('configure a data source in Settings', 'stale');
        if (!this._state.records.length && this.$.main) {
          this.$.main.innerHTML =
            '<h2>Welcome</h2><p style="color:var(--ufo-fg-dim)">' +
            'No data source configured. Open <strong>Settings</strong> to point the widget at a ' +
            'JSON, RSS or Atom feed, or use the demo&rsquo;s <em>Load sample dataset</em> button.</p>';
        }
        return;
      }
      this._setStatus('fetching…');
      const all = [];
      let anyOk = false;
      for (const src of sources) {
        try {
          const r = await fetch(src, { cache: 'no-store' });
          if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
          const ct = r.headers.get('content-type') || '';
          let payload;
          if (ct.includes('xml') || src.endsWith('.xml') || src.endsWith('.rss')) {
            payload = this._parseRss(await r.text());
          } else {
            payload = await r.json();
          }
          const records = normaliseDataset(payload).map((rec) => ({ ...rec, _source: src }));
          all.push(...records);
          anyOk = true;
        } catch (err) {
          this._state.alerts.unshift({
            id: `alert:${Date.now()}:${Math.random()}`, agency: 'SYS', type: 'alert',
            title: `Source unreachable: ${src}`, blurb: String(err.message || err),
            releaseDate: new Date().toISOString(),
          });
          audit('fetch.error', { src, err: String(err) });
        }
      }
      if (anyOk) {
        // Detect new tranche
        const prevIds = new Set(this._state.records.map((r) => r.id));
        const fresh   = all.filter((r) => !prevIds.has(r.id));
        if (fresh.length && this._state.records.length) {
          this._state.alerts.unshift({
            id: `alert:tranche:${Date.now()}`, agency: 'SYS', type: 'alert',
            title: `New tranche detected: ${fresh.length} item(s)`,
            blurb: fresh.slice(0, 5).map((r) => `${r.agency} — ${r.title}`).join(' · '),
            releaseDate: new Date().toISOString(),
          });
          audit('tranche.detected', { count: fresh.length });
        }
        this._state.records     = all;
        this._state.lastFetchAt = Date.now();
        this._state.stale       = false;
        this._state.lastError   = null;
        try { await idbPut('cache', { records: all, ts: this._state.lastFetchAt }, 'dataset'); } catch {}
        this._setStatus(`live · ${fmtDate(this._state.lastFetchAt)} · ${all.length} records`);
        audit('fetch.ok', { count: all.length });
        this._reindex();
      } else {
        this._state.stale     = true;
        this._state.lastError = 'all sources failed';
        this._setStatus('offline · showing cached', 'error');
      }
      this._render();
    }

    _sources() {
      const cfg = (localStorage.getItem(LS_PREFIX + 'sources') || '').trim();
      if (cfg) return cfg.split(/\s+/).filter(Boolean);
      const attr = this._attr('data-source', DEFAULT_SOURCE);
      return attr.split(/\s+/).filter(Boolean);
    }

    _parseRss(text) {
      // Tiny RSS/Atom parser — best-effort, never throws.
      const items = [];
      try {
        const doc = new DOMParser().parseFromString(text, 'application/xml');
        const nodes = [...doc.querySelectorAll('item, entry')];
        for (const n of nodes) {
          const get = (s) => n.querySelector(s)?.textContent?.trim() || '';
          items.push({
            title:       get('title'),
            url:         n.querySelector('link')?.getAttribute('href') || get('link'),
            blurb:       get('description') || get('summary'),
            agency:      get('author') || 'RSS',
            releaseDate: get('pubDate') || get('updated') || get('published'),
            type:        'rss',
          });
        }
      } catch (_) { /* ignore — return whatever we got */ }
      return { items };
    }

    _reindex() {
      this._index = new TfIdfIndex(this._state.records);
    }

    // ── rendering ────────────────────────────────────────────────────────
    _setStatus(text, kind) {
      this.$.status.textContent = text;
      this.$.status.className   = 'status' + (kind ? ' ' + kind : '');
    }

    _setView(view) {
      this._state.view = view;
      this.$tabs.forEach((t) => t.setAttribute('aria-selected', String(t.dataset.view === view)));
      this._render();
    }

    _filtered() {
      const { q, agency, type } = this._state.filter;
      const ql = q.toLowerCase();
      return this._state.records.filter((r) => {
        if (agency && r.agency !== agency) return false;
        if (type   && r.type   !== type)   return false;
        if (!ql) return true;
        return (r.title + ' ' + r.blurb + ' ' + r.agency).toLowerCase().includes(ql);
      });
    }

    _render() {
      this._renderTicker();
      this._renderFeed();
      this._renderMain();
      this._renderAssistantHint();
    }

    _renderTicker() {
      const top = [
        ...this._state.alerts.slice(0, 3),
        ...this._state.records.slice(0, 8),
      ];
      this.$.ticker.innerHTML = top.map((r) => (
        `<span>[${$esc(r.agency)}] ${$esc(r.title)} · ${$esc(fmtDate(r.releaseDate))}</span>`
      )).join('') || '<span>no live items — awaiting first sync</span>';
    }

    _renderFeed() {
      const items = [
        ...this._state.alerts.slice(0, 5).map((a) => ({ ...a, _kind: 'alert' })),
        ...this._state.notes.slice().reverse().map((n) => ({
          id: 'note:' + n.id, agency: 'NOTE', type: 'note',
          title: n.title, blurb: n.body, releaseDate: n.ts, _kind: 'note',
        })),
        ...this._state.records.slice(0, 60).map((r) => ({ ...r, _kind: 'rec' })),
      ];
      if (!items.length) {
        this.$.feed.innerHTML = '<li class="empty">No items yet.</li>';
        return;
      }
      this.$.feed.innerHTML = items.map((r, i) => `
        <li role="listitem" tabindex="0" data-idx="${i}">
          <div class="row1">
            <span class="agency ${r._kind === 'note' ? 'note' : r._kind === 'alert' ? 'alert' : ''}">${$esc(r.agency)}</span>
            <span class="when">${$esc(fmtDate(r.releaseDate))}</span>
          </div>
          <strong class="title">${$esc(r.title)}</strong>
          ${r.blurb ? `<div class="blurb">${$esc(r.blurb.slice(0, 140))}${r.blurb.length > 140 ? '…' : ''}</div>` : ''}
        </li>`).join('');
      // Attach click handlers
      this.$.feed.querySelectorAll('li[data-idx]').forEach((li) => {
        li.addEventListener('click', () => this._openItem(items[+li.dataset.idx]));
        li.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this._openItem(items[+li.dataset.idx]); }
        });
      });
    }

    _openItem(r) {
      if (!r) return;
      this._state.modalRecord = r;
      this._render();
      audit('item.open', { id: r.id, title: r.title });
    }

    _renderMain() {
      const v = this._state.view;
      const m = this.$.main;
      let html = '';
      switch (v) {
        case 'feed':      html = this._viewFeed();      break;
        case 'files':     html = this._viewFiles();     break;
        case 'timeline':  html = this._viewTimeline();  break;
        case 'network':   html = this._viewNetwork();   break;
        case 'analytics': html = this._viewAnalytics(); break;
        case 'audit':     html = this._viewAudit();     break;
        case 'settings':  html = this._viewSettings();  break;
        default:          html = this._viewFeed();
      }
      m.innerHTML = html;
      this._bindMainEvents();
      this._renderModal();
    }

    _renderModal() {
      const r = this._state.modalRecord;
      const existing = this._shadow.querySelector('.modal-backdrop');
      if (existing) existing.remove();
      if (!r) return;
      const div = document.createElement('div');
      div.className = 'modal-backdrop';
      div.setAttribute('role', 'dialog');
      div.setAttribute('aria-modal', 'true');
      div.innerHTML = `<div class="modal">
        <h3>${$esc(r.title)}</h3>
        <div class="meta">${$esc(r.agency)} · ${$esc(fmtDate(r.releaseDate))} · ${$esc(r.type || '—')}</div>
        <p>${$esc(r.blurb || 'No description provided.')}</p>
        ${r.url && r.url !== '#' ? `<p><a href="${$esc(r.url)}" target="_blank" rel="noopener">Open source ↗</a></p>` : ''}
        <div style="text-align:right;margin-top:10px;">
          <button data-act="ask">Ask assistant about this</button>
          <button data-act="close">Close</button>
        </div>
      </div>`;
      div.addEventListener('click', (e) => {
        if (e.target === div) { this._state.modalRecord = null; this._render(); }
      });
      div.querySelector('[data-act="close"]').addEventListener('click', () => {
        this._state.modalRecord = null; this._render();
      });
      div.querySelector('[data-act="ask"]').addEventListener('click', () => {
        this._toggleAssistant(true);
        this.$.asstIn.value = `Tell me about: ${r.title}`;
        this.$.asstIn.focus();
        this._state.modalRecord = null; this._render();
      });
      this._shadow.appendChild(div);
    }

    // ── view templates ───────────────────────────────────────────────────
    _viewFeed() {
      const recs = this._state.records;
      const agencies = [...new Set(recs.map((r) => r.agency))].length;
      const noteCount  = this._state.notes.length;
      const stale = this._state.stale
        ? `<div class="empty" style="background:rgba(255,184,107,0.08);color:var(--ufo-accent-2);border-radius:4px;padding:8px;margin-bottom:10px;">
             Stale data — last successful fetch ${fmtDate(this._state.lastFetchAt)}.
           </div>` : '';
      return `
        <h2>Intelligence Overview</h2>
        ${stale}
        <div class="stats">
          <div class="stat"><div class="v">${recs.length}</div><div class="k">Records</div></div>
          <div class="stat"><div class="v">${agencies}</div><div class="k">Agencies</div></div>
          <div class="stat"><div class="v">${this._state.alerts.length}</div><div class="k">System alerts</div></div>
          <div class="stat"><div class="v">${noteCount}</div><div class="k">Analyst notes</div></div>
        </div>
        <h2>Add analyst note</h2>
        <form id="note-form">
          <input type="text" id="note-title" placeholder="Note title" required style="width:100%;margin-bottom:6px;">
          <textarea id="note-body" rows="3" placeholder="Body — markdown plain text"
                    style="width:100%;"></textarea>
          <div style="margin-top:6px;"><button>Add note</button></div>
        </form>
      `;
    }

    _viewFiles() {
      const recs = this._filtered();
      const agencies = [...new Set(this._state.records.map((r) => r.agency))].sort();
      const types    = [...new Set(this._state.records.map((r) => r.type))].sort();
      return `
        <h2>Files (${recs.length})</h2>
        <div class="toolbar">
          <input type="search" id="filter-q" placeholder="Search title / blurb / agency"
                 value="${$esc(this._state.filter.q)}" aria-label="Search">
          <select id="filter-agency" aria-label="Filter by agency">
            <option value="">All agencies</option>
            ${agencies.map((a) => `<option ${a === this._state.filter.agency ? 'selected' : ''}>${$esc(a)}</option>`).join('')}
          </select>
          <select id="filter-type" aria-label="Filter by type">
            <option value="">All types</option>
            ${types.map((t) => `<option ${t === this._state.filter.type ? 'selected' : ''}>${$esc(t)}</option>`).join('')}
          </select>
        </div>
        <table class="files">
          <thead><tr>
            <th>Date</th><th>Agency</th><th>Type</th><th>Title</th><th>Link</th>
          </tr></thead>
          <tbody>
            ${recs.slice(0, 500).map((r, i) => `
              <tr data-idx="${i}">
                <td>${$esc(fmtDate(r.releaseDate))}</td>
                <td>${$esc(r.agency)}</td>
                <td>${$esc(r.type || '—')}</td>
                <td>${$esc(r.title)}</td>
                <td>${r.url && r.url !== '#' ? `<a href="${$esc(r.url)}" target="_blank" rel="noopener">open ↗</a>` : '—'}</td>
              </tr>`).join('')}
          </tbody>
        </table>
        ${recs.length > 500 ? `<div class="empty">Showing first 500 of ${recs.length}.</div>` : ''}
      `;
    }

    _viewTimeline() {
      const recs = this._filtered().filter((r) => r.releaseDate && !isNaN(Date.parse(r.releaseDate)));
      if (!recs.length) return `<h2>Timeline</h2><div class="empty">No date-bearing records.</div>`;
      const W = 1000, H = 240, P = 30;
      const ts = recs.map((r) => Date.parse(r.releaseDate));
      const tmin = Math.min(...ts), tmax = Math.max(...ts);
      const span = Math.max(1, tmax - tmin);
      const agencies = [...new Set(recs.map((r) => r.agency))];
      const color = (a) => `hsl(${(agencies.indexOf(a) * 47) % 360}, 70%, 60%)`;
      const lanes = agencies;
      const laneH = (H - 2 * P) / Math.max(1, lanes.length);
      const dots = recs.map((r, i) => {
        const x = P + ((Date.parse(r.releaseDate) - tmin) / span) * (W - 2 * P);
        const y = P + lanes.indexOf(r.agency) * laneH + laneH / 2;
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.5"
                  fill="${color(r.agency)}" data-idx="${i}"
                  role="img" aria-label="${$esc(r.title)} ${$esc(fmtDate(r.releaseDate))}">
                  <title>${$esc(r.title)} — ${$esc(r.agency)} — ${$esc(fmtDate(r.releaseDate))}</title>
                </circle>`;
      }).join('');
      const laneLabels = lanes.map((a, i) => `
        <text x="6" y="${(P + i * laneH + laneH / 2 + 3).toFixed(1)}" font-size="10"
              fill="${color(a)}">${$esc(a)}</text>`).join('');
      const ticks = 6;
      const tickEls = Array.from({ length: ticks + 1 }, (_, i) => {
        const t = tmin + (span * i) / ticks;
        const x = P + (i / ticks) * (W - 2 * P);
        return `<line x1="${x}" y1="${P}" x2="${x}" y2="${H - P}" stroke="var(--ufo-grid)"/>
                <text x="${x}" y="${H - P + 14}" font-size="9" fill="var(--ufo-fg-dim)" text-anchor="middle">
                  ${$esc(new Date(t).toISOString().slice(0, 10))}</text>`;
      }).join('');
      return `
        <h2>Timeline (${recs.length})</h2>
        <div class="viz">
          <svg viewBox="0 0 ${W} ${H + 20}" role="img" aria-label="Release timeline">
            ${tickEls}${laneLabels}${dots}
          </svg>
          <div class="legend">
            ${agencies.map((a) => `<span style="--c:${color(a)};"><span style="background:${color(a)}"></span>${$esc(a)}</span>`).join('')}
          </div>
        </div>`;
    }

    _viewNetwork() {
      // Build a co-occurrence graph: nodes = agencies + top keywords; edges where
      // a keyword appears in a record from that agency.
      const recs = this._filtered();
      if (!recs.length) return `<h2>Network</h2><div class="empty">No records to graph.</div>`;
      const idx = new TfIdfIndex(recs);
      const topTerms = idx.topTerms(15).map(([t]) => t);
      const agencies = [...new Set(recs.map((r) => r.agency))];
      const nodes = [
        ...agencies.map((a) => ({ id: 'a:' + a, label: a, kind: 'agency' })),
        ...topTerms.map((t) => ({ id: 'k:' + t, label: t, kind: 'kw' })),
      ];
      const edges = [];
      for (const r of recs) {
        const toks = new Set(tokenise(`${r.title} ${r.blurb}`));
        for (const t of topTerms) if (toks.has(t)) edges.push({ s: 'a:' + r.agency, t: 'k:' + t });
      }
      // Edge weights
      const w = new Map();
      for (const e of edges) { const k = e.s + '|' + e.t; w.set(k, (w.get(k) || 0) + 1); }
      // Simple deterministic layout: agencies on a ring, keywords on inner ring
      const W = 720, H = 420, cx = W / 2, cy = H / 2;
      const r1 = 170, r2 = 90;
      const pos = new Map();
      agencies.forEach((a, i) => {
        const ang = (i / agencies.length) * Math.PI * 2;
        pos.set('a:' + a, [cx + Math.cos(ang) * r1, cy + Math.sin(ang) * r1]);
      });
      topTerms.forEach((t, i) => {
        const ang = (i / topTerms.length) * Math.PI * 2 + 0.2;
        pos.set('k:' + t, [cx + Math.cos(ang) * r2, cy + Math.sin(ang) * r2]);
      });
      // 60 iterations of a tiny force layout for keywords (cheap, deterministic-ish)
      for (let it = 0; it < 40; it++) {
        for (const t of topTerms) {
          const id = 'k:' + t;
          let fx = 0, fy = 0;
          const [x, y] = pos.get(id);
          for (const a of agencies) {
            const k = 'a:' + a + '|' + id;
            const weight = w.get(k) || 0;
            if (!weight) continue;
            const [ax, ay] = pos.get('a:' + a);
            fx += (ax - x) * 0.02 * weight;
            fy += (ay - y) * 0.02 * weight;
          }
          pos.set(id, [x + fx * 0.1, y + fy * 0.1]);
        }
      }
      const edgeEls = [...w.entries()].map(([k, weight]) => {
        const [s, t] = k.split('|');
        const [x1, y1] = pos.get(s), [x2, y2] = pos.get(t);
        return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}"
                       x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"
                       stroke="var(--ufo-accent)" stroke-opacity="${Math.min(0.7, 0.15 + weight * 0.07)}"
                       stroke-width="${Math.min(3, 0.6 + weight * 0.3)}"/>`;
      }).join('');
      const nodeEls = nodes.map((n) => {
        const [x, y] = pos.get(n.id);
        const isAg = n.kind === 'agency';
        return `<g transform="translate(${x.toFixed(1)},${y.toFixed(1)})">
          <circle r="${isAg ? 10 : 6}" fill="${isAg ? 'var(--ufo-accent)' : 'var(--ufo-accent-2)'}"
                  fill-opacity="0.85" stroke="var(--ufo-bg)" stroke-width="1.5"/>
          <text x="${isAg ? 12 : 8}" y="3" font-size="${isAg ? 11 : 10}" fill="var(--ufo-fg)">
            ${$esc(n.label)}</text>
        </g>`;
      }).join('');
      return `
        <h2>Agency × Keyword Network</h2>
        <div class="viz">
          <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Relationship network">
            ${edgeEls}${nodeEls}
          </svg>
          <div class="legend">
            <span style="background:var(--ufo-accent)"></span>Agencies
            <span style="background:var(--ufo-accent-2)"></span>Keywords
          </div>
        </div>`;
    }

    _viewAnalytics() {
      const recs = this._filtered();
      if (!recs.length) return `<h2>Analytics</h2><div class="empty">No records to chart.</div>`;
      // Bar 1: file types per agency (stacked is overkill — show side-by-side bars per agency)
      const byAgency = new Map();
      for (const r of recs) {
        if (!byAgency.has(r.agency)) byAgency.set(r.agency, new Map());
        const m = byAgency.get(r.agency);
        m.set(r.type, (m.get(r.type) || 0) + 1);
      }
      const W = 720, H = 240, P = 30;
      const agencies = [...byAgency.keys()];
      const max = Math.max(...agencies.map((a) => [...byAgency.get(a).values()].reduce((s, v) => s + v, 0)));
      const barW = (W - 2 * P) / Math.max(1, agencies.length);
      const types = [...new Set(recs.map((r) => r.type))];
      const tcolor = (t) => `hsl(${(types.indexOf(t) * 73) % 360}, 60%, 60%)`;
      const bars = agencies.map((a, i) => {
        const x = P + i * barW + 4;
        let y = H - P;
        return types.map((t) => {
          const v = byAgency.get(a).get(t) || 0;
          if (!v) return '';
          const h = (v / max) * (H - 2 * P);
          y -= h;
          return `<rect x="${x}" y="${y}" width="${barW - 8}" height="${h}" fill="${tcolor(t)}">
                    <title>${$esc(a)} · ${$esc(t)} = ${v}</title></rect>`;
        }).join('');
      }).join('');
      const axis = agencies.map((a, i) => `
        <text x="${P + i * barW + barW / 2}" y="${H - P + 14}" font-size="10"
              fill="var(--ufo-fg-dim)" text-anchor="middle">${$esc(a)}</text>`).join('');

      // Bar 2: release cadence (releases per ISO week)
      const weeks = new Map();
      const isoWeek = (dt) => {
        // ISO-8601 week date — week 1 is the week containing the first Thursday.
        const d = new Date(Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth(), dt.getUTCDate()));
        const dayNum = d.getUTCDay() || 7;
        d.setUTCDate(d.getUTCDate() + 4 - dayNum);
        const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
        const week = Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
        return `${d.getUTCFullYear()}-W${String(week).padStart(2, '0')}`;
      };
      for (const r of recs) {
        const d = Date.parse(r.releaseDate);
        if (isNaN(d)) continue;
        const wk = isoWeek(new Date(d));
        weeks.set(wk, (weeks.get(wk) || 0) + 1);
      }
      const wkArr = [...weeks.entries()].sort((a,b) => a[0].localeCompare(b[0]));
      const maxW = Math.max(1, ...wkArr.map((w) => w[1]));
      const W2 = 720, H2 = 160;
      const cellW = (W2 - 2 * P) / Math.max(1, wkArr.length);
      const cad = wkArr.map(([wk, v], i) => {
        const x = P + i * cellW;
        const h = (v / maxW) * (H2 - 2 * P);
        return `<rect x="${x + 1}" y="${H2 - P - h}" width="${Math.max(2, cellW - 2)}" height="${h}"
                      fill="var(--ufo-accent)" fill-opacity="0.7">
                  <title>${$esc(wk)} = ${v}</title></rect>`;
      }).join('');
      // First / middle / last labels
      const labelIdx = wkArr.length > 0 ? [0, Math.floor(wkArr.length/2), wkArr.length-1] : [];
      const cadLabels = labelIdx.map((i) => {
        if (i < 0 || i >= wkArr.length) return '';
        const x = P + i * cellW + cellW / 2;
        return `<text x="${x}" y="${H2 - P + 14}" font-size="10" fill="var(--ufo-fg-dim)"
                       text-anchor="middle">${$esc(wkArr[i][0])}</text>`;
      }).join('');

      return `
        <h2>File-type composition per agency</h2>
        <div class="viz">
          <svg viewBox="0 0 ${W} ${H + 20}" role="img" aria-label="File type composition">${bars}${axis}</svg>
          <div class="legend">
            ${types.map((t) => `<span><span style="background:${tcolor(t)}"></span>${$esc(t)}</span>`).join('')}
          </div>
        </div>
        <h2>Release cadence (per ISO week)</h2>
        <div class="viz">
          <svg viewBox="0 0 ${W2} ${H2 + 20}" role="img" aria-label="Release cadence">${cad}${cadLabels}</svg>
        </div>`;
    }

    async _viewAudit() {
      // (synchronous-looking; we re-render once log is fetched)
      idbAll('audit').then((rows) => {
        const html = `<h2>Audit log (${rows.length})</h2>
          <div class="audit"><pre>${$esc(rows.reverse().map((r) =>
            `${fmtDate(r.ts)}  ${r.event}  ${r.detail ? JSON.stringify(r.detail) : ''}`
          ).join('\n') || '— empty —')}</pre></div>
          <p style="margin-top:8px;color:var(--ufo-fg-dim);font-size:11px;">
            This log lives in your browser only. It is never transmitted.
          </p>`;
        if (this._state.view === 'audit') this.$.main.innerHTML = html;
      });
      return `<h2>Audit log</h2><div class="empty">loading…</div>`;
    }

    _viewSettings() {
      const sources = (localStorage.getItem(LS_PREFIX + 'sources') || this._sources().join('\n'));
      const { provider, model, baseUrl, apiKey } = this._llm;
      return `
        <div class="settings">
          <h2>Data sources</h2>
          <label for="cfg-sources">One URL per line (JSON, RSS or Atom).</label>
          <textarea id="cfg-sources" rows="3">${$esc(sources)}</textarea>

          <h2 style="margin-top:18px;">Refresh interval</h2>
          <label for="cfg-refresh">Milliseconds (min 5000).</label>
          <input type="number" id="cfg-refresh" min="5000" step="1000"
                 value="${$esc(this._attr('refresh-interval', String(DEFAULT_REFRESH_MS)))}">

          <h2 style="margin-top:18px;">Theme</h2>
          <select id="cfg-theme">
            ${['dark','light','compact'].map((t) => `<option ${this.getAttribute('theme')===t?'selected':''}>${t}</option>`).join('')}
          </select>

          <h2 style="margin-top:18px;">LLM provider (BYO key, never transmitted elsewhere)</h2>
          <label for="cfg-provider">Provider</label>
          <select id="cfg-provider">
            <option value="openai"    ${provider==='openai'?'selected':''}>OpenAI</option>
            <option value="anthropic" ${provider==='anthropic'?'selected':''}>Anthropic</option>
            <option value="local"     ${provider==='local'?'selected':''}>Local / OpenAI-compatible</option>
          </select>
          <label for="cfg-model">Model</label>
          <input type="text" id="cfg-model" value="${$esc(model)}" placeholder="e.g. gpt-4o-mini, claude-3-5-sonnet-latest">
          <label for="cfg-base">Base URL (local / proxy only — optional)</label>
          <input type="url" id="cfg-base" value="${$esc(baseUrl || '')}" placeholder="http://localhost:11434">
          <label for="cfg-key">API key</label>
          <input type="password" id="cfg-key" value="${$esc(apiKey || '')}" placeholder="sk-…  or  sk-ant-…">

          <div class="actions">
            <button id="cfg-save">Save</button>
            <button id="cfg-clear-cache">Clear cache</button>
            <button id="cfg-forget-key">Forget API key</button>
          </div>
        </div>`;
    }

    _bindMainEvents() {
      const sh = this._shadow;
      // Note form
      const noteForm = sh.getElementById('note-form');
      if (noteForm) {
        noteForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          const title = sh.getElementById('note-title').value.trim();
          const body  = sh.getElementById('note-body').value.trim();
          if (!title) return;
          const rec = { title, body, ts: new Date().toISOString() };
          try { await idbPut('notes', rec); } catch {}
          this._state.notes.push(rec);
          audit('note.add', { title });
          await this._restoreNotes();
          this._render();
        });
      }
      // Filters
      const q  = sh.getElementById('filter-q');
      const ag = sh.getElementById('filter-agency');
      const tp = sh.getElementById('filter-type');
      if (q)  q.addEventListener('input', debounce((e) => { this._state.filter.q = e.target.value; this._render(); }, 150));
      if (ag) ag.addEventListener('change', (e) => { this._state.filter.agency = e.target.value; this._render(); });
      if (tp) tp.addEventListener('change', (e) => { this._state.filter.type   = e.target.value; this._render(); });
      // File rows clickable
      sh.querySelectorAll('table.files tr[data-idx]').forEach((tr, _, all) => {
        tr.addEventListener('click', () => {
          const idx = +tr.dataset.idx;
          this._openItem(this._filtered()[idx]);
        });
      });
      // Settings
      const save = sh.getElementById('cfg-save');
      if (save) {
        save.addEventListener('click', () => {
          const src   = sh.getElementById('cfg-sources').value.trim();
          const ref   = sh.getElementById('cfg-refresh').value.trim();
          const theme = sh.getElementById('cfg-theme').value;
          localStorage.setItem(LS_PREFIX + 'sources', src);
          if (ref) this.setAttribute('refresh-interval', ref);
          this.setAttribute('theme', theme);
          this._llm.provider = sh.getElementById('cfg-provider').value;
          this._llm.model    = sh.getElementById('cfg-model').value.trim();
          this._llm.baseUrl  = sh.getElementById('cfg-base').value.trim();
          this._llm.apiKey   = sh.getElementById('cfg-key').value.trim();
          this._saveLlm();
          audit('settings.save', { provider: this._llm.provider, model: this._llm.model });
          this._fetch();
          this._setView('feed');
        });
        sh.getElementById('cfg-clear-cache').addEventListener('click', async () => {
          try { await idbPut('cache', null, 'dataset'); } catch {}
          audit('cache.clear');
          this._setStatus('cache cleared');
        });
        sh.getElementById('cfg-forget-key').addEventListener('click', () => {
          this._llm.apiKey = '';
          localStorage.removeItem(LS_PREFIX + 'apiKey');
          sh.getElementById('cfg-key').value = '';
          audit('key.forget');
        });
      }
    }

    // ── exports ──────────────────────────────────────────────────────────
    _exportData() {
      const recs = this._filtered();
      const csv  = this._toCsv(recs);
      const ts   = new Date().toISOString().replace(/[:.]/g, '-');
      this._download(`ufo-intel-${ts}.json`, 'application/json', JSON.stringify(recs, null, 2));
      this._download(`ufo-intel-${ts}.csv`,  'text/csv', csv);
      audit('export.data', { count: recs.length });
    }
    _toCsv(rows) {
      if (!rows.length) return '';
      const cols = ['releaseDate','agency','type','title','url','blurb'];
      const esc  = (v) => `"${String(v == null ? '' : v).replace(/"/g, '""')}"`;
      return [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n');
    }
    _exportTranscript() {
      const txt = this._state.chat.map((m) =>
        `## ${m.role}\n${m.content}${m.sources?.length ? '\n\nSources:\n' + m.sources.map((s) => `- ${s.title} (${s.url})`).join('\n') : ''}`
      ).join('\n\n---\n\n');
      this._download(`ufo-intel-chat-${new Date().toISOString().replace(/[:.]/g,'-')}.md`, 'text/markdown', txt);
      audit('export.transcript');
    }
    _download(name, mime, data) {
      const blob = new Blob([data], { type: mime });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }

    // ── assistant ────────────────────────────────────────────────────────
    _toggleAssistant(force) {
      const open = force === undefined ? !this.$.asst.classList.contains('open') : !!force;
      this.$.asst.classList.toggle('open', open);
      this.$.asst.setAttribute('aria-hidden', String(!open));
      this._shadow.getElementById('btn-asst').setAttribute('aria-expanded', String(open));
      if (open) this._renderAssistant();
    }
    _renderAssistantHint() {
      const hints = SLASH_COMMANDS.map((s) => `${s.cmd}`).join('  ');
      this.$.asstHint.textContent = `Slash commands: ${hints}`;
    }
    _renderAssistant() {
      if (!this._llm.apiKey) {
        this.$.asstMsg.innerHTML = `
          <div class="creds">
            <p><strong>No API key configured.</strong></p>
            <p>The assistant runs entirely in your browser. Open <em>Settings → LLM provider</em>
               and add a key for OpenAI, Anthropic, or a local OpenAI-compatible endpoint
               (Ollama / LM Studio / llama.cpp).</p>
            <p>Your key is stored only in <code>localStorage</code> on this device and is
               only transmitted to the provider URL you configure.</p>
            <p>Once configured, try <code>/summarize</code>, <code>/report</code>,
               <code>/risk anomaly</code>, <code>/timeline</code>, <code>/find</code>.</p>
          </div>`;
        return;
      }
      this.$.asstMsg.innerHTML = this._state.chat.map((m) => `
        <div class="msg ${m.role === 'user' ? 'user' : 'asst'}">
          <div class="who">${m.role === 'user' ? 'you' : 'analyst'}</div>
          <div class="content">${$esc(m.content)}</div>
          ${m.sources?.length ? `<div class="sources">Sources:<br>${
            m.sources.map((s) =>
              `<a href="${$esc(s.url)}" target="_blank" rel="noopener">${$esc(s.title)} — ${$esc(s.agency)}</a>`
            ).join('')
          }</div>` : ''}
        </div>`).join('') || `<div class="hint">Ask anything about the loaded dataset.</div>`;
      this.$.asstMsg.scrollTop = this.$.asstMsg.scrollHeight;
    }

    async _submitChat() {
      if (this._state.streaming) return;
      const text = this.$.asstIn.value.trim();
      if (!text) return;
      if (!this._llm.apiKey) { this._renderAssistant(); return; }
      this.$.asstIn.value = '';
      const ctx = this._buildContext(text);
      this._state.chat.push({ role: 'user', content: text });
      this._state.chat.push({ role: 'assistant', content: '', sources: ctx.sources });
      this._renderAssistant();
      audit('llm.query', { provider: this._llm.provider, model: this._llm.model, q: text.slice(0, 200) });
      const systemPrompt = this._systemPrompt(ctx.contextBlock);
      const messages = [
        { role: 'system', content: systemPrompt },
        ...this._state.chat.slice(0, -1).filter((m) => m.role !== 'system').map((m) => ({
          role: m.role, content: m.content,
        })),
      ];
      const lastIdx = this._state.chat.length - 1;
      this._state.streaming = true;
      try {
        const stream = LLM[this._llm.provider]?.({ ...this._llm, messages });
        if (!stream) throw new Error(`Unknown provider: ${this._llm.provider}`);
        for await (const chunk of stream) {
          this._state.chat[lastIdx].content += chunk;
          this._renderAssistant();
        }
      } catch (err) {
        this._state.chat[lastIdx].content += `\n\n[error: ${err.message || err}]`;
        this._renderAssistant();
        audit('llm.error', { err: String(err) });
      } finally {
        this._state.streaming = false;
      }
    }

    _systemPrompt(ctx) {
      return [
        'You are an intelligence analyst assistant operating inside a browser widget.',
        'You analyse declassified UFO/UAP release metadata. Be precise, concise, and cite sources by title.',
        'Never fabricate file URLs. When unsure, say so. Output plain text or markdown.',
        '',
        '=== CONTEXT (retrieved from the local dataset) ===',
        ctx,
        '=== END CONTEXT ===',
      ].join('\n');
    }

    /** Build retrieval context and rewrite slash-commands into plain prompts. */
    _buildContext(rawQuery) {
      let query = rawQuery;
      const recs = this._state.records;
      // Slash-command expansion
      if (rawQuery.startsWith('/')) {
        const [cmd, ...rest] = rawQuery.split(/\s+/);
        const arg = rest.join(' ');
        switch (cmd) {
          case '/summarize':
            query = 'Summarise the most recent releases. Group by agency, highlight notable items.';
            break;
          case '/compare':
            query = `Compare the release patterns across agencies. ${arg}`;
            break;
          case '/report':
            query = 'Produce a structured intelligence report with the sections: Executive Summary, Key Releases, Agencies Involved, Notable Keywords, Risks, Recommendations.';
            break;
          case '/risk':
            query = `Provide a heuristic risk assessment for the term "${arg || 'anomaly'}" based on its occurrences in the dataset. Include severity rating and reasoning.`;
            break;
          case '/timeline':
            query = 'Describe the timeline of releases chronologically. Note clustering periods.';
            break;
          case '/find':
            query = `Identify recurring patterns or clusters related to: ${arg || 'the overall dataset'}.`;
            break;
        }
      }
      // Retrieval
      let sources = [];
      let contextBlock = '';
      if (this._index && recs.length) {
        const hits = this._index.query(query, 6);
        sources = hits.map((h) => h.record);
        contextBlock = hits.map((h, i) => {
          const r = h.record;
          return `[${i + 1}] (${r.agency}) ${r.title}\n    date: ${r.releaseDate || '?'}\n    url:  ${r.url}\n    blurb: ${(r.blurb || '').slice(0, 320)}`;
        }).join('\n\n');
        if (!contextBlock) {
          // Fall back to a small slice of the freshest items
          sources = recs.slice(0, 6);
          contextBlock = sources.map((r, i) =>
            `[${i + 1}] (${r.agency}) ${r.title} — ${r.releaseDate} — ${r.url}`).join('\n');
        }
      } else {
        contextBlock = '(no dataset loaded)';
      }
      return { query, sources, contextBlock };
    }
  }

  customElements.define('ufo-intel-widget', UfoIntelWidget);
})();
