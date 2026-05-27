# `<ufo-intel-widget>` — embeddable UFO/UAP intelligence widget

A self-contained, zero-backend Web Component that turns a JSON or RSS feed of
declassified UFO / UAP releases into an analyst-grade live intelligence
console, complete with a BYO-key agentic assistant.

> Live demo: <https://dascient.github.io/UACP/widget/>

## Embed

```html
<script type="module"
        src="https://dascient.github.io/UACP/widget/ufo-intel-widget.js"></script>

<ufo-intel-widget
  data-source="https://your-mirror.example/ufo/index.json"
  refresh-interval="30000"
  theme="dark"
  llm-provider="openai"
  llm-model="gpt-4o-mini"></ufo-intel-widget>
```

## Attributes

| Attribute          | Default                    | Notes                                                              |
| ------------------ | -------------------------- | ------------------------------------------------------------------ |
| `data-source`      | demo placeholder           | One or more whitespace-separated URLs (JSON, RSS, Atom).           |
| `refresh-interval` | `30000`                    | Milliseconds between polls (minimum 5 000).                        |
| `theme`            | `dark`                     | `dark`, `light`, or `compact`.                                     |
| `llm-provider`     | `openai`                   | `openai`, `anthropic`, or `local` (OpenAI-compatible).             |
| `llm-model`        | `""`                       | Model identifier passed to the provider.                           |
| `api-key`          | `""` (use Settings dialog) | Optional inline key — prefer the in-widget Settings panel instead. |

## Features

- **Live intelligence feed** merging file releases, analyst notes (added
  in-widget), and system alerts (new-tranche detection, source-down).
- **Deep analytics tabs**: hand-rolled SVG timeline, agency × keyword
  network graph, file-type composition bars, and release-cadence histogram.
  All update live and respect the active filter.
- **Agentic assistant** (slide-out panel) with BYO-key support for OpenAI,
  Anthropic, and any OpenAI-compatible local endpoint (Ollama, LM Studio,
  llama.cpp). Streams responses where the provider supports SSE.
- **Client-side RAG**: TF-IDF retrieval over the loaded dataset (no
  external embeddings model required). Slash commands: `/summarize`,
  `/compare`, `/report`, `/risk <term>`, `/timeline`, `/find <term>`.
- **Offline resilience**: IndexedDB cache, stale banner with last-fetch
  timestamp.
- **Security**: No telemetry, no external CDN dependencies. API keys live
  only in `localStorage` on the user's device.
- **Accessibility**: Shadow-DOM-isolated styles, ARIA roles, keyboard nav
  (`R` refresh, `/` open assistant, `Esc` close modal), `prefers-reduced-motion`.
- **Audit log**: Local-only record of every fetch, query, and user
  action — viewable in the Audit tab, never exfiltrated.
- **Export**: CSV/JSON of the filtered dataset and Markdown chat transcripts.

## Data schema

Each release record is normalised to:

```ts
{ id, title, agency, url, blurb, releaseDate, type }
```

The loader accepts any of:

- `{ pdfs: [...], images: [...], videos: [...] }` (the war.gov-mirror shape),
- `{ items: [...] }`,
- a top-level array `[ … ]`, or
- RSS/Atom XML (auto-detected from `Content-Type` or file extension).

## Privacy & security

- Everything (data fetching, embeddings, retrieval) runs in the user's
  browser. The only outbound network traffic is (a) the configured
  data source URL(s), and (b) the configured LLM provider URL when the
  user submits a chat message.
- The widget never reads or writes cookies. Styles are isolated in
  Shadow DOM so it cannot collide with host CSS.

## Development

The widget is a single dependency-free ES module at
[`ufo-intel-widget.js`](./ufo-intel-widget.js). Open
[`index.html`](./index.html) directly in any modern browser — no build
step required.

Deployed to GitHub Pages via the repository's existing
[`.github/workflows/static_deploy.yml`](../../../../.github/workflows/static_deploy.yml)
which copies `services/web-dashboard/public/` into the published site root.
