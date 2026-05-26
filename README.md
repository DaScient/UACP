# Global UAP Intelligence Hub

> A polyglot, decentralized, open-source platform for ingesting, classifying, and
> publicly visualising Unidentified Anomalous Phenomena (UAP) observations from
> sensors, eyewitnesses, and declassified government data sources.

---

## 1. Polyglot Architecture Philosophy

The hub is intentionally built as a **polyglot, decentralised system** — each
service uses the language best suited to its workload. This avoids the "one
language, one runtime" lock-in that hurts both scientific accuracy and
contributor diversity.

| Service | Language | Why |
| --- | --- | --- |
| `services/math-engine` | **Rust** (→ WASM) | Numerically stable, memory-safe, runs in-browser via WebAssembly for client-side parallax. |
| `services/ingestion-worker` | **Python 3.11** | Best-in-class ML/AI ecosystem (PyTorch, transformers, OpenCV, `unstructured`). |
| `services/api-gateway` | **Go** | High-throughput, low-allocation HTTP/gRPC fan-in for sensor uploads. |
| `services/web-dashboard` | **TypeScript / Next.js** | Static-exportable, deploys cleanly to GitHub Pages. |
| `data-schemas/*.proto` | **Protocol Buffers** | One canonical contract, generated bindings for every language above. |

The single source of truth across all services is
[`data-schemas/telemetry.proto`](data-schemas/telemetry.proto). No service is
allowed to invent its own internal representation of a `UapEvent`.

```
                ┌──────────────────┐
                │  web-dashboard   │  (Next.js static export → GitHub Pages)
                │   (TS / WASM)    │
                └────────▲─────────┘
                         │  REST / static schemas
                ┌────────┴─────────┐
                │   api-gateway    │  (Go)
                └────────▲─────────┘
                         │  gRPC / SQS
        ┌────────────────┼──────────────────┐
        │                │                  │
┌───────▼──────┐ ┌───────▼──────┐  ┌────────▼────────┐
│ ingestion-   │ │  math-engine │  │ object / vector │
│ worker (Py)  │ │   (Rust)     │  │ store (MinIO /  │
│              │ │              │  │  ChromaDB)      │
└──────────────┘ └──────────────┘  └─────────────────┘
```

---

## 2. Prerequisite Installations

You only need *all* of these if you intend to hack on *every* service. For
single-service contributions, install only the runtime for that service.

| Tool | Version | Install |
| --- | --- | --- |
| Python | 3.11+ | <https://www.python.org/downloads/> or `pyenv install 3.11` |
| Node | 20+ (LTS) | <https://nodejs.org/> or `nvm install 20` |
| Rust | stable (1.75+) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| `wasm-pack` | latest | `cargo install wasm-pack` |
| Go | 1.22+ | <https://go.dev/dl/> |
| Docker | 24+ | <https://docs.docker.com/get-docker/> |
| Docker Compose | v2 | bundled with Docker Desktop, or `apt install docker-compose-plugin` |
| `protoc` | 25+ | <https://grpc.io/docs/protoc-installation/> |

### Native libraries (Python ingestion worker)

```bash
# macOS
brew install libmagic ffmpeg poppler tesseract

# Debian / Ubuntu
sudo apt-get install -y libmagic1 ffmpeg poppler-utils tesseract-ocr
```

---

## 3. Single-Command Local Startup

The whole stack — object store, vector DB, ingestion worker, and dashboard —
is orchestrated through Docker Compose. There are no external cloud
dependencies in `dev` mode.

```bash
docker compose -f docker/dev.docker-compose.yml up --build
```

This will:

1. Start **MinIO** (S3-compatible) on `http://localhost:9000` (console `:9001`),
   persisting to `./volumes/local-s3-mock/`.
2. Start **ChromaDB** (vector store) on `http://localhost:8000`,
   persisting to `./volumes/local-vector-db/`.
3. Build & start the **ingestion worker** with the local MinIO/ChromaDB wired
   in via env vars.
4. Build & start the **Next.js dev dashboard** on `http://localhost:3000`.

Tear it down:

```bash
docker compose -f docker/dev.docker-compose.yml down
```

Wipe state:

```bash
rm -rf volumes/local-s3-mock/* volumes/local-vector-db/*
```

---

## 4. Cloud Deployment (Hybrid Architecture)

The hub uses a **hybrid deployment model** — the static, public-facing surface
runs on **GitHub Pages**, while the heavy compute backend runs on conventional
cloud infrastructure.

| Layer | Where it runs | How it's deployed |
| --- | --- | --- |
| Web dashboard (static export) | **GitHub Pages** | [`.github/workflows/static_deploy.yml`](.github/workflows/static_deploy.yml) on every push to `main`. |
| Rust math engine (WASM) | **GitHub Pages** (downloaded by the browser) | Compiled in CI with `wasm-pack`, copied to `gh-pages-root/wasm-engine/`. |
| Protobuf schemas | **GitHub Pages** | Published at `/schemas/telemetry.proto` so anyone can integrate. |
| API gateway | Any cloud (Fargate / Cloud Run / Fly.io) | `docker/prod.docker-compose.yml` |
| Ingestion worker | Any cloud (with SQS / PubSub) | `docker/prod.docker-compose.yml`, Celery worker mode |
| Object store | **AWS S3** / **GCS** / **R2** | env: `OBJECT_STORE_ENDPOINT` |
| Vector store | Pinecone / Weaviate Cloud / self-hosted ChromaDB | env: `VECTOR_DB_HOST` |

### The `.nojekyll` invariant

GitHub Pages defaults to processing the site with **Jekyll**, which silently
**strips any path beginning with an underscore** (e.g. `_next/`, `_headers`).
Next.js static exports rely on `_next/`, so deployments would break invisibly.
The deploy workflow therefore **always** writes an empty `.nojekyll` file at
the publish root. This is mission-critical and is asserted explicitly in
[`static_deploy.yml`](.github/workflows/static_deploy.yml).

---

## 4½. Feature Modules

The following modules implement the immersive sighting-builder, community
triangulation, tactical analytics, and universal data-mapping subsystems
called out in the architecture roadmap.

### D. 3D Spatial Canvas & "Street-View" Reconstruction

[`services/web-dashboard/components/Canvas3D.ts`](services/web-dashboard/components/Canvas3D.ts)

* Pure-TypeScript math layer (`viewportToSkyVector`, `skyVectorToEnu`,
  `evaluateOcclusion`) translates mouse / vector-arrow input on the
  sky-dome canvas into Azimuth (θ), Elevation (φ), and FOV vectors.
* Open-stack adapters wire OpenStreetMap raster/vector tiles, Open-Elevation,
  and OSM 3D building footprints. Heavy 3-D libraries (`three`,
  `@react-three/fiber`, `maplibre-gl`) are loaded **dynamically** so the
  static build remains dependency-free.
* DeviceOrientation API integration captures the witness's true-North heading
  on mobile browsers and serialises a stable JSON payload
  (`serialiseSightingPayload`) that the api-gateway can deduplicate byte-for-byte.

### E. Collaborative Multi-Witness Intersection Router

[`services/api-gateway/triangulation/`](services/api-gateway/triangulation/)

* `Router.Ingest` performs spatio-temporal clustering with ΔX ≤ 50 km and
  ΔT ≤ 10 minutes, fusing matching reports into a **Community Incident Node**.
* When a CIN crosses ≥ 2 unique viewpoints the cluster is forwarded to the
  Rust `math-engine` via the pluggable `MathEngine` interface, and the
  resulting ECEF intercept / altitude / flight vector are written back onto
  the CIN.
* The reference `MeshNotifier` fans out localised WebSocket and WebPush
  alerts to every subscriber inside `NotifyRadiusMeters` of the centroid.
* Unit tests live next to the implementation; run with
  `cd services/api-gateway && go test ./...`.

### F. Tactical "Order of Battle" Analytics Layer

[`services/ingestion-worker/oob/`](services/ingestion-worker/oob/)

* `EntityStateMatrix` indexes every event along three OoB axes: **Domain
  Presence** (space / atmospheric / trans-medium / sub-surface),
  **Kinematic Traits** (loitering / linear / rapid / swarm), and
  **Electronic Signature** (RF/EM spike / grid anomaly / optical cloak).
* `CorrelativeBaselines` cross-references each record against a curated DB
  of military airframes, radar outposts, commercial corridors, and
  experimental launch windows.
* `AnomalousGapIdentifier` flags entries that break conventional physics
  baselines (e.g. hypersonic flight without a thermal signature),
  isolating true unknowns from routine aerospace activity.
* Tests: `cd services/ingestion-worker && python -m unittest oob.test_oob`.

### B. Unified Schema Cross-Mapping

[`data-schemas/cross_map_dictionary.json`](data-schemas/cross_map_dictionary.json)

A self-describing JSON dictionary that maps legacy NUFORC, MUFON, Enigma
Labs, UAPx, and Sky360 records onto the canonical `uacp.telemetry.v1.UapEvent`
proto. Each source declares its `field_map`, `stream_kind`, ingest endpoint,
and post-map rules so the ingestion worker can be configured from data
alone — no per-source code branches required.

---

## 5. Contributing

Please read:

- [`docs/LEGAL_COMPLIANCE.md`](docs/LEGAL_COMPLIANCE.md) — **mandatory** before
  submitting any data.
- [`docs/GOVERNMENT_API_MAP.md`](docs/GOVERNMENT_API_MAP.md) — how to wire in a
  new national UAP data source.

Open issues using the templates in `.github/ISSUE_TEMPLATE/`.

---

## 6. License

This project is released under the MIT License. See `LICENSE` for details.
Contributors retain copyright of their submissions but grant the project an
irrevocable license to redistribute under the same terms.
