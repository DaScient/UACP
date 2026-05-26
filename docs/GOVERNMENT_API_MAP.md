# Government & Multinational UAP Data Source Map

This document is the canonical scaffold for wiring **national / governmental
UAP data sources** into the Global UAP Intelligence Hub. Every connector
**must** be documented here before its code is merged.

---

## How to add a new nation's API

Follow these steps exactly. Open a PR that updates **both** this document and
the corresponding connector in `services/ingestion-worker/connectors/`.

1. **Discovery.** Identify the authoritative agency (military, civil aviation,
   research body) and verify the data they publish is *unclassified*. Cite
   the relevant national FOIA equivalent.
2. **Legal review.** Read [`LEGAL_COMPLIANCE.md`](./LEGAL_COMPLIANCE.md).
   Confirm the dataset can be lawfully redistributed under the project's MIT
   license — most government-produced works are public domain in their
   country of origin, but **not all**.
3. **Schema mapping.** Map every source field to the canonical
   [`UapEvent`](../data-schemas/telemetry.proto) Protobuf schema. Document
   ambiguities and your chosen resolution.
4. **Connector implementation.** Add `services/ingestion-worker/connectors/<iso>.py`
   exposing a single `fetch(since: datetime) -> Iterator[UapEvent]` function.
5. **Rate-limit & robots.txt.** Respect the source's published rate limits.
   Default to ≤ 1 request / second unless the source's docs allow more.
6. **Secret handling.** API keys go in environment variables, never in code.
   Update `docker/dev.docker-compose.yml` and `docker/prod.docker-compose.yml`
   to declare the new var.
7. **Documentation.** Add a new section below using the template in §99.

---

## 1. 🇺🇸 AARO — All-domain Anomaly Resolution Office (USA)

**Status:** ✅ first-class connector

**Authority:** U.S. Department of Defense, established by the 2022 NDAA
(50 U.S.C. § 3373).

**Public portal:** <https://www.aaro.mil/>

**Data character:** Declassified historical case reports (1940s–present),
the AARO case-resolution database, congressional hearing transcripts, and
the periodic Office of the Director of National Intelligence (ODNI) UAP
report addenda.

### 1.1 Setting up access

AARO does **not** (as of writing) expose a self-service REST API. Data is
pulled in three tiers.

#### Tier A — Bulk public archives (no key required)

```bash
# Mirror the AARO declassified case archive (HTML index of PDFs)
mkdir -p volumes/local-s3-mock/aaro-historical
wget --recursive --level=2 --no-parent \
     --accept "*.pdf,*.csv,*.html" \
     --wait=1 --random-wait \
     --user-agent "UACP-research-bot/1.0 (+https://github.com/DaScient/UACP)" \
     -P volumes/local-s3-mock/aaro-historical \
     https://www.aaro.mil/Cases/
```

> Respect the AARO `robots.txt` and the per-request rate limit (≤ 1 req/sec).

#### Tier B — FOIA portal (account required, no API key)

1. Register at <https://efoia.osd.mil/> using your real name and a verifiable
   email address.
2. Submit a FOIA request scoped to UAP/UAS records. Reference DoD Directive
   3000.09 and the relevant fiscal-year NDAA UAP provision.
3. AARO will provide a tracking number. Releases arrive by email (usually as
   PDF). Drop the PDFs into `volumes/local-s3-mock/aaro-foia/` for ingestion.

#### Tier C — Future REST API (placeholder)

Once AARO publishes a machine-readable API, the key will be stored as:

```bash
export AARO_API_KEY="<obtained from https://www.aaro.mil/api/register/>"
export AARO_API_BASE="https://api.aaro.mil/v1"
```

and consumed by `services/ingestion-worker/connectors/usa_aaro.py`.

### 1.2 Pulling declassified historical data (concrete steps)

```bash
# 1. Ensure dev stack is running:
docker compose -f docker/dev.docker-compose.yml up -d minio chromadb

# 2. Run the AARO connector (one-shot historical backfill):
docker compose -f docker/dev.docker-compose.yml run --rm ingestion-worker \
    python -m connectors.usa_aaro --mode=backfill --since=1947-06-24

# 3. Verify objects landed in MinIO:
docker compose -f docker/dev.docker-compose.yml exec minio \
    mc ls local/uap-raw/aaro/
```

### 1.3 Field mapping (AARO PDF case report → `UapEvent`)

| AARO field | Proto field | Notes |
| --- | --- | --- |
| Case ID | `event_id` | Re-hashed to UUIDv5 (namespace `aaro.mil`). |
| "Date of observation" | `timestamp` | Parsed via `dateparser`; default to 12:00 UTC if no time given. |
| "Location" | `location.latitude`, `location.longitude` | Geocoded via `geopy` against Nominatim; `altitude_meters = 0` if unknown. |
| "Sensor" | `sensor_type` | Mapped to `SensorType` enum (e.g. "Radar" → `RADAR`). |
| Narrative | `signature.kinematics_annotated` | Stored as JSON `{"narrative": "...", "source": "AARO"}`. |
| Attachments | `attachments[]` | `cloud://` URIs to ingested PDFs / images. |

---

## 2. 🇫🇷 GEIPAN — Groupe d'Études et d'Informations sur les Phénomènes Aérospatiaux Non identifiés (France)

**Status:** ⏳ scaffold only — connector not yet implemented

**Authority:** CNES (Centre national d'études spatiales).

**Public portal:** <https://geipan.fr/>

**Data character:** Publicly searchable case database (1970s–present),
classified by GEIPAN into A/B/C/D categories.

### 2.1 Setup

```bash
# No API key required for the public catalogue. Optional registration for
# bulk export requests:
export GEIPAN_USER_AGENT="UACP-research-bot/1.0 (contact@example.org)"
```

### 2.2 Status checklist

- [ ] Connector `services/ingestion-worker/connectors/fra_geipan.py`
- [ ] Field mapping table (see §99 template)
- [ ] Robots.txt compliance test
- [ ] Sample backfill job

---

## 3. 🇨🇱 CEFAA — Comité de Estudios de Fenómenos Aéreos Anómalos (Chile)

**Status:** ⏳ scaffold only — connector not yet implemented

**Authority:** Dirección General de Aeronáutica Civil (DGAC).

**Public portal:** <https://www.dgac.gob.cl/cefaa/>

**Data character:** Civil aviation incident reports with anomalous-object
classification; published case summaries in Spanish.

### 3.1 Setup

```bash
export CEFAA_USER_AGENT="UACP-research-bot/1.0 (contact@example.org)"
# CEFAA publishes case summaries as HTML; no API key required.
```

### 3.2 Status checklist

- [ ] Connector `services/ingestion-worker/connectors/chl_cefaa.py`
- [ ] Spanish-language NLP normalisation
- [ ] Field mapping table

---

## 99. Template for a new connector entry

```markdown
## N. 🇽🇽 <AGENCY ACRONYM> — <Full agency name> (<Country>)

**Status:** ⏳ scaffold | ✅ implemented | ⚠️ blocked-on-legal

**Authority:** <ministry / agency>.

**Public portal:** <URL>

**Data character:** <one-paragraph description>.

### N.1 Setting up access

<step-by-step: registration, API key procurement, env vars, rate limits>

### N.2 Pulling data

<concrete shell commands>

### N.3 Field mapping (<source> → `UapEvent`)

| Source field | Proto field | Notes |
| --- | --- | --- |
| ... | ... | ... |
```
