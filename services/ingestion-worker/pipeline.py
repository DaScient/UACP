# services/ingestion-worker/pipeline.py
#
# File-type-agnostic ingestion router for the UAP Intelligence Hub.
# Dispatches on libmagic-derived MIME type (never on file extension) and hands
# the file off to the correct modality processor. Each processor returns a
# JSON-serialisable metadata dictionary that downstream classification and
# storage stages will consume.
"""Ingestion pipeline entry point."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIME detection
# ---------------------------------------------------------------------------

def _detect_mime(file_path: str) -> str:
    """Return the libmagic-derived MIME type of *file_path*.

    Falls back to ``application/octet-stream`` if ``python-magic`` is not
    installed (e.g. in a unit-test environment).
    """
    try:
        import magic  # type: ignore
    except ImportError:  # pragma: no cover - dev-only fallback
        logger.warning("python-magic not installed; using octet-stream fallback")
        return "application/octet-stream"
    return magic.from_file(file_path, mime=True)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------

def process_video(file_path: str, event_id: str) -> Dict[str, Any]:
    """Extract keyframes & luminance spikes from a video using OpenCV (MOG2).

    Returns a metadata dict with the keyframe paths and a frame-level
    luminance trace. Heavy dependencies are imported lazily so unit tests can
    monkeypatch this function.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    out_dir = os.path.join("/tmp", "uacp-keyframes", event_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {file_path}")

    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    keyframes: list[str] = []
    luminance_trace: list[float] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fg = bg.apply(frame)
        mean_lum = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        luminance_trace.append(mean_lum)

        # Keyframe heuristic: large foreground area OR luminance spike vs
        # running mean.
        fg_ratio = float(np.count_nonzero(fg)) / fg.size
        spike = (
            len(luminance_trace) > 5
            and mean_lum > 1.4 * (sum(luminance_trace[-6:-1]) / 5.0)
        )
        if fg_ratio > 0.02 or spike:
            kf_path = os.path.join(out_dir, f"kf_{frame_idx:06d}.png")
            cv2.imwrite(kf_path, frame)
            keyframes.append(kf_path)
        frame_idx += 1

    cap.release()
    return {
        "modality":         "video",
        "frame_count":      frame_idx,
        "keyframes":        keyframes,
        "luminance_trace":  luminance_trace,
    }


def process_document(file_path: str, event_id: str) -> Dict[str, Any]:
    """Extract text from a document and embed it into a 768-d vector."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    from unstructured.partition.auto import partition  # type: ignore

    elements = partition(filename=file_path)
    text = "\n".join(getattr(e, "text", "") or "" for e in elements).strip()

    model  = SentenceTransformer("all-mpnet-base-v2")
    vector = model.encode(text, normalize_embeddings=True).tolist()
    assert len(vector) == 768, "all-mpnet-base-v2 produces 768-dim vectors"

    return {
        "modality": "document",
        "text":     text,
        "vector":   vector,
        "event_id": event_id,
    }


def process_tabular(file_path: str, event_id: str) -> Dict[str, Any]:
    """Normalize a CSV/JSON tabular file against the ``UapEvent`` proto fields."""
    import pandas as pd  # type: ignore

    mime = _detect_mime(file_path)
    if "json" in mime:
        df = pd.read_json(file_path)
    else:
        df = pd.read_csv(file_path)

    # Column aliasing → canonical UapEvent field names.
    alias = {
        "time":      "timestamp",
        "datetime":  "timestamp",
        "lat":       "latitude",
        "lon":       "longitude",
        "lng":       "longitude",
        "long":      "longitude",
        "alt":       "altitude_meters",
        "altitude":  "altitude_meters",
        "altitude_m": "altitude_meters",
    }
    df = df.rename(columns={c: alias.get(c.lower(), c.lower()) for c in df.columns})

    canonical_cols = ["timestamp", "latitude", "longitude", "altitude_meters"]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = None

    return {
        "modality": "tabular",
        "rows":     df[canonical_cols + [c for c in df.columns if c not in canonical_cols]]
                      .to_dict(orient="records"),
        "event_id": event_id,
    }


def process_binary_fallback(file_path: str, event_id: str) -> Dict[str, Any]:
    """Hash + metadata for unknown binary types (raw RF, proprietary sensors)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    stat = os.stat(file_path)
    return {
        "modality":  "binary",
        "sha256":    h.hexdigest(),
        "size_bytes": stat.st_size,
        "mime":      _detect_mime(file_path),
        "event_id":  event_id,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Ordered list of (mime-prefix predicate, handler). First match wins.
_ROUTES: list[tuple[Callable[[str], bool], Callable[[str, str], Dict[str, Any]]]] = [
    (lambda m: m.startswith("video/"),                              process_video),
    (lambda m: m == "application/pdf" or m.startswith("text/"),     process_document),
    (lambda m: m in ("text/csv", "application/json"),               process_tabular),
]


def route_and_process(file_path: str, event_id: str) -> Dict[str, Any]:
    """Detect MIME via libmagic and dispatch to the correct processor.

    All processors return a JSON-serialisable dict that begins with at least
    ``{"modality": <str>, "event_id": <uuid>}``.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    mime = _detect_mime(file_path)
    logger.info("route_and_process event=%s mime=%s path=%s", event_id, mime, file_path)

    for predicate, handler in _ROUTES:
        if predicate(mime):
            result = handler(file_path, event_id)
            result.setdefault("event_id", event_id)
            result["mime"] = mime
            return result

    result = process_binary_fallback(file_path, event_id)
    result["mime"] = mime
    return result
