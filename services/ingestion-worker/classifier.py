# services/ingestion-worker/classifier.py
#
# Extensible rule-based + ML-driven classifier for the UAP Intelligence Hub.
# Consumes the metadata package produced by ``pipeline.route_and_process`` and
# optional output from the Rust math engine (e.g. fused 3D track + ECEF
# velocity), and produces the canonical classification JSON payload.
"""UAP morphology + kinematic classification."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical shape labels. MUST match the ``Shape`` enum in telemetry.proto.
SHAPES: list[str] = ["Tic-Tac", "Sphere", "Disc", "Triangle", "Unknown"]

#: Speed profiles. MUST match the four categories called out in the
#: architecture document.
KINEMATIC_PROFILES: list[str] = [
    "Subsonic",      # M < 1.0
    "Supersonic",    # 1.0 ≤ M < 5.0
    "Hypersonic",    # 5.0 ≤ M
    "Trans-Medium",  # observed crossing air/water or air/space boundary
]

#: Keywords → coarse shape (used when structured shape data is absent).
_SHAPE_KEYWORDS: Dict[str, list[str]] = {
    "Tic-Tac":  ["tic tac", "tic-tac", "pill", "capsule", "lozenge"],
    "Sphere":   ["sphere", "spherical", "ball", "orb", "round"],
    "Disc":     ["disc", "disk", "saucer", "lenticular", "flying saucer"],
    "Triangle": ["triangle", "triangular", "delta", "three-sided", "tri-form"],
}

# ---------------------------------------------------------------------------
# Atmospheric model: simplified ISA → speed of sound
# ---------------------------------------------------------------------------

def _speed_of_sound_mps(altitude_m: float) -> float:
    """Speed of sound (m/s) from a simplified two-layer ISA model.

    * Troposphere (0–11 km): linear lapse rate of 6.5 K/km from 288.15 K.
    * Lower stratosphere (11–25 km): isothermal at 216.65 K.
    * Above 25 km: linear warm-up of 1 K/km up to 271.65 K at 47 km (clipped).
    """
    h = max(0.0, float(altitude_m))
    if h < 11_000.0:
        t_k = 288.15 - 0.0065 * h
    elif h < 25_000.0:
        t_k = 216.65
    else:
        t_k = min(271.65, 216.65 + 0.001 * (h - 25_000.0))
    # a = sqrt(γ R T), γ = 1.4, R_specific(air) = 287.058 J/(kg·K)
    return (1.4 * 287.058 * t_k) ** 0.5


def _kinematic_profile(
    speed_mps: float, altitude_m: float, trans_medium: bool
) -> tuple[str, float]:
    """Return ``(profile_label, mach_number)`` for the given kinematics."""
    if trans_medium:
        return "Trans-Medium", speed_mps / _speed_of_sound_mps(altitude_m)
    a = _speed_of_sound_mps(altitude_m)
    mach = speed_mps / a if a > 0 else 0.0
    if mach < 1.0:
        return "Subsonic", mach
    if mach < 5.0:
        return "Supersonic", mach
    return "Hypersonic", mach


# ---------------------------------------------------------------------------
# Shape classification
# ---------------------------------------------------------------------------

def _shape_from_keywords(text: str) -> Optional[tuple[str, float]]:
    text_l = text.lower()
    for shape, kws in _SHAPE_KEYWORDS.items():
        for kw in kws:
            if kw in text_l:
                return shape, 0.85
    return None


def _shape_from_zero_shot(text: str) -> tuple[str, float]:
    """Use BART-MNLI zero-shot classification to assign a shape."""
    from transformers import pipeline  # type: ignore

    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = [s for s in SHAPES if s != "Unknown"]
    result = clf(text, candidate_labels=labels)
    return result["labels"][0], float(result["scores"][0])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_event(
    metadata_package: Dict[str, Any],
    math_engine_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce the canonical classification JSON payload.

    Parameters
    ----------
    metadata_package
        The dict returned by ``pipeline.route_and_process``. Should contain
        either a structured ``shape`` hint, free-form ``text``, or both.
    math_engine_output
        Optional dict from the Rust math engine. Recognised keys:
          * ``speed_mps`` (float) — fused track speed.
          * ``altitude_m`` (float) — fused track altitude.
          * ``trans_medium`` (bool) — observed crossing air/water boundary.

    Returns
    -------
    dict
        ``{"event_id", "classification_metadata", "storage_routing_path"}``
        as required by the architecture document.
    """
    me = math_engine_output or {}
    event_id = metadata_package.get("event_id", "unknown")

    # --- Shape ------------------------------------------------------------
    assigned_shape: str = "Unknown"
    confidence: float = 0.0

    if "shape" in metadata_package and metadata_package["shape"] in SHAPES:
        assigned_shape = metadata_package["shape"]
        confidence = 1.0
    else:
        text_blob = " ".join(
            str(metadata_package.get(k, ""))
            for k in ("text", "narrative", "description")
            if metadata_package.get(k)
        ).strip()
        if text_blob:
            kw_hit = _shape_from_keywords(text_blob)
            if kw_hit is not None:
                assigned_shape, confidence = kw_hit
            else:
                try:
                    assigned_shape, confidence = _shape_from_zero_shot(text_blob)
                except Exception as exc:  # pragma: no cover - depends on transformers install
                    logger.warning("zero-shot classifier unavailable: %s", exc)

    # --- Kinematics -------------------------------------------------------
    speed = float(me.get("speed_mps", 0.0))
    altitude = float(me.get("altitude_m", metadata_package.get("altitude_m", 0.0)))
    trans_medium = bool(me.get("trans_medium", False))
    speed_profile, mach = _kinematic_profile(speed, altitude, trans_medium)

    # --- Anomaly heuristic ------------------------------------------------
    # Anything ≥ supersonic with no audible sonic-boom report, or any
    # Trans-Medium event, is flagged anomalous.
    is_anomalous = (
        speed_profile in ("Hypersonic", "Trans-Medium")
        or (speed_profile == "Supersonic" and not metadata_package.get("sonic_boom"))
    )

    return {
        "event_id": event_id,
        "classification_metadata": {
            "assigned_shape":   assigned_shape,
            "confidence_score": round(float(confidence), 4),
            "speed_profile":    speed_profile,
            "mach_number":      round(float(mach), 4),
            "altitude_m":       altitude,
            "is_anomalous":     is_anomalous,
        },
        "storage_routing_path":
            f"data/processed/{speed_profile}/{assigned_shape}/{event_id}/",
    }
