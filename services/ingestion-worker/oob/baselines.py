# services/ingestion-worker/oob/baselines.py
#
# Correlative-baseline cross-referencing for the OoB analytics layer.
"""Correlative Baselines.

Cross-references every active OoB record against a curated matrix of:

  * **military_capabilities** — domestic & foreign airframes (with rough
    operating envelope: max Mach, ceiling).
  * **radar_outposts**         — known ground/sea radar coverage circles.
  * **commercial_corridors**   — civil aviation transit corridors.
  * **experimental_launches**  — published space-launch / test windows.

The baseline DB is supplied as plain JSON so it can live in git, be reviewed,
and be overlaid with classified or community contributions at runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .entity_state_matrix import EntityStateRecord


# ---------------------------------------------------------------------------
# Match record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineMatch:
    """A single hit against the baseline DB."""

    event_id: str
    category: str          # "military_capability" | "radar_outpost" | …
    identifier: str        # e.g. "F-22A", "Vandenberg SLC-4"
    confidence: float      # [0.0, 1.0]
    rationale: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "category": self.category,
            "identifier": self.identifier,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Baseline DB
# ---------------------------------------------------------------------------


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_008.8  # mean Earth radius (m)
    d2r = math.pi / 180
    dlat = (lat2 - lat1) * d2r
    dlon = (lon2 - lon1) * d2r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1 * d2r) * math.cos(lat2 * d2r) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


class CorrelativeBaselines:
    """Match OoB records against curated baseline data.

    The DB JSON has the shape::

        {
          "military_capabilities": [
              {"id": "F-22A", "max_mach": 2.25, "ceiling_m": 19_800}, …
          ],
          "radar_outposts": [
              {"id": "CONUS-NE-1", "lat": 41.4, "lon": -70.5, "radius_km": 400}, …
          ],
          "commercial_corridors": [
              {"id": "ATL-LGA", "from": [33.6, -84.4], "to": [40.8, -73.9],
               "width_km": 25}, …
          ],
          "experimental_launches": [
              {"id": "Vandenberg SLC-4", "lat": 34.6, "lon": -120.6,
               "window_open": "2026-05-26T00:00:00Z",
               "window_close": "2026-05-27T00:00:00Z"}, …
          ]
        }
    """

    # Default radius (km) for "near a radar outpost" if not declared on the row.
    DEFAULT_RADAR_RADIUS_KM = 250.0

    # Default temporal slack around an experimental launch window.
    LAUNCH_WINDOW_SLACK = timedelta(hours=2)

    def __init__(self, db: Dict[str, List[Dict[str, Any]]]):
        self.db = db

    # ---- constructors ---------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "CorrelativeBaselines":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def empty(cls) -> "CorrelativeBaselines":
        return cls(
            {
                "military_capabilities": [],
                "radar_outposts": [],
                "commercial_corridors": [],
                "experimental_launches": [],
            }
        )

    # ---- matching -------------------------------------------------------

    def match(self, record: EntityStateRecord) -> List[BaselineMatch]:
        """Return every baseline row this record correlates with."""
        out: List[BaselineMatch] = []
        out.extend(self._match_military(record))
        out.extend(self._match_radar(record))
        out.extend(self._match_corridors(record))
        out.extend(self._match_launches(record))
        return out

    def _match_military(self, record: EntityStateRecord) -> List[BaselineMatch]:
        if record.estimated_speed_mps is None or record.estimated_altitude_m is None:
            return []
        mach = record.estimated_speed_mps / 343.0  # at sea level — approximation
        out: List[BaselineMatch] = []
        for row in self.db.get("military_capabilities", []):
            ceiling = float(row.get("ceiling_m", 0))
            max_mach = float(row.get("max_mach", 0))
            if mach <= max_mach and record.estimated_altitude_m <= ceiling:
                out.append(
                    BaselineMatch(
                        event_id=record.event_id,
                        category="military_capability",
                        identifier=str(row.get("id", "?")),
                        confidence=0.5,
                        rationale=(
                            f"speed M{mach:.2f}≤{max_mach} and "
                            f"altitude {record.estimated_altitude_m:.0f}m≤{ceiling:.0f}m"
                        ),
                    )
                )
        return out

    def _match_radar(self, record: EntityStateRecord) -> List[BaselineMatch]:
        out: List[BaselineMatch] = []
        for row in self.db.get("radar_outposts", []):
            radius_m = float(row.get("radius_km", self.DEFAULT_RADAR_RADIUS_KM)) * 1000
            d = _haversine_m(record.latitude, record.longitude,
                             float(row["lat"]), float(row["lon"]))
            if d <= radius_m:
                out.append(
                    BaselineMatch(
                        event_id=record.event_id,
                        category="radar_outpost",
                        identifier=str(row.get("id", "?")),
                        confidence=0.7,
                        rationale=f"within {d/1000:.1f}km of outpost (radius {radius_m/1000:.0f}km)",
                    )
                )
        return out

    def _match_corridors(self, record: EntityStateRecord) -> List[BaselineMatch]:
        out: List[BaselineMatch] = []
        for row in self.db.get("commercial_corridors", []):
            (lat1, lon1) = row["from"]
            (lat2, lon2) = row["to"]
            width_m = float(row.get("width_km", 25)) * 1000
            d = _point_to_segment_distance_m(
                record.latitude, record.longitude,
                float(lat1), float(lon1), float(lat2), float(lon2),
            )
            if d <= width_m:
                out.append(
                    BaselineMatch(
                        event_id=record.event_id,
                        category="commercial_corridor",
                        identifier=str(row.get("id", "?")),
                        confidence=0.6,
                        rationale=f"{d/1000:.1f}km off corridor centreline (width {width_m/1000:.0f}km)",
                    )
                )
        return out

    def _match_launches(self, record: EntityStateRecord) -> List[BaselineMatch]:
        out: List[BaselineMatch] = []
        ts = record.observed_at.astimezone(timezone.utc)
        for row in self.db.get("experimental_launches", []):
            t_open = datetime.fromisoformat(str(row["window_open"]).replace("Z", "+00:00"))
            t_close = datetime.fromisoformat(str(row["window_close"]).replace("Z", "+00:00"))
            if t_open - self.LAUNCH_WINDOW_SLACK <= ts <= t_close + self.LAUNCH_WINDOW_SLACK:
                d = _haversine_m(record.latitude, record.longitude,
                                 float(row["lat"]), float(row["lon"]))
                # Only count it if the witness was within 1500 km of the pad.
                if d <= 1_500_000:
                    out.append(
                        BaselineMatch(
                            event_id=record.event_id,
                            category="experimental_launch",
                            identifier=str(row.get("id", "?")),
                            confidence=0.8,
                            rationale=(
                                f"observation falls within launch window "
                                f"[{t_open.isoformat()}, {t_close.isoformat()}] and "
                                f"{d/1000:.0f}km from pad"
                            ),
                        )
                    )
        return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _point_to_segment_distance_m(
    p_lat: float, p_lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float,
) -> float:
    """Approximate planar point-to-segment distance, in metres.

    Inputs are WGS-84 (latitude, longitude) pairs. Suitable for
    corridor-membership tests up to ~1000 km long — the small distortion
    from treating lat/lon as a flat plane stays below the corridor width
    threshold (tens of km).
    """
    # Convert to a local metres-ish frame anchored at the segment's midpoint.
    lat0 = (a_lat + b_lat) / 2
    deg_lat_m = 111_320.0
    deg_lon_m = 111_320.0 * math.cos(lat0 * math.pi / 180)

    def lat_lon_to_meters(lat: float, lon: float) -> tuple[float, float]:
        return (lon * deg_lon_m, lat * deg_lat_m)

    px_m, py_m = lat_lon_to_meters(p_lat, p_lon)
    ax_m, ay_m = lat_lon_to_meters(a_lat, a_lon)
    bx_m, by_m = lat_lon_to_meters(b_lat, b_lon)

    dx = bx_m - ax_m
    dy = by_m - ay_m
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 0:
        return math.hypot(px_m - ax_m, py_m - ay_m)
    t = max(0.0, min(1.0, ((px_m - ax_m) * dx + (py_m - ay_m) * dy) / seg_len_sq))
    cx = ax_m + t * dx
    cy = ay_m + t * dy
    return math.hypot(px_m - cx, py_m - cy)
