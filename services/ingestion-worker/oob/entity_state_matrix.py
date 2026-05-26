# services/ingestion-worker/oob/entity_state_matrix.py
#
# Live tactical Order-of-Battle matrix for UAP entities.
"""Entity State Matrix.

Organises every ingested UAP event into a tactical, OoB-style matrix indexed
along three orthogonal axes:

    * Domain Presence    — Space-Based / Atmospheric / Trans-Medium / Sub-Surface
    * Kinematic Traits   — Loitering / Linear / Rapid / Swarm
    * Electronic Signature — RF/EM spikes / Grid anomalies / Optical cloaking

Records flow in via :py:meth:`EntityStateMatrix.add` and can be retrieved as
ordered counts or as a 3-D dense histogram via :py:meth:`as_histogram`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tactical axes
# ---------------------------------------------------------------------------


class DomainPresence(str, Enum):
    """Where in the four-domain envelope was the entity observed?"""

    SPACE_BASED = "space_based"
    ATMOSPHERIC = "atmospheric"
    TRANS_MEDIUM = "trans_medium"
    SUB_SURFACE_OCEANIC = "sub_surface_oceanic"
    UNKNOWN = "unknown"


class KinematicTrait(str, Enum):
    """Macroscopic motion archetype."""

    LOITERING = "loitering"          # station-keeping / hover patterns
    LINEAR_CORRIDOR = "linear"       # steady transit along a heading
    RAPID_DEPLOYMENT = "rapid"       # high-acceleration deployment
    SWARM_ARRAY = "swarm"            # multiple coordinated entities
    UNKNOWN = "unknown"


class ElectronicSignature(str, Enum):
    """Observed RF / EM / optical signature class."""

    RF_EM_SPIKE = "rf_em_spike"
    POWER_GRID_ANOMALY = "power_grid_anomaly"
    OPTICAL_CLOAKING = "optical_cloaking"
    NONE_OBSERVED = "none"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityStateRecord:
    """A single normalised OoB record derived from a `UapEvent`."""

    event_id: str
    observed_at: datetime
    latitude: float
    longitude: float
    domain: DomainPresence
    kinematic: KinematicTrait
    electronic: ElectronicSignature
    estimated_speed_mps: Optional[float] = None
    estimated_altitude_m: Optional[float] = None
    estimated_thermal_w_per_sr: Optional[float] = None
    raw: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "observed_at": self.observed_at.astimezone(timezone.utc).isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "domain": self.domain.value,
            "kinematic": self.kinematic.value,
            "electronic": self.electronic.value,
            "estimated_speed_mps": self.estimated_speed_mps,
            "estimated_altitude_m": self.estimated_altitude_m,
            "estimated_thermal_w_per_sr": self.estimated_thermal_w_per_sr,
        }


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


CellKey = Tuple[DomainPresence, KinematicTrait, ElectronicSignature]


class EntityStateMatrix:
    """Live in-memory OoB matrix.

    Thread-safety is NOT provided — the ingestion worker is expected to drive
    a single matrix instance from a single asyncio loop / worker process. If
    you need to consume from multiple processes, persist the matrix to the
    object store between batches.
    """

    def __init__(self) -> None:
        self._records: List[EntityStateRecord] = []
        self._counts: Counter[CellKey] = Counter()
        self._by_event_id: Dict[str, EntityStateRecord] = {}

    # -- mutation ----------------------------------------------------------

    def add(self, record: EntityStateRecord) -> None:
        if record.event_id in self._by_event_id:
            # Idempotent: same event_id replaces an earlier insertion.
            old = self._by_event_id[record.event_id]
            self._counts[(old.domain, old.kinematic, old.electronic)] -= 1
            self._records = [r for r in self._records if r.event_id != record.event_id]
        self._records.append(record)
        self._by_event_id[record.event_id] = record
        self._counts[(record.domain, record.kinematic, record.electronic)] += 1

    def extend(self, records: Iterable[EntityStateRecord]) -> None:
        for r in records:
            self.add(r)

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[EntityStateRecord]:
        """Read-only view of the records, ordered by insertion."""
        return list(self._records)

    def cell(self, key: CellKey) -> int:
        return int(self._counts.get(key, 0))

    def as_histogram(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Dense 3-D dict-of-dict-of-dicts histogram.

        Convenient for JSON serialisation to the dashboard. Cells with zero
        count are omitted to keep the payload small.
        """
        out: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for (d, k, e), n in self._counts.items():
            if n > 0:
                out[d.value][k.value][e.value] = int(n)
        return {d: {k: dict(v) for k, v in inner.items()} for d, inner in out.items()}

    def filter(
        self,
        *,
        domain: Optional[DomainPresence] = None,
        kinematic: Optional[KinematicTrait] = None,
        electronic: Optional[ElectronicSignature] = None,
    ) -> List[EntityStateRecord]:
        """Return records matching the supplied axes (AND-combined)."""
        out = []
        for r in self._records:
            if domain is not None and r.domain != domain:
                continue
            if kinematic is not None and r.kinematic != kinematic:
                continue
            if electronic is not None and r.electronic != electronic:
                continue
            out.append(r)
        return out
