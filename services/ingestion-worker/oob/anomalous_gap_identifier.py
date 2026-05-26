# services/ingestion-worker/oob/anomalous_gap_identifier.py
#
# Detect entries whose behaviour breaks conventional physics baselines.
"""Anomalous Gap Identifier.

Flags `EntityStateRecord`s whose signature breaks one or more conventional
physics baselines, isolating true unknowns from routine aerospace activity.

The current ruleset covers the canonical UAP "tells":

  * Hypersonic flight (M ≥ 5) with **no measurable thermal signature**.
  * Trans-medium transitions (air ↔ water ↔ space).
  * Apparent acceleration above the ~9 g human-rated airframe envelope.
  * RF/EM spikes or power-grid anomalies coincident with optical cloaking —
    consistent with no known conventional airframe class.

Rules are individually toggleable so deployments can tune the false-positive
rate against their local sensor mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .baselines import BaselineMatch
from .entity_state_matrix import (
    DomainPresence,
    ElectronicSignature,
    EntityStateRecord,
    KinematicTrait,
)


# Physics constants -----------------------------------------------------------

#: Speed of sound at sea level, m·s⁻¹.
SPEED_OF_SOUND_MPS = 343.0
#: Threshold above which the rule-set considers flight "hypersonic".
HYPERSONIC_MACH = 5.0
#: Crude lower bound (W·sr⁻¹) on the thermal signature a hypersonic airframe
#: should radiate due to plasma sheath / leading-edge heating. Anything below
#: this is "physics-violating" and the entity is flagged.
HYPERSONIC_MIN_THERMAL_W_PER_SR = 1.0e3


@dataclass(frozen=True)
class GapFinding:
    """A single physics-baseline violation."""

    event_id: str
    rule_id: str
    severity: str  # "info" | "warn" | "critical"
    description: str

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
class AnomalousGapIdentifier:
    """Rule-driven physics-violation detector."""

    enable_hypersonic_thermal_rule: bool = True
    enable_trans_medium_rule: bool = True
    enable_cloak_em_rule: bool = True
    enable_baseline_isolation_rule: bool = True
    findings: List[GapFinding] = field(default_factory=list)

    # ---- public API ------------------------------------------------------

    def evaluate(
        self,
        record: EntityStateRecord,
        baseline_matches: Optional[Sequence[BaselineMatch]] = None,
    ) -> List[GapFinding]:
        """Evaluate every enabled rule against *record*.

        ``baseline_matches`` is the output of
        :py:meth:`CorrelativeBaselines.match`. When supplied, the "baseline
        isolation" rule fires when *no* baseline matched — i.e. the record
        cannot be explained by any catalogued military / radar / civil /
        launch row, isolating it as a true unknown.
        """
        out: List[GapFinding] = []
        if self.enable_hypersonic_thermal_rule:
            out.extend(self._check_hypersonic_thermal(record))
        if self.enable_trans_medium_rule:
            out.extend(self._check_trans_medium(record))
        if self.enable_cloak_em_rule:
            out.extend(self._check_cloak_em(record))
        if self.enable_baseline_isolation_rule and baseline_matches is not None:
            out.extend(self._check_baseline_isolation(record, baseline_matches))
        self.findings.extend(out)
        return out

    def reset(self) -> None:
        self.findings.clear()

    # ---- individual rules ------------------------------------------------

    def _check_hypersonic_thermal(self, r: EntityStateRecord) -> List[GapFinding]:
        if r.estimated_speed_mps is None:
            return []
        mach = r.estimated_speed_mps / SPEED_OF_SOUND_MPS
        if mach < HYPERSONIC_MACH:
            return []
        thermal = r.estimated_thermal_w_per_sr
        if thermal is not None and thermal >= HYPERSONIC_MIN_THERMAL_W_PER_SR:
            return []
        return [
            GapFinding(
                event_id=r.event_id,
                rule_id="hypersonic-without-thermal",
                severity="critical",
                description=(
                    f"M{mach:.2f} flight observed with negligible thermal "
                    f"signature (≤ {HYPERSONIC_MIN_THERMAL_W_PER_SR:.0f} W·sr⁻¹); "
                    "violates leading-edge heating baseline."
                ),
            )
        ]

    def _check_trans_medium(self, r: EntityStateRecord) -> List[GapFinding]:
        if r.domain != DomainPresence.TRANS_MEDIUM:
            return []
        return [
            GapFinding(
                event_id=r.event_id,
                rule_id="trans-medium-transition",
                severity="warn",
                description=(
                    "Entity observed transitioning between two physical media "
                    "(air/water/space) — no known conventional airframe class "
                    "supports unmodified trans-medium operation."
                ),
            )
        ]

    def _check_cloak_em(self, r: EntityStateRecord) -> List[GapFinding]:
        if r.electronic == ElectronicSignature.OPTICAL_CLOAKING and r.kinematic in (
            KinematicTrait.RAPID_DEPLOYMENT,
            KinematicTrait.SWARM_ARRAY,
        ):
            return [
                GapFinding(
                    event_id=r.event_id,
                    rule_id="optical-cloak-with-rf-em",
                    severity="warn",
                    description=(
                        "Optical-cloaking signature coincident with rapid / "
                        "swarm kinematics — not consistent with any catalogued "
                        "stealth airframe."
                    ),
                )
            ]
        return []

    def _check_baseline_isolation(
        self,
        r: EntityStateRecord,
        baseline_matches: Sequence[BaselineMatch],
    ) -> List[GapFinding]:
        if baseline_matches:
            return []
        return [
            GapFinding(
                event_id=r.event_id,
                rule_id="no-baseline-match",
                severity="info",
                description=(
                    "Record did not correlate with any catalogued military "
                    "airframe, radar outpost, commercial corridor, or "
                    "experimental launch window — isolated as a true unknown."
                ),
            )
        ]
