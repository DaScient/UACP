"""UAP Tactical "Order of Battle" (OoB) analytics.

This subsystem treats UAP activity logs with the structured rigidity of a
military Order of Battle dashboard (architecture doc, section F):

  * :class:`EntityStateMatrix` organises UAP events into a live tactical
    matrix categorising entities by **domain presence**, **kinematic traits**,
    and **electronic signature**.
  * :class:`CorrelativeBaselines` cross-references each active signature
    against a curated matrix of domestic & foreign military capabilities,
    known radar outposts, commercial transit corridors, and experimental
    launch calendars.
  * :class:`AnomalousGapIdentifier` flags signatures whose behaviour breaks
    conventional physics baselines — isolating true unknowns from routine
    aerospace activity.

The module is intentionally framework-agnostic and depends only on the Python
standard library so it can be imported by both the ingestion worker and any
downstream notebook / Streamlit dashboard.
"""

from .entity_state_matrix import (
    DomainPresence,
    KinematicTrait,
    ElectronicSignature,
    EntityStateRecord,
    EntityStateMatrix,
)
from .baselines import CorrelativeBaselines, BaselineMatch
from .anomalous_gap_identifier import AnomalousGapIdentifier, GapFinding

__all__ = [
    "DomainPresence",
    "KinematicTrait",
    "ElectronicSignature",
    "EntityStateRecord",
    "EntityStateMatrix",
    "CorrelativeBaselines",
    "BaselineMatch",
    "AnomalousGapIdentifier",
    "GapFinding",
]
