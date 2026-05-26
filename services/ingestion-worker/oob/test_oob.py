"""Unit tests for the OoB analytics layer.

Runs under plain `python -m unittest`, no third-party deps required.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import os
import sys

# Make the parent (`ingestion-worker`) importable so `oob` resolves whether
# this test is run from the repo root or from inside the package directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import oob  # noqa: E402  pylint: disable=wrong-import-position


class EntityStateMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = oob.EntityStateMatrix()

    def _rec(self, event_id: str, **overrides) -> "oob.EntityStateRecord":
        base = dict(
            event_id=event_id,
            observed_at=datetime.now(timezone.utc),
            latitude=40.0,
            longitude=-74.0,
            domain=oob.DomainPresence.ATMOSPHERIC,
            kinematic=oob.KinematicTrait.LOITERING,
            electronic=oob.ElectronicSignature.NONE_OBSERVED,
        )
        base.update(overrides)
        return oob.EntityStateRecord(**base)

    def test_add_increments_cell(self) -> None:
        self.matrix.add(self._rec("a"))
        self.assertEqual(len(self.matrix), 1)
        self.assertEqual(
            self.matrix.cell(
                (
                    oob.DomainPresence.ATMOSPHERIC,
                    oob.KinematicTrait.LOITERING,
                    oob.ElectronicSignature.NONE_OBSERVED,
                )
            ),
            1,
        )

    def test_idempotent_on_event_id(self) -> None:
        self.matrix.add(self._rec("a"))
        self.matrix.add(self._rec("a", domain=oob.DomainPresence.SPACE_BASED))
        self.assertEqual(len(self.matrix), 1)
        self.assertEqual(
            self.matrix.cell(
                (
                    oob.DomainPresence.ATMOSPHERIC,
                    oob.KinematicTrait.LOITERING,
                    oob.ElectronicSignature.NONE_OBSERVED,
                )
            ),
            0,
        )
        self.assertEqual(
            self.matrix.cell(
                (
                    oob.DomainPresence.SPACE_BASED,
                    oob.KinematicTrait.LOITERING,
                    oob.ElectronicSignature.NONE_OBSERVED,
                )
            ),
            1,
        )

    def test_histogram_excludes_zero_cells(self) -> None:
        self.matrix.add(self._rec("a"))
        hist = self.matrix.as_histogram()
        self.assertIn("atmospheric", hist)
        # no swarm cells should be emitted
        for inner in hist.values():
            self.assertNotIn("swarm", inner)


class CorrelativeBaselinesTests(unittest.TestCase):
    def test_radar_outpost_match_within_radius(self) -> None:
        bl = oob.CorrelativeBaselines(
            {
                "radar_outposts": [
                    {"id": "RX-1", "lat": 40.0, "lon": -74.0, "radius_km": 100}
                ],
            }
        )
        rec = oob.EntityStateRecord(
            event_id="x",
            observed_at=datetime.now(timezone.utc),
            latitude=40.5,
            longitude=-74.0,  # ~55 km north
            domain=oob.DomainPresence.ATMOSPHERIC,
            kinematic=oob.KinematicTrait.LINEAR_CORRIDOR,
            electronic=oob.ElectronicSignature.NONE_OBSERVED,
        )
        matches = bl.match(rec)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].category, "radar_outpost")
        self.assertEqual(matches[0].identifier, "RX-1")

    def test_launch_window_match(self) -> None:
        t = datetime.now(timezone.utc)
        bl = oob.CorrelativeBaselines(
            {
                "experimental_launches": [
                    {
                        "id": "TEST-PAD",
                        "lat": 40.0,
                        "lon": -74.0,
                        "window_open": (t - timedelta(minutes=10)).isoformat(),
                        "window_close": (t + timedelta(minutes=10)).isoformat(),
                    }
                ]
            }
        )
        rec = oob.EntityStateRecord(
            event_id="y",
            observed_at=t,
            latitude=40.0,
            longitude=-74.0,
            domain=oob.DomainPresence.ATMOSPHERIC,
            kinematic=oob.KinematicTrait.RAPID_DEPLOYMENT,
            electronic=oob.ElectronicSignature.RF_EM_SPIKE,
        )
        matches = bl.match(rec)
        self.assertTrue(any(m.category == "experimental_launch" for m in matches))


class AnomalousGapIdentifierTests(unittest.TestCase):
    def test_hypersonic_without_thermal_fires(self) -> None:
        rec = oob.EntityStateRecord(
            event_id="hot",
            observed_at=datetime.now(timezone.utc),
            latitude=0,
            longitude=0,
            domain=oob.DomainPresence.ATMOSPHERIC,
            kinematic=oob.KinematicTrait.LINEAR_CORRIDOR,
            electronic=oob.ElectronicSignature.NONE_OBSERVED,
            estimated_speed_mps=343.0 * 8,  # Mach 8
            estimated_thermal_w_per_sr=10.0,
        )
        agi = oob.AnomalousGapIdentifier()
        findings = agi.evaluate(rec)
        self.assertTrue(any(f.rule_id == "hypersonic-without-thermal" for f in findings))

    def test_no_baseline_match_isolation_rule(self) -> None:
        rec = oob.EntityStateRecord(
            event_id="iso",
            observed_at=datetime.now(timezone.utc),
            latitude=0,
            longitude=0,
            domain=oob.DomainPresence.ATMOSPHERIC,
            kinematic=oob.KinematicTrait.LOITERING,
            electronic=oob.ElectronicSignature.NONE_OBSERVED,
        )
        agi = oob.AnomalousGapIdentifier()
        findings = agi.evaluate(rec, baseline_matches=[])
        self.assertTrue(any(f.rule_id == "no-baseline-match" for f in findings))

    def test_trans_medium_rule(self) -> None:
        rec = oob.EntityStateRecord(
            event_id="tm",
            observed_at=datetime.now(timezone.utc),
            latitude=0,
            longitude=0,
            domain=oob.DomainPresence.TRANS_MEDIUM,
            kinematic=oob.KinematicTrait.RAPID_DEPLOYMENT,
            electronic=oob.ElectronicSignature.NONE_OBSERVED,
        )
        findings = oob.AnomalousGapIdentifier().evaluate(rec)
        self.assertTrue(any(f.rule_id == "trans-medium-transition" for f in findings))


if __name__ == "__main__":
    unittest.main()
