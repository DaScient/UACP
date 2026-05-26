package triangulation

import (
	"context"
	"testing"
	"time"
)

// fakeMath records every Triangulate call and returns a fixed result.
type fakeMath struct {
	calls int
}

func (f *fakeMath) Triangulate(_ context.Context, _ []WitnessReport) (*InterceptResult, error) {
	f.calls++
	return &InterceptResult{ECEF: [3]float64{1, 2, 3}, AltitudeM: 10_000}, nil
}

// fakeNotifier records every Notify call.
type fakeNotifier struct {
	calls int
}

func (f *fakeNotifier) Notify(_ context.Context, _ *CommunityIncidentNode, _ float64) error {
	f.calls++
	return nil
}

func TestHaversineDistanceMeters(t *testing.T) {
	// 1° of latitude ≈ 111 km.
	d := HaversineDistanceMeters(0, 0, 1, 0)
	if d < 110_000 || d > 112_000 {
		t.Fatalf("expected ~111km for 1° lat, got %f", d)
	}
}

func TestIngestSingleReportDoesNotTrigger(t *testing.T) {
	math := &fakeMath{}
	note := &fakeNotifier{}
	rt := NewRouter(math, note)

	_, triggered, err := rt.Ingest(context.Background(), WitnessReport{
		WitnessID:  "alice",
		ObservedAt: time.Now(),
		Latitude:   40.0,
		Longitude:  -74.0,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if triggered {
		t.Fatal("single-report cluster should not trigger triangulation")
	}
	if math.calls != 0 || note.calls != 0 {
		t.Fatalf("downstream invoked too early: math=%d notify=%d", math.calls, note.calls)
	}
}

func TestIngestFusesNearbyReportsAndTriggers(t *testing.T) {
	math := &fakeMath{}
	note := &fakeNotifier{}
	rt := NewRouter(math, note)
	now := time.Now()

	_, _, _ = rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "alice", ObservedAt: now, Latitude: 40.0, Longitude: -74.0,
	})
	// ~30 km away, 2 minutes later — within both bounds.
	cin, triggered, err := rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "bob", ObservedAt: now.Add(2 * time.Minute),
		Latitude: 40.27, Longitude: -74.0,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !triggered {
		t.Fatal("expected triangulation to trigger on 2nd unique viewpoint")
	}
	if math.calls != 1 || note.calls != 1 {
		t.Fatalf("downstream not invoked exactly once: math=%d notify=%d", math.calls, note.calls)
	}
	if cin.UniqueViewpoints() != 2 {
		t.Fatalf("expected 2 unique viewpoints, got %d", cin.UniqueViewpoints())
	}
	if cin.Intercept == nil {
		t.Fatal("intercept result was not stored on CIN")
	}
}

func TestIngestRejectsDistantReports(t *testing.T) {
	rt := NewRouter(nil, nil)
	now := time.Now()
	_, _, _ = rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "alice", ObservedAt: now, Latitude: 40.0, Longitude: -74.0,
	})
	// > 50 km away → should open a NEW CIN, not fuse.
	_, triggered, _ := rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "bob", ObservedAt: now, Latitude: 41.0, Longitude: -74.0,
	})
	if triggered {
		t.Fatal("distant report should not have fused or triggered triangulation")
	}
	if len(rt.Snapshot()) != 2 {
		t.Fatalf("expected 2 distinct CINs, got %d", len(rt.Snapshot()))
	}
}

func TestIngestRejectsTemporallyDistantReports(t *testing.T) {
	rt := NewRouter(nil, nil)
	now := time.Now()
	_, _, _ = rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "alice", ObservedAt: now, Latitude: 40.0, Longitude: -74.0,
	})
	// Same place, but 30 minutes later → different incident.
	_, triggered, _ := rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "bob", ObservedAt: now.Add(30 * time.Minute),
		Latitude: 40.0, Longitude: -74.0,
	})
	if triggered {
		t.Fatal("temporally distant report should not trigger triangulation")
	}
	if len(rt.Snapshot()) != 2 {
		t.Fatalf("expected 2 distinct CINs, got %d", len(rt.Snapshot()))
	}
}

func TestIngestRejectsInvalidReport(t *testing.T) {
	rt := NewRouter(nil, nil)
	_, _, err := rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "alice", ObservedAt: time.Now(), Latitude: 999, Longitude: 0,
	})
	if err != ErrInvalidReport {
		t.Fatalf("expected ErrInvalidReport, got %v", err)
	}
}

func TestReapDropsOldClusters(t *testing.T) {
	rt := NewRouter(nil, nil)
	old := time.Now().Add(-2 * time.Hour)
	rt.clock = func() time.Time { return old }
	_, _, _ = rt.Ingest(context.Background(), WitnessReport{
		WitnessID: "alice", ObservedAt: old, Latitude: 40.0, Longitude: -74.0,
	})
	rt.clock = time.Now
	if removed := rt.Reap(1 * time.Hour); removed != 1 {
		t.Fatalf("expected to reap 1 stale CIN, got %d", removed)
	}
}
