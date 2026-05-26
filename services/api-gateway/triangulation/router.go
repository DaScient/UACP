// Package triangulation implements the Collaborative Multi-Witness Intersection
// Router for the UAP Intelligence Hub api-gateway.
//
// Responsibilities (architecture doc, section E):
//
//  1. Spatio-Temporal Clustering — incoming witness reports are evaluated in
//     real time. When two or more separate reports occur within a bounded
//     geographic radius (ΔX ≤ 50 km) AND overlapping temporal windows
//     (ΔT ≤ 10 minutes), they are fused into a singular Community Incident
//     Node (CIN).
//
//  2. Automated Local Triangulation Trigger — once a CIN accumulates ≥ 2
//     unique viewpoints, the cluster's azimuth/elevation lines of sight are
//     forwarded to the Rust `math-engine` (over its WASM JS bridge in the
//     browser, or its FFI/HTTP shim server-side) to compute the exact 3-D
//     intercept point, target altitude, and flight vector.
//
//  3. Real-Time Mesh Notifications — clients within a down-range radius of
//     the CIN receive a localised push, either via WebSockets (online
//     dashboard sessions) or WebPush (background mobile clients), enabling
//     real-time multi-station observation grids.
//
// The package is deliberately self-contained — it depends only on the Go
// standard library and the canonical telemetry types declared in this file.
// Once the proto-generated Go bindings land at `gen/go/telemetry`, the
// internal `WitnessReport` type can be aliased to `telemetryv1.UapEvent`.
package triangulation

import (
	"context"
	"errors"
	"math"
	"sort"
	"sync"
	"time"
)

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------

const (
	// MaxClusterRadiusMeters bounds the geographic spread of a CIN.
	// ΔX ≤ 50 km per the architecture document.
	MaxClusterRadiusMeters = 50_000.0

	// MaxClusterTemporalWindow bounds the temporal spread of a CIN.
	// ΔT ≤ 10 minutes per the architecture document.
	MaxClusterTemporalWindow = 10 * time.Minute

	// MinReportsForTriangulation is the smallest viewpoint count that will
	// trigger the Rust math-engine intercept calculation.
	MinReportsForTriangulation = 2

	// NotifyRadiusMeters is the default down-range radius within which
	// online clients are notified about a new high-confidence CIN.
	NotifyRadiusMeters = 75_000.0
)

// -----------------------------------------------------------------------------
// Domain types
// -----------------------------------------------------------------------------

// WitnessReport is the subset of `UapEvent` needed by the router. The api-gateway
// layer is expected to translate the proto message into this form so the
// triangulation package can stay independent of the generated proto module
// during early development.
type WitnessReport struct {
	EventID        string
	WitnessID      string // stable per-device / per-account identifier
	ObservedAt     time.Time
	Latitude       float64 // degrees, WGS-84
	Longitude      float64 // degrees, WGS-84
	AltitudeM      float64 // metres
	AzimuthRad     float64 // radians, [0, 2π)
	ElevationRad   float64 // radians, [-π/2, +π/2]
	FieldOfViewDeg float64
}

// InterceptResult mirrors the value returned by the Rust math-engine.
type InterceptResult struct {
	ECEF      [3]float64 // (x, y, z) in metres
	AltitudeM float64    // metres above WGS-84 ellipsoid
	FlightVec [3]float64 // unit vector along the inferred flight path (ECEF)
}

// CommunityIncidentNode is the fused output of the spatio-temporal clusterer.
type CommunityIncidentNode struct {
	IncidentID string
	OpenedAt   time.Time
	UpdatedAt  time.Time

	// Reports is the chronologically-ordered list of fused witness reports.
	Reports []WitnessReport

	// Centroid is the geographic centre of mass of the cluster.
	Centroid struct {
		Latitude  float64
		Longitude float64
	}

	// Intercept, when non-nil, is the latest triangulation solution.
	Intercept *InterceptResult
}

// UniqueViewpoints returns the number of distinct WitnessIDs in the CIN.
func (c *CommunityIncidentNode) UniqueViewpoints() int {
	seen := make(map[string]struct{}, len(c.Reports))
	for _, r := range c.Reports {
		seen[r.WitnessID] = struct{}{}
	}
	return len(seen)
}

// -----------------------------------------------------------------------------
// Pluggable downstream interfaces
// -----------------------------------------------------------------------------

// MathEngine is the contract the Rust math-engine binding (HTTP shim, FFI,
// or in-process WASM) must satisfy.
type MathEngine interface {
	Triangulate(ctx context.Context, reports []WitnessReport) (*InterceptResult, error)
}

// Notifier delivers real-time alerts to down-range clients.
type Notifier interface {
	// Notify is invoked once per CIN that has crossed the triangulation
	// threshold. Implementations are expected to fan out over WebSockets
	// (online dashboards) and WebPush (background mobile clients) in
	// parallel. `audienceRadiusM` is the radius around the CIN centroid
	// within which subscribers should be alerted.
	Notify(ctx context.Context, cin *CommunityIncidentNode, audienceRadiusM float64) error
}

// -----------------------------------------------------------------------------
// Spatio-temporal math
// -----------------------------------------------------------------------------

// HaversineDistanceMeters returns the great-circle distance between two
// WGS-84 surface points, in metres. Sufficient accuracy (< 0.5%) for the
// 50 km clustering window.
func HaversineDistanceMeters(lat1, lon1, lat2, lon2 float64) float64 {
	const earthRadiusM = 6_371_008.8 // mean Earth radius (IUGG)
	const d2r = math.Pi / 180

	dLat := (lat2 - lat1) * d2r
	dLon := (lon2 - lon1) * d2r
	lat1R := lat1 * d2r
	lat2R := lat2 * d2r

	sinDLat := math.Sin(dLat / 2)
	sinDLon := math.Sin(dLon / 2)
	a := sinDLat*sinDLat + math.Cos(lat1R)*math.Cos(lat2R)*sinDLon*sinDLon
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return earthRadiusM * c
}

// withinTemporalWindow returns true if |a - b| ≤ MaxClusterTemporalWindow.
func withinTemporalWindow(a, b time.Time) bool {
	d := a.Sub(b)
	if d < 0 {
		d = -d
	}
	return d <= MaxClusterTemporalWindow
}

// shouldFuse decides whether `r` is close enough (in space *and* time) to
// any report already in `c` to be considered part of the same CIN.
func shouldFuse(c *CommunityIncidentNode, r WitnessReport) bool {
	for _, existing := range c.Reports {
		if !withinTemporalWindow(existing.ObservedAt, r.ObservedAt) {
			continue
		}
		dist := HaversineDistanceMeters(
			existing.Latitude, existing.Longitude,
			r.Latitude, r.Longitude,
		)
		if dist <= MaxClusterRadiusMeters {
			return true
		}
	}
	return false
}

// recomputeCentroid updates the CIN centroid as the arithmetic mean of all
// member reports' coordinates. For 50 km clusters the flat-earth average
// is well within the precision needed for routing notifications.
func recomputeCentroid(c *CommunityIncidentNode) {
	if len(c.Reports) == 0 {
		return
	}
	var lat, lon float64
	for _, r := range c.Reports {
		lat += r.Latitude
		lon += r.Longitude
	}
	n := float64(len(c.Reports))
	c.Centroid.Latitude = lat / n
	c.Centroid.Longitude = lon / n
}

// -----------------------------------------------------------------------------
// Router
// -----------------------------------------------------------------------------

// Router is the live spatio-temporal clusterer. It is safe for concurrent
// use from any number of HTTP handlers.
type Router struct {
	mu        sync.Mutex
	clusters  map[string]*CommunityIncidentNode
	math      MathEngine
	notifier  Notifier
	idFactory func() string
	clock     func() time.Time
}

// NewRouter constructs a fresh router. `math` and `notifier` may be nil during
// unit-testing — the router will then skip the triangulation/notification
// stages and simply emit fused CINs.
func NewRouter(math MathEngine, notifier Notifier) *Router {
	return &Router{
		clusters:  make(map[string]*CommunityIncidentNode),
		math:      math,
		notifier:  notifier,
		idFactory: defaultIncidentID,
		clock:     time.Now,
	}
}

// ErrInvalidReport is returned when a caller submits an obviously malformed
// witness report (e.g. NaN coordinates).
var ErrInvalidReport = errors.New("triangulation: invalid witness report")

func validate(r WitnessReport) error {
	switch {
	case math.IsNaN(r.Latitude), math.IsNaN(r.Longitude):
		return ErrInvalidReport
	case r.Latitude < -90 || r.Latitude > 90:
		return ErrInvalidReport
	case r.Longitude < -180 || r.Longitude > 180:
		return ErrInvalidReport
	case r.ObservedAt.IsZero():
		return ErrInvalidReport
	}
	return nil
}

// Ingest consumes a single witness report. It returns the CIN the report was
// merged into (or freshly opened) along with `triggered` set to true when the
// CIN newly crossed the MinReportsForTriangulation threshold during this call.
func (rt *Router) Ingest(ctx context.Context, r WitnessReport) (*CommunityIncidentNode, bool, error) {
	if err := validate(r); err != nil {
		return nil, false, err
	}

	rt.mu.Lock()
	cin := rt.findOrCreateLocked(r)
	beforeViewpoints := cin.UniqueViewpoints()
	cin.Reports = append(cin.Reports, r)
	cin.UpdatedAt = rt.clock()
	recomputeCentroid(cin)
	afterViewpoints := cin.UniqueViewpoints()
	rt.mu.Unlock()

	triggered := beforeViewpoints < MinReportsForTriangulation &&
		afterViewpoints >= MinReportsForTriangulation

	if triggered {
		if err := rt.triangulateAndNotify(ctx, cin); err != nil {
			return cin, true, err
		}
	}
	return cin, triggered, nil
}

// findOrCreateLocked searches for a CIN this report can fuse into. Caller MUST
// hold rt.mu.
func (rt *Router) findOrCreateLocked(r WitnessReport) *CommunityIncidentNode {
	for _, c := range rt.clusters {
		if shouldFuse(c, r) {
			return c
		}
	}
	id := rt.idFactory()
	c := &CommunityIncidentNode{
		IncidentID: id,
		OpenedAt:   rt.clock(),
		UpdatedAt:  rt.clock(),
		Reports:    make([]WitnessReport, 0, 4),
	}
	c.Centroid.Latitude = r.Latitude
	c.Centroid.Longitude = r.Longitude
	rt.clusters[id] = c
	return c
}

// triangulateAndNotify is invoked the moment a CIN crosses the trigger
// threshold. Errors from the math engine or notifier are returned to the
// caller but do not roll back the CIN — keeping the partial cluster is
// strictly better than losing it.
func (rt *Router) triangulateAndNotify(ctx context.Context, cin *CommunityIncidentNode) error {
	rt.mu.Lock()
	reports := make([]WitnessReport, len(cin.Reports))
	copy(reports, cin.Reports)
	rt.mu.Unlock()

	if rt.math != nil {
		result, err := rt.math.Triangulate(ctx, reports)
		if err != nil {
			return err
		}
		rt.mu.Lock()
		cin.Intercept = result
		rt.mu.Unlock()
	}

	if rt.notifier != nil {
		if err := rt.notifier.Notify(ctx, cin, NotifyRadiusMeters); err != nil {
			return err
		}
	}
	return nil
}

// Snapshot returns a stable, time-ordered copy of the currently active CINs.
// Useful for HTTP endpoints that expose the live incident board.
func (rt *Router) Snapshot() []*CommunityIncidentNode {
	rt.mu.Lock()
	out := make([]*CommunityIncidentNode, 0, len(rt.clusters))
	for _, c := range rt.clusters {
		// Deep-copy the reports slice so callers can't mutate router state.
		copyC := *c
		copyC.Reports = append([]WitnessReport(nil), c.Reports...)
		out = append(out, &copyC)
	}
	rt.mu.Unlock()
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].OpenedAt.Before(out[j].OpenedAt)
	})
	return out
}

// Reap discards CINs whose most recent report is older than `maxAge`.
// Call periodically from a background goroutine to keep the cluster map
// from growing without bound.
func (rt *Router) Reap(maxAge time.Duration) int {
	cutoff := rt.clock().Add(-maxAge)
	rt.mu.Lock()
	defer rt.mu.Unlock()
	removed := 0
	for id, c := range rt.clusters {
		if c.UpdatedAt.Before(cutoff) {
			delete(rt.clusters, id)
			removed++
		}
	}
	return removed
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

func defaultIncidentID() string {
	// Avoids pulling in `crypto/rand` for this scaffold — a strictly
	// monotonic time-based ID is fine for cluster identification, and
	// production deployments can override `Router.idFactory` with a UUID
	// generator if they need globally-unique IDs across instances.
	return "cin-" + time.Now().UTC().Format("20060102T150405.000000000Z")
}
