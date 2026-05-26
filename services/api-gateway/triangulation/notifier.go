// Package triangulation — mesh notification fan-out.
//
// `MeshNotifier` is a reference `Notifier` implementation that delivers
// localised alerts about a newly-triangulated Community Incident Node to:
//
//   * online dashboard sessions (over a `SocketHub` of WebSocket peers), and
//   * background mobile clients (via the `WebPushDispatcher` interface).
//
// Subscribers are filtered by distance from the CIN centroid using the
// Haversine helper from router.go, so the down-range radius (default
// NotifyRadiusMeters) is honoured exactly. Both transports are invoked
// concurrently; the function returns the first transport error, if any,
// after both have completed.
package triangulation

import (
	"context"
	"encoding/json"
	"sync"
)

// SocketSubscriber is one client (dashboard tab or mobile app foreground
// session) currently holding an open WebSocket to the api-gateway.
type SocketSubscriber struct {
	ID        string
	Latitude  float64
	Longitude float64
	// Send delivers a serialised payload. The implementation is expected
	// to handle write deadlines and buffered-channel back-pressure.
	Send func(payload []byte) error
}

// SocketHub holds the live WebSocket subscribers. Implementations must be
// safe for concurrent use; the trivial in-memory hub below already is.
type SocketHub interface {
	Subscribers() []SocketSubscriber
}

// WebPushDispatcher sends a WebPush notification to a single subscription.
// Production implementations would wrap the VAPID-signed HTTP request to
// the user's push service endpoint.
type WebPushDispatcher interface {
	PushSubscribers() []SocketSubscriber
	Push(ctx context.Context, sub SocketSubscriber, payload []byte) error
}

// MeshNotifier implements Notifier.
type MeshNotifier struct {
	Hub  SocketHub
	Push WebPushDispatcher
}

// AlertPayload is the JSON body delivered to every subscriber.
type AlertPayload struct {
	Kind             string  `json:"kind"`
	IncidentID       string  `json:"incident_id"`
	OpenedAt         string  `json:"opened_at"`
	CentroidLat      float64 `json:"centroid_lat"`
	CentroidLon      float64 `json:"centroid_lon"`
	UniqueViewpoints int     `json:"unique_viewpoints"`
	HasIntercept     bool    `json:"has_intercept"`
}

// Notify fans out an alert about `cin` to every subscriber within
// `audienceRadiusM` of the cluster centroid. The two transports run
// concurrently and the first error encountered is returned.
func (m *MeshNotifier) Notify(ctx context.Context, cin *CommunityIncidentNode, audienceRadiusM float64) error {
	payload := AlertPayload{
		Kind:             "uacp.community_incident.opened",
		IncidentID:       cin.IncidentID,
		OpenedAt:         cin.OpenedAt.UTC().Format("2006-01-02T15:04:05.000000000Z"),
		CentroidLat:      cin.Centroid.Latitude,
		CentroidLon:      cin.Centroid.Longitude,
		UniqueViewpoints: cin.UniqueViewpoints(),
		HasIntercept:     cin.Intercept != nil,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	var (
		wg      sync.WaitGroup
		errMu   sync.Mutex
		firstErr error
	)

	record := func(e error) {
		if e == nil {
			return
		}
		errMu.Lock()
		if firstErr == nil {
			firstErr = e
		}
		errMu.Unlock()
	}

	inRange := func(lat, lon float64) bool {
		return HaversineDistanceMeters(
			cin.Centroid.Latitude, cin.Centroid.Longitude, lat, lon,
		) <= audienceRadiusM
	}

	if m.Hub != nil {
		for _, sub := range m.Hub.Subscribers() {
			if !inRange(sub.Latitude, sub.Longitude) {
				continue
			}
			wg.Add(1)
			go func(s SocketSubscriber) {
				defer wg.Done()
				record(s.Send(body))
			}(sub)
		}
	}

	if m.Push != nil {
		for _, sub := range m.Push.PushSubscribers() {
			if !inRange(sub.Latitude, sub.Longitude) {
				continue
			}
			wg.Add(1)
			go func(s SocketSubscriber) {
				defer wg.Done()
				record(m.Push.Push(ctx, s, body))
			}(sub)
		}
	}

	wg.Wait()
	return firstErr
}

// -----------------------------------------------------------------------------
// In-memory hub (development / unit-test convenience)
// -----------------------------------------------------------------------------

// InMemorySocketHub is a goroutine-safe SocketHub useful for tests and the
// local Docker Compose dev environment. Production deployments should swap
// it for a Redis-backed (or NATS-backed) hub when running multi-replica.
type InMemorySocketHub struct {
	mu   sync.RWMutex
	subs map[string]SocketSubscriber
}

func NewInMemorySocketHub() *InMemorySocketHub {
	return &InMemorySocketHub{subs: make(map[string]SocketSubscriber)}
}

func (h *InMemorySocketHub) Add(s SocketSubscriber)    { h.mu.Lock(); h.subs[s.ID] = s; h.mu.Unlock() }
func (h *InMemorySocketHub) Remove(id string)          { h.mu.Lock(); delete(h.subs, id); h.mu.Unlock() }
func (h *InMemorySocketHub) Subscribers() []SocketSubscriber {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make([]SocketSubscriber, 0, len(h.subs))
	for _, s := range h.subs {
		out = append(out, s)
	}
	return out
}
