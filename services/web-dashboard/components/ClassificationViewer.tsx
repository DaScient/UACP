'use client';

/**
 * ClassificationViewer
 * --------------------
 * Demo dashboard component that:
 *   1. Dynamically loads the Rust-generated WebAssembly math engine.
 *   2. Simulates two `StationObservation` records.
 *   3. Calls the WASM `calculateIntersectionGeocentric` function.
 *   4. Displays the resulting ECEF (x, y, z) intersection.
 *   5. Renders a placeholder classification JSON payload, including a red
 *      "ANOMALOUS" badge driven by `classification.anomalous_flag`.
 *
 * Styling: Tailwind CSS, dark/data-dense aesthetic.
 */

import { useCallback, useState } from 'react';

// ---------------------------------------------------------------------------
// Types — match the public WASM bindings in `services/math-engine/src/lib.rs`.
// ---------------------------------------------------------------------------

interface WasmStationObservation {
  free(): void;
}

interface WasmModule {
  default: (input?: unknown) => Promise<unknown>;
  JsStationObservation: new (
    lat_rad: number,
    lon_rad: number,
    alt_m: number,
    azimuth_rad: number,
    elevation_rad: number,
  ) => WasmStationObservation;
  calculateIntersectionGeocentric: (
    a: WasmStationObservation,
    b: WasmStationObservation,
  ) => Float64Array;
}

interface ClassificationPayload {
  event_id: string;
  classification_metadata: {
    assigned_shape:    'Tic-Tac' | 'Sphere' | 'Disc' | 'Triangle' | 'Unknown';
    confidence_score:  number;
    speed_profile:     'Subsonic' | 'Supersonic' | 'Hypersonic' | 'Trans-Medium';
    mach_number:       number;
    altitude_m:        number;
    anomalous_flag:    boolean;
  };
  storage_routing_path: string;
}

// ---------------------------------------------------------------------------
// Static demo data
// ---------------------------------------------------------------------------

const DEMO_STATIONS = [
  { id: 'STA-A', lat: 0,        lon: 0,         alt: 0,  az: 90,  el: 45 },
  { id: 'STA-B', lat: 0,        lon: 0.0898,    alt: 0,  az: 270, el: 45 },
] as const;

const DEMO_PAYLOAD: ClassificationPayload = {
  event_id: 'demo-5b6f2e08-9d8a-4b39-9c0a-3c5d2c5d4e6f',
  classification_metadata: {
    assigned_shape:   'Tic-Tac',
    confidence_score: 0.93,
    speed_profile:    'Hypersonic',
    mach_number:      8.4,
    altitude_m:       12_000,
    anomalous_flag:   true,
  },
  storage_routing_path:
    'data/processed/Hypersonic/Tic-Tac/demo-5b6f2e08-9d8a-4b39-9c0a-3c5d2c5d4e6f/',
};

const deg2rad = (d: number) => (d * Math.PI) / 180.0;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ClassificationViewer({
  wasmModulePath,
}: { wasmModulePath: string }) {

  const [wasm,       setWasm]       = useState<WasmModule | null>(null);
  const [loading,    setLoading]    = useState(false);
  const [intersect,  setIntersect]  = useState<[number, number, number] | null>(null);
  const [error,      setError]      = useState<string | null>(null);

  const loadWasm = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // `@vite-ignore` / `/* webpackIgnore: true */` — we want a runtime path,
      // not a build-time bundle. The path arrives via prop so Pages can host
      // the artefact at `/wasm-engine/uacp_math_engine.js`.
      const mod = (await import(/* webpackIgnore: true */ wasmModulePath)) as WasmModule;
      await mod.default();   // wasm-bindgen init
      setWasm(mod);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [wasmModulePath]);

  const runIntersection = useCallback(() => {
    if (!wasm) return;
    setError(null);
    try {
      const [a, b] = DEMO_STATIONS;
      const sa = new wasm.JsStationObservation(
        deg2rad(a.lat), deg2rad(a.lon), a.alt, deg2rad(a.az), deg2rad(a.el),
      );
      const sb = new wasm.JsStationObservation(
        deg2rad(b.lat), deg2rad(b.lon), b.alt, deg2rad(b.az), deg2rad(b.el),
      );
      const out = wasm.calculateIntersectionGeocentric(sa, sb);
      setIntersect([out[0], out[1], out[2]]);
      sa.free(); sb.free();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [wasm]);

  const anomalous = DEMO_PAYLOAD.classification_metadata.anomalous_flag;

  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">

      {/* ------------------------------------------------------------------ */}
      {/* Left card: math engine                                             */}
      {/* ------------------------------------------------------------------ */}
      <div className="bg-uap-card rounded-2xl p-6 ring-1 ring-white/5 shadow-lg">
        <h2 className="text-lg font-semibold mb-4">Math Engine (Rust → WASM)</h2>

        <div className="flex flex-wrap gap-3 mb-4">
          <button
            onClick={loadWasm}
            disabled={loading || wasm !== null}
            className="px-4 py-2 rounded-md bg-uap-accent/90 hover:bg-uap-accent
                       text-black font-medium disabled:opacity-40"
          >
            {wasm ? '✓ Engine loaded' : loading ? 'Loading…' : 'Load Math Engine'}
          </button>

          <button
            onClick={runIntersection}
            disabled={!wasm}
            className="px-4 py-2 rounded-md border border-uap-accent/40
                       hover:bg-uap-accent/10 disabled:opacity-40"
          >
            Calculate Intersection
          </button>
        </div>

        <table className="w-full text-sm mb-4">
          <thead className="text-gray-400">
            <tr>
              <th className="text-left">Station</th>
              <th className="text-right">lat°</th>
              <th className="text-right">lon°</th>
              <th className="text-right">alt m</th>
              <th className="text-right">az°</th>
              <th className="text-right">el°</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {DEMO_STATIONS.map((s) => (
              <tr key={s.id} className="border-t border-white/5">
                <td className="py-1">{s.id}</td>
                <td className="text-right">{s.lat}</td>
                <td className="text-right">{s.lon}</td>
                <td className="text-right">{s.alt}</td>
                <td className="text-right">{s.az}</td>
                <td className="text-right">{s.el}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="bg-black/40 rounded-md p-4 font-mono text-sm">
          {error ? (
            <span className="text-red-400">⚠ {error}</span>
          ) : intersect ? (
            <>
              <div>x = {intersect[0].toFixed(3)} m</div>
              <div>y = {intersect[1].toFixed(3)} m</div>
              <div>z = {intersect[2].toFixed(3)} m</div>
            </>
          ) : (
            <span className="text-gray-500">No intersection computed yet.</span>
          )}
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Right card: classification payload                                 */}
      {/* ------------------------------------------------------------------ */}
      <div className="bg-uap-card rounded-2xl p-6 ring-1 ring-white/5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Classification Payload (preview)</h2>
          <span
            className={
              'px-2 py-1 rounded text-xs font-bold ' +
              (anomalous
                ? 'bg-red-600 text-white'
                : 'bg-emerald-600 text-white')
            }
          >
            {anomalous ? 'ANOMALOUS' : 'NOMINAL'}
          </span>
        </div>
        <pre className="bg-black/40 rounded-md p-4 text-xs overflow-auto">
{JSON.stringify(DEMO_PAYLOAD, null, 2)}
        </pre>
        <p className="text-gray-500 text-xs mt-3">
          The backend will populate this object via the ingestion-worker
          pipeline. Shape is shown by name; <code>anomalous_flag</code>
          drives the badge colour.
        </p>
      </div>
    </section>
  );
}
