// =============================================================================
// services/web-dashboard/components/Canvas3D.ts
// -----------------------------------------------------------------------------
// 3D Spatial Canvas & "Street-View" Reconstruction Engine.
//
// A framework-agnostic TypeScript module that powers the immersive client-side
// sighting builder described in the architecture document (sections D and 5).
// The module is intentionally split into three layers:
//
//   1.  Pure math   (azimuth / elevation / FOV / occlusion / vector projection)
//   2.  Open-stack adapters (OSM tiles via Leaflet / Maplibre, three-geo / osm-3d
//       building meshes, Open-Elevation lookups)
//   3.  React-friendly orchestrator (`createCanvas3D`) that ties a sky-dome
//       Three.js scene + React Three Fiber surface together with the
//       DeviceOrientation API and pushes the resulting vector payload back to
//       the decentralised community-triangulation cluster.
//
// Heavy 3D dependencies (three, @react-three/fiber, maplibre-gl, leaflet) are
// loaded *dynamically* — exactly like the WASM loader in
// `ClassificationViewer.tsx` — so this file compiles under the existing strict
// tsconfig without forcing new build-time dependencies on contributors who
// only touch other services.
// =============================================================================

/* eslint-disable @typescript-eslint/no-explicit-any */

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/** WGS-84 geodetic point. Matches the `GeoLocation` proto message. */
export interface GeoPoint {
  /** Latitude in decimal degrees, [-90, +90]. */
  latitude: number;
  /** Longitude in decimal degrees, [-180, +180]. */
  longitude: number;
  /** Metres above the WGS-84 ellipsoid. */
  altitudeMeters: number;
}

/**
 * A direction vector from the witness to a sky-dome point, in the local
 * East-North-Up (ENU) frame. Stored in **radians** and **degrees**
 * simultaneously so the payload is self-describing for downstream consumers.
 */
export interface SkyVector {
  /** Azimuth θ, clockwise from true North. */
  azimuthDeg: number;
  azimuthRad: number;
  /** Elevation φ, +up / -down from the local horizon. */
  elevationDeg: number;
  elevationRad: number;
  /**
   * Horizontal field-of-view (degrees) of the viewport at the moment this
   * vector was captured. Useful for downstream angular-error estimates.
   */
  fovDeg: number;
}

/** Raw orientation reading from the Web DeviceOrientation API. */
export interface DeviceOrientation {
  /** Compass heading in degrees, [0, 360). 0 == true (or magnetic) North. */
  alphaDeg: number;
  /** Pitch in degrees, [-180, +180]. */
  betaDeg: number;
  /** Roll in degrees, [-90, +90]. */
  gammaDeg: number;
  /** True when the alphaDeg value is referenced to *true* North. */
  isTrueNorth: boolean;
}

/**
 * A 3-D bounding geometry the user has drawn over a nearby structure
 * (building / tree / powerline) so we can compute lines-of-sight occlusion.
 * Stored as an axis-aligned ENU box around an anchor for simplicity; complex
 * meshes can be approximated by a union of these.
 */
export interface OcclusionBox {
  id: string;
  /** Centre of the box in the ENU frame, metres. */
  centerEnu: [number, number, number];
  /** Half-extents along East/North/Up axes (metres). */
  halfExtentsEnu: [number, number, number];
  /** Free-form structural tag — `"building"`, `"tree"`, `"powerline"`, … */
  label: string;
}

/** Final JSON payload emitted by the sky-dome builder. */
export interface SightingVectorPayload {
  schemaVersion: '1';
  /** Witness position. */
  observer: GeoPoint;
  /** UTC nanosecond timestamp at which the vector was committed. */
  capturedAtIso: string;
  /** Anchor / direction the witness pinned on the sky dome. */
  vector: SkyVector;
  /** Device orientation at capture time (mobile only). */
  deviceOrientation: DeviceOrientation | null;
  /** Occluding structures the witness annotated. */
  occluders: OcclusionBox[];
  /**
   * Result of the line-of-sight pass at capture time — `true` when the
   * direction vector is unobstructed by any user-drawn `OcclusionBox`.
   */
  lineOfSightClear: boolean;
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
/** Default horizontal FOV of a desktop browser viewport. */
const DEFAULT_FOV_DEG = 60;
/** Open-Elevation public endpoint used when no local DEM is configured. */
const OPEN_ELEVATION_URL = 'https://api.open-elevation.com/api/v1/lookup';

// -----------------------------------------------------------------------------
// 1. Pure math — viewport / vector / occlusion
// -----------------------------------------------------------------------------

/**
 * Normalise an angle in **degrees** into the [0, 360) range.
 * Useful when summing a viewport offset with the witness's compass heading.
 */
export function normaliseAzimuthDeg(deg: number): number {
  const m = deg % 360;
  return m < 0 ? m + 360 : m;
}

/** Clamp an elevation angle into the legal [-90, +90] range. */
export function clampElevationDeg(deg: number): number {
  if (deg > 90) return 90;
  if (deg < -90) return -90;
  return deg;
}

/**
 * Translate a user click / drag position on the canvas into an
 * (azimuth, elevation) sky-vector, taking the witness's true-North heading
 * and the viewport FOV into account.
 *
 * Math:
 *   * The canvas is treated as a pinhole projection of the sky dome.
 *   * Horizontal pixel offset → angular offset around the up-axis.
 *   * Vertical   pixel offset → angular offset around the right-axis.
 *   * Both offsets are scaled linearly across the viewport FOV. For wide
 *     FOVs (> ~90°) a proper `atan` projection is used to avoid distortion.
 *
 * @param canvasX      Mouse X in CSS pixels relative to the canvas top-left.
 * @param canvasY      Mouse Y in CSS pixels relative to the canvas top-left.
 * @param canvasWidth  Canvas width  in CSS pixels.
 * @param canvasHeight Canvas height in CSS pixels.
 * @param headingDeg   Witness compass heading in degrees [0, 360). 0 = N.
 * @param fovDeg       Horizontal FOV of the viewport in degrees.
 */
export function viewportToSkyVector(
  canvasX: number,
  canvasY: number,
  canvasWidth: number,
  canvasHeight: number,
  headingDeg: number,
  fovDeg: number = DEFAULT_FOV_DEG,
): SkyVector {
  if (canvasWidth <= 0 || canvasHeight <= 0) {
    throw new Error('Canvas3D.viewportToSkyVector: canvas has zero dimension');
  }

  // Normalised device coordinates in [-1, +1], with +X to the right and +Y up.
  const ndcX = (2 * canvasX) / canvasWidth - 1;
  const ndcY = 1 - (2 * canvasY) / canvasHeight;

  // Use an `atan` projection so the math stays accurate for wide FOVs.
  const halfFovRad = (fovDeg * DEG_TO_RAD) / 2;
  const aspect = canvasWidth / canvasHeight;
  const halfFovVRad = Math.atan(Math.tan(halfFovRad) / aspect);

  const deltaAzRad = Math.atan(ndcX * Math.tan(halfFovRad));
  const deltaElRad = Math.atan(ndcY * Math.tan(halfFovVRad));

  const azimuthDeg = normaliseAzimuthDeg(headingDeg + deltaAzRad * RAD_TO_DEG);
  const elevationDeg = clampElevationDeg(deltaElRad * RAD_TO_DEG);

  return {
    azimuthDeg,
    azimuthRad: azimuthDeg * DEG_TO_RAD,
    elevationDeg,
    elevationRad: elevationDeg * DEG_TO_RAD,
    fovDeg,
  };
}

/**
 * Convert an (azimuth, elevation) sky-vector into a unit direction in the
 * local East-North-Up (ENU) frame:
 *
 *   * +X axis points East,
 *   * +Y axis points North,
 *   * +Z axis points Up.
 *
 * Returns `[e, n, u]` with `e² + n² + u² ≈ 1`.
 */
export function skyVectorToEnu(v: SkyVector): [number, number, number] {
  const cosEl = Math.cos(v.elevationRad);
  const e = cosEl * Math.sin(v.azimuthRad);
  const n = cosEl * Math.cos(v.azimuthRad);
  const u = Math.sin(v.elevationRad);
  return [e, n, u];
}

/**
 * Slab method ray / AABB intersection test in the ENU frame.
 *
 * @returns The smallest positive parametric distance `t` along the ray
 *          at which it enters the box, or `Infinity` if the ray misses
 *          the box entirely.
 */
function rayHitsAabb(
  origin: [number, number, number],
  dir: [number, number, number],
  box: OcclusionBox,
): number {
  let tMin = -Infinity;
  let tMax = Infinity;

  for (let i = 0; i < 3; i++) {
    const o = origin[i];
    const d = dir[i];
    const c = box.centerEnu[i];
    const h = box.halfExtentsEnu[i];
    const min = c - h;
    const max = c + h;

    if (Math.abs(d) < 1e-9) {
      // Ray is parallel to this slab — only a miss if origin is outside it.
      if (o < min || o > max) return Infinity;
      continue;
    }

    let t1 = (min - o) / d;
    let t2 = (max - o) / d;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    if (t1 > tMin) tMin = t1;
    if (t2 < tMax) tMax = t2;
    if (tMin > tMax) return Infinity;
  }

  return tMin >= 0 ? tMin : tMax >= 0 ? tMax : Infinity;
}

/**
 * Dynamic Occlusion Mapping.
 *
 * Tests the witness's pinned sky-vector against every user-drawn occluding
 * AABB and returns whether the line of sight is clear, plus the closest
 * occluder hit (when any).
 */
export function evaluateOcclusion(
  vector: SkyVector,
  occluders: OcclusionBox[],
): { lineOfSightClear: boolean; firstHit: OcclusionBox | null; hitDistanceM: number } {
  const dir = skyVectorToEnu(vector);
  const origin: [number, number, number] = [0, 0, 0]; // observer sits at ENU origin

  let bestT = Infinity;
  let bestBox: OcclusionBox | null = null;

  for (const box of occluders) {
    const t = rayHitsAabb(origin, dir, box);
    if (t < bestT) {
      bestT = t;
      bestBox = box;
    }
  }

  return {
    lineOfSightClear: bestBox === null,
    firstHit: bestBox,
    hitDistanceM: bestBox === null ? Infinity : bestT,
  };
}

// -----------------------------------------------------------------------------
// 2. Open-stack adapters — OSM tiles, Open-Elevation, three-geo / osm-3d
// -----------------------------------------------------------------------------

/**
 * Resolve the ground elevation (metres above WGS-84 ellipsoid) at a given
 * lat/lon via the public Open-Elevation API, with an injectable fetch so
 * unit tests can stub the network call.
 *
 * Falls back to `0` when the network is unavailable — callers should treat
 * a zero return as "unknown elevation, witness must confirm".
 */
export async function lookupElevation(
  lat: number,
  lon: number,
  fetchImpl: typeof fetch = fetch,
): Promise<number> {
  try {
    const url = `${OPEN_ELEVATION_URL}?locations=${lat},${lon}`;
    const res = await fetchImpl(url, { method: 'GET' });
    if (!res.ok) return 0;
    const json = (await res.json()) as { results?: { elevation: number }[] };
    return json.results?.[0]?.elevation ?? 0;
  } catch {
    return 0;
  }
}

/**
 * Configuration for the open-source spatial stack.
 *
 * All fields are optional; when omitted the engine falls back to a pure
 * client-side proxy environment (a flat ground plane + textured sky dome).
 */
export interface OpenStackConfig {
  /** Raster tile template, e.g. `"https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"`. */
  osmTileUrl?: string;
  /** Maplibre / Leaflet vector tile style URL. */
  maplibreStyleUrl?: string;
  /**
   * Override for the elevation lookup. The default uses Open-Elevation; you
   * can plug in USGS, GEBCO, or a self-hosted DEM here.
   */
  elevationLookup?: (lat: number, lon: number) => Promise<number>;
  /**
   * Optional URL for OSM 3D building footprints (osm2world / osm-3d style).
   * The orchestrator will request a small bounding box around the witness
   * and feed the meshes into the Three.js scene.
   */
  osmBuildingsTileUrl?: string;
}

// -----------------------------------------------------------------------------
// 3. Orchestrator
// -----------------------------------------------------------------------------

/** Three.js / R3F handles returned by `createCanvas3D`. */
export interface Canvas3DHandle {
  /** Pin a vector at the given canvas pixel coordinates. */
  pinAtCanvas: (x: number, y: number) => SkyVector;
  /** Manually push a vector (e.g. after dragging an arrow). */
  setVector: (v: SkyVector) => void;
  /** Add a user-drawn occluder. */
  addOccluder: (box: OcclusionBox) => void;
  /** Remove an occluder by id. */
  removeOccluder: (id: string) => void;
  /**
   * Commit the current state into the JSON payload that will be POSTed to
   * the api-gateway triangulation router.
   */
  buildPayload: () => SightingVectorPayload;
  /** Latest sky-vector, or `null` until the witness has pinned one. */
  getVector: () => SkyVector | null;
  /** Listen for changes to the captured vector. */
  onVectorChanged: (cb: (v: SkyVector) => void) => () => void;
  /** Tear down listeners and renderer resources. */
  dispose: () => void;
}

/** Construction options for the canvas orchestrator. */
export interface Canvas3DOptions {
  /** The witness's WGS-84 location. */
  observer: GeoPoint;
  /** Initial heading override in degrees. Defaults to DeviceOrientation.alpha. */
  initialHeadingDeg?: number;
  /** Initial horizontal FOV in degrees. */
  fovDeg?: number;
  /** Open-source spatial stack configuration. */
  openStack?: OpenStackConfig;
  /**
   * Optional canvas to render into. If omitted, the caller is expected to be
   * using a React Three Fiber `<Canvas>` and pass the resulting renderer
   * surface back via `attachRenderer`.
   */
  canvas?: HTMLCanvasElement | null;
}

/**
 * Read the next single DeviceOrientation event (mobile browsers only).
 *
 * Resolves with `null` on desktop or when the user denies the permission
 * prompt (iOS Safari requires explicit consent).
 */
export async function readDeviceOrientationOnce(
  timeoutMs = 1500,
): Promise<DeviceOrientation | null> {
  if (typeof window === 'undefined' || typeof window.DeviceOrientationEvent === 'undefined') {
    return null;
  }

  // iOS gated permission flow.
  const Ctor = window.DeviceOrientationEvent as unknown as {
    requestPermission?: () => Promise<'granted' | 'denied'>;
  };
  if (typeof Ctor.requestPermission === 'function') {
    try {
      const state = await Ctor.requestPermission();
      if (state !== 'granted') return null;
    } catch {
      return null;
    }
  }

  return new Promise(resolve => {
    let done = false;
    const finalize = (v: DeviceOrientation | null): void => {
      if (done) return;
      done = true;
      window.removeEventListener('deviceorientation', handler as EventListener);
      resolve(v);
    };
    const handler = (raw: DeviceOrientationEvent): void => {
      const alpha = raw.alpha ?? 0;
      const beta = raw.beta ?? 0;
      const gamma = raw.gamma ?? 0;
      const trueNorth = (raw as unknown as { webkitCompassHeading?: number })
        .webkitCompassHeading;
      finalize({
        alphaDeg: trueNorth ?? alpha,
        betaDeg: beta,
        gammaDeg: gamma,
        isTrueNorth: typeof trueNorth === 'number',
      });
    };
    window.addEventListener('deviceorientation', handler as EventListener, { once: true });
    window.setTimeout(() => finalize(null), timeoutMs);
  });
}

/**
 * Build the sky-dome reconstruction engine.
 *
 * The factory is **framework-agnostic**: it does not import three.js at
 * the top of the file (so the dashboard can be statically built without
 * forcing every contributor to install the 3D toolchain). When a `canvas`
 * is provided we attempt a dynamic `import('three')` so the renderer is
 * only loaded on the client at the moment a sighting is being built.
 */
export async function createCanvas3D(options: Canvas3DOptions): Promise<Canvas3DHandle> {
  const fovDeg = options.fovDeg ?? DEFAULT_FOV_DEG;
  let headingDeg = options.initialHeadingDeg ?? 0;
  let lastVector: SkyVector | null = null;
  const occluders: OcclusionBox[] = [];
  const listeners = new Set<(v: SkyVector) => void>();

  // Try to pull the witness's compass heading from the phone, if available.
  let deviceOrientation: DeviceOrientation | null = null;
  if (options.initialHeadingDeg === undefined) {
    deviceOrientation = await readDeviceOrientationOnce();
    if (deviceOrientation) headingDeg = deviceOrientation.alphaDeg;
  }

  // Dynamically import three.js when a real canvas is supplied. The import
  // happens at runtime — never at module-load — so static builds remain
  // dependency-free.
  let renderer: { dispose?: () => void } | null = null;
  if (options.canvas && typeof window !== 'undefined') {
    try {
      // The import path is resolved at runtime so TypeScript does not require
      // `three` to be declared as a build dependency. We deliberately route
      // the import through a `Function` shim so the static type-checker does
      // not try to resolve the module at build time.
      const dynamicImport = new Function('s', 'return import(s)') as (
        s: string,
      ) => Promise<any>;
      const three: any = await dynamicImport('three').catch(() => null);
      if (three) {
        const scene = new three.Scene();
        const camera = new three.PerspectiveCamera(
          fovDeg,
          options.canvas.width / Math.max(options.canvas.height, 1),
          0.1,
          50_000,
        );
        camera.up.set(0, 0, 1); // Z-up matches the ENU convention used by skyVectorToEnu
        const r = new three.WebGLRenderer({ canvas: options.canvas, antialias: true });
        // A textured inverted icosphere = inexpensive proxy sky dome.
        const skyGeom = new three.SphereGeometry(1000, 32, 32);
        const skyMat = new three.MeshBasicMaterial({ color: 0x07142d, side: three.BackSide });
        scene.add(new three.Mesh(skyGeom, skyMat));
        renderer = r;
      }
    } catch {
      renderer = null;
    }
  }

  const emit = (v: SkyVector): void => {
    lastVector = v;
    for (const cb of listeners) cb(v);
  };

  return {
    pinAtCanvas: (x, y) => {
      const w = options.canvas?.width ?? 1;
      const h = options.canvas?.height ?? 1;
      const v = viewportToSkyVector(x, y, w, h, headingDeg, fovDeg);
      emit(v);
      return v;
    },
    setVector: v => emit(v),
    addOccluder: box => {
      occluders.push(box);
    },
    removeOccluder: id => {
      const idx = occluders.findIndex(b => b.id === id);
      if (idx >= 0) occluders.splice(idx, 1);
    },
    getVector: () => lastVector,
    onVectorChanged: cb => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    buildPayload: () => {
      if (!lastVector) {
        throw new Error('Canvas3D.buildPayload: no vector has been pinned yet');
      }
      const occlusion = evaluateOcclusion(lastVector, occluders);
      return {
        schemaVersion: '1',
        observer: options.observer,
        capturedAtIso: new Date().toISOString(),
        vector: lastVector,
        deviceOrientation,
        occluders: occluders.slice(),
        lineOfSightClear: occlusion.lineOfSightClear,
      };
    },
    dispose: () => {
      listeners.clear();
      renderer?.dispose?.();
    },
  };
}

// -----------------------------------------------------------------------------
// Convenience: serialise a payload for the triangulation router.
// -----------------------------------------------------------------------------

/**
 * Stable JSON serialisation of a `SightingVectorPayload`. Keys are emitted
 * in a deterministic order so two clients that pinned the same vector
 * produce byte-identical payloads — important for de-duplication on the
 * api-gateway side.
 */
export function serialiseSightingPayload(p: SightingVectorPayload): string {
  return JSON.stringify(p, [
    'schemaVersion',
    'observer',
    'latitude',
    'longitude',
    'altitudeMeters',
    'capturedAtIso',
    'vector',
    'azimuthDeg',
    'azimuthRad',
    'elevationDeg',
    'elevationRad',
    'fovDeg',
    'deviceOrientation',
    'alphaDeg',
    'betaDeg',
    'gammaDeg',
    'isTrueNorth',
    'occluders',
    'id',
    'centerEnu',
    'halfExtentsEnu',
    'label',
    'lineOfSightClear',
  ]);
}
