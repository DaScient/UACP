//! # Multi-station optical parallax (ECEF, WGS-84)
//!
//! This module computes the 3-D intersection of bearing rays from two ground
//! stations that simultaneously observe the same UAP. It is **geocentric**
//! (Earth-Centered, Earth-Fixed) and explicitly accounts for Earth curvature
//! — a flat-Earth approximation is *never* acceptable for stations separated
//! by more than ~10 km, which is the typical baseline for any useful UAP
//! triangulation.
//!
//! ## Geodetic conversions used
//!
//! ### 1. Geodetic → ECEF
//!
//! Given geodetic latitude `φ` (rad), longitude `λ` (rad), and ellipsoidal
//! height `h` (m), the radius of curvature in the prime vertical is
//!
//! ```text
//!         a
//! N(φ) = ───────────────────
//!        √(1 − e² · sin²φ)
//! ```
//!
//! and the ECEF position is
//!
//! ```text
//! x = (N + h) · cos φ · cos λ
//! y = (N + h) · cos φ · sin λ
//! z = (N · (1 − e²) + h) · sin φ
//! ```
//!
//! where `a` = 6378137.0 m (WGS-84 semi-major axis) and
//! `e² = 2f − f²` with `f = 1 / 298.257223563`.
//!
//! ### 2. Local ENU (East-North-Up) unit bearing
//!
//! Azimuth `A` is measured clockwise from true north; elevation `E` is the
//! angle above the local horizon. The unit vector in the local
//! East-North-Up (ENU) frame is:
//!
//! ```text
//! e_ENU = [ cos E · sin A,   // East
//!           cos E · cos A,   // North
//!           sin E         ]  // Up
//! ```
//!
//! ### 3. ENU → ECEF rotation
//!
//! Bearing vectors must be rotated from the station's local tangent plane
//! into the global ECEF frame using the standard ENU → ECEF DCM:
//!
//! ```text
//! R = [ −sin λ            cos λ            0    ]
//!     [ −sin φ · cos λ   −sin φ · sin λ   cos φ ]
//!     [  cos φ · cos λ    cos φ · sin λ   sin φ ]ᵀ
//! ```
//!
//! ### 4. Least-squares ray intersection
//!
//! Given two rays `P_i + t_i · d_i`, the closest-approach midpoint is the
//! least-squares solution to the 2×2 normal system
//!
//! ```text
//! [  d1·d1   −d1·d2 ] [t1]   [ (P2 − P1) · d1 ]
//! [ −d1·d2    d2·d2 ] [t2] = [ (P1 − P2) · d2 ]
//! ```
//!
//! The intersection point returned is the midpoint of the two
//! closest-approach points. If the rays are nearly parallel (angular
//! separation < 1e-3 rad) the system is ill-conditioned and we refuse.

/// WGS-84 semi-major axis (metres).
pub const WGS84_A: f64 = 6_378_137.0;

/// WGS-84 flattening (dimensionless).
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// Minimum angular separation between two bearing rays (rad). Below this the
/// least-squares solution becomes numerically unstable.
pub const NEAR_PARALLEL_RAD: f64 = 1.0e-3;

/// One synchronized observation from a single ground station.
#[derive(Debug, Clone)]
pub struct StationObservation {
    /// Stable identifier for logging / provenance.
    pub station_id:    String,
    /// Geodetic latitude  (radians, WGS-84).
    pub lat_rad:       f64,
    /// Geodetic longitude (radians, WGS-84).
    pub lon_rad:       f64,
    /// Ellipsoidal height (metres above the WGS-84 ellipsoid).
    pub alt_m:         f64,
    /// Bearing azimuth, clockwise from true north (radians).
    pub azimuth_rad:   f64,
    /// Bearing elevation above the local horizon (radians).
    pub elevation_rad: f64,
}

/// Convert WGS-84 geodetic coordinates to ECEF metres.
fn geodetic_to_ecef(lat: f64, lon: f64, h: f64) -> (f64, f64, f64) {
    let e2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = WGS84_A / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let x = (n + h) * cos_lat * lon.cos();
    let y = (n + h) * cos_lat * lon.sin();
    let z = (n * (1.0 - e2) + h) * sin_lat;
    (x, y, z)
}

/// Build the unit bearing vector in the local ENU frame.
fn enu_unit(az: f64, el: f64) -> (f64, f64, f64) {
    let ce = el.cos();
    (ce * az.sin(),   // East
     ce * az.cos(),   // North
     el.sin())        // Up
}

/// Rotate an ENU vector at (lat, lon) into the global ECEF frame.
fn enu_to_ecef(lat: f64, lon: f64, e: f64, n: f64, u: f64) -> (f64, f64, f64) {
    let sl = lat.sin();
    let cl = lat.cos();
    let so = lon.sin();
    let co = lon.cos();
    // Rᵀ · [e, n, u]ᵀ
    let x = -so * e - sl * co * n + cl * co * u;
    let y =  co * e - sl * so * n + cl * so * u;
    let z =            cl       * n + sl       * u;
    (x, y, z)
}

#[inline]
fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

/// Solve the two-ray least-squares intersection in ECEF.
///
/// Returns the midpoint of the closest-approach segment as `(x, y, z)`
/// metres in WGS-84 ECEF, or an `Err` string if the rays are too close to
/// parallel (angular separation < [`NEAR_PARALLEL_RAD`]).
pub fn calculate_intersection_geocentric(
    obs1: &StationObservation,
    obs2: &StationObservation,
) -> Result<(f64, f64, f64), String> {
    // 1. Station positions in ECEF.
    let p1 = geodetic_to_ecef(obs1.lat_rad, obs1.lon_rad, obs1.alt_m);
    let p2 = geodetic_to_ecef(obs2.lat_rad, obs2.lon_rad, obs2.alt_m);

    // 2. Bearing unit vectors: ENU → ECEF.
    let (e1, n1, u1) = enu_unit(obs1.azimuth_rad, obs1.elevation_rad);
    let (e2, n2, u2) = enu_unit(obs2.azimuth_rad, obs2.elevation_rad);
    let d1 = enu_to_ecef(obs1.lat_rad, obs1.lon_rad, e1, n1, u1);
    let d2 = enu_to_ecef(obs2.lat_rad, obs2.lon_rad, e2, n2, u2);

    // 3. Parallel-ray guard. Both d1 and d2 are unit vectors, so
    //    sin(θ) ≥ NEAR_PARALLEL_RAD ⇔ |d1 × d2| ≥ NEAR_PARALLEL_RAD.
    let cross = (
        d1.1 * d2.2 - d1.2 * d2.1,
        d1.2 * d2.0 - d1.0 * d2.2,
        d1.0 * d2.1 - d1.1 * d2.0,
    );
    let sin_theta = (cross.0 * cross.0 + cross.1 * cross.1 + cross.2 * cross.2).sqrt();
    if sin_theta < NEAR_PARALLEL_RAD {
        return Err(format!(
            "rays from stations {} and {} are near-parallel (sin θ = {:.3e} < {:.3e})",
            obs1.station_id, obs2.station_id, sin_theta, NEAR_PARALLEL_RAD
        ));
    }

    // 4. Least-squares 2x2 normal system.
    let d1d1 = dot(d1, d1);   // = 1, but kept symbolic
    let d2d2 = dot(d2, d2);   // = 1
    let d1d2 = dot(d1, d2);
    let r    = (p2.0 - p1.0, p2.1 - p1.1, p2.2 - p1.2);
    let rhs1 =  dot(r, d1);
    let rhs2 = -dot(r, d2);

    let det = d1d1 * d2d2 - d1d2 * d1d2;
    if det.abs() < 1.0e-12 {
        return Err("normal-equations matrix is singular".into());
    }

    let t1 = ( d2d2 * rhs1 + d1d2 * rhs2) / det;
    let t2 = ( d1d2 * rhs1 + d1d1 * rhs2) / det;

    // 5. Closest-approach midpoint in ECEF.
    let q1 = (p1.0 + t1 * d1.0, p1.1 + t1 * d1.1, p1.2 + t1 * d1.2);
    let q2 = (p2.0 + t2 * d2.0, p2.1 + t2 * d2.1, p2.2 + t2 * d2.2);
    Ok((
        0.5 * (q1.0 + q2.0),
        0.5 * (q1.1 + q2.1),
        0.5 * (q1.2 + q2.2),
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two stations 10 km apart, both looking at a target ~5 km above the
    /// midpoint. The intersection should be ~5 km above the surface at the
    /// midpoint baseline.
    #[test]
    fn intersection_above_baseline_midpoint() {
        // Station A at (0°N, 0°E, 0 m), Station B at (0°N, 0.0898°E, 0 m)
        // ≈ 10 km east of A along the equator.
        let a = StationObservation {
            station_id:    "A".into(),
            lat_rad:       0.0,
            lon_rad:       0.0,
            alt_m:         0.0,
            azimuth_rad:   90f64.to_radians(),  // looking due east
            elevation_rad: 45f64.to_radians(),  // 45° up
        };
        let b = StationObservation {
            station_id:    "B".into(),
            lat_rad:       0.0,
            lon_rad:       (10_000.0_f64 / WGS84_A),  // ~10 km east
            alt_m:         0.0,
            azimuth_rad:   270f64.to_radians(), // looking due west
            elevation_rad: 45f64.to_radians(),  // 45° up
        };

        let (x, y, z) = calculate_intersection_geocentric(&a, &b).unwrap();

        // Magnitude of position vector should be ~ a + 5km (roughly).
        let r = (x * x + y * y + z * z).sqrt();
        assert_relative_eq!(r, WGS84_A + 5_000.0, max_relative = 1e-3);
        // The intersection should sit on the equator (z ≈ 0).
        assert!(z.abs() < 1.0);
    }

    #[test]
    fn near_parallel_rays_are_rejected() {
        let a = StationObservation {
            station_id:    "A".into(),
            lat_rad:       0.0, lon_rad: 0.0, alt_m: 0.0,
            azimuth_rad:   0.0, elevation_rad: 45f64.to_radians(),
        };
        // Co-located station with identical bearing → rays are exactly
        // parallel (in fact identical), so the LS system must be rejected.
        let b = StationObservation {
            station_id:    "B".into(),
            lat_rad:       0.0, lon_rad: 0.0, alt_m: 100.0,
            azimuth_rad:   0.0, elevation_rad: 45f64.to_radians(),
        };
        assert!(calculate_intersection_geocentric(&a, &b).is_err());
    }
}
