//! UACP math engine — public crate entry point.
//!
//! This crate compiles to two targets:
//!   * a native Rust library used by `services/ingestion-worker` via PyO3 / FFI,
//!   * a WebAssembly module loaded directly in the browser by the dashboard.

pub mod parallax;

#[cfg(feature = "wasm")]
mod wasm_bindings {
    use crate::parallax::{calculate_intersection_geocentric, StationObservation};
    use wasm_bindgen::prelude::*;

    /// JS-friendly view of [`StationObservation`].
    #[wasm_bindgen]
    #[derive(Clone, Debug)]
    pub struct JsStationObservation {
        pub lat_rad:       f64,
        pub lon_rad:       f64,
        pub alt_m:         f64,
        pub azimuth_rad:   f64,
        pub elevation_rad: f64,
    }

    #[wasm_bindgen]
    impl JsStationObservation {
        #[wasm_bindgen(constructor)]
        pub fn new(
            lat_rad: f64,
            lon_rad: f64,
            alt_m: f64,
            azimuth_rad: f64,
            elevation_rad: f64,
        ) -> Self {
            Self { lat_rad, lon_rad, alt_m, azimuth_rad, elevation_rad }
        }
    }

    /// Solve the two-station ECEF intersection. Returns `[x, y, z]` (metres)
    /// or throws a JS `Error` when the lines are near-parallel.
    #[wasm_bindgen(js_name = calculateIntersectionGeocentric)]
    pub fn js_calculate_intersection_geocentric(
        a: &JsStationObservation,
        b: &JsStationObservation,
    ) -> Result<Box<[f64]>, JsError> {
        let obs1 = StationObservation {
            station_id:    "A".into(),
            lat_rad:       a.lat_rad,
            lon_rad:       a.lon_rad,
            alt_m:         a.alt_m,
            azimuth_rad:   a.azimuth_rad,
            elevation_rad: a.elevation_rad,
        };
        let obs2 = StationObservation {
            station_id:    "B".into(),
            lat_rad:       b.lat_rad,
            lon_rad:       b.lon_rad,
            alt_m:         b.alt_m,
            azimuth_rad:   b.azimuth_rad,
            elevation_rad: b.elevation_rad,
        };
        match calculate_intersection_geocentric(&obs1, &obs2) {
            Ok((x, y, z)) => Ok(Box::new([x, y, z])),
            Err(msg)      => Err(JsError::new(&msg)),
        }
    }
}
