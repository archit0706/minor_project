"""Google Maps Routes API (computeRoutes) client.

Returns real API values only. When there is no key or the call fails, route() raises
GoogleMapsError and the caller runs in offline mode -- no substitute ETA is invented.

Google's `duration` is a traffic-aware ETA for an ordinary car. It is NOT an ambulance
travel time (no siren/priority effect, and no measured ambulance trips exist here).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELDS = ("routes.duration,routes.staticDuration,routes.distanceMeters,"
          "routes.polyline.encodedPolyline,routes.travelAdvisory.speedReadingIntervals,"
          "routes.legs.duration,routes.legs.staticDuration,routes.legs.distanceMeters")


class GoogleMapsError(RuntimeError):
    pass


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not k.strip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def api_key() -> str | None:
    _load_dotenv()
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    return key or None


def _latlng(p):
    return {"location": {"latLng": {"latitude": float(p[0]), "longitude": float(p[1])}}}


def route(origin, destination, via=None, key: str | None = None) -> dict:
    """Traffic-aware car route origin -> [via] -> destination. Points are (lat, lon)."""
    key = key or api_key()
    if not key:
        raise GoogleMapsError("GOOGLE_MAPS_API_KEY not set")
    body = {"origin": _latlng(origin), "destination": _latlng(destination), "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE", "extraComputations": ["TRAFFIC_ON_POLYLINE"]}
    if via:
        body["intermediates"] = [_latlng(via)]
    headers = {"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS}
    t0 = time.perf_counter()
    for attempt in range(3):  # DNS hiccups were observed on this network
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=20)
            break
        except requests.ConnectionError as e:
            if attempt == 2:
                raise GoogleMapsError(f"network error: {e}") from e
    if r.status_code != 200:
        raise GoogleMapsError(f"HTTP {r.status_code}: {r.text[:300]}")
    routes = r.json().get("routes")
    if not routes:
        raise GoogleMapsError("no route returned")
    rt = routes[0]
    speeds = {}
    for iv in rt.get("travelAdvisory", {}).get("speedReadingIntervals", []):
        speeds[iv.get("speed", "UNKNOWN")] = speeds.get(iv.get("speed", "UNKNOWN"), 0) + 1
    sec = lambda v: float(str(v).rstrip("s"))
    duration, static = sec(rt["duration"]), sec(rt["staticDuration"])
    return {
        "duration_s": duration,                    # traffic-aware ETA (car)
        "static_s": static,                        # same route without live traffic
        "traffic_factor": duration / static if static > 0 else 1.0,
        "distance_m": float(rt.get("distanceMeters", 0)),
        "points": decode_polyline(rt["polyline"]["encodedPolyline"]),
        "speed_intervals": speeds,                 # NORMAL / SLOW / TRAFFIC_JAM counts
        "legs": [{"time_s": sec(lg.get("duration", "0s")), "static_s": sec(lg.get("staticDuration", "0s")),
                  "distance_m": float(lg.get("distanceMeters", 0))} for lg in rt.get("legs", [])],
        "request_s": time.perf_counter() - t0,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def decode_polyline(s: str) -> list[tuple[float, float]]:
    pts, i, lat, lon = [], 0, 0, 0
    while i < len(s):
        for which in (0, 1):
            shift = result = 0
            while True:
                b = ord(s[i]) - 63
                i += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if which == 0:
                lat += delta
            else:
                lon += delta
        pts.append((lat / 1e5, lon / 1e5))
    return pts
