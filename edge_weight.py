"""Edge weight = estimated travel time in seconds for one directed road segment e = (u, v).

    w(e) = (L_e / v_e) * T * M_w  +  J * j_e  +  S * s_e

    L_e   segment length, metres (OSM)
    v_e   speed for the segment's road group (major / mid / minor), m/s, capped at the OSM
          maxspeed when tagged. Group speeds are FITTED to Google Maps staticDuration by
          calibrate.py (defaults in config.DEFAULT_SPEEDS until then).
    T     live traffic factor = Google traffic-aware duration / staticDuration for this trip.
          Offline: T = 1 (no live traffic data; NOT a guessed congestion value).
    M_w   weather factor for the rainfall class predicted by weather_model (config.WEATHER_MULTIPLIER).
    J     delay per junction (s), fitted by calibrate.py; j_e = 1 if v joins >= 3 streets.
    S     delay per traffic signal (config.SIGNAL_DELAY_S); s_e = 1 if v is a mapped signal.
"""
from __future__ import annotations

import json

import networkx as nx

from config import CALIBRATION, DEFAULT_JUNCTION_DELAY_S, DEFAULT_SPEEDS, SIGNAL_DELAY_S


def load_calibration() -> dict:
    if CALIBRATION.exists():
        return json.loads(CALIBRATION.read_text())
    return {"speeds_kmh": dict(DEFAULT_SPEEDS), "junction_delay_s": DEFAULT_JUNCTION_DELAY_S,
            "calibrated": False}


def edge_time(length_m, speed_kmh, traffic=1.0, weather=1.0, junction=0, signal=0,
              junction_delay=0.0, signal_delay=SIGNAL_DELAY_S) -> float:
    return length_m / (speed_kmh / 3.6) * traffic * weather + junction_delay * junction + signal_delay * signal


def edge_speed(d: dict, speeds: dict) -> float:
    v = speeds[d["group"]]
    return min(v, d["maxspeed"]) if d.get("maxspeed") else v


def apply_weights(D: nx.DiGraph, traffic=1.0, weather=1.0, cal: dict | None = None) -> nx.DiGraph:
    """Write w(e) into D[u][v]['weight'] (in place) and return D."""
    cal = cal or load_calibration()
    speeds, J = cal["speeds_kmh"], cal["junction_delay_s"]
    for _, _, d in D.edges(data=True):
        d["weight"] = edge_time(d["length"], edge_speed(d, speeds), traffic, weather,
                                d["junction"], d["signal"], J)
    return D


def route_breakdown(D: nx.DiGraph, path, traffic=1.0, weather=1.0, cal: dict | None = None) -> dict:
    """Split a route's time into the formula's terms (for the report)."""
    cal = cal or load_calibration()
    free = sum(d["length"] / (edge_speed(d, cal["speeds_kmh"]) / 3.6)
               for d in (D[u][v] for u, v in zip(path, path[1:])))
    j = sum(D[u][v]["junction"] for u, v in zip(path, path[1:]))
    s = sum(D[u][v]["signal"] for u, v in zip(path, path[1:]))
    return {"free_flow_s": free, "traffic_extra_s": free * (traffic - 1) * weather,
            "weather_extra_s": free * (weather - 1), "junctions": j,
            "junction_s": j * cal["junction_delay_s"], "signals": s, "signal_s": s * SIGNAL_DELAY_S}
