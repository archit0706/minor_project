"""Fit the edge-weight parameters to Google Maps so our free-flow times match Google's.

1. Sample N origin -> hospital trips inside Ranchi (seeded) and query the Routes API once per
   trip. Raw responses are cached in data/google_samples.json (re-runs cost nothing).
2. Route each trip on our graph, then solve (bounded least squares)
       staticDuration ~= sum_g  pace_g * km_g  +  J * junctions  +  S * signals
   for pace (s/km) of each road group and junction delay J. staticDuration is Google's
   no-live-traffic time, so live traffic stays a separate factor T at query time.
3. Re-route with the new weights and refit (routes depend on weights), 4 rounds.
4. Report error on held-out trips, before vs after, then save data/calibration.json.

Usage:  python calibrate.py [--samples 40]
"""
from __future__ import annotations

import argparse
import json
import random

import networkx as nx
import numpy as np
from scipy.optimize import lsq_linear

import google_maps
from config import CALIBRATION, DEFAULT_JUNCTION_DELAY_S, DEFAULT_SPEEDS, GOOGLE_SAMPLES, SIGNAL_DELAY_S
from edge_weight import apply_weights
from road_network import NodeIndex, build_routing_graph, haversine_m, load_hospitals, load_osm_graph, path_length_m

GROUPS = ["major", "mid", "minor"]


def collect(D: nx.DiGraph, n: int) -> list[dict]:
    samples = json.loads(GOOGLE_SAMPLES.read_text()) if GOOGLE_SAMPLES.exists() else []
    if len(samples) >= n:
        return samples[:n]
    rng = random.Random(7 + len(samples))
    idx = NodeIndex(D)
    hosp = [idx.nearest(h["lat"], h["lon"]) for h in load_hospitals()]
    core = [v for v, d in D.nodes(data=True) if 23.32 <= d["y"] <= 23.44 and 85.27 <= d["x"] <= 85.40]
    while len(samples) < n:
        o, t = rng.choice(core), rng.choice(hosp)
        po, pt = (D.nodes[o]["y"], D.nodes[o]["x"]), (D.nodes[t]["y"], D.nodes[t]["x"])
        if not 1000 <= haversine_m(*po, *pt) <= 10000 or not nx.has_path(D, o, t):
            continue
        g = google_maps.route(po, pt)
        g.pop("points")
        samples.append({"o": o, "t": t, "origin": po, "dest": pt, "google": g})
        GOOGLE_SAMPLES.write_text(json.dumps(samples, indent=1))
        print(f"  sample {len(samples)}/{n}: {g['distance_m'] / 1000:.2f} km, static {g['static_s']:.0f} s")
    return samples


def features(D, samples):
    X, paths = [], []
    for s in samples:
        p = nx.dijkstra_path(D, s["o"], s["t"], weight="weight")
        km = {g: 0.0 for g in GROUPS}
        for u, v in zip(p, p[1:]):
            km[D[u][v]["group"]] += D[u][v]["length"] / 1000
        X.append([km[g] for g in GROUPS] + [sum(D[u][v]["junction"] for u, v in zip(p, p[1:])),
                                            sum(D[u][v]["signal"] for u, v in zip(p, p[1:]))])
        paths.append(p)
    return np.array(X), paths


def fit(X, y):
    A = X[:, :4]
    target = y - SIGNAL_DELAY_S * X[:, 4]
    lo = [3600 / 80] * 3 + [0.0]    # speeds <= 80 km/h, J >= 0
    hi = [3600 / 8] * 3 + [60.0]    # speeds >= 8 km/h, J <= 60 s
    sol = lsq_linear(A, target, bounds=(lo, hi)).x
    return {"speeds_kmh": {g: round(float(3600 / sol[i]), 2) for i, g in enumerate(GROUPS)},
            "junction_delay_s": round(float(sol[3]), 2)}


def predict(X, cal):
    pace = np.array([3600 / cal["speeds_kmh"][g] for g in GROUPS])
    return X[:, :3] @ pace + X[:, 3] * cal["junction_delay_s"] + X[:, 4] * SIGNAL_DELAY_S


def calibrate(D, samples, rounds=4):
    cal = {"speeds_kmh": dict(DEFAULT_SPEEDS), "junction_delay_s": DEFAULT_JUNCTION_DELAY_S}
    y = np.array([s["google"]["static_s"] for s in samples])
    for _ in range(rounds):
        apply_weights(D, cal=cal)
        X, paths = features(D, samples)
        cal = fit(X, y)
    apply_weights(D, cal=cal)
    return cal, *features(D, samples)


def errors(D, samples, cal):
    apply_weights(D, cal=cal)
    X, paths = features(D, samples)
    pred = predict(X, cal)
    g = np.array([s["google"]["static_s"] for s in samples])
    T = np.array([s["google"]["traffic_factor"] for s in samples])
    gd = np.array([s["google"]["distance_m"] for s in samples])
    ours_d = np.array([path_length_m(D, p) for p in paths])
    return {"mape_static_%": float(np.mean(np.abs(pred - g) / g) * 100),
            "mape_traffic_eta_%": float(np.mean(np.abs(pred * T - g * T) / (g * T)) * 100),
            "mean_signed_%": float(np.mean((pred - g) / g) * 100),
            "mae_s": float(np.mean(np.abs(pred - g))),
            "distance_mape_%": float(np.mean(np.abs(ours_d - gd) / gd) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    n = ap.parse_args().samples

    D = build_routing_graph(load_osm_graph())
    apply_weights(D, cal={"speeds_kmh": DEFAULT_SPEEDS, "junction_delay_s": DEFAULT_JUNCTION_DELAY_S})
    print(f"Collecting {n} Google Routes API samples (cached in {GOOGLE_SAMPLES.name}) ...")
    samples = collect(D, n)

    rng = random.Random(0)
    order = list(range(len(samples)))
    rng.shuffle(order)
    cut = int(0.7 * len(order))
    train = [samples[i] for i in order[:cut]]
    test = [samples[i] for i in order[cut:]]

    default = {"speeds_kmh": dict(DEFAULT_SPEEDS), "junction_delay_s": DEFAULT_JUNCTION_DELAY_S}
    cal_train, *_ = calibrate(D, train)
    before, after = errors(D, test, default), errors(D, test, cal_train)
    cal_all, *_ = calibrate(D, samples)

    report = {**cal_all, "calibrated": True, "n_samples": len(samples),
              "holdout": {"n_train": len(train), "n_test": len(test),
                          "before": before, "after": after},
              "fit_target": "Google Routes API staticDuration (car, no live traffic)"}
    CALIBRATION.write_text(json.dumps(report, indent=1))

    print(f"\nHeld-out trips ({len(test)}), our estimate vs Google:")
    print(f"  {'':22}{'default':>10}{'calibrated':>12}")
    for k in before:
        print(f"  {k:22}{before[k]:10.1f}{after[k]:12.1f}")
    print(f"\nFinal parameters (all {len(samples)} trips): speeds {cal_all['speeds_kmh']} km/h, "
          f"junction delay {cal_all['junction_delay_s']} s  ->  {CALIBRATION.name}")


if __name__ == "__main__":
    main()
