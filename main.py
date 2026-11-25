"""Ambulance -> patient -> hospital routing for Ranchi: ACO, Dijkstra, A* vs Google Maps.

    python main.py --patient 23.3600,85.3200
    python main.py --hospital "RIMS" --month 7 --offline
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from functools import lru_cache

import networkx as nx

import google_maps
import weather_model
from aco import run_aco
from config import ACO, BBOX, OUTPUT
from edge_weight import apply_weights, load_calibration, route_breakdown
from search import ALGORITHMS as SEARCH
from road_network import NodeIndex, build_routing_graph, load_hospitals, load_osm_graph, path_coords, path_length_m


@lru_cache(maxsize=1)
def network():
    D = build_routing_graph(load_osm_graph())
    return D, NodeIndex(D)


@lru_cache(maxsize=1)
def fixed_ambulances(rows: int = 5, cols: int = 10) -> list[dict]:
    """50 fixed ambulances evenly distributed across Ranchi and snapped to drivable road nodes."""
    D, idx = network()
    w, s, e, n = BBOX
    pad_lat, pad_lon = (n - s) * 0.08, (e - w) * 0.08
    lats = [s + pad_lat + i * (n - s - 2 * pad_lat) / (rows - 1) for i in range(rows)]
    lons = [w + pad_lon + j * (e - w - 2 * pad_lon) / (cols - 1) for j in range(cols)]
    ambulances = []
    for k, (lat, lon) in enumerate(((la, lo) for la in lats for lo in lons), start=1):
        node = idx.nearest(lat, lon)
        ambulances.append({
            "id": f"AMB-{k:02d}",
            "node": node,
            "lat": D.nodes[node]["y"],
            "lon": D.nodes[node]["x"],
        })
    return ambulances


def in_ranchi(lat, lon) -> bool:
    w, s, e, n = BBOX
    return s <= lat <= n and w <= lon <= e


def _join_histories(hs):
    n = max(len(h) for h in hs)
    return [sum(h[min(i, len(h) - 1)] for h in hs) for i in range(n)]


def _graph_route(D, legs_paths, legs_costs, compute_s) -> dict:
    path = legs_paths[0] + legs_paths[1][1:]
    return {"coords": path_coords(D, path), "path": path,
            "time_s": sum(legs_costs), "distance_m": path_length_m(D, path), "compute_s": compute_s,
            "legs": [{"time_s": c, "distance_m": path_length_m(D, p)} for p, c in zip(legs_paths, legs_costs)]}


GRAPH_ALGOS = ("aco", "dijkstra", "astar")


def plan_trip(patient, hospital: str | None = None, month: int | None = None,
              online: bool = True, aco_params: dict | None = None, algorithms=GRAPH_ALGOS) -> dict:
    """Auto-dispatch the fastest of 50 fixed ambulances -> patient -> hospital."""
    if not in_ranchi(*patient):
        raise ValueError("The patient is outside the Ranchi map area")
    D, idx = network()
    cal = load_calibration()
    weather = weather_model.predict(month or datetime.now().month)
    W = weather["multiplier"]
    hospitals = load_hospitals()
    ambulances = fixed_ambulances()

    npt = idx.nearest(*patient)
    pt = lambda n: (D.nodes[n]["y"], D.nodes[n]["x"])

    # Auto-pick the fastest-by-road ambulance (reverse graph from patient) and hospital
    apply_weights(D, 1.0, W, cal)
    to_patient = nx.single_source_dijkstra_path_length(D.reverse(copy=False), npt, weight="weight")
    amb = min(ambulances, key=lambda a: to_patient.get(a["node"], math.inf))
    if to_patient.get(amb["node"], math.inf) == math.inf:
        raise RuntimeError("No fixed ambulance can legally reach this patient location")
    na = amb["node"]

    if hospital:
        matches = ([h for h in hospitals if h["name"] == hospital]
                   or [h for h in hospitals if hospital.lower() in h["name"].lower()])
        if not matches:
            raise ValueError(f"no hospital matching {hospital!r}")
        hosp = matches[0]
    else:
        reach = nx.single_source_dijkstra_path_length(D, npt, weight="weight")
        hosp = min((h for h in hospitals if not h["specialty"]),
                   key=lambda h: reach.get(idx.nearest(h["lat"], h["lon"]), math.inf))
    nh = idx.nearest(hosp["lat"], hosp["lon"])

    # Google first: its live traffic factor T feeds our edge weights
    google, google_error, T = None, None, 1.0
    if online:
        try:
            google = google_maps.route(pt(na), pt(nh), via=pt(npt))
            T = google["traffic_factor"]
        except google_maps.GoogleMapsError as e:
            google_error = str(e)
    apply_weights(D, T, W, cal)

    legs = [(na, npt), (npt, nh)]
    routes = {}

    # Dijkstra always runs: it is the exact optimum every other route is measured against
    wanted = ["dijkstra"] + [a for a in SEARCH if a != "dijkstra" and a in algorithms]
    for name in wanted:
        res = [SEARCH[name](D, s, t) for s, t in legs]
        if any(r["path"] is None for r in res):
            raise RuntimeError("No drivable route between these points (check one-way streets / location)")
        r = _graph_route(D, [x["path"] for x in res], [x["cost_s"] for x in res], sum(x["compute_s"] for x in res))
        r["expanded"] = sum(x["expanded"] for x in res)
        routes[name] = r

    if "aco" in algorithms:
        params = {**ACO, **(aco_params or {})}
        aco_legs = [run_aco(D, s, t, **params) for s, t in legs]
        if any(r["path"] is None for r in aco_legs):
            raise RuntimeError("ACO found no route; try more ants or iterations")
        aco = _graph_route(D, [r["path"] for r in aco_legs], [r["cost_s"] for r in aco_legs],
                           sum(r["runtime_s"] for r in aco_legs))
        aco.update(history=_join_histories([r["history"] for r in aco_legs]),
                   iterations=sum(r["iterations"] for r in aco_legs), ants=params["n_ants"],
                   expanded=sum(r["search_nodes"] for r in aco_legs))
        routes["aco"] = aco

    opt = routes["dijkstra"]["time_s"]
    for r in routes.values():
        r["gap_pct"] = max(0.0, (r["time_s"] - opt) / opt * 100) if opt else 0.0
        r["breakdown"] = route_breakdown(D, r["path"], T, W, cal)

    if google:
        for r in routes.values():
            r["vs_google_pct"] = (r["time_s"] - google["duration_s"]) / google["duration_s"] * 100
        routes["google"] = {"coords": google["points"], "time_s": google["duration_s"],
                            "static_s": google["static_s"], "distance_m": google["distance_m"],
                            "compute_s": google["request_s"], "legs": google["legs"],
                            "traffic_factor": T, "speed_intervals": google["speed_intervals"]}

    for r in routes.values():
        r.pop("path", None)
    return {
        "ambulance": {"id": amb["id"], "lat": amb["lat"], "lon": amb["lon"]},
        "hospital": hosp, "all_hospitals": hospitals, "weather": weather,
        "traffic_factor": T, "calibrated": cal.get("calibrated", False),
        "points": {"ambulance": pt(na), "patient": pt(npt), "hospital": pt(nh)},
        "google_error": google_error, "routes": routes,
    }


def _latlon(s: str):
    lat, lon = (float(x) for x in s.split(","))
    return lat, lon


def main():
    import visualization as viz
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", type=_latlon, default=(23.3600, 85.3200), help="lat,lon")
    ap.add_argument("--hospital", help="name substring (default: fastest general hospital)")
    ap.add_argument("--month", type=int, choices=range(1, 13), metavar="1-12")
    ap.add_argument("--offline", action="store_true", help="skip Google Maps")
    ap.add_argument("--algorithms", default="aco,dijkstra,astar",
                    help="comma list from: " + ",".join(GRAPH_ALGOS))
    a = ap.parse_args()

    res = plan_trip(a.patient, a.hospital, a.month, online=not a.offline,
                    algorithms=a.algorithms.split(","))
    if res["google_error"]:
        print(f"[offline] Google Maps unavailable: {res['google_error']}")
    if not res["calibrated"]:
        print("[note] using default speeds - run `python calibrate.py` to fit them to Google Maps")
    print(f"Dispatched: {res['ambulance']['id']}  ->  Hospital: {res['hospital']['name']}\n")
    table = viz.comparison_table(res)
    print(table.to_string(index=False))

    OUTPUT.mkdir(exist_ok=True)
    viz.route_map(res).save(OUTPUT / "route_map.html")
    viz.charts(res).savefig(OUTPUT / "comparison.png", dpi=150)
    table.to_csv(OUTPUT / "comparison.csv", index=False)
    print(f"\nSaved route_map.html, comparison.png, comparison.csv in {OUTPUT}")


if __name__ == "__main__":
    main()