"""Ranchi road network (OpenStreetMap via OSMnx) and hospital data."""
from __future__ import annotations

import math
import re

import networkx as nx
import numpy as np
import pandas as pd

from config import BBOX, HOSPITALS_CSV, NETWORK_FILE


def load_osm_graph() -> nx.MultiDiGraph:
    """Directed OSM 'drive' network. One-way streets only have their legal direction."""
    import osmnx as ox
    if NETWORK_FILE.exists():
        return ox.load_graphml(NETWORK_FILE)
    G = ox.graph_from_bbox(BBOX, network_type="drive")
    NETWORK_FILE.parent.mkdir(exist_ok=True)
    ox.save_graphml(G, NETWORK_FILE)
    return G


def road_group(highway) -> str:
    hw = (highway[0] if isinstance(highway, list) else str(highway)).strip("[]'\" ").split("'")[0]
    hw = hw.removesuffix("_link")
    if hw in ("motorway", "trunk", "primary"):
        return "major"
    if hw in ("secondary", "tertiary"):
        return "mid"
    return "minor"


def _maxspeed(raw) -> float | None:
    m = re.search(r"\d+(\.\d+)?", str(raw)) if raw else None
    return float(m.group()) if m else None


def build_routing_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse parallel edges (keep shortest) and attach the static per-edge features
    the weight function needs: length, road group, OSM maxspeed, junction/signal at head node."""
    D = nx.DiGraph()
    for n, d in G.nodes(data=True):
        D.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        length = float(d.get("length", 0.0))
        if D.has_edge(u, v) and D[u][v]["length"] <= length:
            continue
        head = G.nodes[v]
        D.add_edge(u, v,
                   length=length,
                   group=road_group(d.get("highway", "")),
                   maxspeed=_maxspeed(d.get("maxspeed")),
                   junction=int(int(head.get("street_count", 0)) >= 3),
                   signal=int(str(head.get("highway", "")) == "traffic_signals"),
                   name=d.get("name", ""))
    return D


class NodeIndex:
    """Nearest-node lookup (equirectangular distance is exact enough at city scale)."""

    def __init__(self, D: nx.DiGraph):
        self.ids = np.array(list(D.nodes))
        self.lat = np.array([D.nodes[n]["y"] for n in self.ids])
        self.lon = np.array([D.nodes[n]["x"] for n in self.ids])

    def nearest(self, lat: float, lon: float):
        dx = (self.lon - lon) * math.cos(math.radians(lat))
        return self.ids[int(np.argmin(dx * dx + (self.lat - lat) ** 2))].item()


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 12_742_000 * math.asin(math.sqrt(a))


def path_length_m(D: nx.DiGraph, path) -> float:
    return sum(D[u][v]["length"] for u, v in zip(path, path[1:]))


def path_coords(D: nx.DiGraph, path) -> list[tuple[float, float]]:
    return [(D.nodes[n]["y"], D.nodes[n]["x"]) for n in path]


def load_hospitals() -> list[dict]:
    df = pd.read_csv(HOSPITALS_CSV, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"hospital name": "name", "latitude": "lat", "longitude": "lon"})
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    if "specialty" not in df.columns:
        df["specialty"] = ""
    return df.to_dict("records")