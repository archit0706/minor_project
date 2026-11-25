"""Smoke checks for the core logic.   python test_core.py"""
import networkx as nx

from aco import run_aco
from search import ALGORITHMS, dijkstra
from edge_weight import edge_time
from google_maps import decode_polyline
from road_network import road_group


def test_edge_time():
    # 1 km at 36 km/h = 100 s; x traffic 1.5 x weather 1.2 = 180 s; + 7 s junction + 30 s signal
    assert abs(edge_time(1000, 36) - 100) < 1e-9
    assert abs(edge_time(1000, 36, 1.5, 1.2, junction=1, signal=1, junction_delay=7, signal_delay=30) - 217) < 1e-9


def test_polyline():
    # Example from Google's polyline algorithm documentation
    pts = decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    assert [tuple(round(c, 3) for c in p) for p in pts] == [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]


def test_road_group():
    assert road_group("primary_link") == "major"
    assert road_group("['tertiary', 'residential']") == "mid"
    assert road_group("residential") == "minor"


def test_aco_respects_one_way_and_finds_optimum():
    # s -> a -> t is cheapest on paper, but a -> t is one-way the other way (only t -> a exists).
    G = nx.DiGraph()
    xy = {"s": (0, 0), "a": (0.001, 0), "b": (0.001, 0.001), "c": (0.002, 0.0005), "t": (0.002, 0)}
    for n, (x, y) in xy.items():
        G.add_node(n, x=85.3 + x, y=23.35 + y)
    for u, v, w in [("s", "a", 1), ("t", "a", 1), ("s", "b", 5), ("b", "t", 5), ("a", "c", 3), ("c", "t", 3)]:
        G.add_edge(u, v, weight=w, length=w * 10)
    best = dijkstra(G, "s", "t")
    res = run_aco(G, "s", "t", n_ants=10, n_iterations=20, seed=1)
    assert best["path"] == ["s", "a", "c", "t"] and best["cost_s"] == 7
    assert res["path"] == best["path"] and res["cost_s"] == best["cost_s"]
    assert run_aco(G, "t", "s")["path"] is None   # no legal way back
    # optimal searches agree with Dijkstra; every search respects the one-way edge
    assert ALGORITHMS["astar"](G, "s", "t")["cost_s"] == 7
    for name, f in ALGORITHMS.items():
        p = f(G, "s", "t")["path"]
        assert all(G.has_edge(u, v) for u, v in zip(p, p[1:])), name
        assert f(G, "t", "s")["path"] is None, name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
