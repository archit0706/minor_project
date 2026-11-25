"""Classical graph search on the same directed, weighted road graph that ACO uses.

dijkstra  uniform-cost search; exact fastest route (the optimum every other result is measured against).
astar     Dijkstra + heuristic h(n) = straight-line distance to goal / fastest edge speed. h never
          overestimates, so A* returns the same optimal time while exploring fewer nodes.

All of them only follow out-edges, so one-way streets are respected. Each returns
{path, cost_s, compute_s, expanded}: cost_s is the route's travel time under the edge weights,
expanded is how many nodes the search took off its frontier.
"""
from __future__ import annotations

import heapq
import math
import time

import networkx as nx

from road_network import haversine_m


def _result(D, path, t0, expanded) -> dict:
    cost = sum(D[u][v]["weight"] for u, v in zip(path, path[1:])) if path else math.inf
    return {"path": path, "cost_s": cost, "compute_s": time.perf_counter() - t0, "expanded": expanded}


def _rebuild(parent: dict, t) -> list:
    path = [t]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def heuristic(D: nx.DiGraph, t):
    """Admissible lower bound on remaining time (s): straight-line metres / fastest edge speed (m/s)."""
    vmax = max(d["length"] / d["weight"] for _, _, d in D.edges(data=True) if d["weight"] > 0)
    yt, xt = D.nodes[t]["y"], D.nodes[t]["x"]
    return lambda n: haversine_m(D.nodes[n]["y"], D.nodes[n]["x"], yt, xt) / vmax


def _best_first(D: nx.DiGraph, s, t, key) -> dict:
    """Shared frontier search; key(g, n) orders the priority queue."""
    t0 = time.perf_counter()
    g, parent, closed, expanded = {s: 0.0}, {s: None}, set(), 0
    heap = [(key(0.0, s), s)]
    while heap:
        _, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == t:
            return _result(D, _rebuild(parent, t), t0, expanded)
        for v, d in D[u].items():
            ng = g[u] + d["weight"]
            if v not in closed and ng < g.get(v, math.inf):
                g[v], parent[v] = ng, u
                heapq.heappush(heap, (key(ng, v), v))
    return _result(D, None, t0, expanded)


def dijkstra(D, s, t):
    return _best_first(D, s, t, lambda g, n: g)


def astar(D, s, t):
    h = heuristic(D, t)
    return _best_first(D, s, t, lambda g, n: g + h(n))


ALGORITHMS = {"dijkstra": dijkstra, "astar": astar}