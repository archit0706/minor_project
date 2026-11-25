"""Web app.   python server.py   ->  http://localhost:8000

Serves web/index.html and a small JSON API (stdlib only):
  GET  /api/init   hospitals, 50 fixed ambulances, map bounds, Google key status, ACO defaults
  POST /api/route  {patient:[lat,lon], hospital, month, algorithms:[...], aco:{...}}
"""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import google_maps
from config import ACO, BBOX
from main import fixed_ambulances, network, plan_trip
from road_network import load_hospitals

WEB = Path(__file__).resolve().parent / "web"
LOCK = threading.Lock()
ALGOS = {"aco", "dijkstra", "astar", "google"}
ACO_LIMITS = {"n_ants": (1, 200, int), "n_iterations": (1, 500, int), "alpha": (0, 5, float),
              "beta": (0, 10, float), "rho": (0.01, 0.99, float)}


def _point(v, name):
    try:
        lat, lon = (float(x) for x in v)
        return lat, lon
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be [lat, lon]")


def route(body: dict) -> dict:
    algos = set(body.get("algorithms") or []) & ALGOS
    if not algos:
        raise ValueError("Select at least one algorithm")
    month = int(body.get("month") or datetime.now().month)
    if not 1 <= month <= 12:
        raise ValueError("month must be 1-12")
    aco = {}
    for k, (lo, hi, cast) in ACO_LIMITS.items():
        if k in (body.get("aco") or {}):
            aco[k] = min(hi, max(lo, cast(body["aco"][k])))
    with LOCK:
        res = plan_trip(_point(body.get("patient"), "patient"),
                        body.get("hospital") or None, month, online="google" in algos,
                        aco_params=aco, algorithms=algos)
    res["routes"] = {k: v for k, v in res["routes"].items() if k in algos}
    res.pop("all_hospitals")
    return res


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/init":
            with LOCK:
                ambs = [{"id": a["id"], "lat": a["lat"], "lon": a["lon"]} for a in fixed_ambulances()]
            self._send(200, {
                "hospitals": [{k: h[k] for k in ("name", "lat", "lon", "specialty")} for h in load_hospitals()],
                "ambulances": ambs,
                "bbox": BBOX, "google": google_maps.api_key() is not None,
                "month": datetime.now().month,
                "aco": {k: ACO[k] for k in ACO_LIMITS},
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/route":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._send(200, route(body))
        except (ValueError, RuntimeError, KeyError) as e:
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    def warm():
        with LOCK:
            network()
            fixed_ambulances()
    threading.Thread(target=warm, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Ranchi Ambulance Router running at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    server.serve_forever()