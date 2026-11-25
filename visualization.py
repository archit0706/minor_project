"""CLI outputs: route map (Folium), comparison table, charts (matplotlib)."""
from __future__ import annotations

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Colour follows the entity everywhere; this order passes the colour-blind checks (dataviz validator).
STYLE = {
    "aco": {"label": "ACO", "color": "#008300"},
    "google": {"label": "Google Maps", "color": "#2a78d6"},
    "dijkstra": {"label": "Dijkstra", "color": "#eb6834"},
    "astar": {"label": "A*", "color": "#1baf7a"},
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def route_map(res: dict) -> folium.Map:
    amb, pat, hosp = res["points"]["ambulance"], res["points"]["patient"], res["hospital"]
    m = folium.Map(location=pat, zoom_start=14, tiles="CartoDB Voyager")
    line = {"dijkstra": dict(weight=10, opacity=0.45),          # wide halo under ACO, so overlap stays visible
            "astar": dict(weight=4, opacity=0.95, dash_array="2 8"),
            "aco": dict(weight=5, opacity=0.95),
            "google": dict(weight=5, opacity=0.95, dash_array="10 8")}
    for key in line:
        r = res["routes"].get(key)
        if r:
            folium.PolyLine(r["coords"], color=STYLE[key]["color"], **line[key],
                            tooltip=f"{STYLE[key]['label']}: {r['distance_m'] / 1000:.2f} km, "
                                    f"{r['time_s'] / 60:.1f} min").add_to(m)
    folium.Marker(amb, tooltip="Ambulance", icon=folium.Icon(color="red", icon="ambulance", prefix="fa")).add_to(m)
    folium.Marker(pat, tooltip="Patient", icon=folium.Icon(color="purple", icon="user", prefix="fa")).add_to(m)
    folium.Marker((hosp["lat"], hosp["lon"]), tooltip=hosp["name"],
                  icon=folium.Icon(color="black", icon="plus", prefix="fa")).add_to(m)
    rows = "".join(
        f'<div><span style="display:inline-block;width:26px;border-top:4px '
        f'{"dashed" if k == "google" else "solid"} {STYLE[k]["color"]};margin-right:6px"></span>'
        f'{STYLE[k]["label"]} · {r["distance_m"] / 1000:.2f} km · {r["time_s"] / 60:.1f} min</div>'
        for k, r in res["routes"].items())
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:18px;left:18px;z-index:999;background:#fff;padding:10px 12px;'
        f'border-radius:8px;font:13px system-ui;color:{INK};box-shadow:0 1px 4px #0003">{rows}</div>'))
    m.fit_bounds([amb, pat, (hosp["lat"], hosp["lon"])], padding=(40, 40))
    return m


def comparison_table(res: dict) -> pd.DataFrame:
    rows = []
    for k in (k for k in STYLE if k in res["routes"]):
        r = res["routes"][k]
        rows.append({
            "Algorithm": STYLE[k]["label"],
            "Time (min)": round(r["time_s"] / 60, 1),
            "Distance (km)": round(r["distance_m"] / 1000, 2),
            "vs Google": f"{r['vs_google_pct']:+.1f}%" if "vs_google_pct" in r else "",
            "Compute (s)": round(r["compute_s"], 2),
            "vs optimum": "" if k == "google" else f"{r['gap_pct']:+.1f}%",
            "Nodes explored": r.get("expanded", ""),
        })
    return pd.DataFrame(rows)


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(res: dict) -> plt.Figure:
    keys = [k for k in STYLE if k in res["routes"]]
    names = [STYLE[k]["label"] for k in keys]
    colors = [STYLE[k]["color"] for k in keys]
    times = [res["routes"][k]["time_s"] / 60 for k in keys]
    dists = [res["routes"][k]["distance_m"] / 1000 for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    for ax, vals, title in ((axes[0], times, "Travel time (min)"), (axes[1], dists, "Route distance (km)")):
        bars = ax.bar(names, vals, color=colors, width=0.55)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=3, color=INK, fontsize=9)
        _style(ax, title)
        ax.margins(y=0.15)

    ax = axes[2]
    aco = res["routes"].get("aco")
    if aco:
        hist = [h / 60 for h in aco["history"]]
        ax.plot(range(1, len(hist) + 1), hist, color=STYLE["aco"]["color"], linewidth=2, label="ACO best so far")
        ax.set_xlabel("iteration", color=MUTED)
    ax.axhline(res["routes"]["dijkstra"]["time_s"] / 60, color=STYLE["dijkstra"]["color"], linewidth=1.5,
               linestyle="--", label="Dijkstra optimum")
    _style(ax, "ACO convergence (min)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    return fig