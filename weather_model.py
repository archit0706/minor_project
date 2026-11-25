"""Monthly rainfall-class model for Ranchi, trained on the provided satellite climate CSV.

Dataset facts (satellite_climate_district_FIXED.csv, District == "Ranchi"):
  168 rows = 14 years (2010-2023) x 12 months. Columns used: Month, LST_C, NDVI, Rainfall_mm.
  Values are MONTHLY aggregates, so the model predicts the expected rainfall class of a
  month, not the weather at a specific hour. It says nothing about traffic or travel time.

Target: Rainfall_mm binned into clear (<10) / light_rain (10-50) / moderate_rain (50-200)
        / heavy_rain (200-400) / extreme (>=400) mm per month.
Features: cyclic month (sin, cos), LST_C, NDVI. Year is dropped: trees can't extrapolate it.
Evaluation: train 2010-2020, test 2021-2023, compared with a "same month's most common
class" baseline, so the report shows whether LST/NDVI add anything beyond the calendar.
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import CLIMATE_CSV, DATA, WEATHER_MODEL, WEATHER_MULTIPLIER

LABELS = list(WEATHER_MULTIPLIER)
BINS = [-np.inf, 10, 50, 200, 400, np.inf]
FEATURES = ["month_sin", "month_cos", "LST_C", "NDVI"]


def load_ranchi() -> pd.DataFrame:
    src = CLIMATE_CSV if CLIMATE_CSV.exists() else DATA / "ranchi_weather.csv"
    df = pd.read_csv(src, usecols=["District", "Year", "Month", "LST_C", "NDVI", "Rainfall_mm"])
    df = df[df.District == "Ranchi"].dropna().copy()
    df["severity"] = pd.cut(df.Rainfall_mm, BINS, labels=LABELS, right=False).astype(str)
    df["month_sin"] = np.sin(2 * np.pi * df.Month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.Month / 12)
    return df


def train() -> dict:
    df = load_ranchi()
    tr, te = df[df.Year <= 2020], df[df.Year >= 2021]

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0)
    rf.fit(tr[FEATURES], tr.severity)
    pred = rf.predict(te[FEATURES])

    month_mode = tr.groupby("Month").severity.agg(lambda s: s.mode()[0])
    base = te.Month.map(month_mode)

    metrics = {
        "train_years": "2010-2020", "test_years": "2021-2023",
        "n_train": len(tr), "n_test": len(te),
        "rf_accuracy": accuracy_score(te.severity, pred),
        "rf_macro_f1": f1_score(te.severity, pred, average="macro"),
        "baseline_accuracy": accuracy_score(te.severity, base),
        "baseline_macro_f1": f1_score(te.severity, base, average="macro"),
        "class_counts": df.severity.value_counts().to_dict(),
    }
    rf.fit(df[FEATURES], df.severity)  # final model uses all years
    monthly = df.groupby("Month")[["LST_C", "NDVI"]].mean()
    WEATHER_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": rf, "monthly_means": monthly, "metrics": metrics}, WEATHER_MODEL)
    return metrics


def _bundle() -> dict:
    if not WEATHER_MODEL.exists():
        train()
    return joblib.load(WEATHER_MODEL)


def predict(month: int, lst_c: float | None = None, ndvi: float | None = None) -> dict:
    """Rainfall class for a month. Missing LST/NDVI fall back to that month's 2010-2023 mean."""
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    b = _bundle()
    mean = b["monthly_means"].loc[int(month)]
    lst = mean.LST_C if lst_c is None else lst_c
    nd = mean.NDVI if ndvi is None else ndvi
    x = pd.DataFrame([[np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12), lst, nd]],
                     columns=FEATURES)
    model = b["model"]
    proba = {str(c): round(float(p), 3) for c, p in zip(model.classes_, model.predict_proba(x)[0])}
    sev = str(model.predict(x)[0])
    return {"severity": sev, "multiplier": WEATHER_MULTIPLIER[sev], "month": int(month),
            "lst_c": round(float(lst), 2), "ndvi": round(float(nd), 3),
            "inputs": "monthly climatology" if lst_c is None or ndvi is None else "user supplied",
            "proba": proba}


if __name__ == "__main__":
    m = train()
    print(f"Ranchi rainfall-class model  (train {m['train_years']}: {m['n_train']} rows, "
          f"test {m['test_years']}: {m['n_test']} rows)")
    print(f"  Random forest      accuracy {m['rf_accuracy']:.3f}   macro-F1 {m['rf_macro_f1']:.3f}")
    print(f"  Month-only baseline accuracy {m['baseline_accuracy']:.3f}   macro-F1 {m['baseline_macro_f1']:.3f}")
    print(f"  Class counts: {m['class_counts']}")
    for mo in (1, 4, 7, 10):
        print(" ", predict(mo), file=sys.stdout)
