"""Settings for the Ranchi ambulance router.

Anything tagged ASSUMPTION is a modelling choice, not measured data.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUTPUT = BASE / "output"
MODELS = BASE / "models"

CLIMATE_CSV = BASE / "satellite_climate_district_FIXED.csv"   # user-provided weather dataset
HOSPITALS_CSV = DATA / "ranchi_hospitals.csv"                  # verified via Google Places
NETWORK_FILE = DATA / "ranchi_network.graphml"                 # OSM drive network (OSMnx)
WEATHER_MODEL = MODELS / "weather_model.pkl"
CALIBRATION = DATA / "calibration.json"                        # written by calibrate.py
GOOGLE_SAMPLES = DATA / "google_samples.json"                  # raw Routes API responses used for calibration

BBOX = (85.24, 23.30, 85.45, 23.46)   # west, south, east, north (covers city + Irba)

# ASSUMPTION: starting free-flow speeds (km/h) per road group, used until calibrate.py
# fits them to Google Maps staticDuration.
DEFAULT_SPEEDS = {"major": 40.0, "mid": 30.0, "minor": 20.0}
DEFAULT_JUNCTION_DELAY_S = 0.0

# ASSUMPTION: delay per signalised junction. Ranchi OSM data has ~1 mapped signal,
# so in practice the calibrated junction delay carries intersection cost.
SIGNAL_DELAY_S = 30.0

# ASSUMPTION: travel-time multiplier per predicted rainfall class. No local study
# links monthly rainfall to road speed, so these are deliberately mild.
WEATHER_MULTIPLIER = {
    "clear": 1.00,
    "light_rain": 1.05,
    "moderate_rain": 1.10,
    "heavy_rain": 1.20,
    "extreme": 1.30,
}

# Tuned on 8 trips against the exact optimum (see aco.py docstring).
ACO = dict(n_ants=30, n_iterations=100, alpha=1.0, beta=1.0, rho=0.2,
           tau0=1.0, tau_min=0.01, tau_max=10.0, patience=25, seed=42)
