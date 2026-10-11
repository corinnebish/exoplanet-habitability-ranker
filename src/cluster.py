"""
Group planets into natural types with K-Means.

Input:  data/processed/planets_clean.csv
Output: data/processed/planets_clustered.csv  (adds `cluster` and `cluster_name`)
        models/scaler.joblib, models/kmeans.joblib
        reports/choosing_k.png
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "planets_clean.csv"
OUT_PATH = ROOT / "data" / "processed" / "planets_clustered.csv"

# Size, mass, year length, starlight received, temperature, and host-star
# temperature: together these describe what kind of world a planet is.
FEATURES = ["log_pl_rade", "log_pl_bmasse", "log_pl_orbper", "log_pl_insol", "pl_eqt", "st_teff"]

# Earth's values, transformed the same way as the planet data.
EARTH = pd.DataFrame([[0.0, 0.0, np.log10(365.25), 0.0, 255.0, 5772.0]], columns=FEATURES)

# K = 4 chosen by comparing K = 2..10 (see reports/choosing_k.png):
#  - K = 2–3 score best on silhouette, but lump every small planet into one
#    group, hiding the rocky-vs-warm split we care about.
#  - K >= 5 splits off a 6-planet group orbiting unusually hot stars
#    (>10,000 K): an outlier pile, not a planet type.
#  - K = 4 sits at the elbow and gives four groups that match types
#    astronomers already recognize.
K = 4
RANDOM_STATE = 42


def name_clusters(df: pd.DataFrame) -> dict:
    """Name each cluster from its median size and temperature."""
    prof = df.groupby("cluster")[["pl_rade", "pl_eqt"]].median()
    names = {}
    for c, row in prof.iterrows():
        big = row["pl_rade"] > 6  # bigger than ~Neptune
        hot = row["pl_eqt"] > 900
        names[c] = {
            (True, True): "Hot Giants",
            (True, False): "Cool Giants",
            (False, True): "Scorched Small Planets",
            (False, False): "Warm Small Planets",
        }[(big, hot)]
    return names


def plot_choosing_k(X: np.ndarray, out: Path) -> None:
    ks = range(2, 11)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_, sample_size=3000, random_state=RANDOM_STATE))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ks, inertia, "o-"); a1.set_title("Elbow method"); a1.set_xlabel("K"); a1.set_ylabel("Inertia")
    a2.plot(ks, sil, "o-"); a2.set_title("Silhouette score"); a2.set_xlabel("K")
    for a in (a1, a2):
        a.axvline(K, color="red", ls="--", lw=1)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)


def main() -> None:
    df = pd.read_csv(IN_PATH)

    # Standardize so every feature counts equally; otherwise temperature
    # (thousands) would outweigh log-radius (around 0–1.5).
    scaler = StandardScaler().fit(df[FEATURES])
    X = scaler.transform(df[FEATURES])

    plot_choosing_k(X, ROOT / "reports" / "choosing_k.png")

    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(X)
    df["cluster"] = km.labels_
    names = name_clusters(df)
    df["cluster_name"] = df["cluster"].map(names)

    earth_cluster = int(km.predict(scaler.transform(EARTH))[0])

    print("Planet groups (median values):\n")
    profile = df.groupby("cluster_name").agg(
        planets=("pl_name", "size"),
        radius_earths=("pl_rade", "median"),
        mass_earths=("pl_bmasse", "median"),
        year_days=("pl_orbper", "median"),
        starlight_vs_earth=("pl_insol", "median"),
        temp_K=("pl_eqt", "median"),
    ).round(1)
    print(profile.to_string())
    print(f"\nEarth would belong to: {names[earth_cluster]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(scaler, ROOT / "models" / "scaler.joblib")
    joblib.dump(km, ROOT / "models" / "kmeans.joblib")


if __name__ == "__main__":
    main()
