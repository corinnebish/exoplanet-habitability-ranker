"""
Download confirmed exoplanets from the NASA Exoplanet Archive.

Why the `pscomppars` table: it has exactly one row per confirmed planet, with
the archive's best value for each parameter. The other main table (`ps`) has
one row per published measurement, so planets would be counted many times.
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns chosen for habitability analysis:
# planet size/mass, orbit, energy received from the star, host star, distance.
COLUMNS = [
    "pl_name",          # planet name
    "hostname",         # host star name
    "pl_rade",          # radius (Earth radii)
    "pl_bmasse",        # mass (Earth masses)
    "pl_orbper",        # orbital period (days)
    "pl_orbsmax",       # orbit size / semi-major axis (AU)
    "pl_orbeccen",      # orbit eccentricity (0 = circle)
    "pl_eqt",           # equilibrium temperature (K)
    "pl_insol",         # insolation: starlight received (Earth = 1)
    "st_teff",          # star temperature (K)
    "st_rad",           # star radius (Sun radii)
    "st_mass",          # star mass (Sun masses)
    "sy_dist",          # distance from Earth (parsecs)
    "ra",               # sky position: right ascension (degrees), used for the 3D map
    "dec",              # sky position: declination (degrees)
    "discoverymethod",  # e.g. Transit, Radial Velocity
    "disc_year",        # year discovered
]

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "exoplanets.csv"


def fetch() -> pd.DataFrame:
    query = f"select {', '.join(COLUMNS)} from pscomppars"
    resp = requests.get(TAP_URL, params={"query": query, "format": "csv"}, timeout=120)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def main() -> None:
    df = fetch()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df):,} planets to {OUT_PATH.relative_to(OUT_PATH.parents[2])}\n")
    print("Missing values by column (%):")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
