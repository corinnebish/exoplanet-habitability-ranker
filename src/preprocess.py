"""
Clean the raw NASA data and build model-ready features.

Output: data/processed/planets_clean.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "exoplanets.csv"
OUT = ROOT / "data" / "processed" / "planets_clean.csv"

PARSEC_TO_LIGHTYEAR = 3.26156

# Features that span several orders of magnitude (e.g. radius 0.3 to 30+ Earths).
# Log-transforming keeps a few giant planets from dominating distance-based
# methods like K-Means.
LOG_FEATURES = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_insol"]


def fill_mass_from_radius(radius: pd.Series) -> pd.Series:
    """
    Estimate mass from radius when mass is missing.

    Uses the piecewise mass-radius relation from Chen & Kipping (2017),
    inverted to give mass in Earth masses:
      rocky   (R < 1.23):         R = M^0.279
      Neptune-like (1.23–14.3):   R = 0.808 * M^0.589
      giants  (R >= 14.3):        mass is nearly flat with radius, so use ~1 Jupiter mass
    """
    r = radius
    return pd.Series(
        np.select(
            [r < 1.23, r < 14.3],
            [r ** (1 / 0.279), (r / 0.808) ** (1 / 0.589)],
            default=317.8,  # 1 Jupiter mass in Earth masses
        ),
        index=r.index,
    )


def main() -> None:
    df = pd.read_csv(RAW)
    print(f"Start:                                   {len(df):>6,} planets")

    # These three describe whether a planet could be Earth-like (size and the
    # energy it gets from its star), so we can't score a planet without them.
    df = df.dropna(subset=["pl_rade", "pl_insol", "pl_eqt"])
    print(f"After dropping missing radius/insol/temp: {len(df):>6,}")

    df = df.dropna(subset=["pl_orbper", "st_teff", "sy_dist"])
    print(f"After dropping missing period/star/dist:  {len(df):>6,}")

    # Flag before filling so the model and the dashboard can show which
    # masses are measured vs. estimated.
    df["mass_imputed"] = df["pl_bmasse"].isna()
    df.loc[df["mass_imputed"], "pl_bmasse"] = fill_mass_from_radius(df.loc[df["mass_imputed"], "pl_rade"])
    print(f"Masses estimated from radius:             {df['mass_imputed'].sum():>6,}")

    # log10 is undefined at 0; a handful of rows have 0 as a placeholder value.
    df = df[(df[LOG_FEATURES] > 0).all(axis=1)]
    print(f"After dropping zero/negative values:      {len(df):>6,}")

    for col in LOG_FEATURES:
        df[f"log_{col}"] = np.log10(df[col])

    df["dist_ly"] = df["sy_dist"] * PARSEC_TO_LIGHTYEAR

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df):,} planets to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
