"""
Score every planet's Earth-likeness (0–100) with logistic regression.

Labels come from an outside expert source, the Planetary Habitability
Laboratory's Habitable Worlds Catalog (University of Puerto Rico at Arecibo,
https://phl.upr.edu/hwc). Using outside labels matters: if we wrote our own
"habitable" rule from these same features, the model would just learn our rule
back.

Input:  data/processed/planets_clustered.csv, data/raw/phl_habitable_worlds.csv
Output: data/processed/planets_scored.csv, models/logreg.joblib
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PLANETS = ROOT / "data" / "processed" / "planets_clustered.csv"
LABELS = ROOT / "data" / "raw" / "phl_habitable_worlds.csv"
OUT = ROOT / "data" / "processed" / "planets_scored.csv"
RANDOM_STATE = 42


def normalize(name: str) -> str:
    """'TRAPPIST-1 e' and 'Trappist-1e' should match."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic regression is linear: it can only learn "more X = more likely".
    Habitability is a sweet spot (too much starlight is bad, too little is
    bad), so we add squared distance from Earth's value. Earth = 0 on every
    log feature, so the square of the log is that distance.
    Linear terms stay in so the model can learn that, e.g., slightly cooler
    than Earth is better than equally hotter.
    """
    return pd.DataFrame({
        "log_radius": df["log_pl_rade"],
        "log_starlight": df["log_pl_insol"],
        "radius_gap_sq": df["log_pl_rade"] ** 2,
        "starlight_gap_sq": df["log_pl_insol"] ** 2,
        "mass_gap_sq": df["log_pl_bmasse"] ** 2,
        "star_temp": df["st_teff"],
    }, index=df.index)


def main() -> None:
    df = pd.read_csv(PLANETS)
    phl = pd.read_csv(LABELS)

    habitable = set(phl["Name"].map(normalize))
    df["is_habitable"] = df["pl_name"].map(normalize).isin(habitable).astype(int)
    unmatched = sorted(set(phl["Name"]) - set(df.loc[df["is_habitable"] == 1, "pl_name"]))
    print(f"Habitable planets in catalog: {len(phl)}, matched: {df['is_habitable'].sum()}")
    print(f"Not matched (removed in cleaning, missing NASA data): {unmatched}\n")

    X, y = build_features(df), df["is_habitable"]

    # class_weight="balanced": only ~1% of planets are habitable, so without it
    # the model can score 99% accuracy by saying "no" to everything.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )

    # 1) Honest check on planets the model never saw.
    # Stratified split keeps the same ~1% habitable share in train and test.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    print("=== Test set (25% held out) ===")
    print(classification_report(y_te, (p_te >= 0.5).astype(int), target_names=["not habitable", "habitable"], digits=3))
    tn, fp, fn, tp = confusion_matrix(y_te, (p_te >= 0.5).astype(int)).ravel()
    print(f"Confusion matrix: found {tp} of {tp + fn} habitable planets, {fp} false alarms, {tn} correct rejections")
    # Precision-recall AUC (average precision) is the right headline metric for
    # rare positives. A random guess would score about the positive rate (~0.01).
    print(f"PR-AUC: {average_precision_score(y_te, p_te):.3f}  (random guessing ≈ {y_te.mean():.3f})\n")

    # 2) 5-fold cross-validation: same check repeated on 5 different splits.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    p_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"=== 5-fold cross-validation PR-AUC: {average_precision_score(y, p_cv):.3f} ===\n")

    # 3) Final model on all data, used to score every planet.
    model.fit(X, y)
    coefs = pd.Series(model[-1].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    print("What the model learned (standardized coefficients; negative = lowers the score):")
    print(coefs.round(2).to_string(), "\n")

    df["earth_likeness_score"] = (model.predict_proba(X)[:, 1] * 100).round(1)

    earth = pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0.0, 5772.0]], columns=X.columns)
    print(f"Sanity check: Earth itself scores {model.predict_proba(earth)[0, 1] * 100:.1f}\n")

    top = df.sort_values("earth_likeness_score", ascending=False).head(15)
    print("Top 15 most Earth-like planets:")
    print(top[["pl_name", "earth_likeness_score", "pl_rade", "pl_insol", "dist_ly", "is_habitable"]]
          .round(2).to_string(index=False))

    df.to_csv(OUT, index=False)
    joblib.dump(model, ROOT / "models" / "logreg.joblib")


if __name__ == "__main__":
    main()
