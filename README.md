# 🪐 Exoplanet Habitability Ranker

**Which of the 5,700+ planets found beyond our solar system are most like Earth?**

[![Launch the app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://exoplanet-ranker.streamlit.app)

### 👉 [**Open the live dashboard**](https://exoplanet-ranker.streamlit.app): no download or setup needed

---

## What it does
Uses machine learning on real NASA data to give every confirmed exoplanet an **Earth-likeness score from 0 to 100**, then lets anyone explore the results in an interactive dashboard: a 3D map of where the planets are, an animated 30-year discovery timeline, and a profile for every planet.

## Key results
- **5,729 planets** analyzed from the NASA Exoplanet Archive
- **4 planet types** found automatically with K-Means clustering; Earth falls in "Warm Small Planets"
- **116 strong candidates** (score ≥ 90); the closest, **Proxima Centauri b**, is 4.2 light-years away
- The model found **16 of 16** habitable planets it had never seen, and scored **52× better than random guessing** (PR-AUC 0.57 vs. 0.01)
- **Earth scores 99.5**, even though the model never saw it during training

## How it works
| Step | What happens | Code |
|---|---|---|
| 1. Collect | Download every confirmed planet from NASA's Exoplanet Archive (`pscomppars` table) | [`src/fetch_data.py`](src/fetch_data.py) |
| 2. Clean | Log-transform skewed features; estimate missing masses from radius (Chen & Kipping 2017) and flag them | [`src/preprocess.py`](src/preprocess.py) |
| 3. Cluster | K-Means on 6 standardized features; K = 4 chosen with elbow + silhouette analysis | [`src/cluster.py`](src/cluster.py) |
| 4. Score | Logistic regression trained on the expert-curated [Habitable Worlds Catalog](https://phl.upr.edu/hwc); balanced class weights for the ~1% positive rate; evaluated with stratified 5-fold cross-validation and PR-AUC | [`src/train_model.py`](src/train_model.py) |
| 5. Dashboard | Interactive Streamlit + Plotly app with a Three.js animated intro | [`app/streamlit_app.py`](app/streamlit_app.py) |

**Key design decision:** habitability labels come from an outside expert source rather than a hand-written rule, so the model isn't just relearning its own definition. Because habitability is a "sweet spot" (not too hot, not too cold), features include squared distance from Earth's values.

## Limitations
- Temperatures exclude atmosphere (Earth would be ≈ 0 °F by this measure)
- Only 65 positive examples, so scores are a ranking rather than a true probability
- Most planets were found by the transit method, which favors large planets close to their stars

## Tech
Python · pandas · scikit-learn · Plotly · Streamlit · Three.js

## Run it yourself (optional)
```bash
pip install -r requirements.txt
python src/fetch_data.py && python src/preprocess.py && python src/cluster.py && python src/train_model.py
streamlit run app/streamlit_app.py
```

---
Built by **Corinne Bish** · USC Marshall MSBA · Data: NASA Exoplanet Archive & PHL Habitable Worlds Catalog
