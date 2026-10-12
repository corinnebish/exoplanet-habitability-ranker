"""
Exoplanet Habitability Ranker: interactive dashboard.

Run locally:  streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "planets_scored.csv"
METRICS = ROOT / "reports" / "metrics.json"
CHOOSING_K = ROOT / "reports" / "choosing_k.png"
HERO = Path(__file__).resolve().parent / "hero.html"

GROUP_COLORS = {
    "Warm Small Planets": "#4ade80",
    "Scorched Small Planets": "#f87171",
    "Hot Giants": "#fb923c",
    "Cool Giants": "#60a5fa",
}
GROUP_INFO = {
    "Warm Small Planets": ("🌍", "Rocky or water worlds a few times Earth's size, getting moderate starlight. "
                           "Earth belongs here, and so do almost all of the best habitable candidates."),
    "Scorched Small Planets": ("🔥", "Small planets hugging their star, with years lasting just days. "
                               "Surfaces hot enough to melt lead, so no liquid water."),
    "Hot Giants": ("☀️", "\"Hot Jupiters\": gas giants orbiting so close they're heated past 1,300 °F. "
                   "The easiest planets to detect, so there are lots of them."),
    "Cool Giants": ("🧊", "Jupiter-like gas giants far from their star. No solid surface, "
                    "but some of their moons could be interesting."),
}

# Unit conversions for the planet profile.
KM_PER_LY = 9.461e12
VOYAGER_KM_S = 17.0  # Voyager 1, the fastest object leaving the solar system
SECONDS_PER_YEAR = 3.156e7
EARTH_EQT_K = 255  # Earth's equilibrium temperature (before greenhouse warming)

PLOT_TEMPLATE = "plotly_dark"
# Hide the toolbar of small icons (zoom, pan, camera...) that confuses non-technical visitors.
CHART_CONFIG = {"displayModeBar": False}

# Log-scale tick labels written as multiples of Earth, instead of Plotly's "100μ" / "10k" shorthand.
RATIO_TICKS = dict(
    tickvals=[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4],
    ticktext=["1/10,000×", "1/1,000×", "1/100×", "1/10×", "1× (Earth)", "10×", "100×", "1,000×", "10,000×"],
)
SIZE_TICKS = dict(tickvals=[0.5, 1, 2, 5, 10, 20], ticktext=["½×", "1× (Earth)", "2×", "5×", "10×", "20×"])
DIST_TICKS = dict(tickvals=[5, 10, 50, 100, 500, 1000, 5000],
                  ticktext=["5", "10", "50", "100", "500", "1,000", "5,000"])

st.set_page_config(page_title="Exoplanet Habitability Ranker", page_icon="🪐", layout="wide")

# Space theme: fonts, nebula background, twinkling stars, glowing glass cards, fade-in animations.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: .04em; }
h2, h3 { background: linear-gradient(90deg, #7dd3fc, #c084fc); -webkit-background-clip: text; color: transparent !important; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 15% 0%, rgba(99,102,241,.20), transparent 50%),
    radial-gradient(ellipse at 90% 25%, rgba(236,72,153,.13), transparent 45%),
    radial-gradient(ellipse at 50% 100%, rgba(14,165,233,.16), transparent 55%),
    #070b18;
}
[data-testid="stAppViewContainer"]::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    radial-gradient(1px 1px at 10% 20%, #fff, transparent), radial-gradient(1px 1px at 30% 70%, #cbd5ff, transparent),
    radial-gradient(1.5px 1.5px at 55% 35%, #fff, transparent), radial-gradient(1px 1px at 75% 80%, #fff, transparent),
    radial-gradient(1px 1px at 90% 10%, #bae6fd, transparent), radial-gradient(1.5px 1.5px at 65% 60%, #fff, transparent),
    radial-gradient(1px 1px at 20% 90%, #fff, transparent), radial-gradient(1px 1px at 45% 5%, #e9d5ff, transparent);
  background-size: 400px 400px; animation: twinkle 6s ease-in-out infinite alternate; opacity: .55;
}
@keyframes twinkle { to { opacity: .2; } }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(9,13,30,.92); border-right: 1px solid rgba(125,211,252,.15); }
.block-container { padding-top: 1rem; }
[data-testid="stMetric"] {
  background: rgba(255,255,255,.04); border: 1px solid rgba(125,211,252,.22); border-radius: 16px;
  padding: 14px 18px; backdrop-filter: blur(8px); box-shadow: 0 0 22px rgba(125,211,252,.07);
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-4px); border-color: rgba(192,132,252,.6);
  box-shadow: 0 0 30px rgba(125,211,252,.30), 0 0 60px rgba(192,132,252,.15); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; line-height: 1.25; }
[data-testid="stMetricValue"] { font-size: clamp(1.3rem, 2.2vw, 2rem); font-family: 'Orbitron', sans-serif;
  background: linear-gradient(90deg, #7dd3fc, #c084fc); -webkit-background-clip: text; color: transparent; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { font-family: 'Orbitron', sans-serif; font-size: .8rem; padding: 8px 14px;
  border-radius: 10px 10px 0 0; transition: background .2s; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(125,211,252,.08); }
.stTabs [aria-selected="true"] { color: #7dd3fc !important; text-shadow: 0 0 12px rgba(125,211,252,.8); }
.stTabs [data-baseweb="tab-panel"] { animation: fadeUp .55s ease both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
[data-testid="stExpander"] { border: 1px solid rgba(125,211,252,.18); border-radius: 12px; background: rgba(255,255,255,.02); }
[data-testid="stAlert"] { border-radius: 14px; }
.stButton button, .stDownloadButton button { border-radius: 999px; border: 1px solid #7dd3fc; transition: all .2s; }
.stButton button:hover, .stDownloadButton button:hover { box-shadow: 0 0 18px rgba(125,211,252,.6); }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA)
    df["expert_label"] = df["is_habitable"].map({1: "Yes", 0: "No"})
    # Floor bubble size so low-scoring planets stay visible on the chart.
    df["bubble"] = df["earth_likeness_score"].clip(lower=8)
    df["gravity_g"] = df["pl_bmasse"] / df["pl_rade"] ** 2  # surface gravity, Earth = 1
    df["temp_f"] = (df["pl_eqt"] - 273.15) * 9 / 5 + 32
    df["voyager_years"] = df["dist_ly"] * KM_PER_LY / VOYAGER_KM_S / SECONDS_PER_YEAR
    return df, json.loads(METRICS.read_text())


df, metrics = load()

if "planet" not in st.session_state:
    st.session_state.planet = df.sort_values("earth_likeness_score", ascending=False).iloc[0]["pl_name"]


def score_label(score: float) -> str:
    if score >= 90:
        return "🟢 Strong candidate"
    if score >= 50:
        return "🟡 Possible candidate"
    if score >= 10:
        return "🟠 Unlikely"
    return "🔴 Not Earth-like"


def planet_card(name: str, where: str) -> None:
    """Profile of one planet, compared against Earth. `where` keeps chart IDs unique per tab."""
    p = df[df["pl_name"] == name].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p["earth_likeness_score"],
            number={"suffix": " / 100"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#7dd3fc"},
                   "steps": [{"range": [0, 10], "color": "#3b1d1d"}, {"range": [10, 50], "color": "#3b2f1d"},
                             {"range": [50, 90], "color": "#3a3a1d"}, {"range": [90, 100], "color": "#1d3b26"}]},
            title={"text": "Earth-likeness score"},
        ))
        gauge.update_layout(template=PLOT_TEMPLATE, height=230, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True, config=CHART_CONFIG, key=f"gauge_{where}")
        st.markdown(f"### {name}\n**{score_label(p['earth_likeness_score'])}**")
        icon, _ = GROUP_INFO[p["cluster_name"]]
        st.markdown(
            f"{icon} **Group:** {p['cluster_name']}  \n"
            f"⭐ **Expert-listed as habitable:** {p['expert_label']}  \n"
            f"🔭 **Discovered:** {int(p['disc_year'])} by {p['discoverymethod'].lower()}  \n"
            f"🏠 **Host star:** {p['hostname']} (surface {(p['st_teff'] - 273.15) * 9 / 5 + 32:,.0f} °F vs. our Sun's 9,930 °F)"
        )

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Distance (light-years)", f"{p['dist_ly']:,.1f}", help="A light-year is a distance, not a time: how far light travels in one year (≈ 5.9 trillion miles).")
        m2.metric("Trip at Voyager speed (years)", f"{p['voyager_years']:,.0f}",
                  help="Voyager 1 travels about 17 km/s, the fastest spacecraft leaving our solar system.")
        m3.metric("One year there (days)", f"{p['pl_orbper']:,.1f}", help="How long it takes to orbit its star. Earth: 365 days.")
        m4, m5, m6 = st.columns(3)
        m4.metric("Gravity (× Earth's)", f"{p['gravity_g']:.2f}", help="Estimated from mass ÷ radius². 2× means you'd weigh double.")
        m5.metric("Temperature (°F)", f"{p['temp_f']:,.0f}",
                  help="Equilibrium temperature: no atmosphere included. Earth's is about 0 °F; greenhouse gases warm it to ~59 °F.")
        m6.metric("Starlight (× Earth's)", f"{p['pl_insol']:.2f}", help="Energy received from its star. Earth = 1.")

        comp = pd.DataFrame({
            "Feature": ["Radius", "Mass", "Starlight", "Gravity", "Temperature"],
            "Ratio": [p["pl_rade"], p["pl_bmasse"], p["pl_insol"], p["gravity_g"], p["pl_eqt"] / EARTH_EQT_K],  # Kelvin ratio, so it is a true multiple
        })
        bar = px.bar(comp, x="Ratio", y="Feature", orientation="h", log_x=True, text=comp["Ratio"].map(lambda v: f"{v:.2f}×"),
                     template=PLOT_TEMPLATE, title="Compared to Earth (dashed line = exactly Earth)")
        bar.add_vline(x=1, line_dash="dash", line_color="gold")
        bar.update_traces(marker_color="#7dd3fc")
        bar.update_layout(height=280, margin=dict(t=40, b=10), xaxis_title="× Earth's value", yaxis_title="")
        bar.update_xaxes(**RATIO_TICKS)
        st.plotly_chart(bar, use_container_width=True, config=CHART_CONFIG, key=f"compare_{where}")
        if p["mass_imputed"]:
            st.caption("ⓘ Mass not measured for this planet; estimated from its radius (Chen & Kipping 2017).")


# ================= Header =================
hero = (HERO.read_text()
        .replace("__N_PLANETS__", str(metrics["n_planets"]))
        .replace("__N_STRONG__", str(int((df["earth_likeness_score"] >= 90).sum())))
        .replace("__NEAREST__", f"{df.loc[df['earth_likeness_score'] >= 90, 'dist_ly'].min():.1f}"))
components.html(hero, height=600)
st.markdown(
    f"Which of the **{metrics['n_planets']:,} known exoplanets** are most like Earth? "
    "Machine learning sorts them into planet types (**K-Means**), then scores each one 0–100 "
    "for Earth-likeness (**logistic regression**). Use the controls below to choose which planets to show."
)

# ================= Filters =================
# On the main page rather than the sidebar: Streamlit collapses the sidebar on smaller screens,
# so many visitors would never see it.
with st.container(border=True):
    f1, f2 = st.columns([1, 1])
    max_dist = f1.slider(
        "Distance from Earth (light-years)", 4, 5000, 100, step=1,
        help="Only show planets within this distance. A light-year ≈ 5.9 trillion miles. "
             "The nearest exoplanet is 4.2 light-years away.",
    )
    show = f2.radio(
        "Show", ["All planets", "Strong candidates (90+)", "Expert-listed only"], horizontal=True,
        help="Strong candidates: Earth-likeness score of 90 or higher. "
             "Expert-listed: the ~70 planets scientists list as potentially habitable.",
    )

view = df[df["dist_ly"] <= max_dist]
if show == "Strong candidates (90+)":
    view = view[view["earth_likeness_score"] >= 90]
elif show == "Expert-listed only":
    view = view[view["is_habitable"] == 1]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Planets shown", f"{len(view):,}", help="Planets matching your distance and Show choices.")
k2.metric("Strong candidates", f"{(view['earth_likeness_score'] >= 90).sum():,}",
          help="Planets scoring 90 or higher out of 100.")
k3.metric("Nearest (light-yrs)", f"{view['dist_ly'].min():.1f}" if len(view) else "—",
          help="Distance to the closest planet matching the filters above. One light-year ≈ 5.9 trillion miles.")
k4.metric("Earth's own score", f"{metrics['earth_score']}", help="Sanity check: the model scores Earth itself as highly Earth-like.")

with st.expander("📘 How to read this dashboard: terms and units"):
    st.markdown(
        "| Term | Meaning |\n|---|---|\n"
        "| **Exoplanet** | A planet orbiting a star other than our Sun. |\n"
        "| **Earth-likeness score (0–100)** | How closely a planet matches the profile of planets scientists consider "
        "potentially habitable. 90+ is a strong candidate. It ranks similarity; it does not mean life exists there. |\n"
        "| **Light-year** | A distance, not a time: how far light travels in one year, ≈ 5.9 trillion miles. |\n"
        "| **× Earth / 1× (Earth)** | Compared with Earth. 2× = twice Earth's value; 1/10× = one-tenth. |\n"
        "| **Starlight received** | Energy a planet gets from its star, compared with what Earth gets from the Sun. |\n"
        "| **Habitable zone** | The range of starlight where liquid water could exist on a planet's surface. |\n"
        "| **Estimated temperature** | Temperature from starlight alone, before any atmosphere. Earth would be ≈ 0 °F "
        "by this measure (actual average ≈ 59 °F). |\n"
        "| **Planet type** | One of four groups found by K-Means clustering, an algorithm that groups similar items. |\n"
        "| **Expert-listed** | On the Habitable Worlds Catalog, a list of ~70 potentially habitable planets maintained by "
        "the Planetary Habitability Laboratory. |\n"
        "| **Log scale** | Each step on the axis is 10× the last, so tiny and huge values fit on one chart. |"
    )

tab_explore, tab_galaxy, tab_time, tab_profile, tab_types, tab_model = st.tabs(
    ["🔭 Explore", "🌌 3D galaxy map", "⏳ Discovery timeline", "🌍 Planet profile", "🧩 Planet types", "🧠 How it works"]
)

# ================= Explore =================
with tab_explore:
    if view.empty:
        st.warning("No planets match. Try a larger distance or set Show to \"All planets\".")
    else:
        chart_view = st.radio(
            "Chart", ["Size vs. starlight", "Distance vs. score", "Temperature vs. size"], horizontal=True,
        )
        st.caption({
            "Size vs. starlight": "Earth-like planets sit near the ⭐: about Earth's size, getting about as much starlight as Earth.",
            "Distance vs. score": "The best nearby candidates are in the top-left: high score, close to Earth.",
            "Temperature vs. size": "Earth-like planets sit near the ⭐: about Earth's size and temperature.",
        }[chart_view])

        x, y, logx, logy, xl, yl = {
            "Size vs. starlight": ("pl_insol", "pl_rade", True, True, "Starlight received (× what Earth gets)", "Radius (× Earth's)"),
            "Distance vs. score": ("dist_ly", "earth_likeness_score", True, False, "Distance from Earth (light-years)", "Earth-likeness score"),
            "Temperature vs. size": ("temp_f", "pl_rade", False, True, "Estimated temperature (°F, excluding atmosphere)", "Radius (× Earth's)"),
        }[chart_view]
        color_args = dict(color="cluster_name", color_discrete_map=GROUP_COLORS)
        fig = px.scatter(
            view, x=x, y=y, size="bubble", size_max=18, log_x=logx, log_y=logy,
            hover_name="pl_name", custom_data=["pl_name"], template=PLOT_TEMPLATE,
            hover_data={"earth_likeness_score": ":.1f", "dist_ly": ":.1f", "pl_rade": ":.2f", "pl_insol": ":.2f",
                        "temp_f": ":.0f", "cluster_name": True, "expert_label": True, "bubble": False},
            labels={"pl_insol": "Starlight (×Earth)", "pl_rade": "Radius (×Earth)", "cluster_name": "Group",
                    "earth_likeness_score": "Score", "dist_ly": "Distance (light-years)", "temp_f": "Temp (°F)",
                    "expert_label": "Expert-listed"},
            **color_args,
        )
        earth_xy = {"Size vs. starlight": (1, 1), "Temperature vs. size": ((EARTH_EQT_K - 273.15) * 9 / 5 + 32, 1)}.get(chart_view)
        if earth_xy:
            fig.add_trace(go.Scatter(
                x=[earth_xy[0]], y=[earth_xy[1]], mode="markers+text", text=["Earth"], textposition="top center",
                marker=dict(symbol="star", size=22, color="gold", line=dict(color="white", width=1)),
                name="Earth", hoverinfo="name", textfont=dict(color="gold"),
            ))
        if chart_view == "Size vs. starlight":
            # Rough habitable zone: liquid water possible at ~0.25–1.5× Earth's starlight.
            fig.add_vrect(x0=0.25, x1=1.5, fillcolor="#4ade80", opacity=0.08, line_width=0,
                          annotation_text="habitable zone", annotation_position="top left")
        fig.update_layout(height=560, xaxis_title=xl, yaxis_title=yl, legend_title_text="", margin=dict(t=30),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        if x == "pl_insol":
            fig.update_xaxes(**RATIO_TICKS)
        if x == "dist_ly":
            fig.update_xaxes(**DIST_TICKS)
        if y == "pl_rade":
            fig.update_yaxes(**SIZE_TICKS)

        event = st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, on_select="rerun",
                                selection_mode="points", key="main_chart")
        st.caption("💡 Hover for details · **click a planet** to select it for the Planet profile tab · drag to zoom · double-click to reset. "
                   "Bigger bubble = higher score. Green band = rough habitable zone (where liquid water is possible).")
        pts = event.selection.points if event and event.selection else []
        if pts and pts[0].get("customdata"):
            st.session_state.planet = pts[0]["customdata"][0]

        st.subheader("🏆 Top candidates")
        st.caption("Click a row to select that planet, then open the **🌍 Planet profile** tab.")
        top = view.sort_values("earth_likeness_score", ascending=False).head(25).reset_index(drop=True)
        table = top[["pl_name", "earth_likeness_score", "dist_ly", "pl_rade", "gravity_g", "temp_f",
                     "pl_orbper", "cluster_name", "expert_label"]].copy()
        table.index = range(1, len(table) + 1)
        sel = st.dataframe(
            table, use_container_width=True, on_select="rerun", selection_mode="single-row", key="top_table",
            column_config={
                "pl_name": "Planet",
                "earth_likeness_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                "dist_ly": st.column_config.NumberColumn("Distance (light-years)", format="%.1f"),
                "pl_rade": st.column_config.NumberColumn("Size (×Earth)", format="%.2f"),
                "gravity_g": st.column_config.NumberColumn("Gravity (×Earth)", format="%.2f"),
                "temp_f": st.column_config.NumberColumn("Temp (°F)", format="%.0f"),
                "pl_orbper": st.column_config.NumberColumn("Year (days)", format="%.1f"),
                "cluster_name": "Group",
                "expert_label": "Expert-listed",
            },
        )
        if sel and sel.selection.rows:
            st.session_state.planet = top.iloc[sel.selection.rows[0]]["pl_name"]
        st.download_button("⬇️ Download this list (CSV)", table.to_csv(index_label="Rank"),
                           "habitable_shortlist.csv", "text/csv")

        if (pts and pts[0].get("customdata")) or (sel and sel.selection.rows):
            st.success(f"**{st.session_state.planet}** selected. Open the **🌍 Planet profile** tab to see its full profile.")

    # ----- Key findings (Explore tab only) -----
    st.divider()
    st.markdown("## 📌 Key findings")
    _strong = df[df["earth_likeness_score"] >= 90]
    _new = _strong[_strong["is_habitable"] == 0]
    # Break distance ties (e.g. planets in the same system) by the higher score.
    _nearest = _strong.sort_values(["dist_ly", "earth_likeness_score"], ascending=[True, False]).iloc[0]
    _top = df.sort_values("earth_likeness_score", ascending=False).head(3)["pl_name"].tolist()
    _warm = df[df["cluster_name"] == "Warm Small Planets"]
    _warm_share = (_strong["cluster_name"] == "Warm Small Planets").mean() * 100
    _transit = (df["discoverymethod"] == "Transit").mean() * 100
    fa, fb = st.columns(2)
    with fa:
        st.markdown(
            f"**1. Earth-like planets are rare.** Only **{len(_strong):,} of {len(df):,}** planets "
            f"({len(_strong) / len(df) * 100:.1f}%) score 90 or higher.\n\n"
            f"**2. The closest strong candidate is {_nearest['pl_name']}**, just **{_nearest['dist_ly']:.1f} light-years** away "
            f"(score {_nearest['earth_likeness_score']:.1f}). Top-ranked overall: {', '.join(_top)}.\n\n"
            f"**3. The model flags {len(_new)} strong candidates that aren't on the expert list**, such as "
            f"{', '.join(_new.sort_values('earth_likeness_score', ascending=False)['pl_name'].head(3))}. Some are promising "
            f"leads; others are near-misses just outside the habitable zone, showing where the model is more generous than the experts.\n\n"
            f"**4. Starlight matters most.** Receiving the right amount of energy from its star is the strongest "
            f"predictor of an Earth-like planet, followed by being close to Earth's size."
        )
    with fb:
        st.markdown(
            f"**5. Two independent methods agree.** K-Means grouped planets without any habitability labels, yet "
            f"**{_warm_share:.0f}%** of strong candidates fall in the same group as Earth (Warm Small Planets, "
            f"{len(_warm):,} planets).\n\n"
            f"**6. The model is reliable on unseen planets.** It found **{metrics['test_found']} of "
            f"{metrics['test_total_habitable']}** habitable planets it had never seen, and scored "
            f"**{metrics['cv_pr_auc'] / metrics['baseline_pr_auc']:.0f}× better than random guessing** (PR-AUC "
            f"{metrics['cv_pr_auc']:.2f} vs. {metrics['baseline_pr_auc']:.2f}).\n\n"
            f"**7. Earth passes the sanity check**, scoring **{metrics['earth_score']}** even though the model never saw it.\n\n"
            f"**8. What we know is shaped by how we look.** {_transit:.0f}% of planets were found by the transit method, "
            f"which favors large planets orbiting close to their stars, so cooler, Earth-like planets are likely undercounted."
        )

# ================= 3D galaxy map =================
with tab_galaxy:
    st.markdown("Every planet placed where it **really is** in space, with **Earth (gold diamond) at the center**. "
                "Brighter, bigger dots are more Earth-like. **Drag** to spin, **scroll** to zoom, or press **▶ Spin the galaxy**. "
                "Increase the distance slider above to see farther out.")

    gv = view.copy()
    ra, dec = np.radians(gv["ra"]), np.radians(gv["dec"])
    # Square-root distances: real ones span 4 to 20,000+ light-years, which would pile nearby planets onto Earth.
    r = np.sqrt(gv["dist_ly"])
    gv["x"], gv["y"], gv["z"] = r * np.cos(dec) * np.cos(ra), r * np.cos(dec) * np.sin(ra), r * np.sin(dec)

    hover = ("<b>%{text}</b><br>Score: %{customdata[0]:.1f}<br>Distance: %{customdata[1]:,.1f} light-years"
             "<br>Group: %{customdata[2]}<extra></extra>")
    gfig = go.Figure()
    gfig.add_trace(go.Scatter3d(
        x=gv["x"], y=gv["y"], z=gv["z"], mode="markers", text=gv["pl_name"],
        customdata=gv[["earth_likeness_score", "dist_ly", "cluster_name"]], hovertemplate=hover,
        marker=dict(size=2.5 + gv["earth_likeness_score"] / 12, color=gv["earth_likeness_score"],
                    colorscale=[[0, "#4c1d95"], [.5, "#c026d3"], [.85, "#22d3ee"], [1, "#a7f3d0"]],
                    cmin=0, cmax=100, opacity=.85, line=dict(width=0),
                    colorbar=dict(title="Earth-likeness<br>score", thickness=12, len=.6)),
        name="Planets",
    ))
    gfig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text", text=["  Earth"], textfont=dict(color="gold", size=14),
        marker=dict(size=9, color="gold", symbol="diamond", line=dict(color="white", width=1)),
        hoverinfo="text", name="Earth",
    ))
    axis = dict(visible=False, showbackground=False)
    lim = float(np.abs(gv[["x", "y", "z"]]).to_numpy().max()) * 1.05 if len(gv) else 1

    # Spin animation: the camera circles Earth when you press play.
    frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.05 * np.cos(a), y=1.05 * np.sin(a), z=.4))))
              for a in np.linspace(0, 2 * np.pi, 90)]
    gfig.frames = frames
    gfig.update_layout(
        template=PLOT_TEMPLATE, height=680, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=dict(axis, range=[-lim, lim]), yaxis=dict(axis, range=[-lim, lim]),
                   zaxis=dict(axis, range=[-lim, lim]), aspectmode="cube", bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=1.05, y=0, z=.4))),
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
        updatemenus=[dict(type="buttons", x=.02, y=.04, xanchor="left", bgcolor="#151c33", bordercolor="#7dd3fc",
                          font=dict(color="#7dd3fc"), active=-1, direction="right", pad=dict(r=6),
                          buttons=[dict(label="▶ Spin the galaxy", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0), mode="immediate")]),
                                   dict(label="⏸ Stop", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
    )
    st.plotly_chart(gfig, use_container_width=True, config=CHART_CONFIG)
    st.caption("💡 The thick streak of planets in one direction is the **Kepler telescope's field of view**: it stared at one "
               "patch of sky for 4 years, so that's where most planets were found. That's detection bias you can see.")

# ================= Discovery timeline =================
with tab_time:
    st.markdown("Press **▶** (below the chart, on the left) to watch planets appear in the year they were discovered, "
                "or drag the year slider to jump to any year. The first planet around a Sun-like star was found in **1995**.")
    first = int(max(df["disc_year"].min(), 1995))
    last = int(df["disc_year"].max())
    tl = df[["pl_name", "pl_insol", "pl_rade", "cluster_name", "disc_year", "earth_likeness_score"]]
    anim = pd.concat([tl[tl["disc_year"] <= y].assign(year=y) for y in range(first, last + 1)], ignore_index=True)
    # Plotly animations only draw groups that exist in the first frame, and 1995 has just one
    # planet. An invisible placeholder per group per year keeps all four colors in every frame.
    placeholders = pd.DataFrame([{"pl_name": f"_{g}", "pl_insol": np.nan, "pl_rade": np.nan, "cluster_name": g, "year": y}
                                 for y in range(first, last + 1) for g in GROUP_COLORS])
    anim = pd.concat([anim, placeholders], ignore_index=True)
    tfig = px.scatter(
        anim, x="pl_insol", y="pl_rade", animation_frame="year", animation_group="pl_name",
        color="cluster_name", color_discrete_map=GROUP_COLORS, hover_name="pl_name",
        log_x=True, log_y=True, range_x=[1e-4, 1e5], range_y=[0.25, 40], template=PLOT_TEMPLATE,
        category_orders={"cluster_name": list(GROUP_COLORS)},
        labels={"pl_insol": "Starlight received (× what Earth gets)", "pl_rade": "Radius (× Earth's)", "cluster_name": ""},
    )
    tfig.update_traces(marker=dict(size=6, opacity=.8, line=dict(width=0)))
    tfig.add_trace(go.Scatter(x=[1], y=[1], mode="markers+text", text=["Earth"], textposition="top center",
                              marker=dict(symbol="star", size=20, color="gold"), textfont=dict(color="gold"),
                              name="Earth", showlegend=False))
    tfig.add_vrect(x0=0.25, x1=1.5, fillcolor="#4ade80", opacity=0.08, line_width=0)
    tfig.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30))
    tfig.update_xaxes(**RATIO_TICKS)
    tfig.update_yaxes(**SIZE_TICKS)
    tfig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 350
    tfig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
    st.plotly_chart(tfig, use_container_width=True, config=CHART_CONFIG)

    per_year = df.groupby("disc_year").size().reset_index(name="planets")
    per_year = per_year[per_year["disc_year"] >= first]
    bar = px.bar(per_year, x="disc_year", y="planets", template=PLOT_TEMPLATE,
                 labels={"disc_year": "Year", "planets": "Planets discovered"})
    bar.update_traces(marker_color="#7dd3fc", marker_line_width=0)
    bar.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
    st.plotly_chart(bar, use_container_width=True, config=CHART_CONFIG)
    st.caption("The 2014 and 2016 spikes are the Kepler mission releasing hundreds of planets at once.")

# ================= Planet profile =================
with tab_profile:
    st.markdown("Pick any of the 5,700+ planets to see how it compares to Earth.")
    names = df.sort_values("earth_likeness_score", ascending=False)["pl_name"].tolist()
    choice = st.selectbox("Search for a planet", names, index=names.index(st.session_state.planet),
                          help="Start typing a name, e.g. 'TRAPPIST' or 'Proxima'.")
    planet_card(choice, "profile")

    st.markdown("#### Famous planets to try")
    st.markdown("**Proxima Cen b** (the closest exoplanet) · **TRAPPIST-1 e** (one of 7 planets around a tiny star) · "
                "**Kepler-452 b** (\"Earth's cousin\") · **51 Peg b** (the first planet found around a Sun-like star)")

# ================= Planet types =================
with tab_types:
    st.markdown(
        "**K-Means clustering** sorted every planet into groups *without being told what the groups were*. "
        "The four groups it found match planet types astronomers already recognize."
    )
    cols = st.columns(4)
    for col, (g, color) in zip(cols, GROUP_COLORS.items()):
        sub = df[df["cluster_name"] == g]
        icon, desc = GROUP_INFO[g]
        with col:
            st.markdown(f"<h4 style='color:{color}'>{icon} {g}</h4>", unsafe_allow_html=True)
            st.metric("Planets", f"{len(sub):,}")
            st.markdown(desc)
            st.markdown(
                f"- Typical size: **{sub['pl_rade'].median():.1f}× Earth**\n"
                f"- Typical temp: **{sub['temp_f'].median():,.0f} °F**\n"
                f"- Typical year: **{sub['pl_orbper'].median():,.1f} days**\n"
                f"- Avg. score: **{sub['earth_likeness_score'].mean():.1f}**"
            )
            if g == "Warm Small Planets":
                st.success("🌍 Earth would be in this group")

    st.markdown("#### Score by group")
    box = px.box(df, x="cluster_name", y="earth_likeness_score", color="cluster_name",
                 color_discrete_map=GROUP_COLORS, points=False, template=PLOT_TEMPLATE,
                 labels={"cluster_name": "", "earth_likeness_score": "Earth-likeness score"})
    box.update_layout(showlegend=False, height=380)
    st.plotly_chart(box, use_container_width=True, config=CHART_CONFIG)
    st.caption("Nearly all high scores come from Warm Small Planets: the two methods agree.")

    with st.expander("Why 4 groups? (how K was chosen)"):
        st.markdown(
            "We tried 2 to 10 groups and scored each. **3 groups** lumped every small planet together, hiding "
            "the difference between scorched and temperate worlds. **5 or more** created a tiny group of just 6 planets "
            "orbiting unusually hot stars, which is an outlier pile, not a real planet type. **4** was the best balance."
        )
        if CHOOSING_K.exists():
            st.image(str(CHOOSING_K), caption="Left: elbow method. Right: silhouette score. Red line = chosen K.")

# ================= How the model works =================
with tab_model:
    st.markdown("### The pipeline")
    s1, s2, s3, s4 = st.columns(4)
    s1.info(f"**1 · Data**\n\n{6445:,} planets from the NASA Exoplanet Archive → {metrics['n_planets']:,} with complete data.")
    s2.info("**2 · Clean**\n\nLog-scale skewed values, estimate missing masses from radius, standardize.")
    s3.info("**3 · Group**\n\nK-Means sorts planets into 4 natural types.")
    s4.info(f"**4 · Score**\n\nLogistic regression trained on {metrics['n_habitable_labels']} expert-labeled habitable planets.")

    st.markdown("### How well does it work?")
    a, b, c = st.columns(3)
    a.metric("Habitable planets found", f"{metrics['test_found']} of {metrics['test_total_habitable']}",
             help="On a test set of planets the model never saw while learning.")
    b.metric("PR-AUC score (0 to 1)", f"{metrics['cv_pr_auc']:.2f}",
             delta=f"{metrics['cv_pr_auc'] / metrics['baseline_pr_auc']:.0f}× better than random", delta_color="normal",
             help="Precision-recall area under the curve: grades how well a model finds rare cases. "
                  "1.0 = perfect; random guessing would score about 0.01 here. Averaged over 5 rounds of testing (cross-validation).")
    c.metric("False alarms", f"{metrics['test_false_alarms']}",
             help="Planets flagged as Earth-like that experts didn't list. Some may be real candidates experts haven't added yet.")

    st.markdown(
        "**Why not just report accuracy?** Only about 1% of planets are habitable. A lazy model that answers "
        "\"not habitable\" every time would be **99% accurate** and completely useless. So the model is told to pay "
        "extra attention to the rare habitable planets (*balanced class weights*), and graded on how well it finds them."
    )

    st.markdown("### What the model learned")
    nice = {
        "starlight_gap_sq": "Starlight far from Earth's", "log_starlight": "More starlight overall",
        "radius_gap_sq": "Size far from Earth's", "mass_gap_sq": "Mass far from Earth's",
        "log_radius": "Bigger overall", "star_temp": "Hotter host star",
    }
    coef = pd.DataFrame({"Factor": [nice[k] for k in metrics["coefficients"]],
                         "Effect": list(metrics["coefficients"].values())}).sort_values("Effect")
    cf = px.bar(coef, x="Effect", y="Factor", orientation="h", template=PLOT_TEMPLATE,
                color=np.where(coef["Effect"] < 0, "Lowers score", "Raises score"),
                color_discrete_map={"Lowers score": "#f87171", "Raises score": "#4ade80"})
    cf.update_layout(height=320, legend_title_text="", yaxis_title="", xaxis_title="Coefficient (standardized)")
    st.plotly_chart(cf, use_container_width=True, config=CHART_CONFIG)
    st.markdown(
        "**In plain English:** getting the *right amount of starlight* matters most. Too much or too little "
        "both drag the score down. Being *Earth-sized* comes second. Planets that get too much starlight are "
        "penalized more than ones getting a bit too little."
    )

    with st.expander("🔑 The key design decision: where the labels come from"):
        st.markdown(
            "Logistic regression needs examples labeled *yes, habitable* and *no*. NASA doesn't provide that. "
            "If we wrote our own rule (e.g. \"Earth-sized and the right temperature = yes\"), the model would just "
            "learn our rule back, which is circular. Instead, labels come from an **outside expert source**: the "
            "[Habitable Worlds Catalog](https://phl.upr.edu/hwc) from the Planetary Habitability Laboratory, "
            "University of Puerto Rico at Arecibo."
        )
    with st.expander("⚠️ Limitations"):
        st.markdown(
            "- **Temperatures ignore atmosphere.** By this measure Earth would be about 0 °F; greenhouse gases warm it to about 59 °F.\n"
            f"- **Only {metrics['n_habitable_labels']} positive examples**, so the score is a *ranking*, not a true probability.\n"
            "- **Detection bias:** most planets are found by the transit method, which favors big planets close to their stars.\n"
            "- **Distance is not used in the score.** A planet 1,000 light-years away can score as high as a nearby one; "
            "use the distance slider to focus on reachable targets."
        )
    with st.expander("📖 Glossary"):
        st.markdown(
            "- **Exoplanet:** a planet orbiting a star other than our Sun.\n"
            "- **Light-year:** a distance, not a time: how far light travels in one year, ≈ 5.9 trillion miles.\n"
            "- **Starlight / insolation:** energy a planet receives from its star. Earth = 1.\n"
            "- **Habitable zone:** the range of distances from a star where liquid water could exist.\n"
            "- **K-Means:** an algorithm that sorts items into K groups of similar items.\n"
            "- **Logistic regression:** a model that outputs the probability of a yes/no outcome.\n"
            "- **PR-AUC:** a score for how well a model finds rare cases; 1.0 is perfect."
        )

st.divider()
st.caption("Built by Corinne Bish · USC Marshall MSBA · Data: NASA Exoplanet Archive & PHL Habitable Worlds Catalog · "
           "[Code on GitHub](https://github.com/corinnebish/exoplanet-habitability-ranker)")
