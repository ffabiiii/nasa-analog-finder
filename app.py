import json
from pathlib import Path
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

from engine.similarity_model import (
    FEATURE_KEYS,
    FEATURE_LABELS_EN,
    normalize_value,
    compute_analog_ranking,
)

st.set_page_config(
    page_title="Planetary Analog Finder | NASA Space Apps",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MASTER_PARQUET = Path("data/master_analogs.parquet")
MASTER_CSV = Path("data/master_analogs.csv")
TARGETS_JSON = Path("data/extraterrestrial_targets.json")


def translate_geology_to_en(text: str) -> str:
    """Ensures any Spanish geological terms in cached CSV/Parquet files render in clean English."""
    if not isinstance(text, str):
        return "Basaltic & Arid Planetary Analog Terrain"
    replacements = {
        "Desierto hiperárido": "Hyper-arid desert",
        "suelos ricos en": "soils rich in",
        "percloratos": "perchlorates",
        "sulfatos": "sulfates",
        "óxidos de hierro": "iron oxides",
        "Cráter de impacto": "Impact crater",
        "brechas": "breccias",
        "permafrost": "permafrost",
        "Valles secos": "Dry valleys",
        "hielo subsuperficial": "subsurface ice",
        "Volcán": "Volcano",
        "basáltico": "basaltic",
        "basáltica": "basaltic",
        "ceniza volcánica": "volcanic ash",
        "flujos de lava": "lava flows",
        "tubos de lava": "lava tubes",
        "Salar": "Salt flat",
        "costra salina": "evaporitic salt crust",
        "borde volcánico": "volcanic rim",
        "Meseta": "High-altitude plateau",
        "planicie": "plain",
        "dunas": "dunes",
        "campo volcánico": "volcanic field",
        "y": "and",
        "con": "with",
        "de": "of",
    }
    out = text
    for es, en in replacements.items():
        out = out.replace(es, en)
    return out


@st.cache_data
def load_data():
    if MASTER_PARQUET.exists():
        df = pd.read_parquet(MASTER_PARQUET)
    elif MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    else:
        st.error("Master dataset not found. Please run `python etl/fetch_terrain_sar.py` first.")
        st.stop()

    if "primary_geology" in df.columns:
        df["primary_geology"] = df["primary_geology"].apply(translate_geology_to_en)

    with open(TARGETS_JSON, "r", encoding="utf-8") as f:
        targets_data = json.load(f)["targets"]
    return df, targets_data


df_sites, targets = load_data()

# --- SIDEBAR: MISSION CONFIGURATION ---
st.sidebar.title("🛰️ Mission Control")
st.sidebar.caption("Planetary Analog Finder • NASA & ESA Official Data")

target_names = [t["name"] for t in targets]
selected_target_name = st.sidebar.selectbox("Extraterrestrial Target", target_names, index=0)
selected_target = next(t for t in targets if t["name"] == selected_target_name)

st.sidebar.info(
    f"**Celestial Body:** {selected_target['body']}\n\n"
    f"**Mission Focus:** {selected_target['mission_focus']}"
)

category_filter = st.sidebar.radio(
    "Terrestrial Candidate Group",
    ["All Sites (15)", "Discovery Only (11)", "Control Only (4)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Subsystem Priority Weights")
st.sidebar.caption("Adjust engineering weights (0.0 – 2.5) to recalculate similarity in real time:")

default_w = selected_target["default_weights"]
weights = {
    "slope_deg": st.sidebar.slider("Landing Safety (Slope)", 0.0, 2.5, float(default_w["slope_deg"]), 0.1),
    "diurnal_temp_range_c": st.sidebar.slider("Thermal Swing (ΔT)", 0.0, 2.5, float(default_w["diurnal_temp_range_c"]), 0.1),
    "aridity_ice_index": st.sidebar.slider("Hyper-Aridity & Ice", 0.0, 2.5, float(default_w["aridity_ice_index"]), 0.1),
    "basalt_oxide_index": st.sidebar.slider("Basalt & Iron Oxides", 0.0, 2.5, float(default_w["basalt_oxide_index"]), 0.1),
    "sar_roughness_index": st.sidebar.slider("SAR Surface Roughness", 0.0, 2.5, float(default_w["sar_roughness_index"]), 0.1),
}

# Compute full ranking first, then apply category filter
ranked_all = compute_analog_ranking(df_sites, selected_target, weights)

if "Discovery" in category_filter:
    display_df = ranked_all[ranked_all["category"] == "Discovery"].reset_index(drop=True)
elif "Control" in category_filter:
    display_df = ranked_all[ranked_all["category"] == "Control"].reset_index(drop=True)
else:
    display_df = ranked_all.copy()

# --- HEADER & KPI CARDS ---
st.title("🪐 Planetary Analog Finder")
st.markdown(
    "Geophysical similarity engine powered by **NASA PDS (LRO Diviner, LOLA, MOLA, CRISM)**, "
    "**NASA POWER**, **NASA Earthdata / Worldview GIBS**, and **ESA Copernicus Sentinel-1**."
)

top_overall = ranked_all.iloc[0]
top_discovery = ranked_all[ranked_all["category"] == "Discovery"].iloc[0]
top_control = ranked_all[ranked_all["category"] == "Control"].iloc[0]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Top Overall Analog", f"{top_overall['similarity_pct']}%", top_overall["site_name"])
kpi2.metric("Top Discovery Site", f"{top_discovery['similarity_pct']}%", top_discovery["site_name"])
kpi3.metric("Top NASA Control Benchmark", f"{top_control['similarity_pct']}%", top_control["site_name"])
kpi4.metric("Evaluated Sites", f"{len(display_df)}", f"Target: {selected_target['body']}")

st.markdown("---")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs([
    "🗺️ Satellite Map & Prioritization Ranking",
    "🔬 Geophysical Diagnosis & Analogy Breakdown",
    "📊 Scientific Validation (Benchmark)",
])

# === TAB 1: MAP & RANKING ===
with tab1:
    col_map, col_table = st.columns([1.15, 1.0])

    with col_map:
        st.subheader("Global Distribution of Candidate Analogs")
        m = folium.Map(location=[5.0, -25.0], zoom_start=2, tiles=None)
        folium.TileLayer(
            tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg",
            attr="NASA Worldview / GIBS (Earthdata)",
            name="NASA Worldview (Shaded Relief)",
            overlay=False,
            control=True,
            max_native_zoom=8,
        ).add_to(m)

        for _, row in display_df.iterrows():
            score = row["similarity_pct"]
            if row["category"] == "Control":
                color = "blue"
            elif score >= 80:
                color = "green"
            elif score >= 68:
                color = "orange"
            else:
                color = "red"

            popup_html = (
                f"<b>#{row['rank']} {row['site_name']}</b><br>"
                f"Category: {row['category']}<br>"
                f"Global Similarity: <b>{score}%</b><br>"
                f"Slope: {row['slope_deg']}° | ΔT: {row['diurnal_temp_range_c']}°C<br>"
                f"Geology: {row['primary_geology']}<br>"
                f"<hr style='margin:4px 0;'>"
                f"<small><b>NASA Earthdata:</b> {row.get('nasa_granule_id', 'NASADEM_HGT')}<br>"
                f"<b>ESA Sentinel-1:</b> {row.get('esa_sar_product_id', 'S1_SAR')}</small>"
            )
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"#{row['rank']} {row['site_name']} ({score}%)",
                icon=folium.Icon(color=color, icon="globe", prefix="fa"),
            ).add_to(m)

        st_folium(m, width=None, height=470, returned_objects=[])
        st.caption(
            "Pin Legend: 🔵 Blue = NASA Control Benchmark | 🟢 Green = High Similarity (≥80%) | "
            "🟠 Orange = Moderate (68–79%) | 🔴 Red = Low (<68%)"
        )

    with col_table:
        st.subheader("Subsystem Prioritization Ranking")
        table_cols = [
            "rank",
            "site_name",
            "category",
            "similarity_pct",
            "landing_topo_score",
            "climate_cryo_score",
            "mineralogy_score",
        ]
        renamed_table = display_df[table_cols].rename(
            columns={
                "rank": "Rank",
                "site_name": "Terrestrial Site",
                "category": "Group",
                "similarity_pct": "Global (%)",
                "landing_topo_score": "Landing/Topo (%)",
                "climate_cryo_score": "Climate/Ice (%)",
                "mineralogy_score": "Mineralogy (%)",
            }
        )
        st.dataframe(renamed_table, use_container_width=True, height=470, hide_index=True)

# === TAB 2: GEOPHYSICAL DIAGNOSIS & RADAR ===
with tab2:
    st.subheader("Multi-Dimensional Geophysical Fingerprint & Analogy Limits")
    site_options = display_df["site_name"].tolist()
    chosen_site_name = st.selectbox("Select a Terrestrial Site to Audit:", site_options, index=0)
    site_row = display_df[display_df["site_name"] == chosen_site_name].iloc[0]

    col_radar, col_diag = st.columns([1.05, 1.0])

    with col_radar:
        categories = [FEATURE_LABELS_EN[k] for k in FEATURE_KEYS]
        earth_vals = [float(site_row[f"norm_{k}"]) for k in FEATURE_KEYS]
        target_vals = [
            normalize_value(float(selected_target["physical_vector"][k]), k)
            for k in FEATURE_KEYS
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=target_vals + [target_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"Target: {selected_target['name']}",
                line_color="#FF6B35",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=earth_vals + [earth_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"Earth: {site_row['site_name']}",
                line_color="#00B4D8",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=430,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_diag:
        st.markdown(f"### {site_row['site_name']} (`{site_row['category']}`)")
        st.markdown(
            f"**Global Similarity Score:** `{site_row['similarity_pct']}%` &nbsp;|&nbsp; "
            f"**Coordinates:** `{site_row['lat']:.3f}, {site_row['lon']:.3f}`"
        )
        st.markdown(f"**Primary Geology:** {site_row['primary_geology']}")
        st.success(f"**Primary Physical Strength:** {site_row['top_strength']}")
        st.warning(f"**Largest Physical Gap:** {site_row['main_gap']}")

        st.markdown("#### Recommended Engineering Field Tests")
        for app_item in selected_target["engineering_applications"]:
            st.markdown(f"- {app_item}")

        st.markdown("#### Where the Analogy Breaks Down (Non-Replicable Space Conditions)")
        for brk_item in selected_target["analogy_breakdowns"]:
            st.markdown(f"- {brk_item}")

        # Generate downloadable English Mission Brief (.md)
        md_report = f"""# Planetary Analog Mission Brief
**Terrestrial Analog Site:** {site_row['site_name']} ({site_row['category']})
**Extraterrestrial Target:** {selected_target['name']} ({selected_target['body']})
**Mission Focus:** {selected_target['mission_focus']}
**Global Similarity Index:** {site_row['similarity_pct']}% (Rank #{site_row['rank']})

## 1. Geophysical Comparison Table
| Parameter | Earth Site Value | Space Target Value | Subsystem Score |
| :--- | :---: | :---: | :---: |
| Topographic Slope | {site_row['slope_deg']}° | {selected_target['physical_vector']['slope_deg']}° | {site_row['landing_topo_score']}% (Landing/Topo) |
| Diurnal Temp Swing (ΔT) | {site_row['diurnal_temp_range_c']}°C | {selected_target['physical_vector']['diurnal_temp_range_c']}°C | {site_row['climate_cryo_score']}% (Climate/Cryo) |
| Hyper-Aridity & Ice Index | {site_row['aridity_ice_index']} | {selected_target['physical_vector']['aridity_ice_index']} | {site_row['climate_cryo_score']}% (Climate/Cryo) |
| Basalt & Iron Oxide Index | {site_row['basalt_oxide_index']} | {selected_target['physical_vector']['basalt_oxide_index']} | {site_row['mineralogy_score']}% (Mineralogy) |
| SAR Surface Roughness | {site_row['sar_roughness_index']} | {selected_target['physical_vector']['sar_roughness_index']} | {site_row['landing_topo_score']}% (Landing/Topo) |

## 2. Official NASA & ESA Data Traceability
- **NASA Earthdata Granule ID:** `{site_row.get('nasa_granule_id', 'NASADEM_HGT')}`
- **ESA Copernicus Sentinel-1 Product:** `{site_row.get('esa_sar_product_id', 'S1_SAR')}`
- **Planetary Ground Truth Sources:** {', '.join(selected_target.get('official_nasa_sources', ['NASA PDS']))}

## 3. Recommended Field Validation Tests
{chr(10).join(['- ' + x for x in selected_target['engineering_applications']])}

## 4. Analogy Breakdowns (Simulation Chamber Required)
{chr(10).join(['- ' + x for x in selected_target['analogy_breakdowns']])}
"""
        st.download_button(
            label="📥 Download Mission Technical Brief (.md)",
            data=md_report,
            file_name=f"analog_brief_{site_row['site_id']}_{selected_target['target_id']}.md",
            mime="text/markdown",
        )

# === TAB 3: SCIENTIFIC VALIDATION ===
with tab3:
    st.subheader("Blind Benchmark Validation: Control vs. Discovery Sites")
    st.markdown(
        "Without receiving prior labels, the mathematical model independently recovers established NASA "
        "analog sites (**Control**) near the top of the ranking before surfacing underexplored **Discovery** candidates."
    )

    fig_bar = px.bar(
        ranked_all.sort_values(by="similarity_pct", ascending=True),
        x="similarity_pct",
        y="site_name",
        color="category",
        orientation="h",
        text="similarity_pct",
        color_discrete_map={"Control": "#1D4ED8", "Discovery": "#10B981"},
        labels={
            "similarity_pct": "Hybrid Similarity Score (%)",
            "site_name": "Terrestrial Location",
            "category": "Site Group",
        },
        height=520,
    )
    fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_bar.update_layout(xaxis_range=[0, 105], margin=dict(l=20, r=40, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)