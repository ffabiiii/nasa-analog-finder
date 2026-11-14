import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

from engine.similarity_model import (
    FEATURES,
    FEATURE_LABELS,
    load_datasets,
    normalize_vector,
    compute_analog_ranking,
    generate_markdown_report,
)

st.set_page_config(
    page_title="Planetary Analog Finder | NASA Space Apps",
    page_icon="🛰️",
    layout="wide",
)

# Carga cacheada para rendimiento instantáneo en la demo
@st.cache_data
def get_data():
    return load_datasets()


df_master, targets_list = get_data()
targets_by_name = {t["name"]: t for t in targets_list}

# --- BARRA LATERAL: CONFIGURACIÓN DE MISIÓN ---
st.sidebar.title("🛰️ Control de Misión")
selected_target_name = st.sidebar.selectbox(
    "1. Selecciona Objetivo Extraterrestre:",
    options=list(targets_by_name.keys()),
)
target = targets_by_name[selected_target_name]

st.sidebar.info(f"**Cuerpo:** {target['body']}\n\n**Enfoque:** {target['mission_focus']}")

category_filter = st.sidebar.radio(
    "2. Filtrar Sitios Terrestres:",
    options=["Todos (Control + Descubrimiento)", "Solo Discovery (Nuevos Sitios)", "Solo Control (Calibración)"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Pesos por Subsistema")
st.sidebar.caption("Ajusta la prioridad según el hardware o fase de misión que deseas validar:")

dw = target["default_weights"]
custom_weights = {
    "slope_deg": st.sidebar.slider("Seguridad de Aterrizaje (Pendiente)", 0.0, 2.5, float(dw["slope_deg"]), 0.1),
    "diurnal_temp_range_c": st.sidebar.slider("Estrés Térmico (ΔT Diaria)", 0.0, 2.5, float(dw["diurnal_temp_range_c"]), 0.1),
    "aridity_ice_index": st.sidebar.slider("Hiperaridez / Hielo Subsuperficial", 0.0, 2.5, float(dw["aridity_ice_index"]), 0.1),
    "basalt_oxide_index": st.sidebar.slider("Mineralogía Basáltica / Óxidos", 0.0, 2.5, float(dw["basalt_oxide_index"]), 0.1),
    "sar_roughness_index": st.sidebar.slider("Rugosidad Radar (SAR)", 0.0, 2.5, float(dw["sar_roughness_index"]), 0.1),
}

# Ejecutar motor matemático
ranked_df = compute_analog_ranking(df_master, target, custom_weights)

if category_filter == "Solo Discovery (Nuevos Sitios)":
    display_df = ranked_df[ranked_df["category"] == "Discovery"].copy()
elif category_filter == "Solo Control (Calibración)":
    display_df = ranked_df[ranked_df["category"] == "Control"].copy()
else:
    display_df = ranked_df.copy()

# --- ENCABEZADO PRINCIPAL ---
st.title("🌍 Planetary Analog Finder: Earth-to-Moon & Mars Twin Engine")
st.markdown(
    f"Identificación sistemática de análogos terrestres mediante datos reales de **NASA POWER**, "
    f"**Copernicus DEM/Sentinel-2** y métricas **SAR** para **{target['name']}**."
)

# Métricas resumen superiores
top_overall = ranked_df.iloc[0]
top_discovery = ranked_df[ranked_df["category"] == "Discovery"].iloc[0]
top_control = ranked_df[ranked_df["category"] == "Control"].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("🥇 Mejor Análogo Global", top_overall["site_name"].split("(")[0].strip(), f"{top_overall['similarity_pct']}% similitud")
m2.metric("🔭 Top Nuevo Descubrimiento", top_discovery["site_name"].split("(")[0].strip(), f"{top_discovery['similarity_pct']}% (#{top_discovery['rank']})")
m3.metric("✅ Calibración (Control NASA)", top_control["site_name"].split("(")[0].strip(), f"{top_control['similarity_pct']}% (#{top_control['rank']})")
m4.metric("📡 Sitios Analizados", f"{len(ranked_df)} ubicaciones", "5 dimensiones físicas")

# --- PESTAÑAS DE EXPLORACIÓN ---
tab_map, tab_diag, tab_val = st.tabs(
    ["🗺️ Mapa Satelital y Ranking", "🔬 Diagnóstico Geofísico y Ruptura", "📊 Validación Científica (Benchmark)"]
)

# TAB 1: MAPA Y RANKING
with tab_map:
    col_map, col_table = st.columns([1.15, 1])

    with col_map:
        st.subheader("Distribución Global de Análogos Candidatos")
        m = folium.Map(location=[5.0, -25.0], zoom_start=2, tiles="Esri.WorldImagery")

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
                f"Categoría: {row['category']}<br>"
                f"Similitud Global: <b>{score}%</b><br>"
                f"Pendiente: {row['slope_deg']}° | ΔT: {row['diurnal_temp_range_c']}°C<br>"
                f"Geología: {row['primary_geology']}"
            )
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"#{row['rank']} {row['site_name']} ({score}%)",
                icon=folium.Icon(color=color, icon="globe", prefix="fa"),
            ).add_to(m)

        st_folium(m, width=None, height=460, returned_objects=[])
        st.caption("Leyenda: 🔵 Sitio de Control (Calibración) | 🟢 Descubrimiento ≥80% | 🟠 Descubrimiento 68–79% | 🔴 Baja similitud")

    with col_table:
        st.subheader("Tabla de Priorización por Similitud")
        cols_show = [
            "rank",
            "site_name",
            "category",
            "similarity_pct",
            "topo_landing_score",
            "thermal_aridity_score",
            "geology_score",
        ]
        st.dataframe(
            display_df[cols_show].rename(
                columns={
                    "rank": "#",
                    "site_name": "Ubicación Terrestre",
                    "category": "Tipo",
                    "similarity_pct": "Similitud (%)",
                    "topo_landing_score": "Topografía (%)",
                    "thermal_aridity_score": "Clima/Hielo (%)",
                    "geology_score": "Mineralogía (%)",
                }
            ),
            use_container_width=True,
            height=460,
            hide_index=True,
        )

# TAB 2: DIAGNÓSTICO DE RUPTURA Y COMPARACIÓN RADAR
with tab_diag:
    selected_site_name = st.selectbox(
        "Selecciona una ubicación terrestre para auditar su transferencia tecnológica:",
        options=display_df["site_name"].tolist(),
    )
    site_row = display_df[display_df["site_name"] == selected_site_name].iloc[0]

    c_radar, c_specs = st.columns([1, 1.15])

    with c_radar:
        st.subheader("Huella Geofísica Normalizada")
        target_norm = normalize_vector(target["physical_vector"])
        site_norm = site_row["norm_vector"]
        labels = [FEATURE_LABELS[f] for f in FEATURES]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=target_norm.tolist() + [target_norm[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=f"Objetivo: {target['name']}",
                line_color="#FF6B35",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=site_norm + [site_norm[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=f"Tierra: {site_row['site_name']}",
                line_color="#00B4D8",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with c_specs:
        st.subheader(f"Auditoría de Ingeniería: {site_row['site_name']}")
        s1, s2, s3 = st.columns(3)
        s1.metric("Similitud Global", f"{site_row['similarity_pct']}%", f"Rank #{site_row['rank']}")
        s2.metric("Aterrizaje / SAR", f"{site_row['topo_landing_score']}%")
        s3.metric("Clima / Hiperaridez", f"{site_row['thermal_aridity_score']}%")

        st.success(
            f"**✅ Dónde funciona la analogía ({site_row['strongest_match']}):**\n\n"
            + "\n".join([f"- {app}" for app in target["engineering_applications"]])
        )

        st.warning(
            f"**⚠️ Dónde se rompe la analogía ({site_row['weakest_match']} + Límites Planetarios):**\n\n"
            + "\n".join([f"- {brk}" for brk in target["analogy_breakdowns"]])
        )

        report_md = generate_markdown_report(site_row, target)
        st.download_button(
            label="📥 Descargar Ficha Técnica de Misión (.md)",
            data=report_md,
            file_name=f"analog_report_{site_row['site_id']}_{target['target_id']}.md",
            mime="text/markdown",
        )

# TAB 3: VALIDACIÓN CIENTÍFICA
with tab_val:
    st.subheader("Prueba de Calibración: Sitios Control vs. Nuevos Descubrimientos")
    st.markdown(
        "Para validar científicamente el algoritmo, incluimos análogos históricos reconocidos por la NASA "
        "(**Grupo Control**: Atacama, Cráter Haughton, Mauna Kea, Valles Secos de McMurdo) junto con regiones "
        "poco exploradas (**Grupo Discovery**). El modelo identifica automáticamente los sitios control en los "
        "primeros puestos y revela candidatos inéditos de alto valor operativo."
    )

    fig_bar = px.bar(
        ranked_df.sort_values("similarity_pct", ascending=True),
        x="similarity_pct",
        y="site_name",
        color="category",
        orientation="h",
        color_discrete_map={"Control": "#3B82F6", "Discovery": "#10B981"},
        labels={
            "similarity_pct": "Índice de Similitud Multidimensional (%)",
            "site_name": "Ubicación Terrestre",
            "category": "Grupo",
        },
        height=520,
    )
    st.plotly_chart(fig_bar, use_container_width=True)