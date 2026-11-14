import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

st.set_page_config(page_title="Planetary Analog Finder", layout="wide")
st.title("🛰️ Planetary Analog Finder: Earth-to-Moon & Mars Twin Engine")

# Datos de ejemplo (El Integrante 1 y 2 reemplazarán esto con master_analogs.parquet)
data = pd.DataFrame([
    {"site": "Cráter Jezero (Marte - Objetivo)", "type": "Target", "slope_deg": 3.2, "diurnal_dt": 65.0, "aridity_idx": 0.99, "basalt_idx": 0.88, "sar_roughness": 0.45},
    {"site": "Desierto de Atacama (Control)", "type": "Control", "slope_deg": 4.1, "diurnal_dt": 24.0, "aridity_idx": 0.98, "basalt_idx": 0.72, "sar_roughness": 0.40},
    {"site": "Pampa de la Joya, Perú (Descubrimiento)", "type": "Discovery", "slope_deg": 2.8, "diurnal_dt": 22.5, "aridity_idx": 0.97, "basalt_idx": 0.81, "sar_roughness": 0.43},
    {"site": "Andahua Volcanic Field, Perú (Descubrimiento)", "type": "Discovery", "slope_deg": 11.5, "diurnal_dt": 21.0, "aridity_idx": 0.82, "basalt_idx": 0.94, "sar_roughness": 0.78},
])

features = ["slope_deg", "diurnal_dt", "aridity_idx", "basalt_idx", "sar_roughness"]

st.sidebar.header("⚙️ Pesos por Subsistema de Misión")
w_slope = st.sidebar.slider("Seguridad de Aterrizaje (Pendiente)", 0.0, 2.0, 1.5)
w_dt = st.sidebar.slider("Estrés Térmico (Amplitud Térmica)", 0.0, 2.0, 1.0)
w_arid = st.sidebar.slider("Hiperaridez / Regolito Seco", 0.0, 2.0, 1.5)
w_basalt = st.sidebar.slider("Mineralogía Basáltica", 0.0, 2.0, 1.2)
w_sar = st.sidebar.slider("Rugosidad Radar (SAR)", 0.0, 2.0, 1.0)

weights = np.array([w_slope, w_dt, w_arid, w_basalt, w_sar])

scaler = MinMaxScaler()
norm_matrix = scaler.fit_transform(data[features]) * weights

target_vec = norm_matrix[0].reshape(1, -1)
scores = cosine_similarity(norm_matrix, target_vec).flatten() * 100
data["similarity_pct"] = np.round(scores, 2)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Ranking de Sitios Análogos en la Tierra")
    ranking = data[data["type"] != "Target"].sort_values(by="similarity_pct", ascending=False)
    st.dataframe(ranking[["site", "type", "similarity_pct"] + features], use_container_width=True)
    
    top_site = ranking.iloc[0]
    st.warning(
        f"**Diagnóstico de Ruptura de Analogía para {top_site['site']}:**\n\n"
        f"✅ **Alta transferencia:** Topografía de aterrizaje y química de suelo hiperárido aptas para pruebas de movilidad de rovers y perforación de regolito.\n\n"
        f"⚠️ **Dónde falla la analogía:** Gravedad terrestre ($9.81\\text{{ m/s}}^2$ vs. $3.72\\text{{ m/s}}^2$ en Marte), presión atmosférica ($1013\\text{{ hPa}}$ vs. $6.1\\text{{ hPa}}$) y ausencia de radiación cósmica galáctica."
    )

with col2:
    st.subheader("Perfil Geofísico Comparativo")
    selected_name = st.selectbox("Comparar objetivo contra:", ranking["site"].tolist())
    sel_idx = data[data["site"] == selected_name].index[0]
    
    norm_unweighted = scaler.fit_transform(data[features])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm_unweighted[0], theta=features, fill="toself", name="Cráter Jezero (Marte)"))
    fig.add_trace(go.Scatterpolar(r=norm_unweighted[sel_idx], theta=features, fill="toself", name=selected_name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)