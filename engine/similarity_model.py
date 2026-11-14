import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "slope_deg",
    "diurnal_temp_range_c",
    "aridity_ice_index",
    "basalt_oxide_index",
    "sar_roughness_index",
]

FEATURE_LABELS = {
    "slope_deg": "Pendiente Topográfica (°)",
    "diurnal_temp_range_c": "Amplitud Térmica Diaria (°C)",
    "aridity_ice_index": "Índice de Hiperaridez / Criósfera",
    "basalt_oxide_index": "Mineralogía Basáltica y Óxidos",
    "sar_roughness_index": "Rugosidad Superficial (SAR)",
}

# Límites físicos globales para proyectar Tierra, Luna y Marte en el mismo espacio [0, 1]
PHYSICAL_BOUNDS = {
    "slope_deg": (0.0, 35.0),
    "diurnal_temp_range_c": (0.0, 145.0),
    "aridity_ice_index": (0.0, 1.0),
    "basalt_oxide_index": (0.0, 1.0),
    "sar_roughness_index": (0.0, 1.0),
}


def load_datasets(
    parquet_path: str = "data/master_analogs.parquet",
    targets_path: str = "data/extraterrestrial_targets.json",
) -> tuple[pd.DataFrame, list[dict]]:
    """Carga el dataset maestro terrestre y los objetivos extraterrestres."""
    df = pd.read_parquet(parquet_path)
    with open(targets_path, "r", encoding="utf-8") as f:
        targets = json.load(f)["targets"]
    return df, targets


def normalize_vector(raw_dict: dict) -> np.ndarray:
    """Normaliza un diccionario de parámetros físicos al rango [0.0, 1.0]."""
    vec = []
    for feat in FEATURES:
        min_v, max_v = PHYSICAL_BOUNDS[feat]
        val = float(raw_dict[feat])
        norm_val = np.clip((val - min_v) / (max_v - min_v), 0.0, 1.0)
        vec.append(norm_val)
    return np.array(vec, dtype=float)


def compute_analog_ranking(
    df: pd.DataFrame, target: dict, custom_weights: dict | None = None
) -> pd.DataFrame:
    """
    Calcula el ranking de similitud multidimensional y métricas por subsistema
    frente a un objetivo lunar o marciano.
    """
    weights_dict = custom_weights if custom_weights else target["default_weights"]
    w = np.array([float(weights_dict[f]) for f in FEATURES], dtype=float)
    if np.sum(w) == 0:
        w = np.ones(len(FEATURES), dtype=float)

    t_norm = normalize_vector(target["physical_vector"])
    t_weighted = t_norm * w
    t_norm_mag = np.linalg.norm(t_weighted)

    results = []
    for _, row in df.iterrows():
        e_norm = normalize_vector(row.to_dict())
        e_weighted = e_norm * w
        e_norm_mag = np.linalg.norm(e_weighted)

        # 1. Similitud del Coseno Ponderada (alineación direccional de perfil)
        if t_norm_mag > 0 and e_norm_mag > 0:
            cos_sim = float(np.dot(t_weighted, e_weighted) / (t_norm_mag * e_norm_mag))
        else:
            cos_sim = 0.0

        # 2. Similitud de Proximidad Euclidiana Ponderada (cercanía en magnitud real)
        max_dist = np.linalg.norm(w)
        euclid_dist = float(np.linalg.norm(t_weighted - e_weighted))
        prox_sim = max(0.0, 1.0 - (euclid_dist / max_dist))

        # Score Híbrido (50% alineación + 50% proximidad física)
        hybrid_score = round((0.5 * cos_sim + 0.5 * prox_sim) * 100.0, 2)

        # Similitudes parciales por subsistema de ingeniería (0 a 100%)
        dim_diffs = np.abs(t_norm - e_norm)
        dim_sims = (1.0 - dim_diffs) * 100.0

        topo_score = round(float(0.6 * dim_sims[0] + 0.4 * dim_sims[4]), 1)
        env_score = round(float(0.45 * dim_sims[1] + 0.55 * dim_sims[2]), 1)
        geo_score = round(float(dim_sims[3]), 1)

        best_idx = int(np.argmax(dim_sims))
        worst_idx = int(np.argmin(dim_sims))

        row_dict = row.to_dict()
        row_dict.update(
            {
                "similarity_pct": hybrid_score,
                "topo_landing_score": topo_score,
                "thermal_aridity_score": env_score,
                "geology_score": geo_score,
                "strongest_match": f"{FEATURE_LABELS[FEATURES[best_idx]]} ({dim_sims[best_idx]:.1f}%)",
                "weakest_match": f"{FEATURE_LABELS[FEATURES[worst_idx]]} ({dim_sims[worst_idx]:.1f}%)",
                "norm_vector": e_norm.tolist(),
            }
        )
        results.append(row_dict)

    ranked_df = pd.DataFrame(results).sort_values(
        by="similarity_pct", ascending=False
    ).reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1
    return ranked_df


def generate_markdown_report(site_row: pd.Series, target: dict) -> str:
    """Genera una ficha técnica descargable en Markdown para planificadores de misión."""
    apps = "\n".join([f"- {item}" for item in target["engineering_applications"]])
    breaks = "\n".join([f"- {item}" for item in target["analogy_breakdowns"]])

    return f"""# Ficha Técnica de Análogo Planetario: {site_row['site_name']}
**ID del Sitio:** `{site_row['site_id']}` | **Categoría:** {site_row['category']} | **País:** {site_row['country']}
**Coordenadas:** `{site_row['lat']}, {site_row['lon']}`
**Objetivo Extraterrestre Evaluado:** {target['name']} ({target['body']})
**Enfoque de Misión:** {target['mission_focus']}

---

## 1. Métricas de Similitud Multidimensional
- **Índice Global de Similitud:** **{site_row['similarity_pct']}%** (Ranking #{site_row['rank']})
- **Subsistema Topografía y Aterrizaje (DEM + SAR):** {site_row['topo_landing_score']}%
- **Subsistema Térmico y Criósfera/Hiperaridez:** {site_row['thermal_aridity_score']}%
- **Subsistema Mineralogía y Composición Superficial:** {site_row['geology_score']}%
- **Mayor Fortaleza Física:** {site_row['strongest_match']}
- **Mayor Brecha Terrestre:** {site_row['weakest_match']}

## 2. Comparativa de Parámetros Reales (Tierra vs. {target['body']})
| Parámetro Geofísico | {site_row['site_name']} (Tierra) | {target['name']} ({target['body']}) |
| :--- | :---: | :---: |
| Pendiente Topográfica (°) | {site_row['slope_deg']}° | {target['physical_vector']['slope_deg']}° |
| Amplitud Térmica Diaria (°C) | {site_row['diurnal_temp_range_c']} °C | {target['physical_vector']['diurnal_temp_range_c']} °C |
| Índice de Hiperaridez / Hielo | {site_row['aridity_ice_index']} | {target['physical_vector']['aridity_ice_index']} |
| Índice Basalto / Óxidos de Hierro | {site_row['basalt_oxide_index']} | {target['physical_vector']['basalt_oxide_index']} |
| Rugosidad Superficial (SAR) | {site_row['sar_roughness_index']} | {target['physical_vector']['sar_roughness_index']} |

## 3. Aplicaciones de Ingeniería Recomendadas
{apps}

## 4. Diagnóstico de Ruptura de Analogía (Limitaciones Físicas)
{breaks}
"""