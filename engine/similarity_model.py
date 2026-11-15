import numpy as np
import pandas as pd

FEATURE_KEYS = [
    "slope_deg",
    "diurnal_temp_range_c",
    "aridity_ice_index",
    "basalt_oxide_index",
    "sar_roughness_index",
]

FEATURE_BOUNDS = {
    "slope_deg": (0.0, 35.0),
    "diurnal_temp_range_c": (0.0, 145.0),
    "aridity_ice_index": (0.0, 1.0),
    "basalt_oxide_index": (0.0, 1.0),
    "sar_roughness_index": (0.0, 1.0),
}

FEATURE_LABELS_EN = {
    "slope_deg": "Topographic Slope",
    "diurnal_temp_range_c": "Diurnal Thermal Swing",
    "aridity_ice_index": "Hyper-Aridity & Cryosphere",
    "basalt_oxide_index": "Basalt & Iron Oxides",
    "sar_roughness_index": "SAR Surface Roughness",
}

FEATURE_UNITS = {
    "slope_deg": "°",
    "diurnal_temp_range_c": "°C",
    "aridity_ice_index": " idx",
    "basalt_oxide_index": " idx",
    "sar_roughness_index": " idx",
}


def normalize_value(val: float, key: str) -> float:
    low, high = FEATURE_BOUNDS[key]
    return float(np.clip((val - low) / (high - low), 0.0, 1.0))


def compute_analog_ranking(
    df: pd.DataFrame,
    target: dict,
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Projects Earth candidate sites and the extraterrestrial target into a 5D normalized
    geophysical vector space and computes a hybrid Cosine + Euclidean similarity index.
    """
    result = df.copy()
    t_phys = target["physical_vector"]

    t_norm = np.array([normalize_value(float(t_phys[k]), k) for k in FEATURE_KEYS], dtype=float)
    w_vec = np.array([float(weights.get(k, 1.0)) for k in FEATURE_KEYS], dtype=float)
    if np.linalg.norm(w_vec) == 0:
        w_vec = np.ones(len(FEATURE_KEYS), dtype=float)

    e_norm = np.zeros((len(result), len(FEATURE_KEYS)), dtype=float)
    for j, k in enumerate(FEATURE_KEYS):
        low, high = FEATURE_BOUNDS[k]
        vals = np.clip((result[k].astype(float).to_numpy() - low) / (high - low), 0.0, 1.0)
        e_norm[:, j] = vals
        result[f"norm_{k}"] = np.round(vals, 3)

    # Weighted vectors
    e_w = e_norm * w_vec
    t_w = t_norm * w_vec

    # 1. Weighted Cosine Similarity
    dot_products = e_w @ t_w
    norms_e = np.linalg.norm(e_w, axis=1)
    norm_t = float(np.linalg.norm(t_w))
    denom = norms_e * norm_t
    cos_sim = np.where(denom > 0, dot_products / denom, 0.0)

    # 2. Normalized Euclidean Proximity
    euclid_dist = np.linalg.norm(e_w - t_w, axis=1)
    euclid_sim = 1.0 - (euclid_dist / np.linalg.norm(w_vec))

    # Hybrid Global Similarity Score (%)
    global_sim = np.clip(100.0 * (0.5 * cos_sim + 0.5 * euclid_sim), 0.0, 100.0)
    result["similarity_pct"] = np.round(global_sim, 1)

    # Per-feature closeness [0, 1]
    feat_closeness = 1.0 - np.abs(e_norm - t_norm)

    # Subsystem partial scores (%)
    result["landing_topo_score"] = np.round(
        100.0 * (0.60 * feat_closeness[:, 0] + 0.40 * feat_closeness[:, 4]), 1
    )
    result["climate_cryo_score"] = np.round(
        100.0 * (0.45 * feat_closeness[:, 1] + 0.55 * feat_closeness[:, 2]), 1
    )
    result["mineralogy_score"] = np.round(100.0 * feat_closeness[:, 3], 1)

    # Automated strength and physical gap diagnostics in English
    strengths = []
    gaps = []
    for i in range(len(result)):
        best_j = int(np.argmax(feat_closeness[i]))
        worst_j = int(np.argmin(feat_closeness[i]))

        best_key = FEATURE_KEYS[best_j]
        worst_key = FEATURE_KEYS[worst_j]

        best_pct = feat_closeness[i, best_j] * 100.0
        e_raw = float(result.iloc[i][worst_key])
        t_raw = float(t_phys[worst_key])
        delta = abs(e_raw - t_raw)

        strengths.append(f"{FEATURE_LABELS_EN[best_key]} ({best_pct:.0f}% match)")
        gaps.append(f"{FEATURE_LABELS_EN[worst_key]} (Δ {delta:.2f}{FEATURE_UNITS[worst_key]})")

    result["top_strength"] = strengths
    result["main_gap"] = gaps

    result = result.sort_values(by="similarity_pct", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result