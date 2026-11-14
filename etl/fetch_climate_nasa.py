import time
import requests
import pandas as pd
from pathlib import Path

INPUT_CSV = Path("data/terrestrial_candidates.csv")
OUTPUT_CSV = Path("data/climate_nasa_real.csv")

def clean_nasa_value(val):
    """NASA POWER usa -999 para datos faltantes. Lo convertimos a None."""
    if val is None or val <= -900:
        return None
    return float(val)

def fetch_site_climate(row: pd.Series) -> dict:
    """Consulta la API NASA POWER para una coordenada (lat, lon)."""
    lat = row["lat"]
    lon = row["lon"]
    site_id = row["site_id"]
    site_name = row["site_name"]

    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN"
        f"&community=RE&longitude={lon}&latitude={lat}&format=JSON"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        params = response.json()["properties"]["parameter"]

        t_max = clean_nasa_value(params["T2M_MAX"].get("ANN"))
        t_min = clean_nasa_value(params["T2M_MIN"].get("ANN"))
        precip = clean_nasa_value(params["PRECTOTCORR"].get("ANN"))
        solar = clean_nasa_value(params["ALLSKY_SFC_SW_DWN"].get("ANN"))

        # Calcular amplitud térmica diaria promedio (°C)
        diurnal_dt = round(t_max - t_min, 2) if (t_max is not None and t_min is not None) else None

        # Calcular índice normalizado de aridez / criósfera (0.0 a 1.0)
        if precip is not None:
            aridity = round(1.0 - min(precip / 5.0, 1.0), 3)
            # Si la temperatura máxima anual es bajo cero, el agua está bloqueada como hielo/permafrost
            if t_max is not None and t_max < 0:
                aridity = max(aridity, 0.92)
        else:
            aridity = None

        return {
            "site_id": site_id,
            "site_name": site_name,
            "lat": lat,
            "lon": lon,
            "temp_max_c": t_max,
            "temp_min_c": t_min,
            "diurnal_temp_range_c": diurnal_dt,
            "precip_mm_day": precip,
            "solar_rad_kwh_m2": solar,
            "aridity_ice_index": aridity
        }

    except Exception as e:
        print(f"[ERROR] Falló la extracción para {site_id} ({site_name}): {e}")
        return {
            "site_id": site_id,
            "site_name": site_name,
            "lat": lat,
            "lon": lon,
            "temp_max_c": None,
            "temp_min_c": None,
            "diurnal_temp_range_c": None,
            "precip_mm_day": None,
            "solar_rad_kwh_m2": None,
            "aridity_ice_index": None
        }

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No se encontró {INPUT_CSV}. Pide al Integrante 3 que lo suba primero.")

    candidates_df = pd.read_csv(INPUT_CSV)
    print(f"Iniciando extracción de datos reales NASA POWER para {len(candidates_df)} sitios...\n")

    results = []
    for idx, row in candidates_df.iterrows():
        print(f"[{idx + 1}/{len(candidates_df)}] Consultando {row['site_id']} - {row['site_name']}...")
        site_data = fetch_site_climate(row)
        results.append(site_data)
        # Pausa breve para no saturar el límite de peticiones de la API de la NASA
        time.sleep(0.5)

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Extracción completada. Archivo guardado en: {OUTPUT_CSV}")
    print(output_df[["site_id", "site_name", "diurnal_temp_range_c", "aridity_ice_index"]].to_string(index=False))

if __name__ == "__main__":
    main()