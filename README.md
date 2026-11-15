# Terra Analog Explore 🌠

**A geophysical similarity engine powered by NASA & ESA satellite archives to discover, quantify, and audit terrestrial analog sites for Lunar and Martian exploration.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Data Sources](https://img.shields.io/badge/Data-NASA%20PDS%20%7C%20Earthdata%20%7C%20ESA%20Copernicus-0B3D91.svg)

---

## 1. Overview & Problem Statement

When space agencies select terrestrial field sites to test rovers, habitats, or spacesuits, relying purely on visual resemblance often leads to costly engineering mismatches. A desert may look like Mars in a photograph while having slopes, surface roughness, or thermal swings incompatible with the specific hardware subsystem being validated.

**Planetary Analog Finder** solves this challenge by projecting both Earth candidate locations and extraterrestrial landing targets into a unified **5-dimensional normalized geophysical vector space**. Built by a team of six Systems Engineering students for the **NASA Space Apps Challenge**, the platform enables mission planners to:
1. **Discover** underexplored high-potential analog sites (e.g., *Pampa de la Joya* and *Andahua* in Peru, or the volcanic rim of *Salar de Uyuni*) alongside established NASA calibration benchmarks (*Atacama*, *Haughton Crater*, *McMurdo Dry Valleys*, *Mauna Kea*).
2. **Customize** subsystem priority weights (landing safety, thermal swing, cryosphere/aridity, mineralogy, and radar roughness) in real time.
3. **Audit** exact satellite granule provenance from NASA Earthdata and ESA Copernicus, as well as physical **analogy breakdowns** (gravity, vacuum, radiation) where simulation chambers are required.

---

## 2. System Architecture & How Each Module Was Built

To ensure sub-second responsiveness on Streamlit Community Cloud without downloading heavy satellite imagery during live user sessions, the application was engineered using a **three-layer decoupled architecture**:

<img width="1090" height="610" alt="image" src="https://github.com/user-attachments/assets/3ea10fce-e0c3-4094-827a-dc577c9760ca" />

### Repository

```text
nasa-analog-finder/
├── data/
│   ├── terrestrial_candidates.csv      # Curated catalog of 15 Earth sites (4 Control, 11 Discovery)
│   ├── extraterrestrial_targets.json   # Planetary ground-truth vectors (Moon & Mars via NASA PDS)
│   ├── climate_nasa_real.csv           # Extracted NASA POWER climatological metrics
│   ├── terrain_sar_real.csv            # Extracted NASA GIBS/CMR & ESA Sentinel-1 metrics
│   ├── master_analogs.parquet          # Unified columnar dataset consumed by the web engine
│   └── master_analogs.csv              # Human-readable mirror of the master dataset
├── engine/
│   ├── __init__.py                     # Package initializer for cloud deployment
│   └── similarity_model.py             # 5D vector normalization, hybrid similarity & diagnostics
├── etl/
│   ├── fetch_climate_nasa.py           # Offline extractor for NASA POWER (MERRA-2 / CERES)
│   └── fetch_terrain_sar.py            # Offline extractor for NASA Earthdata, GIBS & ESA OData
├── app.py                              # Main Streamlit web application
├── requirements.txt                    # Python dependencies for Streamlit Cloud
└── .gitignore                          # Excludes virtual environments and __pycache__
```

---

## 3. Official NASA & ESA Data Sources

Every dataset and map layer in this project is sourced strictly from official space agency repositories:

| Subsystem / Layer | Official Agency Resource | Technical Endpoint / Instrument | Extracted Parameters |
| :--- | :--- | :--- | :--- |
| **Lunar Target**<br>*(Shackleton Crater Rim)* | **NASA PDS Geosciences Node**<br>*(LRO & Diviner)* | `pds-geosciences.wustl.edu/missions/lro/`<br>(`Diviner`, `LOLA`, `Mini-RF`) | Mean slope ($8.5^\circ$), extreme diurnal thermal swing ($140^\circ\text{C}$), polar cold-trap ice index ($0.98$), radar roughness ($0.72$). |
| **Martian Targets**<br>*(Jezero & Arcadia Planitia)* | **NASA Planetary Data System**<br>*(PDS)* | `pds.nasa.gov`<br>(`MOLA`, `THEMIS`, `CRISM`, `SHARAD`) | Deltaic/plain slopes ($3.4^\circ$ / $1.2^\circ$), ferric oxide & basalt index ($0.91$ in Jezero), subsurface ice indicators. |
| **Terrestrial Climatology & Cryosphere** | **NASA Earthdata / LaRC POWER** | `power.larc.nasa.gov/api/temporal/climatology/point`<br>(`MERRA-2` & `CERES`) | `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR` $\rightarrow$ computes `diurnal_temp_range_c` and `aridity_ice_index`. |
| **Granule Provenance** | **NASA Earthdata Search (CMR)** | `cmr.earthdata.nasa.gov/search/granules.json` | Queries exact `NASADEM_HGT` and `MOD11A1` granule IDs intersecting each site's coordinates. |
| **Topography, Mineralogy & Base Map** | **NASA Worldview / GIBS** | `gibs.earthdata.nasa.gov` (`WMS` & `WMTS`) | • `ASTER_GDEM_Greyscale_Shaded_Relief`: $16\times16$ pixel patches processed via `np.gradient` to compute `slope_deg` and `sar_roughness_index`.<br>• `MODIS_Terra_CorrectedReflectance_TrueColor`: Queried at local summer solstice (`2025-07-15` North / `2025-01-15` South) to avoid polar night darkness and compute `basalt_oxide_index`.<br>• `BlueMarble_ShadedRelief_Bathymetry`: Interactive Folium base map. |
| **Radar Backscatter Verification** | **ESA Copernicus Data Space** | `catalogue.dataspace.copernicus.eu/odata/v1/Products` | Verifies and logs official `SENTINEL-1` SAR scene IDs covering each terrestrial site. |

---

## 4. How the Similarity Engine Works (`engine/similarity_model.py`)

### Step 1: 5D Geophysical Vectorization
Each terrestrial site $\vec{E}$ and extraterrestrial target $\vec{T}$ is represented as a 5-dimensional vector normalized to $[0, 1]$ across physical bounds ($0^\circ\text{–}35^\circ$ for slope; $0^\circ\text{C–}145^\circ\text{C}$ for thermal swing):

$$\vec{V} = \left[\text{Slope}, \; \Delta T_{\text{diurnal}}, \; \text{Aridity/Ice}, \; \text{Basalt/Oxides}, \; \text{SAR Roughness}\right]$$

### Step 2: Mission Priority Weighting
User-defined sidebar weights $\vec{w} \in [0.0, 2.5]$ scale each physical dimension according to the hardware subsystem under test:

$$\vec{E}_w = \vec{E} \odot \vec{w}, \qquad \vec{T}_w = \vec{T} \odot \vec{w}$$

### Step 3: Hybrid Similarity Metric
To prevent false positives caused by scale-invariant collinear vectors, the engine combines **Weighted Cosine Similarity** (structural profile alignment) and **Normalized Euclidean Proximity** (absolute physical magnitude closeness) in equal parts:

$$\text{Similarity (\%)} = 100 \times \left(0.5 \cdot \frac{\vec{E}_w \cdot \vec{T}_w}{\|\vec{E}_w\| \|\vec{T}_w\|} + 0.5 \cdot \left(1 - \frac{\|\vec{E}_w - \vec{T}_w\|}{\|\vec{w}\|}\right)\right)$$

In parallel, the engine computes three independent subsystem scores (**Landing & Topography**, **Climate & Cryosphere**, and **Mineralogy**) and identifies the **Primary Physical Strength** and **Largest Physical Gap** for every candidate site.

---

## 5. Interactive Dashboard Features (`app.py`)

* **Mission Control Sidebar:** Allows the user to switch between *Shackleton Crater Rim (Moon)*, *Jezero Crater (Mars)*, and *Arcadia Planitia (Mars)*; filter between *All Sites (15)*, *Discovery Only (11)*, or *Control Only (4)*; and tune the 5 subsystem sliders in real time.
* **Tab 1 — Satellite Map / Ranking:** Displays color-coded markers over NASA Worldview's shaded relief layer (with popups showing NASA Earthdata and ESA Sentinel-1 granule IDs) alongside a dynamic multi-subsystem ranking table.
* **Tab 2 — Geophysical Diagnosis & Analogy Breakdown:** Overlays Earth vs. Target 5D fingerprints on a Plotly polar radar chart, lists recommended field validation tests, explicitly highlights non-replicable space constraints (e.g., $1.62\text{ m/s}^2$ lunar gravity, ultra-high vacuum, cosmic radiation), and exports a downloadable **Mission Technical Brief (`.md`)**.
* **Tab 3 — Scientific Validation (Benchmark):** Demonstrates that without prior labels, the model independently ranks historical NASA analog sites (*Haughton Crater*, *McMurdo Dry Valleys*, *Atacama*) at the top before surfacing new *Discovery* candidates.

---

## 6. Local Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ffabiiii/nasa-analog-finder.git](https://github.com/ffabiiii/nasa-analog-finder.git)
   cd nasa-analog-finder
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* **Re-run the official NASA & ESA ETL pipeline to refresh the master dataset:**
   ```bash
   python etl/fetch_climate_nasa.py
   python etl/fetch_terrain_sar.py
   ```

4. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```
