# 🛰️ Test Satellite Images for SatQuery AI — Small Individual Files

**1-2 images per model, all directly downloadable, no giant datasets.**

---

## 1. VQA & Scene Captioning (RS-VLM)

| # | Image | Source | Size | Download |
|---|---|---|---|---|
| 1 | **Sentinel-2 RGB GeoTIFF** (3-band, Copernicus) | GitHub: mommermi/geotiff_sample | ~1 MB | [⬇️ sample.tif](https://github.com/mommermi/geotiff_sample/raw/master/sample.tif) |
| 2 | **Landsat RGB GeoTIFF** (rasterio test data) | GitHub: rasterio/rasterio | ~500 KB | [⬇️ RGB.byte.tif](https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif) |

**Test queries:** *"What land cover types are visible?"*, *"Describe this scene"*, *"Is there water in this image?"*

---

## 2. Text-Guided Visual Grounding (Grounding DINO + SAM-RS)

| # | Image | Source | Size | Download |
|---|---|---|---|---|
| 3 | **Airport aerial image** (NWPU-RESISC45 sample) | Kaggle | ~50 KB | [⬇️ RESISC45 on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — or use the single image link below |
| 4 | **Harbor with ships** (UC Merced sample) | UC Merced Vision Lab | ~80 KB | [⬇️ UC Merced Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) |

> Since these datasets require clicking through, here are **alternative one-click GeoTIFFs with objects to ground:**

| # | Image | Source | Size | Download |
|---|---|---|---|---|
| 3a | **GeoTIFF with urban buildings** | GeoTIFF/test-data | ~2 MB | [⬇️ wind_direction.tif](https://github.com/GeoTIFF/test-data/raw/main/files/wind_direction.tif) |
| 4a | **Antarctic sea ice GeoTIFF** | GeoTIFF/test-data | ~4 MB | [⬇️ nt_20201024_f18_nrt_s.tif](https://github.com/GeoTIFF/test-data/raw/main/files/nt_20201024_f18_nrt_s.tif) |

**Test queries:** *"Find all buildings in this image"*, *"Locate the water bodies"*, *"Show me the ice boundary"*

---

## 3. Bi-Temporal Change Detection (ChangeFormer / BIT)

For change detection you need **image pairs** (t1 & t2). Here are NASA Worldview snapshots of real events:

| # | Image Pair | Source | Size | Download |
|---|---|---|---|---|
| 5a | **Kerala flood BEFORE (2018-Aug-01)** | NASA Worldview | ~2 MB | [⬇️ Open Worldview snapshot](https://worldview.earthdata.nasa.gov/?v=74.5,8.5,78,12&t=2018-08-01&l=VIIRS_SNPP_CorrectedReflectance_TrueColor) → Screenshot/Export |
| 5b | **Kerala flood AFTER (2018-Aug-22)** | NASA Worldview | ~2 MB | [⬇️ Open Worldview snapshot](https://worldview.earthdata.nasa.gov/?v=74.5,8.5,78,12&t=2018-08-22&l=VIIRS_SNPP_CorrectedReflectance_TrueColor) → Screenshot/Export |
| 6a | **Turkey earthquake BEFORE (2023-Feb-01)** | NASA Worldview | ~2 MB | [⬇️ Open Worldview snapshot](https://worldview.earthdata.nasa.gov/?v=35,36,38,38.5&t=2023-02-01&l=VIIRS_SNPP_CorrectedReflectance_TrueColor) → Screenshot/Export |
| 6b | **Turkey earthquake AFTER (2023-Feb-10)** | NASA Worldview | ~2 MB | [⬇️ Open Worldview snapshot](https://worldview.earthdata.nasa.gov/?v=35,36,38,38.5&t=2023-02-10&l=VIIRS_SNPP_CorrectedReflectance_TrueColor) → Screenshot/Export |

**Test queries:** *"What changed between these two images?"*, *"Has flooding occurred?"*, *"Describe the damage"*

---

## 4. Cross-Modal Fusion (Optical + SAR pair)

| # | Image | Source | Size | Download |
|---|---|---|---|---|
| 7 | **Sentinel-2 optical GeoTIFF** | Copernicus Browser (free account) | ~5-10 MB | [⬇️ Copernicus Browser](https://browser.dataspace.copernicus.eu/) — pick any area, select S2 L2A, download TrueColor GeoTIFF |
| 8 | **Sentinel-1 SAR GeoTIFF** (same area) | Copernicus Browser (free account) | ~5-10 MB | [⬇️ Copernicus Browser](https://browser.dataspace.copernicus.eu/) — same area, select S1 GRD IW, download VV/VH GeoTIFF |

> **Quick alternative** (no login required): Use Sentinel Hub EO Browser playground:
> [⬇️ EO Browser](https://apps.sentinel-hub.com/eo-browser/) — search any location, toggle between Sentinel-1 and Sentinel-2, and export small clips.

**Test queries:** *"Analyze flood extent using both SAR and optical"*, *"What does the SAR reveal that optical cannot?"*

---

## ⚡ Quickest 3 Downloads (No Login, Instant)

These 3 files can be downloaded right now with a single click:

```
1. https://github.com/mommermi/geotiff_sample/raw/master/sample.tif
   → Real Sentinel-2 RGB GeoTIFF (~1 MB) — Test VQA

2. https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif
   → Landsat RGB GeoTIFF (~500 KB) — Test VQA + Grounding

3. https://github.com/GeoTIFF/test-data/raw/main/files/nt_20201024_f18_nrt_s.tif
   → Antarctic sea ice GeoTIFF (~4 MB) — Test Grounding
```

Just paste any of these URLs in your browser to download instantly.

---

## 🔧 Quick Download Script (PowerShell)

Run this to download all 3 instant files into your project:

```powershell
# Download test satellite images
$dest = "d:\ProjectS\data\samples\test_external"
New-Item -ItemType Directory -Force -Path $dest

# 1. Sentinel-2 RGB (VQA testing)
Invoke-WebRequest -Uri "https://github.com/mommermi/geotiff_sample/raw/master/sample.tif" -OutFile "$dest\sentinel2_rgb_sample.tif"

# 2. Landsat RGB (VQA + Grounding testing)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif" -OutFile "$dest\landsat_rgb_sample.tif"

# 3. Sea Ice GeoTIFF (Grounding testing)
Invoke-WebRequest -Uri "https://github.com/GeoTIFF/test-data/raw/main/files/nt_20201024_f18_nrt_s.tif" -OutFile "$dest\seaice_sample.tif"

Write-Host "Downloaded 3 test satellite images to $dest"
```