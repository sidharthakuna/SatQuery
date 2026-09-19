# SatQuery AI — Sensor Data & Benchmark Reference Guide
**Smart India Hackathon (SIH) | Problem Statement ID: 26167**  
**Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)  

---

## 📌 Purpose of this Guide
Remote sensing models require precise handling of physical sensor channels, wavelengths, and radar polarizations. Mixing up bands or failing to calibrate microwave radar values will invalidate evaluation metrics. This guide serves as the definitive reference for all datasets, sensor band maps, and calibration equations used in **SatQuery AI**.

---

## 🛰️ 1. Sensor Band Reference & Spatial Resolutions

### 1.1 Cartosat-2S (ISRO Optical Satellite)
* **Mission:** Very High-Resolution Optical Earth Observation.
* **Bit Depth:** 10-bit or 12-bit Digital Numbers (DN: $0 \text{ to } 1023$ or $0 \text{ to } 4095$).
* **Spatial Resolution (GSD):**
  * Panchromatic (PAN): $\sim 0.65\text{ m}$
  * 4-Band Multispectral (MX): $\sim 1.6\text{ m}$

| Band Index | Band Name | Wavelength ($\mu\text{m}$) | Description & Application |
| :--- | :--- | :--- | :--- |
| **Band 1** | Blue (B2) | 0.45 – 0.52 | Coastal water penetration, soil mapping |
| **Band 2** | Green (B3) | 0.52 – 0.59 | Vegetation vigor, green reflectance |
| **Band 3** | Red (B4) | 0.62 – 0.68 | Chlorophyll absorption, urban infrastructure |
| **Band 4** | Near-Infrared (B5) | 0.77 – 0.86 | Biomass detection, sharp water/land boundary |

---

### 1.2 RISAT-1 / EOS-04 (ISRO Radar Imaging Satellite)
* **Mission:** Active C-band Synthetic Aperture Radar (SAR) operating at **5.35 GHz** ($\lambda \approx 5.6\text{ cm}$).
* **Key Advantage:** Day-and-night imaging; pierces through thick monsoon clouds, haze, and smoke.
* **Operating Modes:** Fine Resolution Stripmap (FRS-1: $\sim 3\text{ m}$), Medium Resolution ScanSAR (MRS: $\sim 25\text{ m}$).
* **Polarizations:**
  * Linear: **HH** (Horizontal Tx/Rx), **HV** (Horizontal Tx, Vertical Rx), **VV**, **VH**.
  * Hybrid/Circular: **RH, RV** (Right-circular transmit, Horizontal/Vertical receive).

---

### 1.3 Sentinel-2 (Copernicus Optical Reference)
Used during domain adaptation with `BigEarthNet.txt`:

| Band | Name | Central Wavelength | GSD (Resolution) | Primary Use |
| :--- | :--- | :--- | :--- | :--- |
| **B02** | Blue | 490 nm | 10 m | Soil, bathymetry |
| **B03** | Green | 560 nm | 10 m | Vegetation health |
| **B04** | Red | 665 nm | 10 m | Chlorophyll absorption |
| **B08** | NIR | 842 nm | 10 m | Leaf canopy, water boundaries |
| **B11** | SWIR-1 | 1610 nm | 20 m | Moisture, snow/cloud discrimination |
| **B12** | SWIR-2 | 2190 nm | 20 m | Geological & burn scar mapping |

---

### 1.4 Sentinel-1 (Copernicus C-band SAR Reference)
* **Frequency:** 5.405 GHz (C-band).
* **Bands:** 
  * Band 1: **VV** (Vertical transmit, Vertical receive) — Excellent for water and rough surfaces.
  * Band 2: **VH** (Vertical transmit, Horizontal receive) — Excellent for volume scattering (forests, crops).

---

## 🔬 2. Sensor Radiometric Calibration Equations

### 2.1 Optical Normalization (Percentile Stretch)
Because 12-bit/16-bit satellite images contain extreme outliers from clouds and reflective roofs, simple min-max scaling washes out the image. We apply a **2%–98% percentile linear stretch**:

$$\text{Norm}(x) = \text{clip}\left(\frac{x - P_2}{P_{98} - P_2}, 0.0, 1.0\right)$$

*Where $P_2$ and $P_{98}$ are the 2nd and 98th percentiles of valid non-zero pixels.*

### 2.2 SAR Backscatter Calibration (dB Conversion)
SAR Digital Numbers ($DN$) represent raw amplitude. To convert to physical backscatter intensity $\sigma^0$ in Decibels (dB):

$$\sigma^0 (\text{dB}) = 10 \cdot \log_{10}(\text{DN}^2 + \epsilon) - K_{\text{cal}}$$

* Where $K_{\text{cal}}$ is the calibration constant found in the RISAT/Sentinel metadata XML.
* Normalized to $[0.0, 1.0]$ using standard terrestrial bounds $[-25.0\text{ dB}, 0.0\text{ dB}]$:

$$\sigma^0_{\text{norm}} = \text{clip}\left(\frac{\sigma^0 (\text{dB}) - (-25.0)}{0.0 - (-25.0)}, 0.0, 1.0\right)$$

---

## 📂 3. Prescribed Benchmark Datasets

### A. `BigEarthNet.txt` (Mandatory Domain Adaptation)
* **Paper:** *BigEarthNet.txt: A Multi-Modal Dataset for Remote Sensing Image-Text Representation Learning* ([arXiv: 2603.29630](https://arxiv.org/abs/2603.29630)).
* **Contents:** Paired Sentinel-1 SAR + Sentinel-2 Optical tiles with multi-label land-cover captions.
* **Usage in SatQuery AI:** Used for LoRA fine-tuning of our vision-language backbone to teach the model radar physics and multi-spectral vocabulary.

### B. `VRSBench` (Visual Remote Sensing Benchmark)
* **Purpose:** Single-image remote sensing benchmark.
* **Evaluates:**
  1. Detailed Scene Captioning.
  2. Text-Guided Visual Grounding (bounding boxes).
  3. Visual Question Answering (VQA).

### C. `RSVQA` (Remote Sensing Visual Question Answering)
* **Variants:** RSVQA-LR (Low Resolution Sentinel-2) and RSVQA-HR (High Resolution aerial imagery).
* **Evaluates:** Single-image count, presence, comparison, and area queries.

### D. `CDVQA` (Change Detection Visual Question Answering)
* **Purpose:** Bi-temporal change question answering over paired dates ($t_1, t_2$).
* **Evaluates:** Questions like *"What replaced the vegetation in the northern zone?"* and *"Has the industrial zone expanded?"*

---

## 🗂️ 4. Recommended Local Directory Structure

Store your sample and evaluation files in the following layout:

```text
data/
├── samples/
│   ├── optical/
│   │   ├── cartosat_sample_01.tif      # 4-band Cartosat-2S MX
│   │   └── sentinel2_sample_01.tif     # 4-band (RGB+NIR) Sentinel-2
│   ├── sar/
│   │   ├── risat_sample_01.tif         # 2-band (VV/VH) RISAT C-band SAR
│   │   └── sentinel1_sample_01.tif     # 2-band (VV/VH) Sentinel-1 SAR
│   ├── cross_modal_pairs/
│   │   ├── pair_01_optical.tif         # Co-registered optical tile
│   │   └── pair_01_sar.tif             # Co-registered SAR tile (cloud penetrating)
│   └── bitemporal_pairs/
│       ├── change_01_t1_2021.tif       # Date 1 GeoTIFF
│       └── change_01_t2_2024.tif       # Date 2 GeoTIFF
│
└── benchmarks/
    ├── vrsbench_test_split.json
    ├── rsvqa_test_split.json
    └── cdvqa_test_split.json
```

---

## 🧪 5. Sample GeoTIFF Verification Script

Run this script to verify that any candidate GeoTIFF contains valid spatial metadata and band counts:

```python
import rasterio
import numpy as np
import sys

def verify_geotiff(file_path: str):
    print(f"--> Inspecting: {file_path}")
    with rasterio.open(file_path) as src:
        print(f"    Dimensions: {src.width} x {src.height}")
        print(f"    Band Count: {src.count}")
        print(f"    Data Types: {src.dtypes}")
        print(f"    CRS:        {src.crs}")
        print(f"    Bounds:     {src.bounds}")
        print(f"    Resolution: {src.res}")

        # Test read band 1
        b1 = src.read(1)
        valid = b1[b1 > 0]
        if len(valid) > 0:
            print(f"    Band 1 Min/Max: {valid.min()} / {valid.max()}")
            print(f"    Band 1 2%/98%:  {np.percentile(valid, (2, 98))}")
            print("    Status: [PASSED] GeoTIFF is well-formed!\n")
        else:
            print("    Status: [WARNING] Band contains only zero / NoData pixels!\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_geotiff(sys.argv[1])
    else:
        print("Usage: python verify_geotiff.py <path_to_geotiff>")
```
