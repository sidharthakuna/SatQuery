# SatQuery AI (ProjectS) — Complete System Architecture & Engineering Manual
**Smart India Hackathon (SIH) | Problem Statement ID: 26167**  
**Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)  
**Theme:** Space Technology | **Category:** Software  
**Repository:** `ProjectS` | **Current Release:** `v0.2.0`  

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement Analysis](#1-executive-summary--problem-statement-analysis)
2. [Input Modalities, Formats & Geospatial Constraints](#2-input-modalities-formats--geospatial-constraints)
3. [Sensor Physics & Calibration Mathematics](#3-sensor-physics--calibration-mathematics)
4. [The 4 Specialist AI Models Suite](#4-the-4-specialist-ai-models-suite)
5. [The 7 Strategic Backend Engineering Innovations](#5-the-7-strategic-backend-engineering-innovations)
6. [Agentic Orchestrator & State Machine](#6-agentic-orchestrator--state-machine)
7. [Geospatial Processing Engine](#7-geospatial-processing-engine)
8. [System Architecture & UML Diagrams](#8-system-architecture--uml-diagrams)
9. [Remote Sensing Grounded VQA Studio & 10 Archetypes](#9-remote-sensing-grounded-vqa-studio--10-archetypes)
10. [Complete Codebase Directory Structure & File Inventory](#10-complete-codebase-directory-structure--file-inventory)
11. [API Contracts, Schemas & Endpoints](#11-api-contracts-schemas--endpoints)
12. [Frontend User Interface & UX Architecture](#12-frontend-user-interface--ux-architecture)
13. [Benchmark Targets & SIH Jury Defense Strategy](#13-benchmark-targets--sih-jury-defense-strategy)
14. [Version Roadmap & Implementation Status](#14-version-roadmap--implementation-status)
15. [Deployment, Environment Setup & Quickstart](#15-deployment-environment-setup--quickstart)

---

# 1. Executive Summary & Problem Statement Analysis

### 1.1 The Core Problem
Remote sensing imagery acquired by Earth Observation (EO) satellites is critical for disaster management, agricultural yield estimation, urban sprawl tracking, maritime surveillance, and border monitoring. However, contemporary EO AI solutions operate in isolated silos:
* Specialized segmentation models exist solely for building footprints.
* Separate classifiers exist solely for crop types.
* Independent change detection scripts require manual pre-processing and tuning.

Non-expert stakeholders (disaster response teams, municipal officers, civil engineers, district collectors) cannot utilize these tools because interacting with them requires in-depth mastery of:
* Geographic Information Systems (GIS) software (QGIS, ArcGIS).
* Coordinate Reference Systems (CRS: UTM, EPSG:4326, EPSG:3857) and reprojections.
* Satellite band mathematics (NIR, RedEdge, SWIR, thermal channels).
* Deep learning hyperparameter tuning.

### 1.2 Why Generic Vision-Language Models (VLMs) Fail
Generic foundational VLMs (such as GPT-4o, Claude 3.5 Sonnet, Google Gemini, or vanilla LLaVA) are fundamentally unsuitable for satellite imagery due to three critical scientific barriers:
1. **Data Formats & Radiometric Bit Depth:**
   * Commercial and web-scale models assume standard 8-bit, 3-channel RGB (JPEG/PNG) representations.
   * Remote sensing satellites generate 10-bit, 12-bit, or 16-bit multi-spectral GeoTIFFs across dozens of discrete spectral wavelengths (e.g., Cartosat-2S, Sentinel-2). Compressing these into 8-bit RGB strips away crucial non-visible bands like Near-Infrared (NIR) and Short-Wave Infrared (SWIR) that indicate chlorophyll absorption and surface moisture.
2. **Microwave SAR Physics & Speckle Noise:**
   * Synthetic Aperture Radar (SAR) sensors (such as ISRO's RISAT-1/EOS-04 and Copernicus Sentinel-1) capture electromagnetic microwave backscatter intensity ($\sigma^0$) and phase across linear (HH, HV, VV, VH) or circular polarizations.
   * Radar measures surface roughness, dielectric permittivity, and geometric moisture—not optical reflectance. Inherent coherent interference creates granular **speckle noise**. Generic VLMs mistake speckle noise for visual texture and hallucinate non-existent features.
3. **Domain Terminology & Scale Ambiguity:**
   * Satellite imagery lacks standard perspective cues (e.g., horizon, human scale). Ground Sampling Distance (GSD) ranges from sub-meter ($0.65\text{ m}$ for Cartosat PAN) to hectometric ($10\text{ m} - 60\text{ m}$ for Sentinel). Generic models suffer severe scale disorientation and hallucinate land-use classifications.

### 1.3 The Solution: SatQuery AI (ProjectS)
SatQuery AI solves this challenge through an **Agentic, Query-Driven Remote Sensing Assistant**. Rather than relying on a single monolithic, black-box VLM, SatQuery AI implements a deterministic, multi-stage state machine that:
* Ingests native, high-bit-depth GeoTIFF rasters without data loss.
* Automatically verifies radiometric calibration, CRS alignment, and sensor compatibility.
* Classifies natural-language user queries into deterministic intent categories.
* Dispatches requests to fine-tuned, specialized remote-sensing neural networks.
* Fuses text and spatial outputs (polygonal GeoJSON vectors, bounding boxes, change masks).
* Emits a 100% transparent, auditable **Execution Trace** detailing selected tools, parameter values, latencies, and confidence scores.

---

# 2. Input Modalities, Formats & Geospatial Constraints

SatQuery AI is engineered to address the three canonical operational scenarios defined by ISRO / SAC:

### 2.1 Scenario 1: Single Image Understanding
* **Input:** A single optical/multispectral raster or a single SAR raster.
* **Mandatory Capability:** Remote Sensing Visual Question Answering (**RS-VQA**). The system accepts queries such as *"How many naval vessels are berthed along the southern pier?"* or *"What is the primary land-use category visible in this sector?"*.
* **Secondary Capability:** **Visual Grounding** and Scene Captioning. Queries like *"Highlight the active runways and taxiways"* produce localized bounding boxes $[x_1, y_1, x_2, y_2]$ and segmentation masks.

### 2.2 Scenario 2: Cross-Modal Sensor Fusion (Optical + SAR)
* **Input:** A co-registered image pair consisting of one Optical/Multispectral raster and one SAR raster covering the identical geographic bounding box at approximately the same timestamp.
* **Operational Purpose:** All-weather, cloud-penetrating surveillance. Optical sensors provide fine spectral classification but are rendered blind by monsoon clouds, smoke, or darkness. SAR microwaves pierce through cloud decks and atmospheric disturbances.
* **Capability:** The cross-modal fusion engine merges the high spectral discrimination of optical bands with the cloud-penetrating structural backscatter of SAR (e.g., detecting flooded riverbanks or camouflaged roads hidden beneath thick cloud cover).

### 2.3 Scenario 3: Bi-Temporal Change Analysis ($t_1 \text{ vs } t_2$)
* **Input:** Two spatially aligned images of the same geographic area acquired at two different dates ($t_1$ and $t_2$).
* **Mandatory Capability:** Change Detection VQA (**CD-VQA**) and Spatial Change Mapping. Queries such as *"Has deforestation occurred along the river basin between 2021 and 2024?"* or *"Quantify urban expansion in hectares"* yield:
  1. A natural language assessment of what changed, where, and by how much.
  2. A pixel-precise binary or multi-class change mask.
  3. Computed metric areas in hectares and percentage of total scene shifted.

---

# 3. Sensor Physics & Calibration Mathematics

### 3.1 Cartosat-2S (Optical Earth Observation)
* **Payload:** Panchromatic (PAN) and 4-Band Multispectral (MX) camera.
* **Radiometric Bit Depth:** 10-bit or 12-bit Digital Numbers (DN: $0 \text{ to } 1023$ or $0 \text{ to } 4095$).
* **Spatial Resolution:** PAN $\sim 0.65\text{ m}$ GSD; MX $\sim 1.6\text{ m}$ GSD.
* **Band Allocation:**
  * Band 1: Blue ($0.45 - 0.52\ \mu\text{m}$) — Coastal water penetration and soil differentiation.
  * Band 2: Green ($0.52 - 0.59\ \mu\text{m}$) — Peak green reflectance, vegetation vigor.
  * Band 3: Red ($0.62 - 0.68\ \mu\text{m}$) — Chlorophyll absorption, built-up urban infrastructure.
  * Band 4: Near-Infrared (NIR) ($0.77 - 0.86\ \mu\text{m}$) — Strong vegetation canopy reflectance, water-land boundary delimitation.
* **Radiometric Normalization (Percentile Linear Stretch):**
  Satellites capture extreme radiometric outliers (e.g., glint from metal roofs, cloud tops). Simple min-max scaling washes out ground details. SatQuery AI applies a **2%–98% percentile linear stretch**:
  $$\text{Norm}(x) = \text{clip}\left(\frac{x - P_2}{P_{98} - P_2}, 0.0, 1.0\right)$$
  *(where $P_2$ and $P_{98}$ are computed exclusively over non-zero, valid raster pixels).*

### 3.2 RISAT-1 / EOS-04 (C-Band Synthetic Aperture Radar)
* **Frequency:** C-band active microwave radar at **5.35 GHz** ($\lambda \approx 5.6\text{ cm}$).
* **Operating Modes:** Fine Resolution Stripmap (FRS-1: $\sim 3\text{ m}$), Medium Resolution ScanSAR (MRS: $\sim 25\text{ m}$).
* **Polarizations:** Linear (HH, HV, VV, VH) and Hybrid Circular (RH, RV).
* **Radiometric Calibration to Sigma-Nought ($\sigma^0$ dB):**
  SAR Digital Numbers ($DN$) represent raw received amplitude. To extract physically meaningful backscatter intensity $\sigma^0$:
  $$\sigma^0 (\text{dB}) = 10 \cdot \log_{10}(\text{DN}^2 + \epsilon) - K_{\text{cal}}$$
  *(where $K_{\text{cal}}$ is the sensor calibration constant from RISAT CEOS/GeoTIFF metadata).*
* **Normalization Scale:** Terrestrial backscatter values span $[-25.0\text{ dB}, 0.0\text{ dB}]$. They are normalized to $[0.0, 1.0]$:
  $$\sigma^0_{\text{norm}} = \text{clip}\left(\frac{\sigma^0 (\text{dB}) - (-25.0)}{0.0 - (-25.0)}, 0.0, 1.0\right)$$
* **Radar Physical Interaction:**
  * **Smooth Water / Runways:** Specular reflection directs pulses away from the receiver $\to$ backscatter is extremely low ($\approx -22\text{ dB}$, rendering pitch black).
  * **Built-up Urban Areas:** Corner reflector effect (double-bounce between ground and vertical walls) directs intense energy back $\to$ very high backscatter ($\approx 0\text{ to } -5\text{ dB}$, rendering brilliant white).
  * **Dense Forest Canopies:** Volume scattering inside leaves and branches creates intermediate diffuse backscatter ($\approx -12\text{ to } -15\text{ dB}$).

### 3.3 Copernicus Reference Constellation
* **Sentinel-2 Optical Bands:**
  * B02 (Blue, 490 nm, 10 m), B03 (Green, 560 nm, 10 m), B04 (Red, 665 nm, 10 m).
  * B08 (NIR, 842 nm, 10 m), B11 (SWIR-1, 1610 nm, 20 m), B12 (SWIR-2, 2190 nm, 20 m).
  * Scaled to Top-Of-Atmosphere (TOA) reflectance by dividing by $10,000$.
* **Sentinel-1 SAR:**
  * 5.405 GHz, dual-polarization (VV + VH). Calibrated to decibels and clipped to $[-25\text{ dB}, 0\text{ dB}]$.

---

# 4. The 4 Specialist AI Models Suite

SatQuery AI replaces single-model failure modes with a modular consortium of 4 fine-tuned neural models:

| Model Specialist | Primary Architecture | Adaptation Dataset / Weights | Primary Output |
| :--- | :--- | :--- | :--- |
| **1. RS-VLM / VQA Specialist** | RS-VLM Transformer + Hybrid Grounded Segmenter | `rs_vlm.pt` (173 domain tokens, trained on 10 EO archetypes) | Authoritative natural-language answer + visual raster overlay mask + metrics |
| **2. Visual Grounding Specialist** | Grounding DINO + SAM-RS (Segment Anything RS) | VRSBench Grounding Split + Pretrained SAM | Normalized pixel bounding boxes $[x_1, y_1, x_2, y_2]$ & binary feature masks |
| **3. Bi-Temporal Change Specialist** | ChangeFormer-V6 / BIT Siamese Transformer | CDVQA / LEVIR-CD+ / WHU-CD | Pixel-level binary difference mask ($t_1 \to t_2$) & change statistics |
| **4. Optical–SAR Cross-Modal Fusion** | Dual-Branch Cross-Attention Network | `BigEarthNet.txt` Paired Multimodal Split | Fused high-confidence land-use / flood delineation under cloud cover |

### 4.1 Specialist 1: RS-VLM Multimodal Architecture & Training
* **Architecture:** `RS_VLM_Transformer` with cross-modal projection layer:
  * **Visual Encoder:** ResNet/ViT feature extractor reducing satellite rasters into 512-dimensional visual tokens.
  * **Language Decoder:** Multi-head self-attention transformer operating over an expanded **173-token Remote Sensing Vocabulary** (`RS_VLM_VOCAB`).
  * **Vocabulary Categories:** Urban morphology (`building`, `residential`, `industrial`), transportation corridors (`highway`, `runway`, `railway`), hydrological features (`estuary`, `basin`, `ndwi`), maritime assets (`vessel`, `berth`, `anchorage`), agricultural parameters (`cultivated`, `irrigation`), and temporal dynamics (`expansion`, `demolished`).
* **Model Checkpoint (`backend/data/checkpoints/rs_vlm.pt`):**
  * **Model Size:** 25.5 MB (6.67M parameters).
  * **Training Protocol:** 20 epochs on CPU/CUDA, Cross-Entropy Loss with Adam optimizer ($\text{lr} = 5\times 10^{-4}$).
  * **Convergence Metrics:** Initial loss `4.1042` $\to$ Final best loss **`0.7019`**, Model Perplexity **`2.02`**.
* **Hybrid Grounded Output:** Unlike text-only LLMs that merely answer with words, the RS-VLM pipeline pairs every textual finding with a **spatial raster overlay** (`VQAGrounding`), classifying the exact method (e.g., `RS-VLM + Segmentation`, `RS-VLM + Water Index (NDWI)`), calibrated confidence score (e.g., 87%), operational notes, and color-coded overlay mask.

---

# 5. The 7 Strategic Backend Engineering Innovations

To meet the rigorous standards of ISRO / SAC and prevent runtime crashes, SatQuery AI incorporates 7 purpose-built engineering subsystems:

### 1. Sliding-Window Chip Inferencer (`app/core/geospatial/tiler.py`)
* **Problem:** Remote sensing GeoTIFFs regularly exceed $12,000 \times 12,000$ pixels ($>1.5\text{ GB}$ uncompressed). Passing these directly into deep learning models causes immediate CUDA Out-Of-Memory (OOM) errors.
* **Solution:** A windowed chip inferencer reading $512 \times 512$ chips with an adjustable stride (default: $384$ px, creating a 128 px overlap).
* **Gaussian Edge Blending:** Overlapping boundary seams are blended using 2D sinusoidal/Gaussian weighting:
  $$W(y, x) = \sin\left(\frac{\pi y}{H_{\text{chip}}}\right) \cdot \sin\left(\frac{\pi x}{W_{\text{chip}}}\right)$$
  This eliminates hard border artifacts and tiling seams across the reconstructed gigapixel canvas.

### 2. Sub-Pixel Co-Registration Refiner (`app/core/geospatial/coregistration.py`)
* **Problem:** Bi-temporal acquisitions from different satellite passes suffer from minute orbital drifts, terrain relief displacements, and tilt angle differences. Even a 2-pixel misalignment generates massive false-positive change detections along every road and building boundary.
* **Solution:** Automated 2D Phase Correlation on normalized gradient images. Computes cross-power spectrum between $t_1$ and $t_2$:
  $$R = \frac{F_1(\xi, \eta) \cdot F_2^*(\xi, \eta)}{|F_1(\xi, \eta) \cdot F_2^*(\xi, \eta)|}$$
  Finding the impulse response peak determines the exact sub-pixel horizontal ($\Delta x$) and vertical ($\Delta y$) translation vectors, automatically warping images into sub-pixel alignment prior to change inference.

### 3. Dynamic GPU VRAM Manager (`app/inference/vram_manager.py`)
* **Problem:** Loading four massive PyTorch vision-language models simultaneously requires $>32\text{ GB}$ of VRAM, exceeding standard workstation GPUs (RTX 3060/4060/4080 with 8GB–16GB VRAM).
* **Solution:** Least-Recently-Used (LRU) model offloader. Only the model active for the current step is resident in GPU VRAM. Inactive models are offloaded to system RAM, followed by immediate calls to `torch.cuda.empty_cache()` and `gc.collect()`.

### 4. High-Speed GeoJSON Vectorizer (`app/core/geospatial/vectorization.py`)
* **Problem:** Sending raw $20\text{ MB}$ raster change masks over HTTP/WebSocket slows down browser map rendering and causes UI stuttering.
* **Solution:** Converts binary raster masks into vector polygons using `rasterio.features.shapes`, simplifies coordinates via the **Douglas-Peucker algorithm** (tolerance: 1.5 pixels), and packages geometries into standard GeoJSON FeatureCollections under $<30\text{ KB}$.

### 5. Dual-Mode Mock/CUDA Engine (`app/inference/factory.py`)
* **Problem:** Developers, CI/CD pipelines, and frontend designers often lack dedicated NVIDIA CUDA GPUs.
* **Solution:** Configurable via `.env`:
  * `INFERENCE_MODE=MOCK`: Generates synthetic domain-grounded responses, realistic bounding boxes, and simulated change masks instantly on CPU with zero heavy dependencies.
  * `INFERENCE_MODE=CUDA`: Loads full PyTorch backbones with GPU acceleration.

### 6. WebSocket Live Execution Streaming (`app/api/v1/endpoints_stream.py`)
* **Benefit:** Instead of blocking HTTP requests during lengthy gigapixel inference runs, the client connects to `ws://host:8000/api/v1/stream/query`. The orchestrator pushes progressive status packets: `INGESTION` $\to$ `VALIDATION` $\to$ `CLASSIFICATION` $\to$ `EXECUTING_TOOL` $\to$ `FUSING_OUTPUTS` $\to$ `FINAL_RESULT`.

### 7. Executive Mission PDF Report Generator (`app/utils/pdf_generator.py`)
* **Benefit:** Mission operators can export a self-contained, publication-quality executive briefing document with a single click. Built with `reportlab`, it compiles query parameters, thumbnail imagery, overlay change masks, detected features, coordinate bounds, and the cryptographic audit trail into a formal ISRO-styled PDF.

---

# 6. Agentic Orchestrator & State Machine

The Agentic Orchestrator is the central brain of SatQuery AI. Unlike unconstrained LLM loops that hallucinate tool names, SatQuery AI employs a **Deterministic Finite State Machine**.

```
  [User Query + GeoTIFF Images]
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 1. Ingestion & Pre-loading   │
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 2. Input Compatibility Guard │ ── (Incompatible CRS/Format) ──► [Aborted / Error]
  └──────────────┬───────────────┘
                 │ (Passed)
                 ▼
  ┌──────────────────────────────┐
  │ 3. Intent & Task Router      │ ──► [SINGLE_VQA | SINGLE_GROUNDING |
  └──────────────┬───────────────┘     BITEMPORAL_CHANGE | CROSS_MODAL_FUSION]
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 4. Tool Registry Dispatch    │ ──► [Execute Model Specialist]
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 5. Spatial & Textual Fusion  │ ──► [Merge Masks, BBoxes & Text Answers]
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 6. Audit Trace Compilation   │ ──► [Compile Immutable ExecutionTrace JSON]
  └──────────────┬───────────────┘
                 │
                 ▼
  [SatQueryResult Response Delivered to Client]
```

### 6.1 State Machine Stages
1. **Input Ingestion:** Ingests 1 or 2 uploaded GeoTIFF IDs, opening file descriptors and verifying raster dimensions.
2. **Input Compatibility Guard (`guard.py`):**
   * Inspects metadata (CRS, band count, height, width).
   * Enforces cross-validation: Single images must be Optical or SAR; Two images must either be identical CRS and dimensions for Bi-temporal change, or complementary Optical+SAR pairs for cross-modal fusion.
3. **Intent Classification & Routing (`router.py`):**
   * Employs heuristic regex rules + keyword embeddings:
     * 2 images (Optical + SAR) $\to$ `CROSS_MODAL_FUSION`.
     * 2 images (Temporal $t_1 + t_2$) $\to$ `BITEMPORAL_CHANGE`.
     * 1 image + spatial keywords (*"where", "highlight", "box", "locate", "outline"*) $\to$ `SINGLE_GROUNDING`.
     * 1 image + general inquiry $\to$ `SINGLE_VQA`.
4. **Tool Execution (`registry.py`):**
   * Calls registered specialist tools decorated with `@register_tool`.
   * Standardizes tool input/output contracts via `ToolInput` and `ToolOutput` objects.
5. **Output Fusion (`agent.py`):**
   * Synthesizes textual findings into natural language.
   * Converts raster masks to saved PNG overlays in `/static/uploads/`.
   * Quantifies spatial metrics (hectares of change, percentage change).
6. **Audit Trace Generation (`tracer.py`):**
   * Packages timestamps, latency in milliseconds, individual pipeline steps, tool names, parameters, and confidence scores into an immutable `ExecutionTrace` structure.

---

# 7. Geospatial Processing Engine

The geospatial engine (`app/core/geospatial/`) bridges pure machine learning and rigorous satellite geodesy:

### 7.1 Component Breakdown
* **`reader.py` (`GeospatialReader`):**
  * Opens multi-band rasters using `rasterio`.
  * Extracts affine transformations, bounding boxes, EPSG CRS, GSD resolution, and band data types.
  * Projects native coordinates to standard WGS84 (`EPSG:4326`) latitude/longitude bounding boxes using `pyproj.Transformer`.
  * Generates optimized 8-bit RGB preview thumbnails for browser display.
* **`calibration.py` (`SensorCalibrator`):**
  * Implements `normalize_optical()` (2%–98% percentile stretch for multi-spectral DNs).
  * Implements `calibrate_sar_db()` (logarithmic conversion of radar power to calibrated $\sigma^0$ dB).
  * Includes automated speckle noise mitigation via spatial filtering.
* **`coregistration.py` (`CoRegistrationEngine`):**
  * Evaluates raster alignment between two acquisition dates or modalities.
  * Performs re-projection warping (`rasterio.warp.reproject`) using bilinear or cubic resampling onto matching spatial grids.
  * Uses phase-correlation peak analysis to eliminate sub-pixel spatial shear.
* **`tiler.py` (`SlidingWindowChipInferencer`):**
  * Tiles gigapixel rasters into arbitrary $512 \times 512$ chips.
  * Reconstructs full-resolution prediction maps with weighted edge-blending matrices.
* **`vectorization.py` (`GeoJSONVectorizer`):**
  * Runs raster contour extraction.
  * Applies the Douglas-Peucker polygon simplification algorithm to keep GeoJSON payloads under bandwidth limits ($<30\text{ KB}$) for smooth rendering in MapLibre.

---

# 8. System Architecture & UML Diagrams

### 8.1 System Architecture Diagram
```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      FRONTEND (React 19 + MapLibre)                    │
 │         Dual-Pane Swipe Compare  •  GeoJSON Overlays  •  Audit UI      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ REST / WebSockets
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      BACKEND GATEWAY (FastAPI)                         │
 │     Multi-part GeoTIFF Upload  •  Task Streaming  •  PDF Briefings     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 AGENTIC ORCHESTRATOR & STATE MACHINE                   │
 │     1. Metadata Guard   2. Intent Classifier   3. Tool Registry        │
 │     4. Param Validator  5. Spatial Fusion      6. Audit Tracer         │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
 ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
 │       SPECIALIST AI SUITE       │   │     GEOSPATIAL DATA ENGINE      │
 │ • RS-VLM (GeoChat + LoRA)       │   │ • GDAL & Rasterio COG Reader    │
 │ • Grounding DINO + SAM-RS       │   │ • Cartosat & RISAT Calibrators  │
 │ • ChangeFormer-V6 Engine        │   │ • Sub-Pixel Co-registration     │
 │ • Optical-SAR Cross-Attention   │   │ • Sliding Window Chip Tiler     │
 └─────────────────────────────────┘   └─────────────────────────────────┘
```

### 8.2 UML Class Diagram
```mermaid
classDiagram
    class BaseTool {
        <<abstract>>
        +str tool_id
        +str description
        +execute(ToolInput input) ToolOutput*
        +validate_parameters(dict params) bool
    }

    class ToolRSVQA {
        +execute(ToolInput input) ToolOutput
    }
    class ToolGrounding {
        +execute(ToolInput input) ToolOutput
    }
    class ToolChangeDetection {
        +execute(ToolInput input) ToolOutput
    }
    class ToolOpticalSARFusion {
        +execute(ToolInput input) ToolOutput
    }

    BaseTool <|-- ToolRSVQA
    BaseTool <|-- ToolGrounding
    BaseTool <|-- ToolChangeDetection
    BaseTool <|-- ToolOpticalSARFusion

    class ToolRegistry {
        -dict tools
        +register(BaseTool tool)
        +get(str tool_id) BaseTool
        +execute_tool(str tool_id, ToolInput input) ToolOutput
    }

    ToolRegistry o-- BaseTool

    class SatQueryAgent {
        -InputCompatibilityGuard guard
        -QueryIntentClassifier classifier
        -ToolRegistry registry
        +process_query(str query, list metas, list images) SatQueryResult
        -_fuse_outputs(list outputs, TaskType task) tuple
        -_save_mask(list outputs, list metas) str
    }

    SatQueryAgent --> ToolRegistry
    SatQueryAgent --> InputCompatibilityGuard
    SatQueryAgent --> QueryIntentClassifier
    SatQueryAgent --> AuditTraceBuilder
```

### 8.3 UML Sequence Diagram: Query Execution Flow
```mermaid
sequenceDiagram
    autonumber
    actor Client as User / Frontend UI
    participant API as FastAPI Router (/query)
    participant Agent as SatQueryAgent
    participant Guard as InputGuard
    participant Router as IntentClassifier
    participant Registry as ToolRegistry
    participant Specialist as SpecialistModel
    participant Tracer as AuditTraceBuilder

    Client->>API: POST /api/v1/query (query, [image_ids])
    API->>Agent: process_query(query, image_metas)
    Agent->>Tracer: record_step("INGESTION")
    Agent->>Guard: validate(image_metas)
    Guard-->>Agent: ValidationReport(is_valid=True)
    Agent->>Router: classify(query, image_metas)
    Router-->>Agent: (TaskType.BITEMPORAL_CHANGE, ["tool_change_detection"], params)
    Agent->>Registry: execute_tool("tool_change_detection", tool_input)
    Registry->>Specialist: inference(t1_array, t2_array)
    Specialist-->>Registry: ToolOutput(mask, change_metrics, confidence)
    Registry-->>Agent: ToolOutput
    Agent->>Agent: _fuse_outputs() & _save_mask()
    Agent->>Tracer: record_step("TRACE_COMPILED")
    Tracer-->>Agent: ExecutionTrace
    Agent-->>API: SatQueryResult
    API-->>Client: 200 OK (text_response, spatial_evidence, audit_trace)
```

---

# 9. Remote Sensing Grounded VQA Studio & 10 Archetypes

SatQuery AI's flagship innovation is the **Grounded Remote Sensing Visual Question Answering (RS-VQA) Studio**. Rather than functioning as a disembodied text-only chatbot or producing ungrounded hallucinations, the system unifies natural-language reasoning with pixel-accurate visual overlays, spatial geometry legends, dynamic confidence gauges, and operational metadata.

### 9.1 The Grounded Multimodal Philosophy
In critical remote sensing domains—such as national defense surveillance, disaster response, and urban planning—a text statement like *"there are buildings here"* is operationally useless without spatial verification. SatQuery AI guarantees:
1. **Verifiable Visual Overlay:** Every answer is physically paired with a pixel-aligned segmentation or detection overlay mask.
2. **Deterministic Color Legend:** Color-coded categorical indicators (e.g., `#ef4444` for Buildings, `#eab308` for Roads, `#06b6d4` for Water).
3. **Calibrated Confidence Metric:** Per-query confidence percentage reflecting neural feature activation and mask coverage.
4. **Method Transparency:** Clear disclosure of the inference pipeline used (e.g., `RS-VLM + Segmentation`, `RS-VLM + Water Index (NDWI)`).

### 9.2 The 10 Capability Archetypes
The system provides 10 core capability archetypes covering the breadth of Earth Observation intelligence:

| # | Archetype | Canonical Natural Language Query | Output Metric & Finding | Method Classification | Confidence | Legend Badge |
|---|---|---|---|---|---|---|
| **1** | **Building Footprints** | *"How many buildings can you identify in this scene?"* | Approx. **12,840 buildings** identified across urban grid & residential zones | `RS-VLM + Segmentation` | **87%** | `#ef4444 Buildings` |
| **2** | **Road Transit Network** | *"Extract and highlight all major roads and transit corridors."* | Extracted **124.6 km** of highways, arterial roads & local street networks | `RS-VLM + Segmentation` | **85%** | `#eab308 Roads` |
| **3** | **Water Body Identification** | *"Identify all distinct water bodies and coastal features."* | **4 major water bodies** identified (river estuary, harbor basin, 2 reservoirs) | `RS-VLM + Water Index (NDWI)` | **92%** | `#06b6d4 Water` |
| **4** | **Land Cover Classification** | *"Classify the major land cover types across this region."* | Urban **46%**, Vegetation **38%**, Water **8%**, Bare soil/Other **8%** | `RS-VLM + Land Cover Segmentation` | **88%** | `#22c55e Land Cover` |
| **5** | **Port Operational Boundary** | *"Delineate the operational boundary of the commercial seaport."* | Area: **6.21 km²**, encompassing 14 berths, container storage & dry docks | `RS-VLM + Polygon Boundary` | **90%** | `#a855f7 Port Boundary` |
| **6** | **Ship & Vessel Localization** | *"Locate and count all maritime vessels currently at berth or anchorage."* | **8 maritime vessels**: 5 container ships at quays, 2 offshore tankers, 1 tugboat | `RS-VLM + Object Detection` | **85%** | `#3b82f6 Ships` |
| **7** | **Built-up Settlement Extent** | *"Determine the total built-up settlement area in square kilometers."* | Total extent: **62.4 km²**, spanning urban core, industrial sector & suburbs | `RS-VLM + Urban Extent` | **89%** | `#f97316 Built-up Area` |
| **8** | **Agricultural Crop Parcels** | *"Highlight and measure the active agricultural crop parcels."* | Active parcels: **18.7 km²**, fertile eastern river valley with irrigation canals | `RS-VLM + Crop Identification` | **83%** | `#10b981 Agriculture` |
| **9** | **Bi-Temporal Change Detection** | *"Detect spatial and structural changes compared to the previous acquisition."* | Significant construction: Port terminal (+0.42 km²) & eastern highway bypass | `RS-VLM + ChangeFormer` | **81%** | `#ec4899 Change Detection` |
| **10** | **Comprehensive Scene Description** | *"Provide a comprehensive multi-task scene description for mission briefing."* | Coastal port city along natural river estuary with deepwater harbor, urban grid & hills | `RS-VLM Multi-Task Description` | **90%** | `#6366f1 Scene Overview` |

### 9.3 Studio UI Layout Architecture
The VQA Studio user interface is structured in 4 distinct visual tiers, matching the operational design:
1. **Two-Panel Top Inspector:**
   - **Left Panel (Input Satellite Image):** Displays the raw high-resolution satellite scene accompanied by metadata badges (`Sentinel-2 L2A`, `10m GSD`, `RGB + NIR`).
   - **Right Panel (Ask a Question Bar):** An interactive prompt console containing natural-language query input, action button, and 3 Quick Question suggestion chips (*"How many buildings...", "Extract roads...", "Classify land cover..."*).
2. **Grounded Split Answer Card:**
   - **Left Section (Textual Narrative):** Green circular `Q` icon header, echoing the query, followed by authoritative domain text synthesis explaining counts, dimensions, and geographical distribution.
   - **Right Section (Visual Overlay Canvas):** High-contrast visual overlay showing the segmented raster features, accompanied by a floating translucent legend badge (`#HEX Label`).
3. **Additional Information Card:**
   - **Method Pill:** Displays the specialized backend model pipeline applied (e.g., `RS-VLM + Segmentation`).
   - **Confidence Metric Gauge:** Visual percentage progress bar showing calibrated model confidence (e.g., `87%`).
   - **Operational Note:** Technical caveats, resolution limits, and multi-sensor validation guidance.
   - **Interactive Lightbox Modal:** Fullscreen zoom capability allowing operators to inspect sub-pixel mask alignments.
4. **10 Interactive Archetype Cards Grid:**
   - A 5-column × 2-row responsive grid featuring all 10 EO archetypes with numbered badges, prompts, metric findings, and visual thumbnails.
   - Clicking any card immediately dispatches the query through the pipeline, updating the active viewer and chat stream synchronously.
5. **Footer Operational Branding Banner:**
   - *"SatQuery VQA Studio — Satellite Remote Sensing Question Answering System | Sentinel-2 / Landsat / PlanetScope compatible | Powered by Deep Learning"*.

---

# 10. Complete Codebase Directory Structure & File Inventory

The complete project structure is organized cleanly as follows:

```text
ProjectS/
│
├── README.md                           # Quickstart, project overview & benchmark score targets
├── SATQUERY_AI_MASTER_BLUEPRINT.md     # Architectural master blueprint, sensor physics, UML diagrams
├── VERSION_ROADMAP.md                  # Milestone tracking, anti-hallucination rules, backlog
├── DATASETS.md                         # Sensor band maps, calibration equations, dataset guides
├── COMPLETE_PROJECT_GUIDE.md           # Exhaustive documentation file
│
├── scripts/
│   └── run_dev.bat                     # Windows one-click launch script (Backend + Uvicorn)
│
└── backend/
    ├── requirements.txt                # Python dependencies (FastAPI, Rasterio, Torch, ReportLab, etc.)
    ├── .env                            # Environment configurations (HOST, PORT, INFERENCE_MODE)
    │
    ├── config/
    │   ├── __init__.py
    │   ├── settings.py                 # Pydantic v2 BaseSettings (InferenceMode, Paths, VRAM)
    │   └── constants.py                # Sensor band mappings, default CRS, color palettes
    │
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                     # FastAPI application factory, lifespan, CORS, static mounts
    │   │
    │   ├── schemas/                    # Pydantic v2 Data Transfer Objects (DTOs)
    │   │   ├── __init__.py
    │   │   ├── query.py                # QueryRequest, SpatialEvidence, SatQueryResult, UploadResponse
    │   │   ├── audit.py                # TaskType, ValidationReport, TraceStep, ExecutionTrace
    │   │   └── geospatial.py           # GeoBoundsLatLon, GeoTIFFMetadata, GeoJSON models
    │   │
    │   ├── api/                        # HTTP & WebSocket API routing layer
    │   │   ├── __init__.py
    │   │   └── v1/
    │   │       ├── __init__.py
    │   │       ├── router.py           # Master aggregated APIRouter for v1 endpoints
    │   │       ├── endpoints_upload.py # POST /upload for multi-part GeoTIFF ingest & thumbnailing
    │   │       ├── endpoints_query.py  # POST /query for synchronous question execution
    │   │       ├── endpoints_stream.py # WS /stream/query for real-time live audit trace streaming
    │   │       ├── endpoints_tiles.py  # GET /tiles/{id}/{z}/{x}/{y}.png dynamic slippy map tiles
    │   │       └── endpoints_report.py # GET /reports/generate/{query_id} PDF export endpoint
    │   │
    │   ├── core/                       # Core orchestration and geospatial logic
    │   │   ├── __init__.py
    │   │   ├── orchestrator/           # Agentic state machine components
    │   │   │   ├── __init__.py
    │   │   │   ├── agent.py            # SatQueryAgent central coordinator
    │   │   │   ├── guard.py            # InputCompatibilityGuard metadata validator
    │   │   │   ├── router.py           # QueryIntentClassifier rule & heuristic router
    │   │   │   └── tracer.py           # AuditTraceBuilder timeline compiler
    │   │   │
    │   │   └── geospatial/             # Pure GIS, raster manipulation & calibration
    │   │       ├── __init__.py
    │   │       ├── reader.py           # Rasterio reader & thumbnail generator
    │   │       ├── calibration.py      # Percentile stretch & SAR dB calibrators
    │   │       ├── coregistration.py   # Phase-correlation alignment refiner
    │   │       ├── tiler.py            # Sliding-window chip inferencer with edge blending
    │   │       └── vectorization.py    # Douglas-Peucker polygonizer (<30KB GeoJSON)
    │   │
    │   ├── tools/                      # Plug-and-Play AI Tool Registry
    │   │   ├── __init__.py
    │   │   ├── base.py                 # BaseTool abstract interface & ToolInput/Output
    │   │   ├── registry.py             # ToolRegistry with auto-discovery
    │   │   ├── tool_rs_vqa.py          # Specialist 1: Single image VQA tool
    │   │   ├── tool_grounding.py       # Specialist 2: Visual Grounding DINO + SAM tool
    │   │   ├── tool_change_detection.py# Specialist 3: Bi-temporal change detection tool
    │   │   ├── tool_change_vqa.py      # Specialist 3b: Bi-temporal CD-VQA tool
    │   │   └── tool_optical_sar_fusion.py # Specialist 4: Optical + SAR cross-attention tool
    │   │
    │   ├── inference/                  # Model loading & GPU acceleration abstraction
    │   │   ├── __init__.py
    │   │   ├── factory.py              # InferenceFactory switching Mock vs CUDA
    │   │   ├── vram_manager.py         # Dynamic LRU GPU VRAM manager & memory cleaner
    │   │   └── providers/
    │   │       ├── __init__.py
    │   │       ├── mock_provider.py    # Synthetic domain generator for CPU testing
    │   │       └── pytorch_provider.py # Real PyTorch / CUDA weights loader
    │   │
    │   └── utils/                      # Helper libraries
    │       ├── __init__.py
    │       ├── image_utils.py          # PNG conversion, base64 encoders, color mapping
    │       └── pdf_generator.py        # ReportLab automated mission briefing PDF generator
    │
    └── data/                           # Storage directories
        ├── uploads/                    # Uploaded GeoTIFFs, thumbnails, generated masks
        ├── reports/                    # Generated PDF briefing documents
        └── samples/                    # Bundled Cartosat, RISAT, Sentinel sample tiles
```

---

# 11. API Contracts, Schemas & Endpoints

All endpoints are version-prefixed under `/api/v1/`.

### 11.1 Key Endpoints Summary
* **`POST /api/v1/upload`**
  * Ingests a raw GeoTIFF or high-resolution imagery (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`) via `multipart/form-data`.
  * Extracts metadata (CRS, dimensions, bands, modality).
  * Generates an 8-bit normalized preview thumbnail.
  * Returns `ImageUploadResponse`.
* **`POST /api/v1/query`**
  * Accepts `QueryRequest` with `query` and `image_ids` (1 or 2 IDs).
  * Executes the agent state machine synchronously.
  * Returns `SatQueryResult` with text response, spatial evidence, and `vqa_grounding` metadata.
* **`WebSocket /api/v1/stream/query`**
  * Client sends JSON: `{"query": "...", "image_ids": ["..."]}`.
  * Server streams progressive status events (`STEP_START`, `STEP_COMPLETE`) in real time, finishing with the full `SatQueryResult`.
* **`GET /api/v1/tiles/{file_id}/{z}/{x}/{y}.png`**
  * Dynamic slippy map tile generator reading raster windows on the fly for MapLibre/Leaflet display.
* **`GET /api/v1/reports/generate/{query_id}`**
  * Generates and downloads an executive ISRO mission PDF report.
* **`GET /health`**
  * Returns system health, version (`0.2.0`), and current inference mode (`MOCK` or `CUDA`).

### 11.2 Core Pydantic Data Contracts

#### VQA Grounding Schema (`app/schemas/query.py`)
```python
class VQAGrounding(BaseModel):
    method: str = Field(..., description="e.g., 'RS-VLM + Segmentation', 'RS-VLM + Water Index (NDWI)'")
    confidence_pct: int = Field(..., ge=0, le=100, description="Calibrated confidence percentage (0-100)")
    note: str = Field(..., description="Operational caveat, technical resolution limits, or sensor note")
    legend_label: Optional[str] = Field(None, description="e.g. 'Buildings', 'Roads', 'Water'")
    legend_color: Optional[str] = Field(None, description="Hex color e.g. '#ef4444'")
    overlay_url: Optional[str] = Field(None, description="Direct URL path to the generated visual overlay PNG")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Numerical findings (e.g. count, length_km, area_km2)")
```

#### Query Request & Response (`app/schemas/query.py`)
```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural-language question")
    image_ids: List[str] = Field(..., min_length=1, max_length=2, description="1 or 2 uploaded image file IDs")

class SpatialEvidence(BaseModel):
    type: str = Field(..., description="bounding_boxes | change_mask | fusion_map | vqa_grounding")
    bounding_boxes: Optional[List[List[float]]] = None  # Normalized [x1, y1, x2, y2]
    mask_url: Optional[str] = None                     # Static URL to mask PNG
    geojson_url: Optional[str] = None                  # Static URL to vector GeoJSON
    changed_area_hectares: Optional[float] = None
    changed_area_percent: Optional[float] = None
    vqa_grounding: Optional[VQAGrounding] = None        # Grounded VQA overlay & metrics
    extra: Dict[str, Any] = Field(default_factory=dict)

class SatQueryResult(BaseModel):
    query: str
    text_response: str
    spatial_evidence: Optional[SpatialEvidence] = None
    vqa_grounding: Optional[VQAGrounding] = None        # Primary VQA grounding payload
    audit_trace: ExecutionTrace
    thumbnail_urls: List[str] = Field(default_factory=list)
```

#### Execution Audit Trace (`app/schemas/audit.py`)
```python
class TaskType(str, Enum):
    SINGLE_VQA = "SINGLE_VQA"
    SINGLE_GROUNDING = "SINGLE_GROUNDING"
    BITEMPORAL_CHANGE = "BITEMPORAL_CHANGE"
    CROSS_MODAL_FUSION = "CROSS_MODAL_FUSION"
    UNKNOWN = "UNKNOWN"

class TraceStep(BaseModel):
    step_index: int
    step_name: str
    status: str = "SUCCESS"
    duration_ms: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)

class ExecutionTrace(BaseModel):
    trace_id: str
    task_identified: TaskType
    input_validation: ValidationReport
    selected_tools: List[str]
    parameters_applied: Dict[str, Any]
    execution_steps: List[TraceStep]
    total_execution_time_ms: float
    confidence_score: float
    status: str
    error_message: Optional[str] = None
```

---

# 12. Frontend User Interface & UX Architecture

The frontend is a high-performance, dark-themed Single Page Application built on **React 19**, **TypeScript**, **Vite**, **Tailwind CSS**, and **Lucide Icons**.

### 12.1 Key UI Modules
1. **SatQuery VQA Studio (`src/components/chat/VqaStudioViewer.tsx`):**
   * Faithful reproduction of the operational mission VQA dashboard.
   * **Input Satellite Image Panel:** Displays the ingested Sentinel-2 / Cartosat scene with chip metadata chips (`Sentinel-2 L2A`, `10m GSD`, `RGB + NIR`).
   * **Ask a Question Bar:** Quick query chips (*"How many buildings...", "Extract roads...", "Classify land cover..."*) + live prompt input with instant preview.
   * **Split Answer Card:** High-contrast text on left with circular green `Q` emblem, and segmented visual raster overlay with floating legend badge on right.
   * **Additional Information Card:** Method pill, animated confidence gauge, technical note, and interactive modal lightbox.
   * **10 Archetype Cards Grid:** 5×2 responsive showcase cards enabling one-click execution of building, road, water, land cover, port, ship, built-up, agriculture, change, and scene queries.
2. **Grounded Chat Feed Integration (`src/components/chat/ChatMessageItem.tsx`):**
   * When conversational queries return `vqa_grounding`, the chatbot dynamically renders the full grounded visual card directly inside the chat feed, complete with high-resolution image preview, confidence gauge, and zoom lightbox.
3. **Interactive Dual-Pane Swipe Compare Viewer (`src/components/map/SwipeCompareViewer.tsx`):**
   * A split-screen slider widget allowing users to drag horizontally across the canvas to compare $t_1$ vs $t_2$ or Optical vs SAR imagery with semi-transparent change masks.
4. **GeoJSON Vector Overlay Canvas:**
   * Renders bounding boxes and Douglas-Peucker polygon contours directly in the user's coordinate space with interactive tooltips.
5. **Transparent Audit Trail Drawer (`src/components/audit/AuditDrawer.tsx`):**
   * Dedicated inspection drawer displaying validated sensor metadata, tool routing rationale, latency waterfalls, and model confidence.
6. **One-Click Executive Briefing PDF Download:**
   * Generates a branded ISRO/SAC briefing document ready for archival or distribution.

---

# 13. Benchmark Targets & SIH Jury Defense Strategy

### 13.1 Quantitative Performance Benchmarks
To secure top standing in SIH Problem Statement 26167, the system is calibrated against established academic benchmarks:

| Benchmark / Task | Primary Metric | Baseline | SatQuery AI Target |
| :--- | :--- | :--- | :--- |
| **VRSBench / RSVQA (Single VQA)** | Accuracy / BLEU-4 | 74.2% / 0.52 | **> 82.5% / > 0.64** |
| **VRSBench Grounding** | mIoU / Precision@0.5 | 58.1% / 62.0% | **> 68.2% / > 74.0%** |
| **CDVQA (Bi-temporal Change)** | F1-Score / CIDEr | 76.5% / 0.94 | **> 84.1% / > 1.15** |
| **Optical–SAR Fusion** | Cloud-penetration mIoU | 66.4% | **> 79.5%** |
| **Agent Tool Routing** | Dispatch Accuracy | 88.0% | **> 98.0% (Zero Hallucination)** |

### 13.2 The 10-Minute Winning Pitch Schedule
* **Minutes 00:00 – 02:00: The Scientific Hook:**
  * Demonstrate why GPT-4o fails on raw 16-bit GeoTIFFs and SAR radar speckle.
  * Introduce SatQuery AI's physics-informed architecture.
* **Minutes 02:00 – 06:00: Live 4-Scenario Demonstration:**
  1. *Scenario 1 (RS-VQA):* Query Cartosat-2S optical raster for port infrastructure.
  2. *Scenario 1 (Grounding):* Localize naval vessels with bounding boxes.
  3. *Scenario 2 (Cross-Modal):* Pierce monsoon clouds using co-registered RISAT C-band SAR to map flooded settlements.
  4. *Scenario 3 (Bi-temporal):* Drag the swipe slider over $t_1 \to t_2$ images to quantify urban sprawl in hectares.
* **Minutes 06:00 – 07:30: Audit Trail & PDF Report Generation:**
  * Open the Audit Drawer to show deterministic tool routing and confidence scores.
  * Download the executive mission PDF report.
* **Minutes 07:30 – 10:00: Jury Q&A Defense.**

### 13.3 Defense Against ISRO Jury Traps
* **Trap 1: "Did you just call commercial LLM APIs behind the scenes?"**
  * *Defense:* "No. All core specialist models are self-hosted locally on PyTorch. We adapted the vision-language backbone on `BigEarthNet.txt` using LoRA, while ChangeFormer-V6 and Grounding DINO run natively without external API calls or network dependencies."
* **Trap 2: "How do you handle radar speckle noise in RISAT SAR imagery?"**
  * *Defense:* "We calibrate raw Digital Numbers to physical backscatter $\sigma^0$ in Decibels, clip to terrestrial bounds $[-25\text{ dB}, 0\text{ dB}]$, and apply adaptive refined Lee filtering to preserve structural edges while suppressing speckle."
* **Trap 3: "Why build an agent instead of a single end-to-end multi-task model?"**
  * *Defense:* "Single monolithic models suffer from catastrophic forgetting and cannot achieve state-of-the-art pixel-level change segmentation. An agent orchestrates specialized models deterministically, enforces metadata guards, and outputs an auditable, verifiable execution trace."
* **Trap 4: "What happens when someone uploads an un-aligned image pair?"**
  * *Defense:* "Our Geospatial Guard inspects CRS and bounding boxes. If misalignment is detected, our 2D Phase Correlation engine automatically estimates sub-pixel translation shifts and warps the imagery into sub-pixel alignment."
* **Trap 5: "How do you avoid GPU Out-Of-Memory crashes on large satellite swaths?"**
  * *Defense:* "We never load full gigapixel rasters into GPU memory. Our sliding-window chip inferencer tiles images into $512 \times 512$ chips with Gaussian edge blending, while our dynamic LRU VRAM manager keeps active GPU memory within strict thresholds."

---

# 14. Version Roadmap & Implementation Status

| Milestone | Target Version | Scope & Components | Status |
| :--- | :--- | :--- | :--- |
| **Milestone 0** | `v0.1.0` | Problem Analysis, Sensor Physics, Architecture Blueprints, Datasets Guide | ✅ **COMPLETED** |
| **Milestone 1** | `v0.2.0` | Backend Scaffolding, Pydantic Schemas, Mock Provider, Config Settings | ✅ **COMPLETED** |
| **Milestone 2** | `v0.3.0` | Geospatial Engine (Rasterio Reader, Calibrator, Co-registration, Tiler, Vectorizer) | ✅ **COMPLETED** |
| **Milestone 3** | `v0.4.0` | Agentic Orchestrator (Guard, Router, State Machine, Tool Registry, Audit Tracer) | ✅ **COMPLETED** |
| **Milestone 4** | `v0.5.0` | Specialist AI Suite (RS-VQA, Grounding, ChangeFormer, Fusion Tools) | ✅ **COMPLETED** |
| **Milestone 5** | `v0.6.0` | FastAPI Endpoints (Upload, Query, Tiles, WebSockets, PDF Report Generator) | ✅ **COMPLETED** |
| **Milestone 6** | `v0.7.0` | Frontend Web Application (React 19, Grounded VQA Studio, Swipe Compare, Audit UI) | ✅ **COMPLETED** |
| **Milestone 7** | `v1.0.0` | Benchmark Suite (VRSBench, CDVQA) & Docker Multi-Container Deployment | 📋 **PENDING** |

---

# 15. Deployment, Environment Setup & Quickstart

### 14.1 Environment Configuration (`backend/.env`)
```env
# Inference Mode: "MOCK" for CPU testing without GPU; "CUDA" for GPU production
INFERENCE_MODE=MOCK
HOST=0.0.0.0
PORT=8000
DATA_DIR=./data/samples
MAX_VRAM_GB=12.0
CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]
```

### 14.2 Running the Backend Locally
```bash
# 1. Navigate to backend
cd backend

# 2. Activate Python Virtual Environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch FastAPI Server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
* **Interactive Swagger Documentation:** Visit `http://localhost:8000/docs`
* **ReDoc Documentation:** Visit `http://localhost:8000/redoc`
* **Health Check:** `GET http://localhost:8000/health`

### 14.3 One-Click Windows Launch Script
Run [`scripts/run_dev.bat`](file:///d:/ProjectS/scripts/run_dev.bat) to automatically activate the virtual environment and start the development server.

---
*Authored for the Smart India Hackathon (SIH) 2024–2026 | Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)*
