# SatQuery AI — Versioning, State & Implementation Roadmap
**Repository Anchor & Context Ground-Truth Document**  
**Problem Statement:** ISRO / SAC — Problem Statement ID 26167  
**Last Updated:** 2026-09-08 | **Current Phase:** Milestones 1–5 Backend Completed (v0.6.0), Ready for Milestone 6 Frontend

---

## 📌 Purpose of this Document
This file is the **single source of truth** for project development. It tracks:
1. **Current System State:** Exactly what has been completed, designed, and approved.
2. **Active Version:** Version numbering and release milestones.
3. **Execution Backlog:** Ordered list of components to build with file paths and dependencies.
4. **Anti-Hallucination Anchors:** Non-negotiable technical constraints to prevent deviation during implementation.

---

## 🚦 High-Level Status Dashboard

| Milestone | Phase Description | Target Version | Status |
| :--- | :--- | :--- | :--- |
| **Milestone 0** | System Design, Architecture, and Master Documentation | `v0.1.0` | ✅ **COMPLETED** |
| **Milestone 1** | Backend Scaffolding, Schemas, & Mock Inference Engine | `v0.2.0` | ✅ **COMPLETED** |
| **Milestone 2** | Geospatial Engine (GeoTIFF, Cartosat, RISAT, Tiling) | `v0.3.0` | ✅ **COMPLETED** |
| **Milestone 3** | Agentic Orchestrator, State Machine & Audit Tracer | `v0.4.0` | ✅ **COMPLETED** |
| **Milestone 4** | Specialist Models Integration (VQA, Grounding, Change, Fusion)| `v0.5.0` | ✅ **COMPLETED** |
| **Milestone 5** | FastAPI Endpoints, WebSockets & PDF Report Generator | `v0.6.0` | ✅ **COMPLETED** |
| **Milestone 6** | Frontend Web App (Dual-Pane Swipe Viewer & Audit UI) | `v0.7.0` | ⏳ **READY TO START** |
| **Milestone 7** | Benchmark Test Suite (VRSBench, CDVQA) & Dockerization | `v1.0.0` | 📋 Pending (Final) |

---

## 🧠 Anti-Hallucination Ground-Truth Rules
Every agent and developer working on this codebase **must strictly follow these rules**:

1. **Format Constraint:**
   * Primary inputs are **GeoTIFF / TIFF**. Never assume simple 8-bit 3-channel JPEGs/PNGs.
   * Optical inputs can be 12-bit/16-bit (Cartosat-2S / Sentinel-2). Always normalize using 2%–98% percentile stretching or division by 10,000.
   * SAR inputs (RISAT / Sentinel-1) must be converted to **Decibels ($\sigma^0$ dB)** and normalized $[-25\text{ dB}, 0\text{ dB}]$.
2. **Agentic Rule (No Black-Box Reasoning):**
   * Do **NOT** send raw internal reasoning text to the user.
   * Always output a structured, observable **Execution Trace** containing: `task_identified`, `selected_tools`, `parameters_applied`, `execution_time_ms`, `confidence_score`.
3. **Memory Rule (No OOM Crashes):**
   * Never run `src.read()` on full multi-gigabyte rasters directly into GPU tensors. Use the **Sliding Window Chip Inferencer (`tiler.py`)**.
4. **Dual-Mode Inference:**
   * All tools must support `INFERENCE_MODE=MOCK` (for local CPU/laptop testing without GPU) and `INFERENCE_MODE=CUDA` (for GPU production).

---

## 🗺️ Detailed Milestone Breakdown & Task Backlog

### Milestone 0: Architecture & Master Blueprint (Completed ✅)
* [x] Problem Statement analysis and ISRO requirements decompiled.
* [x] Sensor physics defined (Optical reflectance vs. SAR $\sigma^0$ backscatter).
* [x] 4 Specialist models selected (`RS-VLM`, `Grounding DINO + SAM`, `ChangeFormer`, `Cross-Attention Fusion`).
* [x] Master design document generated: `SATQUERY_AI_MASTER_BLUEPRINT.md`.

---

### Milestone 1: Backend Scaffolding & Mock Inference Engine (Target: `v0.2.0`) ✅ COMPLETED
**Objective:** Set up project structure, configuration, Pydantic schemas, and a functional mock engine so the API works immediately without GPU dependencies.

* [x] Create folder structure under `backend/`.
* [x] Create `config/settings.py` (Pydantic v2 `BaseSettings` for env vars, mode, ports).
* [x] Create `config/constants.py` (Sensor band maps, default CRS, color palettes).
* [x] Create `app/schemas/`:
  * `schemas/query.py` (`QueryRequest`, `SatQueryResult`).
  * `schemas/audit.py` (`TaskType`, `ExecutionTrace`, `ValidationReport`).
  * `schemas/geospatial.py` (`GeoBoundsLatLon`, `GeoJSONFeatureCollection`).
* [x] Create `app/inference/`:
  * `inference/factory.py` (Toggles Mock vs. CUDA).
  * `inference/providers/mock_provider.py` (Generates synthetic bounding boxes, change masks, and domain text for testing).
* [x] Create `requirements.txt` and `run_dev.bat` / `run_dev.sh`.

---

### Milestone 2: Geospatial & Sensor Engine (Target: `v0.3.0`) ✅ COMPLETED
**Objective:** Ingest, calibrate, and co-register real Cartosat-2S and RISAT GeoTIFFs without memory issues.

* [x] `app/core/geospatial/reader.py`: Windowed COG/GeoTIFF reader using `rasterio`.
* [x] `app/core/geospatial/calibration.py`: Optical percentile stretch & SAR dB calibration.
* [x] `app/core/geospatial/coregistration.py`: Sub-pixel phase-correlation alignment refiner.
* [x] `app/core/geospatial/tiler.py`: Sliding window chip inferencer ($512 \times 512$ with Gaussian blending).
* [x] `app/core/geospatial/vectorization.py`: Douglas-Peucker polygonizer converting masks into `<30KB` GeoJSON.

---

### Milestone 3: Agentic Orchestrator & Tool Registry (Target: `v0.4.0`) ✅ COMPLETED
**Objective:** Build the deterministic state machine, query router, and auditable trace compiler.

* [x] `app/tools/base.py`: `BaseTool` abstract interface + `@register_tool` decorator.
* [x] `app/tools/registry.py`: Dynamic tool auto-discovery engine.
* [x] `app/core/orchestrator/guard.py`: Metadata & CRS compatibility guard.
* [x] `app/core/orchestrator/router.py`: Rule-heuristic + semantic intent classifier.
* [x] `app/core/orchestrator/tracer.py`: Immutable audit execution trace builder.
* [x] `app/core/orchestrator/agent.py`: `SatQueryAgent` state machine coordinating the workflow.

---

### Milestone 4: Specialist Models Suite (Target: `v0.5.0`) ✅ COMPLETED
**Objective:** Plug in the 4 remote sensing AI models with dynamic VRAM offloading.

* [x] `app/inference/vram_manager.py`: LRU dynamic GPU VRAM manager with CPU offloading.
* [x] `app/tools/tool_rs_vqa.py`: Remote Sensing VQA tool (GeoChat / LoRA on BigEarthNet.txt).
* [x] `app/tools/tool_grounding.py`: Visual Grounding tool (Grounding DINO + SAM-RS).
* [x] `app/tools/tool_change_detection.py`: Bi-temporal change detection (ChangeFormer-V6).
* [x] `app/tools/tool_change_vqa.py`: Change description generator.
* [x] `app/tools/tool_optical_sar_fusion.py`: Optical-SAR complementary cross-attention tool.

---

### Milestone 5: FastAPI Gateway & PDF Report Generator (Target: `v0.6.0`) ✅ COMPLETED
**Objective:** Expose clean async REST endpoints, WebSocket live streaming, and downloadable briefings.

* [x] `app/api/v1/endpoints_upload.py`: Handles multi-part GeoTIFF ingestion.
* [x] `app/api/v1/endpoints_query.py`: Synchronous query endpoint returning answer + trace.
* [x] `app/api/v1/endpoints_stream.py`: WebSocket endpoint streaming trace steps in real time.
* [x] `app/api/v1/endpoints_tiles.py`: Slippy map raster tile server for frontend map zoom/pan.
* [x] `app/utils/pdf_generator.py`: Mission briefing PDF export using `reportlab`.
* [x] `app/main.py`: FastAPI server mounting CORS and routers.

---

### Milestone 6: Frontend Web Application (Target: `v0.7.0`)
**Objective:** Build a responsive, state-of-the-art interactive web application.

* [ ] Initialize React 19 + Vite + TailwindCSS.
* [ ] `DualPaneViewer.jsx`: Swipe slider comparing $t_1$ vs $t_2$ or Optical vs SAR.
* [ ] `MapCanvas.jsx`: MapLibre GL / Canvas rendering GeoJSON bounding boxes and masks.
* [ ] `AuditSidebar.jsx`: Live step-by-step audit trace inspector.
* [ ] `QueryChatBar.jsx`: Chat input with quick-prompt chips.
* [ ] `ExportButton.jsx`: Trigger and download the PDF briefing.

---

### Milestone 7: Benchmarks & Production Packaging (Target: `v1.0.0`)
**Objective:** Finalize verification scripts, automated evaluation, and Docker containers.

* [ ] `benchmarks/run_vrsbench.py`: Evaluates Single-Image VQA & Grounding metrics.
* [ ] `benchmarks/run_cdvqa.py`: Evaluates Bi-temporal Change VQA metrics.
* [ ] `benchmarks/evaluate_isro_set.py`: Test harness for Cartosat-2S & RISAT pairs.
* [ ] `Dockerfile`: Multi-stage GPU-accelerated Docker build (CUDA 12.2).
* [ ] `docker-compose.yml`: Multi-container deployment (Frontend + Backend + Redis + Nginx).

---

## 📂 Active Workspace File Inventory

| File Path | Description | Status |
| :--- | :--- | :--- |
| `d:/ProjectS/SATQUERY_AI_MASTER_BLUEPRINT.md` | Master System Architecture, UML Diagrams, Code Blueprints | ✅ Verified |
| `d:/ProjectS/VERSION_ROADMAP.md` | Single Source of Truth for Progress Tracking & Backlog | ✅ Active |
| `d:/ProjectS/backend/app/main.py` | FastAPI application gateway with lifecycle, CORS, routes | ✅ Completed |
| `d:/ProjectS/backend/app/core/geospatial/` | Geospatial raster engine (windowed read, calibration, tiler, vectorizer) | ✅ Completed |
| `d:/ProjectS/backend/app/core/orchestrator/` | Agentic guard, router, tracer, and SatQueryAgent state machine | ✅ Completed |
| `d:/ProjectS/backend/app/tools/` | Specialist tool plugin suite (VQA, Grounding, Change, Fusion) | ✅ Completed |
| `d:/ProjectS/backend/app/inference/` | Dynamic inference factory (MOCK provider + CUDA skeleton + VRAM manager) | ✅ Completed |
| `d:/ProjectS/backend/app/schemas/` | Pydantic v2 schemas for queries, audit traces, geospatial GeoJSON | ✅ Completed |
| `d:/ProjectS/backend/app/api/v1/` | Endpoints (upload, query, stream/WebSocket, report, tiles) | ✅ Completed |
| `d:/ProjectS/backend/app/utils/` | PDF report generator (ISRO branding) and image processing utilities | ✅ Completed |
| `d:/ProjectS/scripts/run_dev.bat` | Windows startup launcher script | ✅ Completed |

---

## 🔄 How to Use this File During Development
* Whenever an agent or engineer resumes work, **read this file first**.
* Only work on tasks marked in the current active milestone.
* When a task is completed, check off its box (`[x]`) and update the current version badge.
* This guarantees zero hallucination, zero lost context, and smooth multi-step progress.
