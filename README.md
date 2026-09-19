# SatQuery AI — Vision-Language Assistant for Remote Sensing
**Smart India Hackathon (SIH) | Problem Statement ID: 26167**  
**Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)  
**Theme:** Space Technology | **Category:** Software  

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![Rasterio](https://img.shields.io/badge/Rasterio-GDAL-blue)](https://rasterio.readthedocs.io/)
[![React](https://img.shields.io/badge/React-19+-61DAFB?style=flat&logo=react)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC?style=flat&logo=tailwind-css)](https://tailwindcss.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker)](https://www.docker.com/)

---

## 🛰️ Overview
**SatQuery AI** is an interactive, agentic vision-language platform engineered for multimodal remote sensing image analysis through natural-language queries. 

Unlike generic vision-language models that fail on satellite formats and radar physics, SatQuery AI features a deterministic **Agentic Orchestrator** coupled with a suite of **specialist remote-sensing AI models** adapted on `BigEarthNet.txt`, handling:
1. **Single-Image Understanding:** RS-VQA (Mandatory) and Text-Guided Visual Grounding (bounding boxes/masks).
2. **Cross-Modal Sensor Fusion:** Co-registered Optical/Multispectral + Synthetic Aperture Radar (SAR) for cloud-penetrating analysis.
3. **Bi-Temporal Change Analysis:** Change description, Change-VQA (CD-VQA), and high-resolution spatial change maps.
4. **Observable Execution Tracing:** Completely inspectable, tamper-proof execution logs of selected models, parameters, confidence metrics, and processing latency.

---

## 🌟 Key Capabilities
* **Grounded Remote Sensing VQA Studio:** Interactive Earth Observation intelligence cockpit supporting 10 operational capability archetypes (buildings, roads, water bodies, land cover, port boundary, ships, built-up area, agriculture, change detection, and scene description). Pairs natural-language answers with instant pixel-level visual segmentation overlays, color-coded legend tags, and calibrated confidence gauges.
* **Fine-Tuned Domain RS-VLM Transformer:** Dedicated vision-language checkpoint (`rs_vlm.pt`, 25.5MB) trained over 20 epochs across an expanded 173-token remote sensing vocabulary down to loss `0.7019` and perplexity `2.02`.
* **Native GeoTIFF Ingestion:** Full support for multi-band 12-bit/16-bit optical rasters (Cartosat-2S, Sentinel-2), C-band microwave SAR rasters (RISAT, Sentinel-1), and standard web imagery formats (`.png`, `.jpg`).
* **Sliding-Window Chip Inferencer:** Processes arbitrary gigapixel ($>12,000 \times 12,000$) satellite tiles without CUDA Out-Of-Memory (OOM) crashes.
* **Dual-Pane Swipe Comparison:** Interactive slider interface allowing visual inspection between Before/After ($t_1$ vs $t_2$) or Optical vs SAR.
* **Sub-Pixel Co-Registration:** Automated 2D phase-correlation alignment refiner eliminating false change alarms.
* **Dual-Mode Inference Engine:** Supports `INFERENCE_MODE=MOCK` (for rapid laptop/CPU development without GPU) and `INFERENCE_MODE=CUDA` (for production GPU acceleration).
* **Automated Mission Intelligence Reports:** One-click generation of downloadable executive PDF briefing documents.

---

## 🏛️ System Architecture

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

---

## ⚡ 3-Step Quickstart Guide

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/your-org/satquery-ai.git
cd satquery-ai

# Create and activate Python virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the `backend/` directory:
```env
# Inference Mode: "MOCK" for CPU testing without GPU, "CUDA" for GPU execution
INFERENCE_MODE=MOCK
HOST=0.0.0.0
PORT=8000
DATA_DIR=./data/samples
MAX_VRAM_GB=12.0
```

### 3. Run the Development Servers
**Option A: Windows Convenience Script**
```bash
# Double-click or run:
scripts\run_dev.bat
```

**Option B: Manual Launch**
```bash
# Terminal 1: Backend API
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend Web App
cd frontend
npm install
npm run dev
```

Open your browser at **`http://localhost:5173`** to access the interactive web application.

---

## 🐳 Docker Deployment (One-Command Launch)
To spin up the complete production stack (Backend + Frontend + Redis + Nginx) with GPU passthrough:
```bash
docker compose up --build
```

---

## 📊 Evaluation & Benchmark Targets

| Task / Benchmark | Primary Metric | Target Winning Score |
| :--- | :--- | :--- |
| **VRSBench / RSVQA (Single VQA)** | Accuracy / BLEU-4 | **Acc > 82.5%**, **BLEU-4 > 0.64** |
| **VRSBench Grounding** | mIoU / Precision@0.5 | **mIoU > 68.2%**, **P@0.5 > 74.0%** |
| **CDVQA (Change Detection)** | F1-Score / CIDEr | **F1 > 84.1%**, **CIDEr > 1.15** |
| **Cross-Modal Fusion** | Cloud-Penetration mIoU | **mIoU > 79.5%** |
| **Agent Tool Routing** | Tool Selection Accuracy | **> 98.0% (Zero Hallucination)** |

---

## 📚 Documentation Index
* 📘 **[SATQUERY_AI_MASTER_BLUEPRINT.md](file:///d:/ProjectS/SATQUERY_AI_MASTER_BLUEPRINT.md)** — Complete system architecture, UML diagrams, code recipes, and SIH jury pitch defense.
* 🚦 **[VERSION_ROADMAP.md](file:///d:/ProjectS/VERSION_ROADMAP.md)** — Single source of truth for version milestones, anti-hallucination rules, and development state.
* 🛰️ **[DATASETS.md](file:///d:/ProjectS/DATASETS.md)** — Comprehensive sensor band mappings, calibration formulas, and dataset download guide.

---

## 👥 Contributors & SIH Team
* **Organization:** Indian Space Research Organisation (ISRO)
* **Center:** Space Applications Centre (SAC), Ahmedabad
* **Problem Statement ID:** 26167
