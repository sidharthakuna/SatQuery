# SatQuery AI — 2-Minute Disaster Situational Assessment Demo Script
**Target Duration:** Exactly 2:00 Minutes (~270 spoken words at an authoritative 135 wpm)  
**Topic:** Real-Time Bi-Temporal Flood Inundation, Infrastructure Threat & Safe Evacuation Routing  
**Mission Case:** Cyclone/Monsoon Flood Inundation Assessment over Mahanadi River Delta (`20.25°N, 85.85°E`)  
**Target Audience:** ISRO/SAC Jury, Disaster Management Authorities (NDRF/SDMA), Geospatial Engineers  
**Reference UI:** Live SatQuery AI Workspace (`BITEMPORAL_CHANGE` Engine — 91% Confidence)  

---

## ⏱️ Timeline & Scene Overview

```
00:00 - 00:20 (20s) | Scene 1: The Crisis — Uploading Pre & Post-Flood Satellite Passes
00:20 - 00:40 (20s) | Scene 2: Natural Query to Multi-Model Orchestration (ChangeNet + RS-VLM)
00:40 - 01:15 (35s) | Scene 3: Quantitative Damage Assessment (32,829 Hectares Inundated)
01:15 - 01:45 (30s) | Scene 4: Cartographic Hotspots — Red Danger Zones vs Green Safe Corridors
01:45 - 02:00 (15s) | Scene 5: Field Rescue Directives & One-Click NDRF Executive Dossier
```

---

## 🎬 Shot-by-Shot Director's Guide

### Scene 1: The Crisis — Uploading Pre & Post-Flood Passes (00:00 – 00:20)
| Timing | Screen Action | Spoken Narration (Voiceover) |
| :---: | :--- | :--- |
| **0:00 – 0:10** | Ingest both passes into the input bar:<br>1. `pre_flood_t1.tif` [OPTICAL] (768×768, 1.7 MB)<br>2. `post_flood_t2.tif` [OPTICAL] (768×768, 1.7 MB)<br>Badge indicates: **"Pair Ready for Fusion / Change Analysis"**. | *"When catastrophic monsoon flooding strikes, rescue commanders cannot wait hours for manual GIS mapping. They need instant situational awareness.*<br><br>*Here, we ingest two temporal Sentinel-2 passes over a 665 square kilometer delta basin: the pre-flood baseline on the left, and the submerged landscape on the right."* |
| **0:10 – 0:20** | Show Preprocessing Check table in UI: EPSG:4326 CRS valid, 10m spatial resolution, 94% data quality verified. | *"SatQuery AI instantly validates projection, alignment, and data quality across both rasters in milliseconds."* |

---

### Scene 2: Natural Query to Neural Orchestration (00:20 – 00:40)
| Timing | Screen Action | Spoken Narration (Voiceover) |
| :---: | :--- | :--- |
| **0:20 – 0:30** | Type or display the query:<br>`"Detect flooded area, assess infrastructure damage and show safe zones"`<br>Press Enter. | *"The operator simply asks: 'Detect flooded area, assess infrastructure damage, and show safe zones'."* |
| **0:30 – 0:40** | Audit Trace highlights step-by-step (+160ms total compile):<br>`AgentIntentNet -> BITEMPORAL_CHANGE (91% Match)`<br>Orchestrating `ChangeNet` + `RS-VLM` segmentation. | *"Watch the Audit Trace: AgentIntentNet decomposes the command into a bi-temporal change task with 91% confidence, fusing SiameseChangeNet with our Remote Sensing Vision-Language Model in under 3 seconds."* |

---

### Scene 3: Quantitative Damage Breakdown (00:40 – 01:15)
| Timing | Screen Action | Spoken Narration (Voiceover) |
| :---: | :--- | :--- |
| **0:40 – 0:55** | Scroll through the **Land Transformation Breakdown** & **Quantitative Analysis** table:<br>- Active Transformation: **32,829.3 ha** (70.59% of area)<br>- Newly Flooded Area: **239.49 km²** (51.5% of AOI)<br>- Submerged Built-Up: **26.34 km²** | *"The results are stark and mathematically precise. The system isolates 32,829 hectares of active ground transformation.*<br><br>*239.5 square kilometers of newly flooded territory — covering 51.5% of the monitored region. Most critically, 26.3 square kilometers of inhabited settlements and roads are confirmed underwater."* |
| **0:55 – 01:15** | Pan across **Model-Wire Results**: Optical Model, SAR backscatter model, ChangeNet pixel mask, and the combined 4-color Fusion Map. | *"Four neural pipelines cross-validate the evidence: optical water indices, microwave radar attenuation, and pixel-level change deltas agree with 89% cross-sensor fidelity."* |

---

### Scene 4: Red Danger Zones vs Green Safe Corridors (01:15 – 01:45)
| Timing | Screen Action | Spoken Narration (Voiceover) |
| :---: | :--- | :--- |
| **1:15 – 1:30** | Hover over the **Impacted Areas Table**:<br>- `20.20°N, 85.83°E`: 59.07 km² Settlements (CRITICAL DANGER - RED, 92%)<br>- `20.23°N, 85.86°E`: 47.09 km² Staging Corridor (PRIMARY SAFE ZONE - GREEN, 96%) | *"SatQuery doesn't just display a heat map; it generates tactical coordinates.*<br><br>*Here in Red: Critical Danger Zone 1 at 20.20 degrees North — 59 square kilometers of submerged residential blocks requiring immediate boat evacuation.*<br><br>*And in Green: Safe Zone Alpha — 19,800 hectares of dry, elevated plateau at 96% confidence."* |
| **1:30 – 1:45** | Click **"Swipe Compare"** or look at the Studio viewer showing designated safe ridge boundaries. | *"Using the interactive Studio swipe tool, operators can verify that transit berms and staging corridors remain completely above the flood line."* |

---

### Scene 5: Field Rescue Directives & PDF Dossier (01:45 – 02:00)
| Timing | Screen Action | Spoken Narration (Voiceover) |
| :---: | :--- | :--- |
| **1:45 – 1:53** | Highlight the **Executive Strategic Assessment & Field Rescue Directives** card.<br>Click the glowing blue **"Executive Briefing Dossier" / "PDF Dossier"** button. | *"Under Executive Directives, actionable rescue protocols are summarized for field deployment. With one click, SatQuery compiles the full analysis, GPS coordinates, and hazard classifications into an ISRO-standard PDF Dossier."* |
| **1:53 – 2:00** | Final pan of the SatQuery AI dark command dashboard.<br>End title: **SatQuery AI — Autonomous Disaster Response Intelligence**. | *"From raw satellite rasters to tactical evacuation intelligence in under two minutes. That is SatQuery AI."* |

---

## 🎙️ Teleprompter Continuous Voiceover Script (120 Seconds)

```text
[00:00] When catastrophic monsoon flooding strikes, rescue commanders cannot wait hours for manual GIS mapping. They need instant situational awareness.

[00:10] Here, we ingest two temporal Sentinel-2 passes over a 665 square kilometer delta basin: the pre-flood baseline on the left, and the submerged landscape on the right. SatQuery AI instantly validates projection, alignment, and data quality across both rasters in milliseconds.

[00:25] The operator simply asks: "Detect flooded area, assess infrastructure damage, and show safe zones".

[00:33] Watch the Audit Trace: AgentIntentNet decomposes the command into a bi-temporal change task with 91% confidence, fusing SiameseChangeNet with our Remote Sensing Vision-Language Model in under 3 seconds.

[00:46] The results are stark and mathematically precise. The system isolates 32,829 hectares of active ground transformation. 239.5 square kilometers of newly flooded territory — covering 51.5% of the monitored region. Most critically, 26.3 square kilometers of inhabited settlements and roads are confirmed underwater.

[01:06] Four neural pipelines cross-validate the evidence: optical water indices, microwave radar attenuation, and pixel-level change deltas agree with 89% cross-sensor fidelity.

[01:18] SatQuery doesn't just display a heat map; it generates tactical coordinates. Here in Red: Critical Danger Zone 1 at 20.20 degrees North — 59 square kilometers of submerged residential blocks requiring immediate boat evacuation. And in Green: Safe Zone Alpha — 19,800 hectares of dry, elevated plateau at 96% confidence.

[01:36] Using the interactive Studio swipe tool, operators can verify that transit berms and staging corridors remain completely above the flood line.

[01:46] Under Executive Directives, actionable rescue protocols are summarized for field deployment. With one click, SatQuery compiles the full analysis, GPS coordinates, and hazard classifications into an ISRO-standard PDF Dossier.

[01:54] From raw satellite rasters to tactical evacuation intelligence in under two minutes. That is SatQuery AI.
```

---

## 💡 Key High-Impact Visual Highlights for Recording
1. **The 32,829 Hectare Influx Metric (`0:48`):** Keep the cursor over the `32829.3 ha` / `70.59%` metric row for 3 seconds—it immediately demonstrates quantitative precision.
2. **The Red vs Green Delineations (`1:22`):** Show the spatial contrast between the flooded red danger zone (`20.2000° N, 85.8300° E`) and the safe green staging corridor (`20.2300° N, 85.8600° E`).
3. **One-Click Dossier Click (`1:48`):** Conclude by clicking **"Executive Briefing Dossier"** to show instant operational readiness for disaster response teams.
