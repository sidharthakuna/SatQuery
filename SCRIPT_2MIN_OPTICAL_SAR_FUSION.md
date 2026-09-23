# SatQuery AI — 2-Minute Optical + SAR Cloud Removal Demo Script
**Target Duration:** Exactly 2:00 Minutes (~270 spoken words at a clean, professional 135 wpm)  
**Topic:** Real-Time Cloud Penetration & Clear-Sky Optical Reconstruction via Cross-Modal SAR Fusion  
**Target Audience:** ISRO/SAC Evaluators, Hackathon Jury, Defense Analysts, YouTube Tech Community  
**Reference UI:** Live SatQuery AI Dashboard (`CROSS_MODAL_FUSION` Engine)  

---

## ⏱️ Video Timeline & Scene Breakdown

```
00:00 - 00:20 (20s) | Scene 1: The Problem & Dual-Sensor Ingestion
00:20 - 00:40 (20s) | Scene 2: Natural Language Command & Neural Routing
00:40 - 01:15 (35s) | Scene 3: Output Verification — Cloudy Optical vs SAR vs Clear Reconstruction
01:15 - 01:45 (30s) | Scene 4: Interactive Analysis — Swipe Compare & Studio Telemetry
01:45 - 02:00 (15s) | Scene 5: Automated Sector Delineation & PDF Dossier Export
```

---

## 🎬 Shot-by-Shot Director's Guide

### Scene 1: The Problem & Dual-Sensor Ingestion (00:00 – 00:20)
| Timing | Screen Action | Narration (Voiceover) |
| :---: | :--- | :--- |
| **0:00 – 0:10** | Zoom in on the input bar: drag and drop `sentinel2_optical_cloudy.png` & `sentinel1_sar_backscatter.png`.<br>Two chips appear with the green **"Pair Ready for Fusion / Change Analysis"** badge. | *"Over 60% of optical satellite imagery in tropical and coastal zones is compromised by cloud cover. Here, nearly half of this vital port is completely obscured.*<br><br>*To pierce this cloud deck, we load two simultaneous passes into SatQuery AI: a cloudy Sentinel-2 optical raster, paired with an all-weather Sentinel-1 SAR microwave pass."* |
| **0:10 – 0:20** | Cursor hovers over the two chips showing `768x768` dimensions and sensor badges (`[OPTICAL]` & `[SAR]`). | *"Our system instantly recognizes the multi-sensor pair and prepares the cross-modal pipeline."* |

---

### Scene 2: Natural Language Command & Neural Routing (00:20 – 00:40)
| Timing | Screen Action | Narration (Voiceover) |
| :---: | :--- | :--- |
| **0:20 – 0:30** | Type into the prompt bar:<br>`"Remove cloud cover and reconstruct clear sky optical view"`<br>Press **Enter**. | *"Instead of scripting complex SAR-to-optical radiometric transforms, the analyst simply asks: 'Remove cloud cover and reconstruct clear sky optical view'."* |
| **0:30 – 0:40** | The right panel flickers to **Audit Steps (8)** showing real-time routing:<br>`AgentIntentNet -> CROSS_MODAL_FUSION (98.4%)`<br>Streaming status indicators light up. | *"In the background, our 49,000-parameter AgentIntentNet routes the prompt directly to our PyTorch Cross-Attention Fusion Network — initiating cross-sensor alignment and deep feature extraction."* |

---

### Scene 3: Output Verification & Physics Explanation (00:40 – 01:15)
| Timing | Screen Action | Narration (Voiceover) |
| :---: | :--- | :--- |
| **0:40 – 0:55** | Smooth scroll through the 3-section structured response card:<br>- Image A: `OPTICAL MULTISPECTRAL (Cloud Contaminated)`<br>- Image B: `MICROWAVE SAR BACKSCATTER (All-Weather Penetration)` | *"In under 3 seconds, SatQuery delivers our structured intelligence card.*<br><br>*Section 1 juxtaposes the inputs: optical wavelengths between 0.4 and 0.7 microns scatter against cloud water droplets. But Sentinel-1 C-band microwaves at 5.4 gigahertz penetrate the cloud deck effortlessly, recording true surface roughness and shoreline geometry."* |
| **0:55 – 01:15** | Pan to Section 3:<br>`C. RECONSTRUCTED CLOUD-FREE OPTICAL SATELLITE IMAGE (CLEAR SKY)`<br>Zoom into the restored docks and urban streets. | *"Section 3 shows the result: a pristine, clear-sky optical reconstruction. 100% of the obscured terrain is recovered with correct optical spectral colors, crisp road networks, and intact coastal wharves — without AI hallucination."* |

---

### Scene 4: Interactive Analysis & Studio Telemetry (01:15 – 01:45)
| Timing | Screen Action | Narration (Voiceover) |
| :---: | :--- | :--- |
| **1:15 – 1:30** | Click **"Swipe Compare"** on the top toolbar.<br>Drag the interactive swipe slider across the image, revealing cloudy optical on the left and cloud-free reconstruction on the right. | *"Using the built-in 'Swipe Compare' tool, analysts can directly verify feature alignment. Notice how every pier and vessel boundary in the reconstructed scene matches the radar geometry perfectly."* |
| **1:30 – 1:45** | Focus on the right **Geospatial Workspace**:<br>- **89% Calibrated Confidence**<br>- Core Sensor Synthesis Insights: 48.0% cloud penetration, 100% terrain restored. | *"The right-hand Studio panel documents the full biophysical audit trail — calculating 48% cloud contamination, 100% terrain recovery, and an 89% calibrated confidence score."* |

---

### Scene 5: Sector Delineation & PDF Dossier Export (01:45 – 02:00)
| Timing | Screen Action | Narration (Voiceover) |
| :---: | :--- | :--- |
| **1:45 – 1:53** | Show delineated bounding boxes in the Studio viewer:<br>`Sector A: 5.1 ha`, `Sector B: 4.6 ha`, `Sector C: 3.5 ha`.<br>Click the **"Executive PDF" / "PDF Dossier"** button. | *"The system automatically delineates operational sectors with precise acreage calculations, and compiles the entire finding into an ISRO-standard Executive PDF Dossier with one click."* |
| **1:53 – 2:00** | Quick cut to the clean dashboard with terminal background.<br>End title card: **SatQuery AI — All-Weather Geospatial Intelligence**. | *"All running locally, in real time, with zero external cloud dependencies. That is SatQuery AI."* |

---

## 🎙️ Teleprompter Continuous Voiceover Script

```text
[00:00] Over 60% of optical satellite imagery in tropical and coastal zones is compromised by cloud cover. Here, nearly half of this vital port is completely obscured.

[00:10] To pierce this cloud deck, we load two simultaneous passes into SatQuery AI: a cloudy Sentinel-2 optical raster, paired with an all-weather Sentinel-1 SAR microwave pass. Our system instantly recognizes the multi-sensor pair and prepares the cross-modal pipeline.

[00:23] Instead of scripting complex SAR-to-optical radiometric transforms, the analyst simply asks: "Remove cloud cover and reconstruct clear sky optical view".

[00:33] In the background, our 49,000-parameter AgentIntentNet routes the prompt directly to our PyTorch Cross-Attention Fusion Network — initiating cross-sensor alignment and deep feature extraction.

[00:44] In under 3 seconds, SatQuery delivers our structured intelligence card.

[00:50] Section 1 juxtaposes the inputs: optical wavelengths between 0.4 and 0.7 microns scatter against cloud water droplets. But Sentinel-1 C-band microwaves at 5.4 gigahertz penetrate the cloud deck effortlessly, recording true surface roughness and shoreline geometry.

[01:06] Section 3 shows the result: a pristine, clear-sky optical reconstruction. 100% of the obscured terrain is recovered with correct optical spectral colors, crisp road networks, and intact coastal wharves — without AI hallucination.

[01:20] Using the built-in 'Swipe Compare' tool, analysts can directly verify feature alignment. Notice how every pier and vessel boundary in the reconstructed scene matches the radar geometry perfectly.

[01:32] The right-hand Studio panel documents the full biophysical audit trail — calculating 48% cloud contamination, 100% terrain recovery, and an 89% calibrated confidence score.

[01:44] The system automatically delineates operational sectors with precise acreage calculations, and compiles the entire finding into an ISRO-standard Executive PDF Dossier with one click.

[01:54] All running locally, in real time, with zero external cloud dependencies. That is SatQuery AI.
```

---

## 🎬 3 Tips for a Flawless 2-Minute Recording
1. **Swipe Divider (1:22):** Hold the swipe bar still right in the middle of the port for 2 seconds. The visual contrast between cloudy whiteout on the left and crisp coastal wharves on the right is your strongest demonstration shot.
2. **Smooth Cursor Motion:** Avoid rapid jittery mouse movements. Glide smoothly between the main card and the right Studio workspace.
3. **Audio Delivery:** Keep a steady, confident cadence. Every technical specification (5.4 GHz, C-band, 89% confidence, 48% cloud penetration) matches your live screen exactly.
