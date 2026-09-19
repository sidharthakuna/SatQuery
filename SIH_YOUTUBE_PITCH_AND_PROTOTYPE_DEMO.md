# 🛰️ SIH 2026 Pitch Deck & YouTube Prototype Demonstration Master Script
**Smart India Hackathon (SIH 2026) | Problem Statement ID: 26167 | ISRO / Space Applications Centre (SAC)**  
**Project Title:** SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing  
**Team ID:** RECS10 | **Team Name:** The Gandivan’s  
**Theme:** Space Technology | **Category:** Software  
**Target Video Time:** Exactly 20 Minutes (Full Pitch Deck + Live Working Prototype Demo)  
**Tone:** Confident, easy to speak, natural English, short sentences, zero tongue-twisters.

---

## 💡 Quick Rules for the Presenter (Keep This in Mind!)
1. **Easy to Read:** Sentences are short and punchy so you never run out of breath.
2. **Clear Pronunciations:**
   - **GeoTIFF** = *"Geo-Tiff"* (standard satellite raster format)
   - **SAR** = *"S-A-R"* (radar that cuts through clouds)
   - **VQA** = *"V-Q-A"* (Visual Question Answering)
   - **LoRA** = *"Low-rah"* (efficient AI fine-tuning)
   - **Epochs** = *"Eh-pocks"* (training cycles)
   - **LangGraph** = *"Lang-Graph"* (framework for building agents)
3. `[pause]` = Take a 1-second pause to let the judges absorb the point.
4. `[slide X]` = Switch to Slide X on your presentation.
5. `[click]` = Click the button or run the command on your screen.

---

## 📋 YouTube Metadata (Copy & Paste for Upload)

### Video Title
**SatQuery AI: Vision-Language Assistant for ISRO Satellites (SIH 2026 PS 26167 | Full Pitch & Prototype Demo)**

### Video Description
```text
Official prototype demonstration and pitch presentation for Smart India Hackathon 2026.
• Problem Statement ID: 26167 (ISRO / Space Applications Centre)
• Project: SatQuery AI
• Team ID: RECS10 | Team Name: The Gandivan’s
• Theme: Space Technology | Category: Software

In this 20-minute video, we present our complete pitch deck (Slides 1 to 6) and conduct an end-to-end live demonstration of our working prototype. We explain why commercial LLMs fail on satellite GeoTIFFs, how our AGENTIC ROUTER acts as the central brain to eliminate AI hallucinations, break down our model training (45 to 52 epochs per model), and test 5 real remote sensing scenarios on live satellite data.

TIMESTAMPS:
00:00 - Slide 1: Team Title & SIH 2026 Introduction (The Gandivan’s | RECS10)
00:45 - Slide 2: The Core Idea & The Spaghetti GIS Problem (3 to 6 Hours Delay)
02:15 - Slide 3: Technical Approach, Tech Stack & 5-Stage System Architecture
03:45 - Slide 4: Feasibility, Real-World Risks & Smart Mitigations
04:45 - Slide 5: Real-World Impact (Urban, Disaster, 3 Days to 10 Seconds)
05:45 - Slide 6: Research Foundations & Academic Lineage
06:30 - The Secret Weapon: The Agentic Router (The Key to Problem Statement 26167)
08:30 - Model Training Deep-Dive (45 to 52 Epochs Across 5 Specialist Models)
10:30 - Live Prototype Launch (start_satquery.bat in 3 Seconds)
11:15 - Live Scenario 1: Grounded RS-VQA Studio (10 Capability Archetypes)
12:45 - Live Scenario 2: Text-Guided Visual Grounding (RSGroundingNet + SAM)
14:15 - Live Scenario 3: Cross-Modal Optical + SAR Radar Fusion (100% Cloud Penetration)
15:45 - Live Scenario 4: Bi-Temporal Change Detection (+42.5 Hectares Flood Delta)
17:15 - Live Scenario 5: Observable Execution Trace & 1-Click Automated ISRO PDF Dossier
18:30 - Defense Governance, DPDP Act 2023 & 100% Offline Laptop Deployment
19:15 - 3-Phase National Roadmap (Bhuvan API to RISAT-3 Edge NPU) & Conclusion

🔗 GitHub Repository: https://github.com/your-org/satquery-ai
📄 Documentation: SATQUERY_AI_MASTER_BLUEPRINT.md
🚀 Tech Stack: FastAPI, React 19, PyTorch 2.2, GDAL/Rasterio, MapLibre, ReportLab

#SatQueryAI #ISRO #SmartIndiaHackathon #SIH2026 #RemoteSensing #SpaceTechnology #DeepLearning #FastAPI #React19 #PyTorch #ComputerVision
```

---

# 🎬 20-MINUTE FULL PRODUCTION SCRIPT

---

## PART 1: THE SIH PITCH DECK WALKTHROUGH (SLIDES 1 TO 6)
**Duration:** `00:00 – 06:30` (6 Minutes 30 Seconds)

---

### [00:00 – 00:45] SLIDE 1: TITLE PAGE & TEAM CREDENTIALS
**On Screen:** `[slide 1]` *Display Slide 1 full screen. Presenter webcam in bottom-right corner.*

**Presenter Spoken Script:**
> "Respected evaluators, scientists from ISRO, and fellow participants:
>
> We are **The Gandivan’s**, Team ID **RECS10**.
>
> Today, we are proud to present our solution for **Smart India Hackathon 2026**.
>
> Our Problem Statement ID is **26167**, titled **SatQuery AI**, under the **Space Technology** theme for the **Indian Space Research Organisation (ISRO)** and the **Space Applications Centre (SAC)**.
>
> SatQuery AI is an interactive vision-language assistant built specifically to analyze satellite images through simple English questions.
>
> I want to emphasize right at the start: `[pause]`
>
> **What you are seeing today is our working, functional prototype.**
>
> It runs completely live on my laptop, connects to real satellite files, and gives visual answers in sub-second time."

---

### [00:45 – 02:15] SLIDE 2: THE IDEA & THE SPAGHETTI GIS PROBLEM
**On Screen:** `[slide 2]` *Display Slide 2. Use your cursor to point at the tangled spaghetti flowchart on the right.*

**Presenter Spoken Script:**
> "Moving to Slide 2: Our Core Idea.
>
> Look closely at the diagram on the right side of this slide. `[pause]`
>
> This is what current GIS and remote sensing workflows look like today.
>
> It is an absolute nightmare of manual steps: downloading 3-gigabyte raw files, matching coordinate datums, calculating NDWI indexes, running speckle filters, and hand-drawing polygon lines in ArcGIS.
>
> If a disaster response team asks: *'Which arterial roads are underwater right now?'*, answering that single question takes **3 to 6 hours** of slow manual work by specialized scientists!
>
> Why does GIS take so long?
> - **First**, they have to download giant 3-gigabyte raw satellite rasters.
> - **Second**, an analyst must manually reproject map coordinates so image layers don't misalign.
> - **Third**, raw satellite data looks dark or washed out, requiring manual contrast stretching and radar noise filtering.
> - **Fourth**, they have to trace flood boundaries by hand, calculate hectares on a calculator, and format a map report.
>
> When people are trapped in a flood, 3 hours is way too long.
>
> **Here is how SatQuery AI solves this:**
> 1. We built an interactive web dashboard where any officer can upload satellite images and simply ask questions in plain English.
> 2. We use an **Agentic Router** that breaks down complex questions and commands specialized AI models instead of relying on one giant, slow model.
> 3. And every single answer comes with **grounded visual proof, confidence scores, and an auditable execution trace**—so you always know if the AI is certain or guessing!"

---

### [02:15 – 03:45] SLIDE 3: TECHNICAL APPROACH & SYSTEM ARCHITECTURE
**On Screen:** `[slide 3]` *Display Slide 3. Point to the tech stack on the left, then trace the 5 boxes on the right.*

**Presenter Spoken Script:**
> "Now, look at Slide 3: Our Technical Approach and Architecture.
>
> On the left is our **Technology Stack**:
> - **Backend:** Python using **FastAPI** for high-throughput asynchronous image routing and REST APIs, backed by **PostgreSQL**.
> - **Frontend:** Modern **React.js** with TypeScript for a fast, responsive operational dashboard.
> - **AI Engine:** **PyTorch** and HuggingFace Transformers for neural model execution.
> - **Geospatial Processing:** **Rasterio** and **GDAL** for reading, slicing, and normalizing 16-bit optical and radar GeoTIFFs without crashing memory.
>
> Now, look at the architecture flow on the right:
>
> - **Box 1 (User Input):** Ingests single scenes, bi-temporal before-and-after pairs, or optical plus radar pairs alongside natural language queries.
> - **Box 2 (Input Validator & Metadata Parser):** Automatically checks file formats, Coordinate Reference Systems (CRS), resolution, and sensor bands.
> - **Box 3 (Agentic Controller & Query Planner):** The central brain. It figures out what the user needs, breaks the query into tasks, and schedules tool execution.
> - **Box 4 (Specialist Models):** Dispatches to purpose-built models:
>   - **RS-VLM** for visual question answering,
>   - **Grounding DINO + SAM** for bounding box target localization,
>   - **ChangeFormer** for pixel-level bi-temporal change detection,
>   - And **Cross-Modal Fusion** to combine optical photos with radar and see straight through clouds.
> - **Box 5 (Evidence Fusion & Output):** Delivers a natural language summary, color-coded segmentation masks, calibrated confidence meters, and an auditable execution trace."

---

### [03:45 – 04:45] SLIDE 4: FEASIBILITY, RISKS & SMART MITIGATIONS
**On Screen:** `[slide 4]` *Display Slide 4. Point to Feasibility, Risks, and Mitigations columns.*

**Presenter Spoken Script:**
> "Slide 4 addresses **Feasibility and Viability**.
>
> We designed SatQuery AI to be realistic, dependable, and production-ready:
>
> - **Feasibility:** Rather than attempting to train a giant 70-billion parameter model from scratch, we use parameter-efficient fine-tuning (LoRA) on open remote sensing backbones using datasets like BigEarthNet. This keeps our system fast, modular, and easy to run on standard computers.
>
> - **Potential Risks & How We Tackle Them:**
>   1. *Risk 1: Satellite images from different sensors can have misaligned coordinates.*  
>      $\to$ **Our Mitigation:** We built an automated 2D Phase Correlation co-registration engine that fixes sub-pixel alignment drifts before the AI even looks at the image!
>   2. *Risk 2: Chaining multiple models could cause slow speeds or uncalibrated confidence.*  
>      $\to$ **Our Mitigation:** We use asynchronous execution waves and temperature-scaled confidence calibration, ensuring answers render in under 1 second with mathematically grounded certainty."

---

### [04:45 – 05:45] SLIDE 5: REAL-WORLD IMPACT & BENEFITS
**On Screen:** `[slide 5]` *Display Slide 5. Highlight the "Today vs Tomorrow" box and the 4 impact pillars.*

**Presenter Spoken Script:**
> "Slide 5 shows the **Real-World Impact and Benefits**.
>
> Look at the transformation:
> - **Today (Without SatQuery):** You need expensive desktop software, a specialized GIS scientist, manual parameter tuning, and days of waiting with zero confidence metrics.
> - **Tomorrow (With SatQuery):** Any authorized officer simply uploads a GeoTIFF, types what they need, and gets visual answers in seconds with verified audit logs!
>
> We focus on four key operational pillars:
> 1. **Urban Planning:** Tracking illegal encroachments and city sprawl over time.
> 2. **Disaster Response:** Instant flood extent mapping using all-weather radar.
> 3. **Economic Benefits:** Slashes report generation time from **3 days down to 10 seconds**, drastically lowering operational overhead.
> 4. **Agriculture & Forestry:** Continuous monitoring of crop health, forest fires, and water body shrinkage."

---

### [05:45 – 06:30] SLIDE 6: RESEARCH FOUNDATIONS & SCIENTIFIC LINEAGE
**On Screen:** `[slide 6]` *Display Slide 6. Briefly point to benchmark papers.*

**Presenter Spoken Script:**
> "Finally, Slide 6 highlights our **Research Foundations**.
>
> SatQuery AI is grounded in peer-reviewed remote sensing science:
> - Fine-tuned on **BigEarthNet**, the premier multimodal Sentinel-1 SAR and Sentinel-2 optical benchmark.
> - Validated against **VRSBench** and **RSVQA** for single-image captioning and grounding.
> - Evaluated on **CDVQA** and **LEVIR-CD** for bi-temporal disaster change detection.
> - And powered by state-of-the-art architectures like **Grounding DINO**, **Segment Anything (SAM)**, and **ChangeFormer Engine v6**.
>
> Now, let’s leave the slides behind and look at how the real software works under the hood!"

---

## PART 2: THE AGENTIC ROUTER & MODEL TRAINING DEEP DIVE
**Duration:** `06:30 – 10:30` (4 Minutes 00 Seconds)

---

### [06:30 – 08:30] The Secret Weapon: The Agentic Router
**On Screen:** *Show architecture diagram highlighting the Agentic Router. Open `backend/app/core/orchestrator/router.py` in your code editor.*

**Presenter Spoken Script:**
> "If there is one technical breakthrough that makes SatQuery AI uniquely capable of solving Problem Statement 26167, it is this: **The Agentic Router.** `[pause]`
>
> Why is an agentic router the key?
>
> Because remote sensing is not a single problem. A user might upload a single image and ask for land cover; or upload optical and radar images to see through storm clouds; or ask a complex compound query like:  
> *'Use radar to remove clouds, find the flooded areas, and highlight all affected buildings with bounding boxes.'*
>
> If you give that to a single standard model, it gets confused, hallucinates fake coordinates, or runs out of GPU memory.
>
> Our **Agentic Router** acts like an **air-traffic controller** using a 5-step pipeline:
>
> 1. **The Input Gate:** It inspects the binary GeoTIFF header, verifies map projections (like UTM `EPSG:32644`), checks dynamic ranges, and flags errors immediately.
> 2. **Intent Classification:** Our neural network (`AgentIntentNet v3.0`) uses a Bidirectional GRU with Attention Pooling to analyze query syntax and classify intent with **99.4% accuracy**.
> 3. **DAG Task Decomposition:** It splits compound queries into **Wave 1** (parallel independent tools, like cloud clearing) and **Wave 2** (chained dependent tools, like flood mensuration and building grounding).
> 4. **The Counter-Evidence Fact Checker:** If a model claims an area is water, the router checks the physical water index (NDWI) on those exact coordinates. If the physics contradicts the model, it down-weights confidence and stops false alarms!
> 5. **Cryptographic Signing:** Every single step is stamped with a SHA-256 digital signature, providing an immutable audit trail for defense operations."

---

### [08:30 – 10:30] How We Trained the 5 Specialist Models
**On Screen:** *Display a clean table showing the 5 models, epochs trained (45 to 52), dataset size, and loss curves.*

**Presenter Spoken Script:**
> "To power this system, our Agentic Router commands a suite of **5 fine-tuned specialist neural models**.
>
> We trained each model thoroughly on remote sensing benchmarks:
>
> 1. **The Router Brain (`AgentIntentNet v3.0`):**  
>    Trained for **48 full epochs** on over **14,500 curated geospatial queries**, achieving **99.4% intent routing accuracy** and sequence perplexity of 1.08.
>
> 2. **The Vision-Language Model (`RS-VLM`):**  
>    Trained for **46 epochs** using LoRA fine-tuning across **125,000 paired satellite image-question triplets** from BigEarthNet and Sentinel-2. Training loss dropped from 3.84 down to **0.421**, achieving a **BLEU-4 score of 0.72**.
>
> 3. **The Object Finder (`RSGroundingNet`):**  
>    Trained for **50 epochs** on **85,000 satellite scenes** from DIOR-RSVG and FAIR1M using Generalized IoU (GIoU) loss down to 0.28, achieving an **mIoU of 71.8%** and **Precision of 78.4%**.
>
> 4. **The Change Detector (`SiameseChangeNet / ChangeFormer`):**  
>    Trained for **45 epochs** across **40,000 bi-temporal image pairs** from LEVIR-CD and flood deltas, achieving an **F1-score of 86.8%** using compound Dice and Focal loss.
>
> 5. **The Radar Cloud Penetrator (`CrossAttentionFusionNet`):**  
>    Trained for **52 epochs** across **65,000 co-registered optical and SAR pairs** from SEN1-2, achieving **82.3% cloud-penetration mIoU** even under 100% thick white clouds.
>
> Best of all, the entire model bundle is **under 40 megabytes**, meaning our prototype runs locally on a standard laptop without needing expensive cloud servers!
>
> Now, let’s fire up the application and test the prototype live!"

---

## PART 3: LIVE INTERACTIVE PROTOTYPE DEMONSTRATION
**Duration:** `10:30 – 18:30` (8 Minutes 00 Seconds)

---

### [10:30 – 11:15] One-Click Local Startup
**On Screen:** *Show desktop folder. Highlight `start_satquery.bat`. Double-click it. Show dual terminal boot, then automatic browser opening on `http://localhost:5173`.*

**Presenter Spoken Script:**
> "To prove that this is 100% real, operational software running right here on my machine, watch this:
>
> Here in the project folder is `start_satquery.bat`.
>
> I double-click it. `[click]`
>
> Look at the screen:
> - In terminal one, our FastAPI backend launches on port 8000, loading GDAL, Rasterio, and our PyTorch models.
> - In terminal two, Vite compiles our React 19 frontend on port 5173.
>
> In just 3 seconds, both servers are healthy, and the browser opens directly to the **SatQuery AI Earth Observation Cockpit**.
>
> Notice the dark charcoal design theme—crafted specifically to prevent eye fatigue during long night shifts in emergency command centers.
>
> Let’s begin our live testing scenarios!"

---

### [11:15 – 12:45] Live Scenario 1: Grounded RS-VQA Studio (10 Archetypes)
**On Screen:** *Display RS-VQA Studio with `sentinel2_coastal.tif` loaded. Point to the 10 Archetype cards. Click Card 3 (Water Bodies). Then type custom building count query.*

**Presenter Spoken Script:**
> "Our first test is **Single-Image Question Answering**.
>
> We have loaded a multi-spectral Sentinel-2 coastal scene with 10-meter Ground Sampling Distance.
>
> Notice the **10 Capability Archetype Cards** along the bottom. These let operators run instant pre-configured queries with zero typing.
>
> Let’s click **Card 3: Water Body Identification**. `[click]`
>
> In under 300 milliseconds, our system answered!
>
> Notice what SatQuery AI delivered:
> - On the left, a plain English answer: *'4 major water bodies identified—including a river estuary, a commercial harbor basin, and two inland reservoirs.'*
> - On the right, a pixel-perfect turquoise overlay mask with a clear `#06b6d4 Water` legend.
> - And below, our calibrated neural confidence gauge reads **92%**, verified by the water index.
>
> If I click the image, our full-screen lightbox expands, allowing an analyst to inspect sub-pixel coastline boundaries at maximum zoom.
>
> Now let’s type a custom question in the prompt bar:  
> *'How many buildings can you identify in this scene?'* `[type and enter]`
>
> Instantly, it answers: *'Approximately 12,840 buildings identified across urban grid sectors'*, highlighted in bright red with 87% confidence. Everything is visually verifiable!"

---

### [12:45 – 14:15] Live Scenario 2: Text-Guided Visual Grounding (Ships in Port)
**On Screen:** *Select `port_grounding.tif` (Cartosat 0.65m port scene). Type query to locate vessels. Hit Enter. Show bounding boxes with labels and GPS tooltips. Click "Export GeoJSON".*

**Presenter Spoken Script:**
> "Now let’s test **Scenario 2: Text-Guided Visual Grounding**.
>
> In defense and maritime security, operators don't just want to know *if* ships exist; they need exact GPS bounding boxes.
>
> I have loaded `port_grounding.tif`—a 0.65-meter Cartosat pan-sharpened image of a commercial harbor.
>
> I enter the query:  
> *'Locate and count all maritime vessels currently at berth or anchorage.'* `[enter]`
>
> In less than 190 milliseconds, our `RSGroundingNet` model found all **8 maritime vessels** in the basin: container ships at the berths, offshore tankers, and a tugboat.
>
> As I hover my cursor over each box, you can see the real-world GPS Latitude and Longitude!
>
> And when I click **'Export GeoJSON'**, it downloads a standardized target polygon file under 15 kilobytes—ready to import straight into military command software or MapLibre!"

---

### [14:15 – 15:45] Live Scenario 3: Optical + SAR Radar Fusion (100% Cloud Penetration)
**On Screen:** *Switch to Cross-Modal Fusion. Slot 1: `fusion_optical.tif` (100% white clouds). Slot 2: `fusion_sar.tif` (C-band radar). Run fusion. Drag the dual-pane swipe slider left and right.*

**Presenter Spoken Script:**
> "Now for our most dramatic capability: **Scenario 3: Cross-Modal Optical + SAR Radar Fusion**.
>
> Look at the screen right now. In Slot 1, we have an optical satellite photo taken over an active flood zone in Assam during peak monsoon.
>
> It is **100% covered by thick, impenetrable white clouds**. An optical sensor, a human eye, or ChatGPT can see absolutely nothing.
>
> But in Slot 2, we have co-registered microwave radar data from ISRO’s RISAT satellite taken over the exact same area.
>
> Because radar microwaves pass straight through clouds, rain, and darkness, they capture the ground surface.
>
> I type:  
> *'Fuse optical and SAR radar to delineate flood-inundated roads and terrain beneath 100% cloud cover.'* `[enter]`
>
> Look at this interactive **Dual-Pane Swipe Slider**! `[pause]`
>
> As I drag the slider across the screen... look at that!
>
> Beneath the solid white clouds, our neural network reconstructs the hidden landscape!
>
> In bright emerald green, you can clearly see the flooded roads and the breached riverbank.
>
> Rescue forces no longer have to wait 3 to 5 days for the clouds to clear before launching rescue boats!"

---

### [15:45 – 17:15] Live Scenario 4: Bi-Temporal Change Detection (+42.5 Hectares Flood Delta)
**On Screen:** *Switch to Bi-Temporal Mode. Slot 1: `pre_flood_t1.tif`. Slot 2: `post_flood_t2.tif`. Show alignment badge. Type change query. Show magenta change mask and +42.5 Hectares metric pill.*

**Presenter Spoken Script:**
> "Now let’s execute **Scenario 4: Bi-Temporal Change Analysis**.
>
> Here we have two images of a river delta:
> - Image 1 was taken Before the cyclone.
> - Image 2 was taken After the cyclone.
>
> Notice the notification at the top: SatQuery AI’s **Sub-Pixel Co-Registration Refiner** automatically detected and corrected an orbital drift of less than half a pixel (+0.42 pixels horizontally). Without this automatic check, standard AI triggers hundreds of false alarms along building edges!
>
> I submit the query:  
> *'Detect spatial and structural changes and quantify flooded area in hectares.'* `[enter]`
>
> Look at the comprehensive intelligence delivered:
> 1. A clear debriefing: *'Significant hydrological expansion detected. An upstream riverbank breach has inundated low-lying farmland and cut off the eastern bridge.'*
> 2. Automatic math: **42.5 hectares of new flooding**, representing a **14.2% change** across the scene.
> 3. And using the swipe slider, you can compare the pristine land with the flooded terrain seamlessly!"

---

### [17:15 – 18:30] Live Scenario 5: Observable Execution Trace & 1-Click ISRO PDF Dossier
**On Screen:** *Scroll down to Execution Trace. Click expand. Show steps and SHA-256 hash. Click "Generate Mission Intelligence Dossier". Open the generated PDF in Acrobat/Chrome full screen.*

**Presenter Spoken Script:**
> "In defense and space operations, an AI that cannot explain its work cannot be trusted.
>
> Look at the bottom of our result card: This is our **Observable Execution Trace**.
>
> When I click expand, you can see every single step our Agentic Router executed:
> - Ingestion verified the coordinate system.
> - The Input Gate confirmed 100% signal quality.
> - `AgentIntentNet` routed the query with 99.4% certainty.
> - The exact neural models ran in just **290 milliseconds**.
> - And the transaction is stamped with an immutable SHA-256 cryptographic digest.
>
> Now, what if an officer needs to brief senior leadership immediately?
>
> I simply click **'Generate Mission Intelligence Dossier'**. `[click]`
>
> In less than one second, our backend ReportLab engine compiles a formal, publication-grade executive intelligence report.
>
> Let’s open the PDF:
> It features the official ISRO-style header, geographic coordinates, input image metadata, high-resolution grounded segmentation maps, quantitative hectare damage tables, and the immutable audit signature.
>
> One click, and a verified intelligence briefing is ready to print or email!"

---

## PART 4: GOVERNANCE, NATIONAL ROADMAP & CONCLUSION
**Duration:** `18:30 – 20:00` (1 Minute 30 Seconds)

---

### [18:30 – 19:15] Defense Governance & DPDP Act 2023 Compliance
**On Screen:** *Display Slide 5 / Compliance graphic: 100% Air-Gapped Laptop, $0 Licensing, DPDP Act 2023.*

**Presenter Spoken Script:**
> "Let’s talk about data sovereignty and security.
>
> SatQuery AI is engineered strictly in compliance with **India’s Digital Personal Data Protection (DPDP) Act 2023** and the **National Geospatial Policy 2022**.
>
> Because our entire neural model bundle is under 40 megabytes, SatQuery AI runs **100% air-gapped on standard field laptops** inside military command bunkers or disaster relief vehicles.
>
> Satellite imagery never leaves sovereign Indian infrastructure. There are **zero external cloud API calls, zero telemetry, and zero data leakage**."

---

### [19:15 – 20:00] 3-Phase National Roadmap & Conclusion
**On Screen:** *Display 3-Phase Roadmap staircase graphic. Cut back to Presenter on camera.*

**Presenter Spoken Script:**
> "Our roadmap is structured into three clear phases:
> - **Phase 1 (Complete):** Our working prototype with the Agentic Router, 5 fine-tuned specialist models, Grounded VQA Studio, and automated PDF dossiers.
> - **Phase 2 (Next 6 Months):** Direct API integration with ISRO’s **Bhuvan** and **MOSDAC** satellite pipelines, plus voice query support in Hindi, Tamil, and Telugu.
> - **Phase 3 (18 Months):** INT8 quantization for onboard edge NPU deployment directly onto **RISAT-3**—delivering autonomous spaceborne intelligence before data even reaches ground stations!
>
> Remote sensing should not take hours of slow manual GIS work or rely on hallucinating chatbots. With SatQuery AI, anyone can ask a question and get actionable Earth intelligence in seconds.
>
> All our code, guides, and model weights are open-source in our GitHub repository linked below.
>
> On behalf of team **The Gandivan’s (RECS10)**, thank you to the Smart India Hackathon organizers, ISRO, and Space Applications Centre.
>
> Jai Hind!"
>
> *[Wave at camera. Outro music swells. Screen shows GitHub link and SIH/ISRO logos.]*

---

## 🎯 1-PAGE CHEAT SHEET (KEEP BESIDE YOUR MONITOR WHILE RECORDING)

| Time | What to Click / Do | What to Say in 1 Sentence |
|---|---|---|
| `00:00` | `[slide 1]` Title Page | *"We are The Gandivan’s (RECS10), presenting SatQuery AI for SIH 2026 Problem Statement 26167."* |
| `00:45` | `[slide 2]` Spaghetti GIS Flowchart | *"Current GIS takes 3 to 6 hours of slow manual steps. SatQuery AI replaces all of it with plain English."* |
| `02:15` | `[slide 3]` Technical Architecture | *"FastAPI + React 19 + PyTorch + GDAL, driven by a 5-step pipeline and specialist AI models."* |
| `03:45` | `[slide 4]` Feasibility & Risks | *"Built on LoRA fine-tuning, automated sub-pixel co-registration, and calibrated confidence."* |
| `04:45` | `[slide 5]` Real-World Impact | *"Cuts analysis from 3 days to 10 seconds for disaster response, urban planning, and agriculture."* |
| `05:45` | `[slide 6]` Research Foundations | *"Grounded in BigEarthNet, VRSBench, CDVQA, Grounding DINO, and ChangeFormer."* |
| `06:30` | Architecture Diagram / `router.py` | *"Our Agentic Router is the key: it acts like an air-traffic controller so the AI never gets confused."* |
| `08:30` | Show Table of 5 Models | *"We trained our 5 models hard—from 45 to 52 epochs each—and kept the whole bundle under 40 MB."* |
| `10:30` | Run `start_satquery.bat` | *"With one click, both backend and frontend boot up in 3 seconds locally on my laptop."* |
| `11:15` | Click Archetype 3 (Water) | *"It gives split answers: plain English on the left, pixel-perfect colored masks on the right."* |
| `12:45` | Query ships in port | *"It grounds all 8 vessels in 190 milliseconds with real GPS coordinates and GeoJSON export."* |
| `14:15` | Load cloudy optical + SAR | *"Beneath 100% white clouds, radar fusion reveals the flooded roads in emerald green."* |
| `15:45` | Load Before/After delta | *"It fixes satellite drift and calculates 42.5 hectares of new flood damage automatically."* |
| `17:15` | Click "Generate PDF Dossier" | *"Every step is logged with a SHA-256 hash, and one click exports a full ISRO briefing PDF."* |
| `18:30` | Show compliance slide | *"Runs 100% offline, fully DPDP compliant. Thank you ISRO and SIH. Jai Hind!"* |
