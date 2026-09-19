# 🛰️ SatQuery AI: 20-Minute Master YouTube Video Script
**Smart India Hackathon 2026 | Problem Statement ID: 26167 | ISRO / Space Applications Centre (SAC)**  
**Project:** SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing  
**Team ID:** RECS10 | **Team Name:** The Gandivan’s  
**Structure:** Exactly **5 Minutes for Problem Statement** & **15 Minutes for Live Working Prototype Demo** (Total: 20:00)  
**Style:** Professional, highly engaging, easy to articulate, clear speech rhythm (135–140 words per minute).

---

## ⏱️ Exact Video Timing Breakdown

| Part | Section | Time Window | Duration | Focus Area |
|:---:|---|:---:|:---:|---|
| **PART 1** | **The Problem Statement** | `00:00 – 05:00` | **5:00** | **Why satellite intelligence takes hours, why commercial LLMs fail, and the ISRO mandate** |
| | 1.1 The Hook & Team Credentials | `00:00 – 01:00` | 1:00 | The high-stakes emergency dilemma; Team The Gandivan's introduction |
| | 1.2 The "Spaghetti GIS" Bottleneck | `01:00 – 02:30` | 1:30 | The 4 excruciating manual steps; why answers take 3 to 6 hours today |
| | 1.3 Why Commercial LLMs & ChatGPT Fail | `02:30 – 03:45` | 1:15 | Discarding infrared bands, zero georeferencing, 34% hallucinations, cloud blindness |
| | 1.4 The SIH PS 26167 Mandate & SatQuery | `03:45 – 05:00` | 1:15 | The official ISRO/SAC challenge and how SatQuery AI solves it |
| **PART 2** | **The Live Prototype Demo** | `05:00 – 20:00` | **15:00** | **End-to-end interactive demonstration across 5 live operational missions** |
| | 2.1 Architecture, Tech Stack & 3-Second Boot | `05:00 – 06:45` | 1:45 | Agentic Router, 5 models (<40MB), `start_satquery.bat`, UI tour |
| | 2.2 Mission 1: Multi-Spectral RS-VQA | `06:45 – 09:15` | 2:30 | Sentinel-2 coastal scene, 10 archetype cards, building segmentation, 92% confidence |
| | 2.3 Mission 2: Text-Guided Visual Grounding | `09:15 – 11:45` | 2:30 | Cartosat 0.65m port scene, detecting 8 ships, GPS tooltips, GeoJSON export |
| | 2.4 Mission 3: Optical + SAR Radar Fusion | `11:45 – 14:30` | 2:45 | Assam 100% storm clouds, RISAT radar fusion, dual-pane slider cloud wipe |
| | 2.5 Mission 4: Bi-Temporal Change Detection | `14:30 – 17:00` | 2:30 | Pre/Post cyclone pair, sub-pixel co-registration, +42.5 Ha flooded delta |
| | 2.6 Mission 5: Execution Trace & ISRO PDF | `17:00 – 18:45` | 1:45 | Observable audit trace, SHA-256 hash, 1-click official intelligence PDF report |
| | 2.7 Security, National Roadmap & Sign-Off | `18:45 – 20:00` | 1:15 | 100% air-gapped laptop, DPDP compliance, Bhuvan to RISAT-3 edge NPU |

---

## 📋 YouTube Metadata (Copy & Paste for Upload)

### Video Title
**Inside SatQuery AI: Interactive Vision-Language Assistant for ISRO Satellites (SIH 2026 PS 26167)**

### Video Description
```text
Official prototype demonstration and pitch presentation for Smart India Hackathon 2026.
• Problem Statement ID: 26167 (ISRO / Space Applications Centre)
• Project: SatQuery AI
• Team ID: RECS10 | Team Name: The Gandivan’s
• Theme: Space Technology | Category: Software

TIMESTAMPS:
PART 1: THE PROBLEM STATEMENT (5 MINUTES)
00:00 - The Hook: The Golden Hour Dilemma in Remote Sensing
01:00 - The "Spaghetti GIS" Bottleneck (Why Satellite Answers Take 3 to 6 Hours)
02:30 - Why ChatGPT & Commercial Vision-LLMs Fail on Satellite Data
03:45 - The ISRO/SAC PS 26167 Challenge & The SatQuery AI Vision

PART 2: THE WORKING PROTOTYPE DEMONSTRATION (15 MINUTES)
05:00 - Architecture, 5 Specialist Models & 3-Second 1-Click Launch (start_satquery.bat)
06:45 - Mission 1: Multi-Spectral RS-VQA & Grounded Building Counting (Sentinel-2)
09:15 - Mission 2: Text-Guided Visual Grounding & Maritime Ship Detection (Cartosat 0.65m)
11:45 - Mission 3: Cross-Modal Optical + SAR Radar Fusion (Wiping 100% Cloud Cover)
14:30 - Mission 4: Bi-Temporal Change Detection & Automated Hectare Mensuration (+42.5 Ha)
17:00 - Mission 5: Observable Execution Trace & 1-Click Automated ISRO PDF Dossier
18:45 - 100% Air-Gapped Security, DPDP Compliance & 3-Phase National Roadmap

🔗 GitHub Repository: https://github.com/your-org/satquery-ai
📄 Architecture & Guides: SATQUERY_AI_MASTER_BLUEPRINT.md
💻 Tech Stack: FastAPI, React 19, PyTorch 2.2, GDAL/Rasterio, MapLibre, ReportLab

#SatQueryAI #ISRO #SmartIndiaHackathon #SIH2026 #RemoteSensing #SpaceTechnology #DeepLearning #FastAPI #React19 #PyTorch #ComputerVision
```

---

# 🎬 FULL PRODUCTION SCRIPT (20 MINUTES)

---

## 🔴 PART 1: THE PROBLEM STATEMENT (00:00 – 05:00)
**Goal:** Hook the viewer, vividly illustrate the 3 to 6-hour GIS crisis, explain why commercial AI fails, and present the ISRO Problem Statement 26167 mandate.  
**Tone:** Urgent, authoritative, clear, and empathetic.

---

### [00:00 – 01:00] The Hook: The Golden Hour Dilemma in Remote Sensing
**Visuals on Screen:**
- *00:00 – 00:20*: High-resolution satellite view of India zooming into an active coastal cyclone. Cut to footage of floodwater rushing into residential streets and disaster rescue personnel huddled around laptops.
- *00:20 – 00:45*: Cut to Presenter on webcam (bottom-right) or center frame with title slide behind.
- *00:45 – 01:00*: Display Team Intro Slide: The Gandivan’s | Team ID: RECS10 | SIH 2026 Problem Statement 26167 (ISRO / SAC).

**Presenter Spoken Dialogue:**
> "Right now, as we record this video, dozens of earth observation satellites from ISRO—like Cartosat, Resourcesat, and RISAT—are orbiting hundreds of kilometers above us, capturing terabytes of crystal-clear imagery every single hour.
>
> These satellites hold the exact ground truth needed during catastrophic emergencies:
> - Which roads are still navigable for rescue boats?
> - Where has the river breached its embankment?
> - And how many residential homes are submerged right now?
>
> But here is the tragic paradox of remote sensing today: `[pause]`
>
> **Having terabytes of satellite data is completely useless if the people on the ground cannot get answers in time.**
>
> In disaster management, the first 6 hours are called the 'Golden Hours'. Yet today, answering a single basic question from a satellite image takes anywhere between **3 to 6 hours** of agonizing manual GIS processing!
>
> Respected evaluators, scientists from ISRO, and hackathon judges:  
> We are **The Gandivan’s**, Team ID **RECS10**.  
> And today, we present **SatQuery AI**—our submission for **Smart India Hackathon 2026**, Problem Statement **26167** under the **Indian Space Research Organisation** and the **Space Applications Centre**."

---

### [01:00 – 02:30] The "Spaghetti GIS" Bottleneck: Why It Takes 3 to 6 Hours
**Visuals on Screen:**
- *01:00 – 01:40*: Display the "Spaghetti GIS Flowchart" diagram showing a tangle of desktop software (ArcGIS, QGIS, ENVI), manual CLI commands, and frustrated analysts.
- *01:40 – 02:30*: On-screen graphics illustrating the 4 manual hurdles: (1) 3GB file download, (2) Coordinate alignment, (3) Dark raw image contrast stretching, (4) Manual polygon tracing.

**Presenter Spoken Dialogue:**
> "To understand why we built SatQuery AI, you have to look at what an earth observation analyst actually goes through during an emergency.
>
> Imagine an NDRF commander contacts the GIS cell:  
> *'We need to know which bridges in this district are submerged.'*
>
> What happens next is what we call the **Spaghetti GIS Bottleneck**:
>
> - **First: The Massive Data Ingestion.**  
>   Satellite rasters aren't small phone JPEGs. They are massive multi-gigabyte 16-bit GeoTIFF files containing multiple spectral bands. Just downloading, unzipping, and loading them into heavy desktop software takes 30 to 45 minutes.
>
> - **Second: The Coordinate Nightmare.**  
>   Satellite sensors capture data at varying angles and orbital trajectories. If an analyst overlays a pre-flood map with a post-flood map, the coordinates often suffer from datum mismatches and sub-pixel drift. If you don't spend an hour manually reprojecting coordinates into UTM zones, your maps misalign by dozens of meters.
>
> - **Third: Radiometric Calibration and Noise.**  
>   Raw satellite sensors capture physical radiance values. To human eyes or standard software, the raw photo appears almost pitch black. Analysts must manually stretch contrast histograms, calibrate sensor gain, and run despeckling filters over radar signals.
>
> - **Fourth: Manual Digitization and Polygon Math.**  
>   Finally, the scientist must sit with a mouse, hand-trace the flood perimeter polygon by polygon, calculate the pixel count, convert it to hectares using a calculator, and manually compile an intelligence brief.
>
> By the time that manual workflow is completed?  
> **Three to six hours have vanished.**  
>
> When floodwaters are rising by 20 centimeters an hour, waiting six hours for a report is not just inefficient—it can cost lives."

---

### [02:30 – 03:45] Why Commercial AI & General LLMs Fail on Satellite Data
**Visuals on Screen:**
- *02:30 – 03:00*: Split-screen comparison. Show ChatGPT / Claude failing to open a `.tif` file, outputting an error or hallucinating fake bounding boxes.
- *03:00 – 03:45*: Infographic showing the 5 failure points of general LLMs: (1) Spectral loss, (2) No georeferencing, (3) 34% hallucination rate, (4) Cloud blindness, (5) Data sovereignty violation.

**Presenter Spoken Dialogue:**
> "Now, you might ask: *'We live in the era of generative AI. Why can't an officer simply upload a satellite photo to ChatGPT, Claude, or Gemini and ask questions?'*
>
> That was the very first avenue our team investigated. And commercial vision-language models fail catastrophically for five fundamental reasons:
>
> 1. **They Destroy Invisible Spectral Bands:**  
>    Everyday cameras only capture red, green, and blue light. But remote sensing relies on Shortwave Infrared, Near-Infrared, and thermal bands. That invisible light is how scientists detect soil moisture, plant health, and water boundaries. Commercial LLMs force images into standard 8-bit RGB, literally throwing away the scientific data before analysis even starts!
>
> 2. **Zero Spatial Georeferencing:**  
>    A commercial chatbot does not understand Coordinate Reference Systems, Ground Sampling Distance, or UTM projections. It cannot calculate real-world hectares or output geographic coordinates.
>
> 3. **The 34% Hallucination Danger:**  
>    Academic benchmarks show general multimodal LLMs hallucinate spatial relationships in overhead imagery up to **34% of the time**. In defense or disaster response, you cannot deploy rescue teams to hallucinated coordinates!
>
> 4. **Total Cloud Blindness:**  
>    During severe storms, optical cameras capture 100% thick white clouds. Standard AI looks at the clouds and says: *'I only see white haze.'* It cannot ingest synthetic aperture radar to see beneath the storm.
>
> 5. **Data Sovereignty Violations:**  
>    Under **India's DPDP Act 2023** and the **National Geospatial Policy 2022**, sensitive sovereign satellite data cannot be uploaded to foreign cloud APIs."

---

### [03:45 – 05:00] The SIH PS 26167 Mandate & The SatQuery Solution
**Visuals on Screen:**
- *03:45 – 04:20*: Display the official Smart India Hackathon Problem Statement 26167 slide with ISRO / SAC emblem. Highlight key requirements: Multimodal RS-VQA, Spatial Grounding, Radar-Optical Fusion, Sub-second Inference, Verifiable Traceability.
- *04:20 – 05:00*: Introduce SatQuery AI graphic: The Agentic Router commanding 5 lightweight models (<40MB total) running 100% locally on a standard laptop.

**Presenter Spoken Dialogue:**
> "This brings us directly to the mandate of **Smart India Hackathon 2026, Problem Statement 26167**, titled:  
> *'Interactive Vision-Language Assistant for Multimodal Remote Sensing'*—framed directly by **ISRO's Space Applications Centre**.
>
> ISRO laid down an ambitious engineering challenge:
> - Enable non-GIS experts to query earth observation rasters using conversational English.
> - Deliver pixel-level spatial grounding and real-world coordinate localization.
> - Enable all-weather vision through optical and radar cross-modal fusion.
> - Quantify disaster change over time in exact physical hectares.
> - And guarantee full explainability with zero black-box hallucinations.
>
> To solve this challenge, our team built **SatQuery AI**.
>
> Instead of relying on a single slow, uncalibrated giant model, SatQuery AI is powered by an **Agentic Router** that commands **5 purpose-built specialist neural models**—all fine-tuned on real satellite benchmarks, all bundled into **under 40 megabytes**, and capable of running **100% offline on a standard field laptop**!
>
> That is the problem statement and our design philosophy.
>
> Now, let's leave the slides behind. For the next 15 minutes, we will demonstrate our fully functional prototype live across five real-world operational missions!"

---

## 🟢 PART 2: THE LIVE PROTOTYPE DEMONSTRATION (05:00 – 20:00)
**Goal:** Show the working prototype running live on a standard laptop, proving 1-click startup, sub-second execution, and flawless performance across 5 missions.  
**Tone:** Confident, enthusiastic, technical yet accessible, driving home real-world utility.

---

### [05:00 – 06:45] Architecture, Tech Stack & 3-Second 1-Click Launch
**Visuals on Screen:**
- *05:00 – 05:35*: Display high-level Architecture Diagram highlighting: FastAPI Backend, React 19 Frontend, PyTorch Engine, Rasterio/GDAL Pipeline, and the Agentic Router.
- *05:35 – 06:05*: Minimize presentation to Windows Desktop. Highlight `start_satquery.bat`. Double-click it. Show dual terminal windows launching: Backend on port 8000 and Vite Frontend on port 5173.
- *06:05 – 06:45*: Chrome automatically opens to `http://localhost:5173`. Tour the interface: Header telemetry, Map viewport, Chat prompt bar, and the 10 Archetype cards along the bottom.

**Presenter Spoken Dialogue:**
> "Welcome to the live prototype demonstration of **SatQuery AI**.
>
> Before we click a single button, I want to emphasize our technical foundation:
> - Our backend is built on **Python and FastAPI** for high-concurrency asynchronous task dispatching.
> - Our frontend is built with **React 19 and TypeScript**, offering a modern, zero-latency reactive dashboard.
> - For geospatial manipulation, we interface directly with **Rasterio and GDAL** to slice multi-band 16-bit GeoTIFFs without blowing up laptop RAM.
> - And our neural brain is powered by **PyTorch 2.2**, orchestrating 5 specialist models that trained between 45 and 52 epochs each.
>
> Watch how easy it is to deploy in the field:
>
> Right here on my Windows desktop is a single script: `start_satquery.bat`.
>
> I double-click it. `[click]`
>
> Watch both terminals boot up in real time:
> - Terminal 1 initializes our FastAPI server on `port 8000`, loading our model weights and geospatial libraries into memory.
> - Terminal 2 compiles our Vite React server on `port 5173`.
>
> **In exactly 3.2 seconds, both servers are healthy**, and our browser automatically opens to the **SatQuery AI Operational Cockpit**!
>
> Notice the visual design:
> We engineered a custom **Dark Charcoal Command Center Theme**. Why? Because in actual defense command bunkers and disaster operations centers, personnel work 12-hour night shifts. A blinding white UI causes severe eye fatigue.
>
> Across the top, you see the active scene metadata: sensor type, coordinate projection, and resolution.  
> On the right is our natural language conversational terminal.  
> And along the bottom, we have **10 Capability Archetype Cards**—allowing operators to launch critical queries with a single mouse click!
>
> Let's run our first operational mission."

---

### [06:45 – 09:15] Mission 1: Multi-Spectral RS-VQA & Grounded Building Counting
**Visuals on Screen:**
- *06:45 – 07:15*: Load `sentinel2_coastal.tif` (10m Multi-Spectral Sentinel-2 scene). Point out metadata strip: UTM Zone 44N, 10m GSD.
- *07:15 – 07:45*: Click **Card 3: Water Body Identification**. Watch the screen update in under 300 milliseconds.
- *07:45 – 08:15*: Point out the split results: Plain-English summary on left, turquoise `#06b6d4` water segmentation mask on right, and the 92% calibrated confidence badge. Click image to trigger full-screen lightbox zoom.
- *08:15 – 09:15*: Type query into chat bar: *"How many buildings can you identify in this scene?"* Hit Enter. Show instant response: 12,840 buildings segmented in red with 87% confidence.

**Presenter Spoken Dialogue:**
> "Our first mission represents **Single-Scene Visual Question Answering and Infrastructure Grounding**.
>
> Imagine an emergency coordination team arriving at an unfamiliar coastal district during a flood alert.
>
> We ingest a multi-spectral Sentinel-2 satellite scene with a 10-meter ground sampling distance.
>
> An operator doesn't need to write complex SQL or GIS scripts. Look at the bottom dock:  
> I simply click **Card 3: Water Body Identification**. `[click]`
>
> **Look at that response speed—under 300 milliseconds!**
>
> Notice the multi-modal evidence SatQuery AI delivers:
> 1. On the left, an unambiguous plain English briefing:  
>    *'4 major water bodies identified—including a primary river estuary, a commercial harbor basin, and two inland drainage reservoirs.'*
> 2. On the right, our neural segmentation engine paints a pixel-perfect turquoise overlay mask directly on the water bodies, accompanied by a dynamic legend.
> 3. Down here, look at our **Calibrated Confidence Meter: 92%**.  
>    Our Agentic Router automatically calculated the physical Normalized Difference Water Index (NDWI) on these exact pixels. Because the physical index matched the neural prediction, confidence is verified!
>
> If I click the image, it opens in our full-screen inspection lightbox. An analyst can pan and zoom down to individual shoreline pixels.
>
> Now, let's take it a step further. An officer needs to know the population risk:  
> I type into the conversational prompt bar:  
> *'How many buildings can you identify in this scene?'* `[type and enter]`
>
> Instantly, SatQuery AI responds:  
> *'Approximately 12,840 buildings identified across urban grid sectors'*, and colors every single residential and commercial rooftop in bright red with 87% confidence.
>
> In less than 30 seconds and two simple interactions, a commander knows exactly where the water is and exactly where the vulnerable settlements are located."

---

### [09:15 – 11:45] Mission 2: Text-Guided Visual Grounding & Maritime Target Detection
**Visuals on Screen:**
- *09:15 – 09:45*: Select `port_grounding.tif` from scene selector (Cartosat 0.65m pan-sharpened harbor scene).
- *09:45 – 10:30*: Type query: *"Locate and count all maritime vessels currently at berth or anchorage."* Press Enter.
- *10:30 – 11:00*: Watch `RSGroundingNet` detect all 8 ships in 190ms. Show green bounding boxes with confidence scores. Hover cursor over boxes to reveal tooltips with real-world GPS Latitude/Longitude.
- *11:00 – 11:45*: Click the blue button **'Export GeoJSON'**. Show file download (`vessels_detected.geojson`, 12 KB). Open file in VS Code or text editor to show valid GeoJSON FeatureCollection with polygon coordinates.

**Presenter Spoken Dialogue:**
> "Now let's move to **Mission 2: Text-Guided Visual Grounding for Maritime Security and Defense**.
>
> In coastal surveillance, intelligence officers don't just want a text description saying 'there are ships in the harbor'. They need precise, geolocated bounding boxes that can be fed into patrol boat GPS systems.
>
> We load `port_grounding.tif`—a 0.65-meter sub-meter resolution Cartosat scene of a busy port.
>
> I enter our natural language query:  
> *'Locate and count all maritime vessels currently at berth or anchorage.'* `[enter]`
>
> In just **190 milliseconds**, our fine-tuned `RSGroundingNet` model executes:
>
> Look at the canvas!  
> It has located and bounded all **8 maritime vessels** across the harbor:
> - Five container cargo ships berthed along the northern wharf.
> - Two commercial petroleum tankers anchored in the outer channel.
> - And one service tugboat maneuvering near the dock!
>
> Now watch this: When I hover my cursor over any of these bounding boxes... `[hover mouse]`  
> **A floating telemetry tooltip pops up showing the real-world GPS coordinates:** Latitude 18.924 degrees North, Longitude 72.836 degrees East!
>
> How does SatQuery AI do this? It reads the affine transformation matrix embedded in the GeoTIFF header and maps image pixel coordinates directly to real-world WGS84 geographic coordinates.
>
> And when the officer needs to dispatch the Coast Guard?  
> Simply click this button: **'Export GeoJSON'**. `[click]`
>
> In one second, a lightweight, 12-kilobyte GeoJSON vector file downloads to the laptop. That file can be imported immediately into QGIS, MapLibre, or military tactical screens.
>
> Zero manual digitizing. Complete operational automation."

---

### [11:45 – 14:30] Mission 3: All-Weather Optical + SAR Radar Cross-Modal Fusion
**Visuals on Screen:**
- *11:45 – 12:15*: Switch to the **Cross-Modal Fusion** tab.
- *12:15 – 12:45*: Show Slot 1 (`fusion_optical.tif`): A satellite view during peak Assam monsoon, 100% obscured by thick white cumulus storm clouds. Show Slot 2 (`fusion_sar.tif`): C-band synthetic aperture radar from ISRO's RISAT satellite.
- *12:45 – 13:15*: Type query: *"Fuse optical and SAR radar to delineate flood-inundated roads beneath 100% cloud cover."* Press Enter.
- *13:15 – 14:30*: The interactive **Dual-Pane Swipe Slider** appears. Presenter grabs the slider handle with the cursor and drags it slowly back and forth across the screen. Show clouds literally vanishing, revealing flooded roads and breached dikes highlighted in vibrant emerald green.

**Presenter Spoken Dialogue:**
> "Now, we arrive at what I believe is the most technologically advanced capability in SatQuery AI: **Mission 3: Cross-Modal Optical and SAR Radar Fusion**.
>
> Every year during the Indian monsoon, states like Assam, Bihar, and Odisha face catastrophic flooding. But when rescue teams open their optical satellite feed, look at what they see on screen in **Slot 1**:  
> **100% dense, impenetrable white clouds.**  
> An optical satellite camera, a human eye, or a commercial AI model like ChatGPT can see absolutely nothing. To standard tools, the ground does not exist.
>
> But in **Slot 2**, we load co-registered microwave Synthetic Aperture Radar (SAR) from ISRO's **RISAT** satellite, captured at the exact same hour.
>
> Radar microwaves operate at centimeter wavelengths. They cut effortlessly through clouds, heavy rain, atmospheric haze, and pitch darkness!
>
> We issue the command:  
> *'Fuse optical and SAR radar to delineate flood-inundated roads beneath 100% cloud cover.'* `[enter]`
>
> Our specialized neural model—`CrossAttentionFusionNet`, trained for 52 epochs on 65,000 multimodal pairs—fuses both rasters in memory.
>
> Look at this interactive **Dual-Pane Swipe Slider** on my screen! `[pause]`
>
> Watch what happens as I drag this divider line across the frame: `[drag slider]`
>
> **The storm clouds literally disappear before your eyes!**
>
> Look underneath the white cloud cover:
> - In vibrant emerald green, our model has reconstructed the submerged terrain!
> - You can see the national highway cut in half by floodwaters.
> - You can clearly trace the breached riverbank where water is pouring into agricultural villages.
>
> Think about the human impact of this feature:  
> Rescue teams no longer have to wait 3 to 5 days for the storm to clear before deploying rescue boats. With SatQuery AI, they have all-weather ground vision in real time."

---

### [14:30 – 17:00] Mission 4: Bi-Temporal Change Detection & Automated Hectare Mensuration
**Visuals on Screen:**
- *14:30 – 15:00*: Switch to the **Bi-Temporal Mode** tab. Load `pre_flood_t1.tif` (dry baseline) into Slot 1, and `post_flood_t2.tif` (post-cyclone) into Slot 2.
- *15:00 – 15:30*: Point out the green badge: *"Sub-Pixel Co-Registration Refiner: +0.42px drift corrected"*. Explain why this prevents false change alarms.
- *15:30 – 16:15*: Enter query: *"Detect spatial and structural changes and quantify flooded area in hectares."* Press Enter.
- *16:15 – 17:00*: Display results: The three-part output. (1) Debriefing narrative, (2) Metric pills (+42.5 Hectares / 14.2% delta), (3) Magenta change mask with swipe comparison.

**Presenter Spoken Dialogue:**
> "Let's proceed to **Mission 4: Bi-Temporal Disaster Change Analysis and Area Mensuration**.
>
> Once an immediate disaster stabilizes, district authorities and insurance bodies need quantitative metrics:  
> *'Exactly how many hectares of farmland were flooded, and which transport links were severed?'*
>
> In the Bi-Temporal view, we load two satellite scenes of a river basin:
> - **Image T1** was taken on August 10th, before the cyclone.
> - **Image T2** was taken on August 24th, after the cyclone.
>
> Look at this green indicator banner at the top of the viewport:  
> **'Sub-Pixel Co-Registration Refiner: +0.42px orbital drift corrected.'**
>
> This is a critical geospatial innovation. When satellites revisit an area two weeks apart, slight orbital oscillations cause images to shift by a fraction of a pixel. If an AI doesn't correct this sub-pixel drift, it flags thousands of false changes along crisp road lines and roof edges. SatQuery AI automatically aligns both rasters using 2D phase correlation before running AI analysis!
>
> Now, we submit our query:  
> *'Detect spatial and structural changes and quantify flooded area in hectares.'* `[enter]`
>
> Look at the comprehensive intelligence delivered by our `SiameseChangeNet` model:
>
> 1. **The Executive Debriefing:**  
>    *'Catastrophic hydrological expansion detected. An upstream riverbank breach has flooded surrounding low-lying agricultural fields and inundated the eastern arterial roadway.'*
> 2. **Exact Geospatial Mensuration:**  
>    Look at these metric badges: **+42.5 Hectares of newly inundated terrain**, representing a **14.2% change** across the surveyed area!
> 3. **The Visual Delta:**  
>    A high-contrast magenta mask covers the exact flood extension. And using the swipe tool, an analyst can slide back and forth to visually verify the dry pre-cyclone land against the flooded post-cyclone reality.
>
> No manual formulas. No calculator errors. Exact physical metrics in 2 seconds."

---

### [17:00 – 18:45] Mission 5: Observable Execution Trace & 1-Click Automated ISRO PDF Dossier
**Visuals on Screen:**
- *17:00 – 17:40*: Scroll to the bottom of the result panel. Click **'Expand Execution Trace'**. Walk through the 5 chronological execution steps, showing millisecond timings and the SHA-256 digital signature.
- *17:40 – 18:10*: Click the prominent button: **'Generate Mission Intelligence Dossier'**. Show instant toast notification *"Dossier compiled in 820ms"*.
- *18:10 – 18:45*: The browser opens the generated PDF in Acrobat/Chrome full-screen. Scroll through the PDF: Official ISRO-style header, map boundaries, sensor metadata table, high-resolution before/after images, change mask, quantitative hectare table, and cryptographic stamp.

**Presenter Spoken Dialogue:**
> "In defense and high-stakes government operations, black-box AI is unacceptable. If an AI cannot explain its reasoning and prove its data integrity, commanders cannot rely on it.
>
> Look at the bottom of our output card: This is our **Observable Execution Trace**.
>
> When I click expand... `[click]`  
> You can audit every microsecond of computation:
> - **Step 1 (Ingestion):** GeoTIFF validated in 12ms; UTM EPSG:32644 datum confirmed.
> - **Step 2 (Routing):** `AgentIntentNet` classified the intent as `CHANGE_DETECTION_QUANTITATIVE` with 99.4% certainty.
> - **Step 3 (Co-Registration):** Sub-pixel drift corrected in 45ms.
> - **Step 4 (Neural Inference):** PyTorch engine executed in 280ms.
> - **Step 5 (Audit Stamp):** An immutable **SHA-256 cryptographic digest** was generated, guaranteeing this intelligence report cannot be tampered with!
>
> Now, imagine an officer has 5 minutes before briefing the Cabinet Secretary or ISRO Director. They don't have time to take screenshots and format Word documents.
>
> Watch this: I click **'Generate Mission Intelligence Dossier'**. `[click]`
>
> In just **820 milliseconds**, our backend ReportLab engine compiles a formal, publication-grade executive PDF.
>
> Let's look at the generated report on screen:
> - Official ISRO/SAC mission header and timestamp.
> - Geographic bounding box coordinates and sensor sensor metadata.
> - High-resolution side-by-side satellite imagery with grounded segmentation overlays.
> - The quantified hectare damage breakdown table.
> - And at the bottom: our cryptographic verification stamp and model version logs.
>
> In one single click, a verified intelligence briefing is ready to print, sign, and act upon!"

---

### [18:45 – 20:00] 100% Air-Gapped Security, DPDP Compliance & National Roadmap
**Visuals on Screen:**
- *18:45 – 19:20*: Display the Governance & Deployment slide: Highlight "100% Air-Gapped Laptop", "$0 Cloud Fees", and "DPDP Act 2023 & Geospatial Policy 2022 Compliance".
- *19:20 – 19:45*: Display the 3-Phase National Roadmap: Phase 1 (Working Prototype), Phase 2 (Bhuvan/MOSDAC + Indic Voice), Phase 3 (RISAT-3 Onboard NPU).
- *19:45 – 20:00*: Cut back to Presenter on webcam full-screen. Display GitHub QR code and repository link. Presenter delivers closing sign-off.

**Presenter Spoken Dialogue:**
> "To conclude our demonstration, let's address data security and real-world deployment.
>
> SatQuery AI was engineered from day one to comply fully with **India's Digital Personal Data Protection (DPDP) Act 2023** and the **National Geospatial Policy 2022**.
>
> Because our entire suite of 5 specialist models occupies **under 40 megabytes**, SatQuery AI runs **100% air-gapped on a standard field laptop**.
>
> There are zero external cloud API calls, zero third-party telemetry, and zero data leakage. Classified defense imagery never leaves sovereign Indian premises!
>
> Our deployment roadmap is structured into three phases:
> - **Phase 1 (Complete):** Our fully functional prototype demonstrated today with the Agentic Router, 5 fine-tuned models, RS-VQA, grounding, SAR fusion, and 1-click PDF dossiers.
> - **Phase 2 (Next 6 Months):** Direct API integration with ISRO's **Bhuvan** and **MOSDAC** data portals, alongside multilingual voice querying in Hindi, Tamil, and Telugu.
> - **Phase 3 (18 Months):** INT8 quantization for onboard edge NPU deployment directly onto **RISAT-3**—allowing satellites to analyze their own imagery in orbit and beam down instant intelligence alerts directly to ground stations!
>
> Remote sensing intelligence should not be trapped behind 6 hours of manual GIS workflows or untrustworthy chatbots. With SatQuery AI, any authorized officer can ask a question in plain English and receive verified Earth intelligence in seconds.
>
> All of our code, trained weights, and technical blueprints are completely open-source in our GitHub repository linked below.
>
> On behalf of team **The Gandivan’s (RECS10)**, we thank the Smart India Hackathon organizers, the Indian Space Research Organisation, and the Space Applications Centre for this incredible problem statement.
>
> Jai Hind, and thank you!"
>
> *[Presenter waves at camera. Outro graphic displays with GitHub link, ISRO emblem, and SIH 2026 logos. Video fades to black.]*

---

## 🎯 1-Page Teleprompter Cheat Sheet (Print or Keep on Monitor)

| Timestamp | Visual Action | Core Spoken Line / Hook |
|:---:|---|---|
| `00:00` | Cyclone satellite zoom $\to$ Webcam | *"Having terabytes of satellite data is useless if rescue teams cannot get answers in time."* |
| `01:00` | Spaghetti GIS Flowchart | *"Downloading 3GB files, coordinate datums, and hand-tracing polygons takes 3 to 6 hours!"* |
| `02:30` | Commercial LLM failure graphic | *"ChatGPT strips away infrared bands, has no georeferencing, and hallucinates 34% of the time."* |
| `03:45` | SIH PS 26167 Slide & Solution | *"ISRO demanded conversational VQA, grounding, and SAR fusion. SatQuery AI delivers it in <40MB."* |
| `05:00` | Double-click `start_satquery.bat` | *"Both FastAPI and React 19 boot up locally in 3.2 seconds on my laptop."* |
| `06:45` | Click Card 3 (Water), Query buildings | *"In under 300ms, it delivers plain English, turquoise water masks, and 12,840 buildings in red."* |
| `09:15` | Query ships on Cartosat 0.65m | *"Finds all 8 vessels with real GPS coordinates and exports a 12KB GeoJSON file in 1 second."* |
| `11:45` | Optical + SAR slider drag | *"As I drag the slider, 100% storm clouds disappear to reveal flooded roads in emerald green!"* |
| `14:30` | Bi-Temporal Pre/Post cyclone pair | *"It corrects sub-pixel drift and calculates 42.5 hectares of new flooding automatically."* |
| `17:00` | Expand trace $\to$ Click "Generate PDF" | *"Every step is stamped with SHA-256, and 1 click compiles an official ISRO briefing PDF in 820ms."* |
| `18:45` | Compliance & Roadmap $\to$ Sign-off | *"Runs 100% offline, zero cloud leaks. Thank you ISRO and SIH. Jai Hind!"* |
