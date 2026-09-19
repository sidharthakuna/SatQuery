# 🛰️ SatQuery AI — 20-Minute YouTube Script (Super Simple English)
**Smart India Hackathon 2026 | Problem: 26167 (ISRO / SAC)**  
**Team:** The Gandivan’s (Team ID: RECS10)  
**Style:** Very simple English words only. Short lines. Easy to read. Easy to write.

---

## ⏱️ Video Time Plan
* **Part 1 (Under 4 Minutes):**  
  * Who we are, the problem, why ChatGPT fails, our Agentic Router, and the 5 models.
* **Part 2 (15 Minutes):**  
  * Live demo on laptop: 1-click start, finding water and houses, finding ships, seeing through clouds with radar, flood damage in hectares, and the 1-click PDF report.

---

## 📋 YouTube Details (Copy & Paste)

### Title
**We Built an AI for ISRO Satellites: Inside SatQuery AI Prototype (SIH 2026 PS 26167)**

### Description
```text
Getting quick answers from satellite images during a flood takes hours of slow manual work. 

For Smart India Hackathon 2026 (Problem Statement 26167 for ISRO / Space Applications Centre), our team—The Gandivan’s (RECS10)—built SatQuery AI. It lets anyone ask questions about satellite pictures in plain English.

In the first 4 minutes, I explain the problem, why ChatGPT fails, and how our AGENTIC ROUTER works. Then, for the next 15 minutes, we test our working prototype live on my laptop across 5 real missions!

TIMESTAMPS:
00:00 - The Problem: Why Satellite Answers Take 6 Hours
01:15 - Why ChatGPT Fails & How the Agentic Router Works
02:30 - How We Trained Our 5 Models (45 to 52 Rounds, Under 40 MB)
03:45 - Live Launch: Starts in 3 Seconds & Dashboard Tour
05:00 - Test 1: Coastal Water & Finding Houses at Risk
07:30 - Test 2: Finding Ships in Port with GPS Boxes
10:00 - Test 3: Wiping Away 100% Storm Clouds with Radar
12:45 - Test 4: Measuring Flood Damage in Hectares
15:30 - Test 5: The Step-by-Step AI Trace & 1-Click PDF Report
17:45 - 100% Offline Mode & Running on Future Satellites

🔗 GitHub: https://github.com/your-org/satquery-ai
```

---

# 🎙️ SCRIPT (VERY SIMPLE WORDS)

---

## PART 1: THE BASICS (UNDER 4 MINUTES)
**Time:** `00:00 – 03:45`

---

### [00:00 – 01:15] The Problem: Why Satellites Are Slow Today
**Screen:** *Show satellite image of India, show clouds, then show your face.*

> "Right now, hundreds of satellites are flying over Earth.
>
> Satellites from ISRO take thousands of sharp pictures every day.
>
> And these pictures have the answers to big problems:
> - Where is the flood water?
> - Are farm crops drying up?
> - Which bridges are broken after a storm?
>
> But think about this: `[pause]`
>
> **Having thousands of satellite photos is useless if you cannot get an answer fast.**
>
> Imagine you are on a rescue team during a cyclone.
>
> And you need to know right now:
> *'Which main bridges in this town are underwater?'*
>
> You cannot just type that into a computer.
>
> Today, getting that answer takes **3 to 6 hours** of slow manual work.
>
> **Have you ever wondered why it takes so long?** `[pause]`
>
> Well, look at what an expert has to do today:
> First, they have to download giant 3-gigabyte files.
> Then, they open complex tools and fix the map lines by hand so the pictures line up.
> Raw satellite photos look completely black, so they have to fix the brightness.
> And then, they have to trace the water by hand on the screen and count hectares on a calculator.
>
> By the time all that slow work is done?
>
> **Three to six hours are gone!**
>
> When people are stuck in a flood, three hours is too long.
>
> My name is [Your Name], from team **The Gandivan’s** (Team ID: **RECS10**).
>
> We built **SatQuery AI** for the **Smart India Hackathon 2026**, for **ISRO** and the **Space Applications Centre**."

---

### [01:15 – 02:30] Why ChatGPT Fails & How Our Router Works
**Screen:** *Show ChatGPT failing to open file. Show simple architecture picture.*

> "Now, I know what you might be thinking:
> *'Why not just send these pictures to ChatGPT and let it answer?'*
>
> Honestly, that was the first thing we tried!
>
> But it fails right away for simple reasons:
> Phone photos are normal light.
> Satellite photos have **invisible light**, like infrared.
> That invisible light is how we see if plants are alive or if soil is wet.
> When you send it to ChatGPT, that light is thrown away!
>
> And during storms, normal cameras only see white clouds.
> ISRO uses radar to see through clouds.
> But radar looks like black-and-white dots.
> Normal chatbots get confused by the dots and make up fake things!
> In fact, chatbots guess wrong locations **34% of the time**.
> You cannot send rescue boats to a fake location!
>
> **So we built our secret weapon:** The **Agentic Router**. `[pause]`
>
> Think of our router like a **smart traffic police**.
>
> If you ask one AI to do three hard jobs at once—like clearing clouds, finding floods, and finding houses—its brain freezes.
>
> Our router breaks the job into simple steps:
> First, it checks the file to make sure it is clean.
> Then, it sends the cloudy photo to our radar tool to clean the clouds.
> Next, it sends that clean photo to our flood tool to find the water.
> And before it shows you the answer, it double-checks the math so there are no fake answers!
>
> It never gets confused, because it gives each job to the right tool."

---

### [02:30 – 03:45] The 5 Models We Trained
**Screen:** *Show simple table of the 5 models.*

> "Under the router, we have **5 small specialist models**.
>
> We trained them on real satellite pictures:
>
> 1. **The Router Brain:**  
>    Trained **48 times** on 14,000 questions. It picks the right tool **99% of the time**.
>
> 2. **The Question Answering Model:**  
>    Trained **46 times** on 125,000 satellite questions. It talks in simple, clear English.
>
> 3. **The Box Model:**  
>    Trained **50 times** on 85,000 pictures. It draws boxes around ships and runways.
>
> 4. **The Change Model:**  
>    Trained **45 times** on 40,000 storm photos. It spots real flood damage.
>
> 5. **The Radar Model:**  
>    Trained **52 times** on 65,000 radar pairs. It sees through 100% white clouds.
>
> **And here is the best part:** `[pause]`  
> The whole pack is **under 40 megabytes!**
>
> It runs right on my laptop without internet or cloud servers.
>
> Now, let's look at the **live working prototype**!"

---

## PART 2: THE LIVE PROTOTYPE DEMO (15 MINUTES)
**Time:** `03:45 – 19:30`

---

### [03:45 – 05:00] Starting in 3 Seconds & Screen Tour
**Screen:** *Show desktop. Double click `start_satquery.bat`. Show browser open.*

> "Watch how fast this starts up:
>
> Here is our start file: `start_satquery.bat`.
>
> I double-click it. `[click]`
>
> Look at the screen:
> In window one, our backend starts on port 8000.
> In window two, our frontend starts on port 5173.
>
> In just 3 seconds, both are ready, and the browser opens to **SatQuery AI**!
>
> We made the screen dark gray.
> We did this because if you work late at night during an emergency, a bright white screen hurts your eyes.
>
> On the top-left, we see the photo details and map numbers.
> On the top-right, we have the chat bar where you can type in simple English.
> And along the bottom, we have **10 quick cards** so you can click with no typing needed.
>
> Now, let's run 5 real tests!"

---

### [05:00 – 07:30] Test 1: Coastal Water & Finding Houses in Danger
**Screen:** *Load coastal photo. Click Card 3 (Water). Show blue mask. Ask building question.*

> "Put yourself in the shoes of a rescue team arriving at a coast.
>
> We load a satellite picture of the coast.
>
> Our first question is simple: *'Where is the water around this town?'*
>
> Along the bottom, I click **Card 3: Water Body Identification**. `[click]`
>
> **Look at that speed:** Under 1 second!
> - On the left, it tells us: *'4 water bodies found—a river, a harbor, and two lakes.'*
> - On the right, it paints bright blue over the exact water edges, with **92% certainty**.
>
> If I click the picture, it opens full screen so you can zoom in close.
>
> Now, as an officer, your next question is:
> *'If the water rises tonight, how many houses are in danger?'*
>
> So I ask in the chat:
> *'How many buildings can you identify in this scene?'* `[type and enter]`
>
> Look at the answer:
> It replies: *'About 12,840 buildings found'*, and colors every single roof in red!
>
> In two quick questions, an officer knows where the water is, and where the houses are."

---

### [07:30 – 10:00] Test 2: Finding Ships in Port with GPS Boxes
**Screen:** *Load port photo. Ask ship question. Show green boxes and GPS tooltips. Click "Export GeoJSON".*

> "Now, let's look at port security.
>
> Suppose we get an alert:
> *'Unknown ships or oil tankers have entered the harbor.'*
>
> In the past, an officer had to search this huge picture by hand for two hours.
>
> With SatQuery, I just ask:
> *'Locate and count all maritime vessels currently at berth or anchorage.'* `[enter]`
>
> In less than 1 second...
> It found all **8 ships** in the harbor!
>
> It drew green boxes on all 5 cargo ships, 2 tankers, and 1 small boat.
>
> When I move my mouse over any box, look at the text box:
> It shows the real GPS numbers!
>
> And when I need to give this to our patrol boats...
> I click **'Export GeoJSON'**. `[click]`
>
> In one second, it downloads a small map file that patrol boats can open right on their GPS screens."

---

### [10:00 – 12:45] Test 3: Wiping Away 100% Storm Clouds with Radar
**Screen:** *Switch to Optical + SAR mode. Slot 1: clouds. Slot 2: radar. Drag the slider.*

> "Now, look at this test—this is the most exciting part.
>
> What happens when a big flood hits, but the ground is hidden by clouds?
>
> Look at **Picture 1**:
> This is a normal satellite photo over Assam during a flood.
> It is **100% covered in thick white clouds**.
> If you ask ChatGPT: *'Where is the flood?'*, it says: *'I only see clouds.'* That leaves rescue teams blind.
>
> But in **Picture 2**, we have radar from ISRO's RISAT satellite taken at the exact same time.
> Radar cuts right through clouds, rain, and dark nights.
>
> So we ask SatQuery:
> *'Fuse optical and SAR radar to delineate flood-inundated roads beneath 100% cloud cover.'* `[enter]`
>
> **Look at this slider on screen:** `[pause]`
>
> On the left is the cloudy photo.
> On the right is our rebuilt photo.
>
> Watch what happens when I drag the slider across...
>
> **The clouds literally disappear!**
>
> Look under the clouds:
> In bright green, you can see the flooded roads and the broken river wall that were totally hidden before!
>
> Rescue boats do not have to wait 4 days for the storm to pass. They can see the flood right now."

---

### [12:45 – 15:30] Test 4: Measuring Flood Damage in Hectares
**Screen:** *Switch to Before/After mode. Slot 1: Before. Slot 2: After. Ask change question. Show +42.5 Ha.*

> "Now, let's jump two weeks forward after the storm.
>
> The rescue team needs real numbers:
> *'How much farm land was flooded, and what bridges were broken?'*
>
> We load the picture Before the storm, and the picture After the storm.
>
> Before answering, SatQuery checks the pictures and fixes any camera drift so there are no fake alarms.
>
> Now we ask:
> *'Detect spatial and structural changes and quantify flooded area in hectares.'* `[enter]`
>
> Look at the answer:
> 1. In plain English: *'Big flood found. The river broke its wall, flooded farm fields, and cut off the eastern bridge.'*
> 2. In exact numbers: **42.5 hectares of new flooding**—a **14% change** across the area.
> 3. And with the slider, you can slide between dry land and water easily."

---

### [15:30 – 17:45] Test 5: The AI Trace & 1-Click PDF Report
**Screen:** *Expand trace. Show speed. Click "Generate Mission Intelligence Dossier". Open PDF full screen.*

> "In defense and space missions, an AI must show how it got its answer.
>
> Look at the bottom: This is our **Step-by-Step AI Trace**.
>
> When I click expand, you see every step:
> - It checked the map numbers.
> - The AI models ran in less than **1 second**.
> - And it gave a digital security stamp so nobody can fake the results.
>
> Now, what if an officer needs to send a report to their boss right now?
>
> I just click this blue button: **'Generate Mission Intelligence Dossier'**. `[click]`
>
> In less than one second, it makes an official PDF report.
>
> Look at the report:
> Official ISRO header, before-and-after pictures, flood maps, hectare tables, and the digital stamp.
>
> In one click, an official report is ready to print or email!"

---

### [17:45 – 19:30] 100% Offline Safety & What's Next
**Screen:** *Show roadmap slide. Cut to your face for the sign-off.*

> "Working with satellite pictures should not take hours of slow work.
>
> SatQuery AI follows all data safety rules.
>
> Because our models are under 40 megabytes, this whole app runs **100% on a laptop without internet**.
>
> No data leaves your machine. Zero leaks.
>
> **Here is what we are building next:**
> - **Phase 1 (Done):** Our working prototype with the Agentic Router, 5 trained models, and instant PDF reports.
> - **Phase 2 (Next 6 Months):** Connecting directly to ISRO’s **Bhuvan** satellite map, plus voice in Hindi, Tamil, and Telugu.
> - **Phase 3 (18 Months):** Putting our models directly on the chip inside ISRO's upcoming **RISAT-3 satellite** in space!
>
> All our code and guides are open-source on GitHub below.
>
> On behalf of team **The Gandivan’s (RECS10)**, thank you to the Smart India Hackathon, ISRO, and Space Applications Centre.
>
> Jai Hind!"
>
> *[Wave at camera. Outro music plays. Screen shows GitHub link.]*

---

## 🎯 1-PAGE CHEAT SHEET (KEEP BESIDE YOUR SCREEN WHILE RECORDING)

| Time | What to Do on Screen | What to Say in 1 Simple Sentence |
|---|---|---|
| `00:00` | Satellite zoom $\to$ camera | *"Getting answers from satellite data today takes 3 to 6 hours. Have you ever wondered why it takes so long?"* |
| `01:15` | Show rejected file on ChatGPT | *"Why not use ChatGPT? Because it loses invisible light and makes up fake locations."* |
| `02:30` | Show Table of 5 Models | *"We trained our 5 models hard—from 45 to 52 rounds each—and kept the whole bundle under 40 MB."* |
| `03:45` | Double click `start_satquery.bat` | *"Watch how fast this starts: with one click, both servers boot up in 3 seconds on my laptop."* |
| `05:00` | Click Water Card, query buildings | *"In two quick questions, an officer knows where the water is and which houses are in danger."* |
| `07:30` | Query ships in port | *"In the past, you searched for 2 hours. SatQuery finds all 8 ships in 1 second with GPS."* |
| `10:00` | Load cloudy optical + radar | *"Look at this slider: as I drag it across, the storm clouds disappear and reveal the flooded roads!"* |
| `12:45` | Load Before/After delta | *"It fixes satellite drift and calculates 42.5 hectares of new flood damage in exact numbers."* |
| `15:30` | Click "Generate PDF Dossier" | *"Now imagine briefing your boss in 5 minutes: one click builds an official ISRO report."* |
| `17:45` | Show roadmap slide, say goodbye | *"Runs 100% on a laptop without internet. Thank you ISRO and SIH. Jai Hind!"* |
