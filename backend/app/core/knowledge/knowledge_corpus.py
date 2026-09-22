"""
SatQuery AI — Remote Sensing & ISRO Earth Observation Knowledge Corpus
Curated, authenticated scientific knowledge passages covering satellite constellations,
sensor physics, radiometric indices, disaster protocols, and geospatial science.
"""

from typing import Dict, List, Any


KNOWLEDGE_PASSAGES: List[Dict[str, Any]] = [
    # ── 0. Provenance & Creators ──────────────────────────────────
    {
        "id": "team_gandivan",
        "category": "system_provenance",
        "title": "The Gandivan’s (Team ID: RECS10) — Creators of SatQuery AI",
        "keywords": [
            "who created you", "who made you", "who built you", "who developed you",
            "creator", "creators", "developer", "developers", "team gandivan",
            "gandivan", "gandivans", "the gandivans", "the gandivan's", "recs10",
            "sih 26167", "who made this", "who are you"
        ],
        "content": (
            "SatQuery AI was designed and engineered by Team The Gandivan’s (Team ID: RECS10) for the Smart India Hackathon (SIH 2026), "
            "addressing Problem Statement ID 26167 issued by the Indian Space Research Organisation (ISRO) and Space Applications Centre (SAC). "
            "The Gandivan's engineered SatQuery AI as an on-device, verifiable vision-language copilot capable of multi-sensor Earth observation analysis, "
            "optical-SAR cloud-penetrating fusion, disaster flood mapping, and text-guided visual grounding with mathematical auditability."
        ),
    },

    # ── 1. ISRO Satellite Constellations & Sensors ────────────────
    {
        "id": "isro_cartosat",
        "category": "isro_missions",
        "title": "ISRO Cartosat Satellite Constellation",
        "keywords": ["cartosat", "cartosat-1", "cartosat-2", "cartosat-3", "sub-meter", "panchromatic", "stereo", "dem"],
        "content": (
            "ISRO's Cartosat series represents India's dedicated high-resolution optical Earth observation constellation. "
            "Cartosat-1 provided 2.5m stereo panchromatic imagery for 3D Digital Elevation Models (DEMs). "
            "Cartosat-2 series enhanced spatial resolution to 0.8m (panchromatic) and 2m (4-band multispectral VNIR: Blue, Green, Red, NIR). "
            "Cartosat-3 provides cutting-edge 0.28m panchromatic and 1.12m multispectral resolution with a 17km swath, "
            "enabling cadastral mapping, urban infrastructure planning, defense surveillance, and precise boundary delineation."
        ),
    },
    {
        "id": "isro_risat",
        "category": "isro_missions",
        "title": "ISRO RISAT Series & Microwave Radar (EOS-04)",
        "keywords": ["risat", "risat-1", "risat-2", "eos-04", "sar", "c-band", "radar", "cloud penetration", "all-weather"],
        "content": (
            "RISAT (Radar Imaging Satellite) and EOS-04 are ISRO's active microwave Synthetic Aperture Radar (SAR) spacecraft. "
            "Operating at C-band frequency (5.405 GHz, ~5.5cm wavelength), RISAT microwave pulses penetrate cloud decks, fog, haze, "
            "and darkness. RISAT supports multiple polarization modes (co-polarized VV/HH, cross-polarized VH/HV, and circular polarimetry) "
            "with spatial resolutions ranging from 1m (spotlight mode) to 50m (Scansar mode). It is indispensable for monsoon flood inundation, "
            "soil moisture estimation, coastal defense, and crop acreage monitoring."
        ),
    },
    {
        "id": "isro_resourcesat",
        "category": "isro_missions",
        "title": "ISRO Resourcesat (LISS-3, LISS-4, AWiFS)",
        "keywords": ["resourcesat", "resourcesat-2", "resourcesat-2a", "liss-3", "liss-4", "awifs", "multispectral", "agriculture"],
        "content": (
            "Resourcesat-2 and 2A carry three multi-spectral sensors tailored for agriculture and natural resources: "
            "1. LISS-4 (Linear Imaging Self-Scanning Sensor): 5.8m high spatial resolution in Green, Red, and NIR bands with 70km swath. "
            "2. LISS-3: 23.5m spatial resolution in 4 spectral bands (Green, Red, NIR, and Shortwave Infrared SWIR: 1.55-1.70 um) with 141km swath. "
            "3. AWiFS (Advanced Wide Field Sensor): 56m spatial resolution with an expansive 740km swath and 5-day revisit for national crop forecasting."
        ),
    },
    {
        "id": "isro_oceansat",
        "category": "isro_missions",
        "title": "ISRO Oceansat Series & Ocean Color",
        "keywords": ["oceansat", "oceansat-2", "oceansat-3", "eos-06", "ocm", "scatterometer", "ocean color", "chlorophyll"],
        "content": (
            "Oceansat-2 and EOS-06 (Oceansat-3) monitor maritime dynamics, chlorophyll concentration, and sea surface conditions. "
            "The Ocean Color Monitor (OCM-3) features 13 spectral bands in visible and near-infrared (400-1010nm) at 360m resolution. "
            "The Ku-band pencil-beam Scatterometer measures ocean surface wind vector speed (4-24 m/s) and direction with 25km resolution, "
            "critical for cyclone tracking in the Bay of Bengal and Arabian Sea."
        ),
    },
    {
        "id": "isro_nisar",
        "category": "isro_missions",
        "title": "NASA-ISRO SAR (NISAR) Dual-Frequency Radar Mission",
        "keywords": ["nisar", "l-band", "s-band", "dual-frequency", "interferometry", "insar", "deformation", "ecosystem"],
        "content": (
            "NISAR is a groundbreaking joint mission between NASA and ISRO featuring dual-frequency SweepSAR radar. "
            "NASA provides the L-band SAR (1.25 GHz, ~24cm wavelength) for penetrating dense forest canopies and deep soil. "
            "ISRO provides the S-band SAR (3.2 GHz, ~9cm wavelength) optimized for light vegetation, snow, and surface moisture. "
            "NISAR maps global land and ice deformation with sub-centimeter interferometric precision every 12 days."
        ),
    },

    # ── 2. Sensor Physics & Electromagnetic Spectrum ──────────────
    {
        "id": "physics_optical_vs_sar",
        "category": "sensor_physics",
        "title": "Optical Passive Solar Reflectance vs Microwave SAR Radar",
        "keywords": ["optical vs sar", "difference optical sar", "passive vs active", "radar vs camera", "microwave", "penetration"],
        "content": (
            "Optical remote sensing is passive, capturing reflected sunlight across Visible (400-700nm), NIR (700-1000nm), and SWIR (1000-2500nm). "
            "Optical wavelengths are shorter than atmospheric water droplet radii, causing complete scattering and obstruction by clouds. "
            "Synthetic Aperture Radar (SAR) is active, transmitting microwave electromagnetic pulses (1GHz to 10GHz, wavelengths 3cm to 30cm). "
            "Because microwave wavelengths are orders of magnitude larger than cloud droplets, SAR pulses penetrate cloud decks, rain, smoke, "
            "and darkness, interacting directly with surface geometry and dielectric permittivity."
        ),
    },
    {
        "id": "physics_sar_polarization",
        "category": "sensor_physics",
        "title": "SAR Polarimetric Backscatter Principles (VV, VH, HH, HV)",
        "keywords": ["polarization", "vv", "vh", "hh", "hv", "dual-pol", "backscatter", "sigma nought", "specular"],
        "content": (
            "SAR antennas transmit and receive polarized electric fields in horizontal (H) or vertical (V) planes. "
            "1. Co-polarization (VV/HH): Transmitted and received in the same plane. VV is sensitive to surface roughness and vertical structures. "
            "2. Cross-polarization (VH/HV): Incident wave undergoes multiple volume scattering inside vegetation canopies or rough terrain, rotating polarization. "
            "3. Specular Reflection: Smooth calm water acts as a mirror, reflecting radar pulses away from the receiver and producing near-zero backscatter (very dark, -22 to -28 dB). "
            "4. Double-Bounce Scattering: Orthogonal angles between ground and vertical building walls reflect energy back to the antenna (very bright, > -5 dB)."
        ),
    },

    # ── 3. Biophysical Spectral Indices & Mathematical Formulas ───
    {
        "id": "formula_ndvi",
        "category": "spectral_indices",
        "title": "Normalized Difference Vegetation Index (NDVI)",
        "keywords": ["ndvi", "ndvi formula", "vegetation index", "chlorophyll", "nir red", "vegetation health"],
        "content": (
            "NDVI quantifies live green vegetative canopy vigor and biomass using the formula: "
            "NDVI = (NIR - Red) / (NIR + Red). "
            "Live photosynthetic leaves absorb red light (660nm) via chlorophyll pigments for photosynthesis and strongly scatter "
            "near-infrared light (842nm) through spongy mesophyll cell structure. "
            "NDVI values range from -1.0 to +1.0: "
            "- Deep open water: negative values (-0.2 to -0.05). "
            "- Bare soil and rocks: low positive values (0.05 to 0.20). "
            "- Sparse scrub or stressed crops: moderate values (0.20 to 0.45). "
            "- Dense, healthy active forest canopy: high values (0.60 to 0.88)."
        ),
    },
    {
        "id": "formula_ndwi",
        "category": "spectral_indices",
        "title": "Normalized Difference Water Index (NDWI) & MNDWI",
        "keywords": ["ndwi", "mndwi", "water index", "water formula", "green nir", "green swir", "flood water"],
        "content": (
            "NDWI (McFeeters, 1996) delineates open water bodies using: NDWI = (Green - NIR) / (Green + NIR). "
            "Water reflects moderately in green wavelengths (560nm) and absorbs almost all near-infrared radiation (842nm), "
            "yielding positive values (> 0.0) for water bodies while terrestrial vegetation and built land remain negative. "
            "Modified NDWI (MNDWI, Xu 2006) replaces NIR with Shortwave Infrared (SWIR): MNDWI = (Green - SWIR) / (Green + SWIR), "
            "effectively suppressing built-up impervious noise and cleanly isolating turbid floodwaters."
        ),
    },
    {
        "id": "formula_ndbi",
        "category": "spectral_indices",
        "title": "Normalized Difference Built-up Index (NDBI)",
        "keywords": ["ndbi", "built-up index", "urban formula", "swir nir", "impervious surface", "buildings"],
        "content": (
            "NDBI isolates built infrastructure, urban settlements, and concrete impervious surfaces using: "
            "NDBI = (SWIR - NIR) / (SWIR + NIR). "
            "Urban materials like concrete, brick, asphalt, and metal rooftops have higher reflectance in the Shortwave Infrared band "
            "(SWIR 1.6 um) than in the Near-Infrared band (NIR 0.84 um), resulting in positive NDBI values (> 0.1) across developed urban grids."
        ),
    },

    # ── 4. Disaster Operations & Flood Protocols ──────────────────
    {
        "id": "disaster_ndma_flood",
        "category": "disaster_protocols",
        "title": "NDMA & CWC Flood Inundation Assessment Protocols",
        "keywords": ["ndma", "cwc", "flood guidelines", "evacuation zone", "safe zone buffer", "flood protocol", "inundation mapping"],
        "content": (
            "Under National Disaster Management Authority (NDMA) and Central Water Commission (CWC) guidelines, "
            "satellite-derived flood maps support rapid rescue and relief logistics: "
            "1. Safe Zone Buffer Threshold: Relocation camps and civilian muster points must maintain a minimum 500m horizontal buffer "
            "and > 3m elevation clearance beyond the delineated peak flood inundation boundary. "
            "2. Critical Transportation Lifelines: Highway and rail bridge approaches with standing water depth > 0.3m are categorized as severed. "
            "3. SAR Reconnaissance Cycle: Daily microwave SAR passes (Sentinel-1 / RISAT) provide uninterrupted flood extent progression "
            "independent of monsoonal overcast weather."
        ),
    },
    {
        "id": "disaster_flood_spectral",
        "category": "disaster_protocols",
        "title": "Satellite Inundation Delineation Methodology",
        "keywords": ["flood mapping method", "submerged land", "flood detection", "change former flood", "flood extent"],
        "content": (
            "Multi-temporal flood inundation mapping combines baseline pre-flood imagery (T1) with peak inundation imagery (T2). "
            "In optical pairs, flood water is identified by sharp drops in near-infrared reflectance and spikes in NDWI/MNDWI. "
            "In microwave SAR pairs, inundated land produces specular forward scattering resulting in backscatter drop of -6 dB to -14 dB "
            "compared to dry ground. Submerged area in hectares is computed as: Area (ha) = (Pixel Count * GSD_m^2) / 10,000."
        ),
    },

    # ── 5. Agricultural Phenology & Cadastral Monitoring ──────────
    {
        "id": "agri_phenology",
        "category": "agriculture",
        "title": "Indian Agricultural Phenology & Satellite Monitoring",
        "keywords": ["kharif", "rabi", "crop cycle", "cadastral", "agricultural monitoring", "irrigation", "crop health"],
        "content": (
            "India's agricultural monitoring revolves around two primary seasons: "
            "1. Kharif (Monsoon crops: Rice, Cotton, Maize, Soybeans) sown in June-July, peak vegetative NDVI in August-September, harvested October. "
            "2. Rabi (Winter crops: Wheat, Mustard, Gram) sown in October-November, peak vegetative NDVI in January-February, harvested March-April. "
            "Cadastral field boundaries exhibit regular geometric parcels (0.5 to 5 hectares) with contiguous perimeter bunds and irrigation canals."
        ),
    },
    # ── 6. NASA & International Earth Observation Systems ────────
    {
        "id": "nasa_landsat",
        "category": "international_missions",
        "title": "NASA / USGS Landsat Constellation (Landsat-8 & Landsat-9)",
        "keywords": ["landsat", "landsat-8", "landsat-9", "oli", "tirs", "nasa", "usgs", "30m", "16-day", "thermal"],
        "content": (
            "NASA and USGS operate Landsat-8 and Landsat-9 in sun-synchronous polar orbits at 705km altitude. "
            "Both carry the Operational Land Imager (OLI/OLI-2) with 9 spectral bands (30m multispectral VNIR/SWIR and 15m panchromatic band 8) "
            "and the Thermal Infrared Sensor (TIRS/TIRS-2) measuring dual longwave thermal emission (10.6-12.5 um) at 100m GSD (resampled to 30m). "
            "Combined, Landsat-8 and Landsat-9 provide an 8-day repeat cycle with a 185km swath, delivering the gold standard for global terrestrial change."
        ),
    },
    {
        "id": "esa_sentinel",
        "category": "international_missions",
        "title": "ESA Copernicus Sentinel Constellation (Sentinel-1 & Sentinel-2)",
        "keywords": ["sentinel", "sentinel-1", "sentinel-2", "copernicus", "msi", "sar", "10m", "5-day", "red edge"],
        "content": (
            "The European Space Agency (ESA) Copernicus program operates Sentinel-1 (C-band SAR at 5.405 GHz with 10m resolution in VV/VH dual-pol) "
            "and Sentinel-2 (Multispectral Instrument MSI with 13 bands: 10m visible/NIR, 20m red-edge and SWIR, and 60m atmospheric bands). "
            "With twin satellites (A & B) for each constellation, Sentinel-2 achieves a 5-day revisit at the equator and Sentinel-1 provides "
            "all-weather day-and-night radar imaging, widely utilized alongside ISRO and NASA data for disaster monitoring and land use."
        ),
    },
]


def get_all_passages() -> List[Dict[str, Any]]:
    return KNOWLEDGE_PASSAGES

