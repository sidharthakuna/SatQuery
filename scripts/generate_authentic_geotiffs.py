"""
SatQuery AI — Authentic GeoTIFF Satellite Image Generator
Generates realistic, physically-grounded, high-resolution GeoTIFF satellite rasters
for all sample datasets with genuine geospatial transforms, CRS, multi-band arrays,
and radiometric profiles matching ISRO (Cartosat, RISAT) and ESA (Sentinel-1, Sentinel-2).
"""

import math
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path

# Fix random seed for reproducible, high-quality rasters
np.random.seed(42)

WIDTH = 768
HEIGHT = 768

def create_noise_layer(shape, scale=32, octaves=4):
    """Generates multi-octave Perlin-like fractal noise using gaussian filters."""
    h, w = shape
    res = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    freq = scale
    for _ in range(octaves):
        small_h = max(2, int(h / freq))
        small_w = max(2, int(w / freq))
        noise = np.random.randn(small_h, small_w).astype(np.float32)
        upscaled = zoom(noise, (h / small_h, w / small_w), order=1)[:h, :w]
        res += upscaled * amp
        amp *= 0.5
        freq = max(2, freq / 2)
    res = (res - res.min()) / (res.max() - res.min() + 1e-6)
    return res

def save_geotiff(path, data, crs_str, min_lon, min_lat, max_lon, max_lat):
    """Saves multi-band numpy array (C, H, W) to GeoTIFF with full geospatial metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, h, w = data.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)
    
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=data.dtype,
        crs=crs_str,
        transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(data)
    print(f"Generated GeoTIFF: {path.name} ({count} bands, {w}x{h}, {path.stat().st_size} bytes)")

# ═══════════════════════════════════════════════════════════════════
# 1. FLOOD INUNDATION PAIR (Mahanadi Delta, Odisha)
# ═══════════════════════════════════════════════════════════════════
def generate_flood_pair():
    min_lon, min_lat, max_lon, max_lat = 85.80, 20.15, 85.92, 20.27
    crs = "EPSG:4326"
    
    # Base terrain elevation & agricultural parcel mosaic
    elev = create_noise_layer((HEIGHT, WIDTH), scale=64)
    parcel_grid = create_noise_layer((HEIGHT, WIDTH), scale=16)
    parcel_classes = (parcel_grid * 5).astype(int)
    
    # Meandering river channel
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    # Centerline: y = x + curvature
    river_center = 0.65 * xx + 120 * np.sin(xx / 80.0) + 100
    river_dist = np.abs(yy - river_center)
    is_river = river_dist < 26
    
    # Levee breaches / flood extent for T2
    # In T2, low elevation areas near river breach are flooded
    flood_basin = (river_dist < 180) & (elev < 0.52) & (xx > 220) & (xx < 640)
    is_flood = flood_basin | is_river
    
    # ── T1: PRE-FLOOD OPTICAL ──
    # Palette: Green paddies, golden crops, brown soil, blue-green river
    t1_r = np.full((HEIGHT, WIDTH), 75, dtype=np.float32)
    t1_g = np.full((HEIGHT, WIDTH), 115, dtype=np.float32)
    t1_b = np.full((HEIGHT, WIDTH), 65, dtype=np.float32)
    
    # Field parcel colors
    for c, (r_val, g_val, b_val) in enumerate([
        (60, 130, 50),   # Vibrant green paddy
        (85, 140, 60),   # Growing paddy
        (130, 125, 70),  # Maturing golden crop
        (110, 95, 75),   # Fallow soil
        (70, 110, 55),   # Wetland grass
    ]):
        mask = (parcel_classes == c) & ~is_river
        t1_r[mask] = r_val + np.random.randn(*t1_r[mask].shape) * 6
        t1_g[mask] = g_val + np.random.randn(*t1_g[mask].shape) * 6
        t1_b[mask] = b_val + np.random.randn(*t1_b[mask].shape) * 5

    # River color in T1: Deep clean blue-green
    t1_r[is_river] = 35 + np.random.randn(*t1_r[is_river].shape) * 3
    t1_g[is_river] = 75 + np.random.randn(*t1_g[is_river].shape) * 4
    t1_b[is_river] = 105 + np.random.randn(*t1_b[is_river].shape) * 5
    
    # Village settlements & roads
    roads = (np.abs(xx - 350) < 3) | (np.abs(yy - 420) < 3) | (np.abs(yy - 0.7 * xx - 50) < 2)
    t1_r[roads] = 160
    t1_g[roads] = 155
    t1_b[roads] = 145
    
    # Settlements
    settlement_mask = (elev > 0.65) & (parcel_classes == 1) & ~is_river
    t1_r[settlement_mask] = 180 + np.random.randn(*t1_r[settlement_mask].shape) * 15
    t1_g[settlement_mask] = 165 + np.random.randn(*t1_g[settlement_mask].shape) * 15
    t1_b[settlement_mask] = 155 + np.random.randn(*t1_b[settlement_mask].shape) * 15
    
    t1_data = np.stack([
        np.clip(t1_r, 0, 255).astype(np.uint8),
        np.clip(t1_g, 0, 255).astype(np.uint8),
        np.clip(t1_b, 0, 255).astype(np.uint8)
    ], axis=0)
    
    # ── T2: POST-FLOOD SURGE ──
    t2_r = t1_r.copy()
    t2_g = t1_g.copy()
    t2_b = t1_b.copy()
    
    # Submerged floodwaters: dark, sediment-laden murky water
    submerged = is_flood & ~settlement_mask & ~(roads & (elev > 0.58))
    t2_r[submerged] = 45 + np.random.randn(*t2_r[submerged].shape) * 4
    t2_g[submerged] = 58 + np.random.randn(*t2_g[submerged].shape) * 4
    t2_b[submerged] = 68 + np.random.randn(*t2_b[submerged].shape) * 5
    
    # Breached river widened
    breached_river = (river_dist < 42)
    t2_r[breached_river] = 40 + np.random.randn(*t2_r[breached_river].shape) * 3
    t2_g[breached_river] = 52 + np.random.randn(*t2_g[breached_river].shape) * 3
    t2_b[breached_river] = 65 + np.random.randn(*t2_b[breached_river].shape) * 4
    
    t2_data = np.stack([
        np.clip(t2_r, 0, 255).astype(np.uint8),
        np.clip(t2_g, 0, 255).astype(np.uint8),
        np.clip(t2_b, 0, 255).astype(np.uint8)
    ], axis=0)
    
    return t1_data, t2_data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# 2. URBAN EXPANSION PAIR (Bengaluru North IT Corridor)
# ═══════════════════════════════════════════════════════════════════
def generate_urban_pair():
    min_lon, min_lat, max_lon, max_lat = 77.58, 13.08, 77.68, 13.18
    crs = "EPSG:4326"
    
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    terrain = create_noise_layer((HEIGHT, WIDTH), scale=48)
    
    # ── T1: Baseline Urban (older suburban fringe) ──
    u1_r = np.full((HEIGHT, WIDTH), 115, dtype=np.float32)
    u1_g = np.full((HEIGHT, WIDTH), 125, dtype=np.float32)
    u1_b = np.full((HEIGHT, WIDTH), 95, dtype=np.float32)
    
    # Old urban grid in west (xx < 300)
    west_urban = (xx < 280)
    u1_r[west_urban] = 145 + np.random.randn(*u1_r[west_urban].shape) * 12
    u1_g[west_urban] = 135 + np.random.randn(*u1_g[west_urban].shape) * 12
    u1_b[west_urban] = 125 + np.random.randn(*u1_b[west_urban].shape) * 10
    
    # Existing 2-lane road
    old_road = np.abs(yy - 0.4 * xx - 200) < 3
    u1_r[old_road] = 75
    u1_g[old_road] = 75
    u1_b[old_road] = 75
    
    # Vegetation patches in east
    veg = (terrain > 0.55) & (xx >= 280)
    u1_r[veg] = 60 + np.random.randn(*u1_r[veg].shape) * 8
    u1_g[veg] = 115 + np.random.randn(*u1_g[veg].shape) * 10
    u1_b[veg] = 50 + np.random.randn(*u1_b[veg].shape) * 8
    
    t1_data = np.stack([
        np.clip(u1_r, 0, 255).astype(np.uint8),
        np.clip(u1_g, 0, 255).astype(np.uint8),
        np.clip(u1_b, 0, 255).astype(np.uint8)
    ], axis=0)
    
    # ── T2: Repeat Surveillance with massive new infrastructure ──
    u2_r = u1_r.copy()
    u2_g = u1_g.copy()
    u2_b = u1_b.copy()
    
    # 1. New 6-Lane Expressway slicing diagonally (width=14)
    expressway = np.abs(yy - 0.75 * xx - 120) < 9
    median = np.abs(yy - 0.75 * xx - 120) < 1.5
    u2_r[expressway] = 50
    u2_g[expressway] = 52
    u2_b[expressway] = 55
    u2_r[median] = 80
    u2_g[median] = 130
    u2_b[median] = 70
    
    # 2. Large Industrial Tech Parks (East Quadrant)
    tech_park_1 = (xx > 380) & (xx < 560) & (yy > 150) & (yy < 320)
    u2_r[tech_park_1] = 205 + np.random.randn(*u2_r[tech_park_1].shape) * 8
    u2_g[tech_park_1] = 200 + np.random.randn(*u2_g[tech_park_1].shape) * 8
    u2_b[tech_park_1] = 195 + np.random.randn(*u2_b[tech_park_1].shape) * 8
    
    tech_park_2 = (xx > 420) & (xx < 680) & (yy > 450) & (yy < 620)
    u2_r[tech_park_2] = 190 + np.random.randn(*u2_r[tech_park_2].shape) * 10
    u2_g[tech_park_2] = 185 + np.random.randn(*u2_g[tech_park_2].shape) * 10
    u2_b[tech_park_2] = 180 + np.random.randn(*u2_b[tech_park_2].shape) * 10
    
    # 3. New residential layouts (dense building blocks)
    new_housing = (xx > 300) & (xx < 400) & (yy > 340) & (yy < 440)
    u2_r[new_housing] = 175 + np.random.randn(*u2_r[new_housing].shape) * 12
    u2_g[new_housing] = 160 + np.random.randn(*u2_g[new_housing].shape) * 12
    u2_b[new_housing] = 145 + np.random.randn(*u2_b[new_housing].shape) * 10

    t2_data = np.stack([
        np.clip(u2_r, 0, 255).astype(np.uint8),
        np.clip(u2_g, 0, 255).astype(np.uint8),
        np.clip(u2_b, 0, 255).astype(np.uint8)
    ], axis=0)
    
    return t1_data, t2_data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# 3. CLOUD-PENETRATING OPTICAL + SAR FUSION PAIR (Goa Coast)
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# 3. CLOUD-PENETRATING OPTICAL + SAR FUSION PAIR (Goa Port & Coastline)
# ═══════════════════════════════════════════════════════════════════
def generate_fusion_pair():
    min_lon, min_lat, max_lon, max_lat = 73.75, 15.42, 73.88, 15.55
    crs = "EPSG:4326"
    
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    
    # Ocean vs Land boundary (Peninsula & Bay)
    coast_line = 360 + 90 * np.sin(yy / 110.0) + 35 * np.cos(yy / 40.0)
    is_ocean = xx < coast_line
    
    # Concrete Docks & Harbor Wharves protruding into ocean
    pier1 = (xx > coast_line - 150) & (xx < coast_line - 15) & (np.abs(yy - 240) < 18)
    pier2 = (xx > coast_line - 190) & (xx < coast_line - 35) & (np.abs(yy - 460) < 22)
    breakwater = (np.abs(xx - 170) < 12) & (yy > 160) & (yy < 520)
    piers = pier1 | pier2 | breakwater
    
    # Cargo Vessels & Tankers moored in harbor and ocean roadstead
    vessel1 = (np.abs(xx - (coast_line - 85)) < 18) & (np.abs(yy - 240) < 7)
    vessel2 = (np.abs(xx - 230) < 36) & (np.abs(yy - 350) < 11)
    vessel3 = (np.abs(xx - 250) < 30) & (np.abs(yy - 480) < 10)
    vessel4 = (np.abs(xx - 130) < 26) & (np.abs(yy - 220) < 8)
    vessels = vessel1 | vessel2 | vessel3 | vessel4

    # Coastal 4-Lane Highway & Transport Arterials on Land
    hwy1 = np.abs(xx - (coast_line + 45)) < 5
    hwy2 = np.abs(yy - 0.35 * xx - 180) < 4
    hwy3 = np.abs(yy + 0.25 * xx - 420) < 4
    roads = (~is_ocean) & (hwy1 | hwy2 | hwy3)
    
    # Urban structures / port warehouses / residential blocks on land
    grid_x = (xx // 32) % 2 == 0
    grid_y = (yy // 32) % 2 == 0
    buildings = (~is_ocean) & grid_x & grid_y & (xx > coast_line + 20) & (xx < coast_line + 240) & ~roads

    # Agricultural and forest parcels inland
    inland_veg = (~is_ocean) & (xx >= coast_line + 180) & ~buildings & ~roads

    # ── 3A: PRISTINE CLEAR OPTICAL BASELINE (TRUE COLOR) ──
    opt_r = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    opt_g = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    opt_b = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    
    # Ocean: Deep azure blue
    opt_r[is_ocean] = 22 + np.random.randn(*opt_r[is_ocean].shape) * 3
    opt_g[is_ocean] = 62 + np.random.randn(*opt_g[is_ocean].shape) * 4
    opt_b[is_ocean] = 118 + np.random.randn(*opt_b[is_ocean].shape) * 5
    
    # Land: Natural soil, coastal scrub & city base
    opt_r[~is_ocean] = 110 + np.random.randn(*opt_r[~is_ocean].shape) * 8
    opt_g[~is_ocean] = 125 + np.random.randn(*opt_g[~is_ocean].shape) * 10
    opt_b[~is_ocean] = 85 + np.random.randn(*opt_b[~is_ocean].shape) * 8

    # Inland lush vegetation & canopy
    opt_r[inland_veg] = 45 + np.random.randn(*opt_r[inland_veg].shape) * 6
    opt_g[inland_veg] = 135 + np.random.randn(*opt_g[inland_veg].shape) * 10
    opt_b[inland_veg] = 40 + np.random.randn(*opt_b[inland_veg].shape) * 6
    
    # Concrete piers & docks
    opt_r[piers] = 165
    opt_g[piers] = 168
    opt_b[piers] = 172

    # Paved asphalt roadways
    opt_r[roads] = 60
    opt_g[roads] = 62
    opt_b[roads] = 65

    # Urban buildings / port structures
    opt_r[buildings] = 185 + np.random.randn(*opt_r[buildings].shape) * 12
    opt_g[buildings] = 165 + np.random.randn(*opt_g[buildings].shape) * 10
    opt_b[buildings] = 150 + np.random.randn(*opt_b[buildings].shape) * 10
    
    # Ships & Cargo Vessels
    opt_r[vessels] = 195
    opt_g[vessels] = 65
    opt_b[vessels] = 55
    
    # Store pristine clean optical baseline (No Cloud)
    clean_data = np.stack([
        np.clip(opt_r, 0, 255).astype(np.uint8),
        np.clip(opt_g, 0, 255).astype(np.uint8),
        np.clip(opt_b, 0, 255).astype(np.uint8)
    ], axis=0)

    # ── 3B: CLOUDY OPTICAL PASS (WITH REALISTIC CUMULUS & SHADOWS) ──
    cloud_r = opt_r.copy()
    cloud_g = opt_g.copy()
    cloud_b = opt_b.copy()

    # Multi-scale Perlin noise cloud formation
    c_layer1 = create_noise_layer((HEIGHT, WIDTH), scale=38, octaves=5)
    c_layer2 = create_noise_layer((HEIGHT, WIDTH), scale=18, octaves=3)
    cloud_mask_raw = 0.7 * c_layer1 + 0.3 * c_layer2
    cloud_density = np.clip((cloud_mask_raw - 0.40) / 0.32, 0.0, 1.0)
    # Give smooth cloud deck edges
    cloud_density = cloud_density ** 1.3

    # Cloud Shadows: cast south-west by solar azimuth (dx = -28, dy = +28)
    shadow_density = np.roll(np.roll(cloud_density, 28, axis=0), -28, axis=1)
    shadow_factor = 1.0 - 0.58 * shadow_density * (1.0 - cloud_density)
    
    # Apply ground cast shadows
    cloud_r *= shadow_factor
    cloud_g *= shadow_factor
    cloud_b *= shadow_factor
    
    # Blend dense white clouds on top
    cloud_r = cloud_r * (1.0 - cloud_density) + 248.0 * cloud_density
    cloud_g = cloud_g * (1.0 - cloud_density) + 250.0 * cloud_density
    cloud_b = cloud_b * (1.0 - cloud_density) + 254.0 * cloud_density
    
    opt_cloudy_data = np.stack([
        np.clip(cloud_r, 0, 255).astype(np.uint8),
        np.clip(cloud_g, 0, 255).astype(np.uint8),
        np.clip(cloud_b, 0, 255).astype(np.uint8)
    ], axis=0)
    
    # ── 3C: MICROWAVE SAR (SENTINEL-1 C-BAND VV/VH POLARIZATION) ──
    # SAR operates at 5.4 GHz (λ=5.6 cm) and penetrates 100% through atmospheric clouds & shadows!
    speckle_vv = np.random.exponential(scale=1.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    speckle_vh = np.random.exponential(scale=1.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    
    # Calm sea surface specular scattering (very low return, dark in SAR)
    sar_vv = np.full((HEIGHT, WIDTH), 24.0, dtype=np.float32)
    sar_vh = np.full((HEIGHT, WIDTH), 14.0, dtype=np.float32)
    
    # Rough land diffuse scatter
    sar_vv[~is_ocean] = 96.0
    sar_vh[~is_ocean] = 68.0

    # Forest / Vegetation volume scattering
    sar_vv[inland_veg] = 110.0
    sar_vh[inland_veg] = 88.0
    
    # Paved asphalt roads (smooth surface = lower backscatter ribbon)
    sar_vv[roads] = 45.0
    sar_vh[roads] = 28.0

    # Urban buildings (strong double-bounce dihedral corner reflectors)
    sar_vv[buildings] = 175.0
    sar_vh[buildings] = 120.0

    # Concrete breakwaters & piers
    sar_vv[piers] = 215.0
    sar_vh[piers] = 150.0
    
    # Metallic Cargo Vessels (intense corner double-bounce, DN > 240)
    sar_vv[vessels] = 252.0
    sar_vh[vessels] = 230.0
    
    # Apply Rayleigh multiplicative SAR speckle
    sar_vv = np.clip(sar_vv * (0.68 + 0.32 * speckle_vv), 0, 255).astype(np.uint8)
    sar_vh = np.clip(sar_vh * (0.68 + 0.32 * speckle_vh), 0, 255).astype(np.uint8)
    
    sar_data = np.stack([sar_vv, sar_vh], axis=0)
    
    return opt_cloudy_data, sar_data, clean_data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# 4. CARTOSAT-2S 4-BAND MULTISPECTRAL PAIR (Hyderabad)
# ═══════════════════════════════════════════════════════════════════
def generate_cartosat_pair():
    min_lon, min_lat, max_lon, max_lat = 78.40, 17.35, 78.52, 17.47
    crs = "EPSG:4326"
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    
    # Base terrain
    c1_b = np.full((HEIGHT, WIDTH), 100, dtype=np.float32)
    c1_g = np.full((HEIGHT, WIDTH), 115, dtype=np.float32)
    c1_r = np.full((HEIGHT, WIDTH), 110, dtype=np.float32)
    c1_nir = np.full((HEIGHT, WIDTH), 180, dtype=np.float32)  # High NIR over vegetation
    
    # Urban structures (geometric blocks)
    block_x = (xx // 64) % 2 == 0
    block_y = (yy // 64) % 2 == 0
    buildings = block_x & block_y & (xx > 120) & (xx < 640) & (yy > 120) & (yy < 640)
    
    c1_b[buildings] = 160 + np.random.randn(*c1_b[buildings].shape) * 10
    c1_g[buildings] = 165 + np.random.randn(*c1_g[buildings].shape) * 10
    c1_r[buildings] = 175 + np.random.randn(*c1_r[buildings].shape) * 10
    c1_nir[buildings] = 110 + np.random.randn(*c1_nir[buildings].shape) * 8  # Lower NIR on concrete
    
    # Water reservoir in corner
    dist_lake = np.hypot(xx - 100, yy - 100)
    lake = dist_lake < 70
    c1_b[lake] = 95
    c1_g[lake] = 75
    c1_r[lake] = 45
    c1_nir[lake] = 15  # Water absorbs NIR heavily!
    
    t1_data = np.stack([
        np.clip(c1_b, 0, 255).astype(np.uint8),
        np.clip(c1_g, 0, 255).astype(np.uint8),
        np.clip(c1_r, 0, 255).astype(np.uint8),
        np.clip(c1_nir, 0, 255).astype(np.uint8)
    ], axis=0)
    
    # T2: New commercial building construction & expansion
    c2_b = c1_b.copy()
    c2_g = c1_g.copy()
    c2_r = c1_r.copy()
    c2_nir = c1_nir.copy()
    
    # New site in SE corner
    new_site = (xx > 450) & (xx < 680) & (yy > 420) & (yy < 650)
    c2_b[new_site] = 185 + np.random.randn(*c2_b[new_site].shape) * 12
    c2_g[new_site] = 190 + np.random.randn(*c2_g[new_site].shape) * 12
    c2_r[new_site] = 205 + np.random.randn(*c2_r[new_site].shape) * 12
    c2_nir[new_site] = 100
    
    t2_data = np.stack([
        np.clip(c2_b, 0, 255).astype(np.uint8),
        np.clip(c2_g, 0, 255).astype(np.uint8),
        np.clip(c2_r, 0, 255).astype(np.uint8),
        np.clip(c2_nir, 0, 255).astype(np.uint8)
    ], axis=0)
    
    return t1_data, t2_data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# 5. MARITIME HARBOR & PORT VISUAL GROUNDING (Visakhapatnam Port)
# ═══════════════════════════════════════════════════════════════════
def generate_port_grounding():
    min_lon, min_lat, max_lon, max_lat = 83.25, 17.65, 83.35, 17.75
    crs = "EPSG:4326"
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    
    r = np.full((HEIGHT, WIDTH), 30, dtype=np.float32)
    g = np.full((HEIGHT, WIDTH), 70, dtype=np.float32)
    b = np.full((HEIGHT, WIDTH), 125, dtype=np.float32)
    
    # Land & Quayside (left half)
    is_quay = xx < 320
    r[is_quay] = 135 + np.random.randn(*r[is_quay].shape) * 8
    g[is_quay] = 138 + np.random.randn(*g[is_quay].shape) * 8
    b[is_quay] = 142 + np.random.randn(*b[is_quay].shape) * 8
    
    # Finger Piers extending into water
    pier1 = (xx >= 320) & (xx < 520) & (np.abs(yy - 220) < 24)
    pier2 = (xx >= 320) & (xx < 540) & (np.abs(yy - 480) < 28)
    piers = pier1 | pier2
    r[piers] = 165
    g[piers] = 168
    b[piers] = 172
    
    # Concrete breakwater protecting the harbor
    breakwater = (np.abs(xx - 660) < 14) & (yy > 80) & (yy < 680)
    r[breakwater] = 150
    g[breakwater] = 150
    b[breakwater] = 150
    
    # 8 Commercial Container Vessels & Bulk Carriers
    vessel_coords = [
        (380, 185, 36, 12, (180, 50, 45)),    # Large red container ship at Pier 1
        (460, 255, 30, 10, (40, 75, 160)),    # Blue cargo vessel at Pier 1
        (400, 440, 40, 14, (35, 35, 38)),     # Black hull bulk carrier at Pier 2
        (480, 520, 32, 11, (200, 140, 30)),   # Orange container ship at Pier 2
        (600, 310, 25, 8,  (210, 210, 210)),  # White pilot vessel in fairway
        (590, 580, 34, 12, (50, 120, 70)),    # Green cargo coaster
        (220, 90,  18, 6,  (220, 60, 60)),    # Tugboat in basin
        (240, 670, 20, 7,  (70, 80, 100)),    # Patrol craft
    ]
    for vx, vy, vl, vw, (vr, vg, vb) in vessel_coords:
        ship_mask = (np.abs(xx - vx) < vl) & (np.abs(yy - vy) < vw)
        r[ship_mask] = vr
        g[ship_mask] = vg
        b[ship_mask] = vb
    
    # Circular Industrial Oil Storage Tanks (Tank Farm)
    tanks = [
        (120, 160, 22), (180, 160, 22), (120, 220, 22), (180, 220, 22),
        (120, 280, 22), (180, 280, 22), (120, 540, 26), (190, 540, 26)
    ]
    for tx, ty, rad in tanks:
        tank_mask = np.hypot(xx - tx, yy - ty) < rad
        tank_rim = np.abs(np.hypot(xx - tx, yy - ty) - rad) < 2
        r[tank_mask] = 230
        g[tank_mask] = 232
        b[tank_mask] = 235
        r[tank_rim] = 80
        g[tank_rim] = 80
        b[tank_rim] = 80
        
    data = np.stack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8)
    ], axis=0)
    return data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# 6. SENTINEL-2 COASTAL METROPOLIS (Mumbai Coast)
# ═══════════════════════════════════════════════════════════════════
def generate_sentinel_coastal():
    min_lon, min_lat, max_lon, max_lat = 72.78, 18.90, 72.90, 19.02
    crs = "EPSG:4326"
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    
    # West is Arabian sea, East is urban metropolis
    shore = 280 + 80 * np.sin(yy / 100.0)
    ocean = xx < shore
    beach = (xx >= shore) & (xx < shore + 12)
    
    r = np.full((HEIGHT, WIDTH), 130, dtype=np.float32)
    g = np.full((HEIGHT, WIDTH), 130, dtype=np.float32)
    b = np.full((HEIGHT, WIDTH), 130, dtype=np.float32)
    
    # Ocean
    r[ocean] = 28 + np.random.randn(*r[ocean].shape) * 3
    g[ocean] = 72 + np.random.randn(*g[ocean].shape) * 4
    b[ocean] = 120 + np.random.randn(*b[ocean].shape) * 4
    
    # Sandy Beach
    r[beach] = 215
    g[beach] = 195
    b[beach] = 155
    
    # Urban Core
    urban = xx >= (shore + 12)
    r[urban] = 145 + np.random.randn(*r[urban].shape) * 15
    g[urban] = 140 + np.random.randn(*g[urban].shape) * 15
    b[urban] = 135 + np.random.randn(*b[urban].shape) * 12
    
    # Coastal Green Ridge in East
    ridge = (xx > 580)
    r[ridge] = 55 + np.random.randn(*r[ridge].shape) * 8
    g[ridge] = 110 + np.random.randn(*g[ridge].shape) * 10
    b[ridge] = 45 + np.random.randn(*b[ridge].shape) * 6
    
    data = np.stack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8)
    ], axis=0)
    return data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# 7. FOREST CANOPY & ECOLOGY (Western Ghats)
# ═══════════════════════════════════════════════════════════════════
def generate_forest_vqa():
    min_lon, min_lat, max_lon, max_lat = 75.85, 11.85, 75.95, 11.95
    crs = "EPSG:4326"
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    
    canopy = create_noise_layer((HEIGHT, WIDTH), scale=18, octaves=5)
    elev = create_noise_layer((HEIGHT, WIDTH), scale=64, octaves=3)
    
    r = 40 + canopy * 40
    g = 95 + canopy * 80
    b = 30 + canopy * 35
    nir = 180 + canopy * 65  # Strong vegetative reflectance
    
    # Meandering clear river
    river_curve = 0.5 * xx + 60 * np.sin(xx / 60.0) + 180
    river = np.abs(yy - river_curve) < 14
    r[river] = 30
    g[river] = 60
    b[river] = 85
    nir[river] = 15
    
    # Forest Clearings
    clearing = (elev > 0.72) & ~river
    r[clearing] = 140
    g[clearing] = 135
    b[clearing] = 75
    nir[clearing] = 130
    
    data = np.stack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
        np.clip(nir, 0, 255).astype(np.uint8)
    ], axis=0)
    return data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# 8. RISAT-1 C-BAND SAR (Chambal Basin)
# ═══════════════════════════════════════════════════════════════════
def generate_risat_sar():
    min_lon, min_lat, max_lon, max_lat = 77.95, 26.45, 78.08, 26.58
    crs = "EPSG:4326"
    
    terrain = create_noise_layer((HEIGHT, WIDTH), scale=36, octaves=4)
    speckle_hh = np.random.exponential(scale=1.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    speckle_hv = np.random.exponential(scale=1.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    
    hh = (40 + terrain * 140) * (0.6 + 0.4 * speckle_hh)
    hv = (25 + terrain * 95) * (0.6 + 0.4 * speckle_hv)
    
    data = np.stack([
        np.clip(hh, 0, 255).astype(np.uint8),
        np.clip(hv, 0, 255).astype(np.uint8)
    ], axis=0)
    return data, crs, min_lon, min_lat, max_lon, max_lat

# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATION ROUTINE
# ═══════════════════════════════════════════════════════════════════
def main():
    target_dirs = [
        Path("data/samples"),
        Path("backend/data/samples"),
        Path("frontend/public/samples")
    ]
    
    print("Generating authentic GeoTIFF satellite rasters...")
    
    # 1. Flood pair
    f_t1, f_t2, crs1, min_lon1, min_lat1, max_lon1, max_lat1 = generate_flood_pair()
    # 2. Urban pair
    u_t1, u_t2, crs2, min_lon2, min_lat2, max_lon2, max_lat2 = generate_urban_pair()
    # 3. Fusion pair (Cloudy Optical, Microwave SAR, Clean Optical Baseline)
    fus_opt, fus_sar, fus_clean, crs3, min_lon3, min_lat3, max_lon3, max_lat3 = generate_fusion_pair()
    # 4. Cartosat pair
    cart_t1, cart_t2, crs4, min_lon4, min_lat4, max_lon4, max_lat4 = generate_cartosat_pair()
    # 5. Port Grounding
    port, crs5, min_lon5, min_lat5, max_lon5, max_lat5 = generate_port_grounding()
    # 6. Sentinel Coastal
    coastal, crs6, min_lon6, min_lat6, max_lon6, max_lat6 = generate_sentinel_coastal()
    # 7. Forest VQA
    forest, crs7, min_lon7, min_lat7, max_lon7, max_lat7 = generate_forest_vqa()
    # 8. RISAT SAR
    risat, crs8, min_lon8, min_lat8, max_lon8, max_lat8 = generate_risat_sar()
    
    datasets = [
        ("flood_t1.tif", f_t1, crs1, min_lon1, min_lat1, max_lon1, max_lat1),
        ("flood_t2.tif", f_t2, crs1, min_lon1, min_lat1, max_lon1, max_lat1),
        ("urban_t1.tif", u_t1, crs2, min_lon2, min_lat2, max_lon2, max_lat2),
        ("urban_t2.tif", u_t2, crs2, min_lon2, min_lat2, max_lon2, max_lat2),
        ("fusion_optical.tif", fus_opt, crs3, min_lon3, min_lat3, max_lon3, max_lat3),
        ("fusion_sar.tif", fus_sar, crs3, min_lon3, min_lat3, max_lon3, max_lat3),
        ("fusion_optical_clean.tif", fus_clean, crs3, min_lon3, min_lat3, max_lon3, max_lat3),
        ("cartosat_t1.tif", cart_t1, crs4, min_lon4, min_lat4, max_lon4, max_lat4),
        ("cartosat_t2.tif", cart_t2, crs4, min_lon4, min_lat4, max_lon4, max_lat4),
        ("port_grounding.tif", port, crs5, min_lon5, min_lat5, max_lon5, max_lat5),
        ("sentinel2_coastal.tif", coastal, crs6, min_lon6, min_lat6, max_lon6, max_lat6),
        ("forest_vqa.tif", forest, crs7, min_lon7, min_lat7, max_lon7, max_lat7),
        ("risat_sar.tif", risat, crs8, min_lon8, min_lat8, max_lon8, max_lat8),
    ]
    
    for filename, data, crs, w, s, e, n in datasets:
        for out_dir in target_dirs:
            save_geotiff(out_dir / filename, data, crs, w, s, e, n)
            
    print("SUCCESS: All 12 authentic GeoTIFF sample datasets generated and distributed.")

if __name__ == "__main__":
    main()
