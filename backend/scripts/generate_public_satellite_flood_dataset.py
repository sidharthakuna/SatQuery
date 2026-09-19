"""
Generate Publicly Available Satellite Imagery Samples for Cloud-Penetrating Flood Analysis
Simulates authentic Copernicus Sentinel-1 C-Band SAR and Sentinel-2 MultiSpectral Optical rasters
covering an active river flood disaster event (e.g. Mahanadi / Brahmaputra basin) with dense storm cloud cover.
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds

def generate_public_satellite_flood_rasters():
    # Set coordinates for active river basin flood event (e.g. Brahmaputra / Mahanadi delta)
    # Bounding box in EPSG:4326
    left, bottom, right, top = 85.75, 20.15, 85.95, 20.35
    h, w = 768, 768
    transform = from_bounds(left, bottom, right, top, w, h)

    np.random.seed(42)

    # ─────────────────────────────────────────────────────────────
    # 1. Base Terrain & Ground Hydrology Model (Ground Truth)
    # ─────────────────────────────────────────────────────────────
    # Create realistic terrain elevation gradient and river meander
    yy, xx = np.mgrid[0:h, 0:w]
    
    # Main winding river channel
    river_center = w * 0.45 + (w * 0.15) * np.sin(yy / 100.0) + (w * 0.05) * np.cos(yy / 45.0)
    dist_to_river = np.abs(xx - river_center)

    # Flood inundation plain (low-lying alluvial basin flooded by monsoonal surge)
    flood_basin_mask = (dist_to_river < 110) | ((yy > 320) & (yy < 580) & (xx > 180) & (xx < 560))
    # Secondary flooded tributaries and low-lying agricultural depressions
    tributary = (np.abs(yy - (h * 0.55 + 0.3 * (xx - w * 0.5) ** 2 / 100.0)) < 35) & (xx < w * 0.6)
    flood_basin_mask = flood_basin_mask | tributary

    # Safe high ground / elevated ridge regions (elevated terraces > 300m away from flood zone)
    high_ground_alpha = ((xx > 580) & (yy < 300)) # North-East elevated ridge
    high_ground_beta = ((xx < 160) & (yy < 280))  # North-West plateau
    high_ground_gamma = ((xx > 590) & (yy > 520)) # South-East municipal upland

    # Ground reflectance (optical clear sky: vegetation green/NIR, urban grey, soil brown, water dark blue)
    ground_opt = np.zeros((3, h, w), dtype=np.uint8)
    
    # Active vegetation background (paddy fields, forest)
    ground_opt[0] = np.clip(60 + np.random.normal(0, 10, (h, w)), 30, 90)   # Red
    ground_opt[1] = np.clip(130 + np.random.normal(0, 15, (h, w)), 80, 180) # Green
    ground_opt[2] = np.clip(55 + np.random.normal(0, 10, (h, w)), 25, 85)   # Blue

    # Settlements / infrastructure on dry ground
    settlement_mask = (high_ground_alpha | high_ground_beta | high_ground_gamma) & (np.random.rand(h, w) > 0.82)
    ground_opt[0, settlement_mask] = 160
    ground_opt[1, settlement_mask] = 155
    ground_opt[2, settlement_mask] = 150

    # Normal clear river / flood water (optical appearance: dark muddy blue/cyan)
    ground_opt[0, flood_basin_mask] = 35
    ground_opt[1, flood_basin_mask] = 75
    ground_opt[2, flood_basin_mask] = 110

    # ─────────────────────────────────────────────────────────────
    # 2. Optical Imagery with Heavy Storm Cloud Cover (Sentinel-2 L2A)
    # ─────────────────────────────────────────────────────────────
    # Dark cumulus and dense convective clouds obscuring 45-60% of the flooded scene
    cloud_density = np.zeros((h, w), dtype=np.float32)
    
    # Generate realistic multi-scale cloud clusters
    cloud_centers = [(h * 0.35, w * 0.40, 160), (h * 0.55, w * 0.60, 210), (h * 0.70, w * 0.30, 150)]
    for cy, cx, r in cloud_centers:
        dist_sq = ((yy - cy) / r) ** 2 + ((xx - cx) / (r * 1.3)) ** 2
        cloud_density += np.exp(-dist_sq * 1.8)
    
    # Add high-frequency wisps
    noise = np.random.normal(0, 0.15, (h, w))
    cloud_density = np.clip(cloud_density + noise, 0.0, 1.0)
    cloud_mask = cloud_density > 0.35

    # Cloud appearance: very high reflectance, near-white to pale grey storm clouds
    cloudy_opt = ground_opt.copy()
    for b in range(3):
        cloudy_opt[b] = np.clip(
            (1.0 - cloud_density) * ground_opt[b] + cloud_density * 240 + np.random.normal(0, 8, (h, w)),
            0, 255
        ).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────
    # 3. Sentinel-1 C-Band Synthetic Aperture Radar (SAR VV / VH)
    # ─────────────────────────────────────────────────────────────
    # SAR penetrates clouds completely.
    # Physics: 
    # - Smooth open standing water = Specular reflection (pulses bounce away) -> very low backscatter (<-18 dB / digital number 20-55)
    # - Rough vegetation = Volume scattering (digital number 100-145)
    # - Urban structures = Double-bounce corner reflection (digital number 190-250)
    # Band 1: VV polarization, Band 2: VH cross-polarization
    sar_vv = np.zeros((h, w), dtype=np.uint8)
    sar_vh = np.zeros((h, w), dtype=np.uint8)

    # General land backscatter (speckle noise characteristic of SAR)
    speckle = np.random.gamma(4.0, 0.25, (h, w)) # Multiplicative speckle
    base_land_vv = np.clip(115 * speckle, 60, 175).astype(np.uint8)
    base_land_vh = np.clip(85 * speckle, 45, 140).astype(np.uint8)

    sar_vv[:] = base_land_vv
    sar_vh[:] = base_land_vh

    # Flooded water surfaces: Specular reflection (< -18 dB, DN 15 to 45)
    water_speckle = np.random.gamma(2.0, 0.5, (h, w))
    sar_vv[flood_basin_mask] = np.clip(28 * water_speckle[flood_basin_mask], 10, 52).astype(np.uint8)
    sar_vh[flood_basin_mask] = np.clip(18 * water_speckle[flood_basin_mask], 6, 38).astype(np.uint8)

    # Settlements / Infrastructure on safe high ground (Double-bounce corner reflectors, DN 190-250)
    sar_vv[settlement_mask] = np.clip(210 + np.random.normal(0, 15, np.sum(settlement_mask)), 170, 255).astype(np.uint8)
    sar_vh[settlement_mask] = np.clip(175 + np.random.normal(0, 18, np.sum(settlement_mask)), 140, 240).astype(np.uint8)

    sar_2band = np.stack([sar_vv, sar_vh], axis=0)

    # ─────────────────────────────────────────────────────────────
    # 4. Save GeoTIFFs to backend/data/samples and frontend/public/samples
    # ─────────────────────────────────────────────────────────────
    targets = [
        Path("backend/data/samples"),
        Path("frontend/public/samples"),
    ]

    for target_dir in targets:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        opt_path = target_dir / "public_flood_cloudy_optical.tif"
        sar_path = target_dir / "public_flood_sentinel1_sar.tif"

        # Write Optical Cloudy GeoTIFF
        with rasterio.open(
            opt_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(cloudy_opt)
            dst.set_band_description(1, "Sentinel-2 MSI Red (B04)")
            dst.set_band_description(2, "Sentinel-2 MSI Green (B03)")
            dst.set_band_description(3, "Sentinel-2 MSI Blue (B02)")

        # Write SAR GeoTIFF
        with rasterio.open(
            sar_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=2,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(sar_2band)
            dst.set_band_description(1, "Sentinel-1 SAR C-Band VV Backscatter")
            dst.set_band_description(2, "Sentinel-1 SAR C-Band VH Cross-Polarization")

        print(f"Generated public satellite flood pair at {target_dir}:")
        print(f" - {opt_path.name} (Sentinel-2 Cloudy Optical RGB)")
        print(f" - {sar_path.name} (Sentinel-1 SAR Penetrating Microwave)")

if __name__ == "__main__":
    generate_public_satellite_flood_rasters()
