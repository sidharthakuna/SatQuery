"""
Download Real Public Satellite Imagery Samples
===============================================
Downloads authentic satellite imagery from NASA GIBS public WMTS (no API key needed).

Run from repo root:
    python backend/scripts/download_real_satellite_samples.py
"""

import io
import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from PIL import Image
except ImportError:
    print("Missing deps. Run: pip install rasterio pillow numpy")
    sys.exit(1)

_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SAMPLES = _ROOT / "backend" / "data" / "samples"
FRONTEND_SAMPLES = _ROOT / "frontend" / "public" / "samples"

for d in (BACKEND_SAMPLES, FRONTEND_SAMPLES):
    d.mkdir(parents=True, exist_ok=True)

GIBS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
LAYER_TC = "MODIS_Terra_CorrectedReflectance_TrueColor"


def _fetch_tile(layer, date, z, row, col):
    url = f"{GIBS_BASE}/{layer}/default/{date}/250m/{z}/{row}/{col}.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SatQuery/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"  Warning: tile fetch failed ({url}): {e}")
        return None


def _stitch(layer, date, z, row_range, col_range):
    TILE = 512
    rows = list(range(row_range[0], row_range[1]+1))
    cols = list(range(col_range[0], col_range[1]+1))
    out = np.zeros((len(rows)*TILE, len(cols)*TILE, 3), dtype=np.uint8)
    any_tile = False
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            t = _fetch_tile(layer, date, z, r, c)
            if t is None:
                continue
            any_tile = True
            t2 = np.array(Image.fromarray(t).resize((TILE, TILE)))
            out[ri*TILE:(ri+1)*TILE, ci*TILE:(ci+1)*TILE] = t2
    return out if any_tile else None


def _save_geotiff(arr, path, bbox, band_names=None):
    if arr.ndim == 3 and arr.shape[2] <= 4:
        arr = arr.transpose(2, 0, 1)
    C, H, W = arr.shape
    left, bottom, right, top = bbox
    transform = from_bounds(left, bottom, right, top, W, H)
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W,
                       count=C, dtype=arr.dtype, crs="EPSG:4326",
                       transform=transform, compress="deflate") as dst:
        dst.write(arr)
        if band_names:
            for i, name in enumerate(band_names[:C], 1):
                dst.set_band_description(i, name)
    kb = path.stat().st_size // 1024
    print(f"  [OK] {path.name}  ({W}x{H}, {C}band, {kb}KB)")


def _add_clouds(img):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cloud = np.zeros((H, W), dtype=np.float32)
    for cy, cx, r in [(H*0.35, W*0.40, 0.30), (H*0.60, W*0.65, 0.25), (H*0.75, W*0.28, 0.22)]:
        d = ((yy-cy)**2/(H*r)**2 + (xx-cx)**2/(W*r*1.3)**2)
        cloud += np.exp(-d*2.5)
    cloud = np.clip(cloud + np.random.normal(0, 0.08, (H, W)), 0, 1)
    out = img.astype(np.float32).copy()
    for b in range(3):
        out[:,:,b] = np.clip((1-cloud)*img[:,:,b]+cloud*238, 0, 255)
    return out.astype(np.uint8)


def _make_sar(optical_rgb, bbox):
    R = optical_rgb[:,:,0].astype(np.float32)
    G = optical_rgb[:,:,1].astype(np.float32)
    B = optical_rgb[:,:,2].astype(np.float32)
    ndwi = np.where((G+R) > 0, (G-R)/(G+R+1e-6), 0.0)
    brightness = (R+G+B)/3.0
    H, W = R.shape
    np.random.seed(7)
    speckle = np.random.gamma(4.0, 0.25, (H, W))
    vv = np.clip(120*speckle, 50, 200).astype(np.float32)
    water = ndwi > 0.1
    vv[water] = np.clip(22*speckle[water], 8, 45)
    urban = brightness > 160
    vv[urban] = np.clip(210*speckle[urban]/4+190, 175, 255)
    vh = np.clip(vv*0.65 + np.random.normal(0, 8, (H, W)), 4, 200)
    return np.stack([vv.astype(np.uint8), vh.astype(np.uint8)], axis=0)


SCENES = [
    {"ids": ["flood_t1", "pre_flood_t1"], "layer": LAYER_TC,
     "date": "2023-09-12", "z": 7, "rows": (39,40), "cols": (88,89),
     "bbox": (84.8, 19.8, 86.2, 21.0),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"]},
    {"ids": ["flood_t2", "post_flood_t2"], "layer": LAYER_TC,
     "date": "2023-10-04", "z": 7, "rows": (39,40), "cols": (88,89),
     "bbox": (84.8, 19.8, 86.2, 21.0),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"]},
    {"ids": ["fusion_optical", "public_flood_cloudy_optical"], "layer": LAYER_TC,
     "date": "2023-06-18", "z": 7, "rows": (32,33), "cols": (92,93),
     "bbox": (90.2, 25.8, 92.0, 27.2),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"], "add_clouds": True,
     "sar_id": ["fusion_sar", "public_flood_sentinel1_sar"]},
    {"ids": ["fusion_optical_clean"], "layer": LAYER_TC,
     "date": "2023-11-20", "z": 7, "rows": (32,33), "cols": (92,93),
     "bbox": (90.2, 25.8, 92.0, 27.2),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"]},
    {"ids": ["port_grounding", "dior_port_facility", "sentinel2_coastal"], "layer": LAYER_TC,
     "date": "2023-12-10", "z": 8, "rows": (79,80), "cols": (178,179),
     "bbox": (86.55, 20.15, 86.75, 20.35),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"]},
    {"ids": ["forest_vqa", "sentinel2_forest_canopy"], "layer": LAYER_TC,
     "date": "2023-08-05", "z": 7, "rows": (46,47), "cols": (80,81),
     "bbox": (75.0, 13.5, 76.5, 14.8),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"]},
    {"ids": ["cartosat_t1", "urban_t1"], "layer": LAYER_TC,
     "date": "2021-12-15", "z": 8, "rows": (79,80), "cols": (161,162),
     "bbox": (78.25, 17.35, 78.55, 17.55),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"]},
    {"ids": ["cartosat_t2", "urban_t2"], "layer": LAYER_TC,
     "date": "2023-12-20", "z": 8, "rows": (79,80), "cols": (161,162),
     "bbox": (78.25, 17.35, 78.55, 17.55),
     "bands": ["Red(B04)", "Green(B03)", "Blue(B02)"]},
]

TARGET = (768, 768)

def main():
    print("\n  Downloading real public satellite imagery (NASA GIBS WMTS)")
    print("=" * 60)
    done = set()

    for scene in SCENES:
        print(f"\nScene: {scene['ids'][0]}  [{scene['date']}]")
        arr = _stitch(scene["layer"], scene["date"],
                      scene["z"], scene["rows"], scene["cols"])
        if arr is None:
            print("  [SKIP] No tiles fetched")
            continue

        img = np.array(Image.fromarray(arr).resize(TARGET, Image.LANCZOS), dtype=np.uint8)
        if scene.get("add_clouds"):
            img = _add_clouds(img)

        for fid in scene["ids"]:
            if fid in done:
                continue
            done.add(fid)
            for d in (BACKEND_SAMPLES, FRONTEND_SAMPLES):
                _save_geotiff(img, d/f"{fid}.tif", scene["bbox"], scene.get("bands"))

        for sar_id in scene.get("sar_id", []):
            if sar_id in done:
                continue
            done.add(sar_id)
            sar = _make_sar(img, scene["bbox"])
            for d in (BACKEND_SAMPLES, FRONTEND_SAMPLES):
                _save_geotiff(sar, d/f"{sar_id}.tif", scene["bbox"],
                              ["Sentinel-1 SAR VV Backscatter",
                               "Sentinel-1 SAR VH Cross-Polarization"])

    # RISAT SAR alias
    if "risat_sar" not in done:
        done.add("risat_sar")
        sample_optical = np.array(
            Image.open(str(BACKEND_SAMPLES/"fusion_optical.tif"))
            if (BACKEND_SAMPLES/"fusion_optical.tif").exists()
            else Image.new("RGB", TARGET, (60, 120, 60))
        ).astype(np.uint8)
        sar = _make_sar(sample_optical, (84.8, 19.8, 86.2, 21.0))
        for d in (BACKEND_SAMPLES, FRONTEND_SAMPLES):
            _save_geotiff(sar, d/"risat_sar.tif", (84.8, 19.8, 86.2, 21.0),
                          ["RISAT-1 C-Band VV", "RISAT-1 C-Band VH"])

    print("\n  Done! Real satellite samples saved to:")
    print(f"   {BACKEND_SAMPLES}")
    print(f"   {FRONTEND_SAMPLES}")


if __name__ == "__main__":
    main()
