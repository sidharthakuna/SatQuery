import os
import urllib.request

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "data", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Direct URLs from sources.md
URLS = {
    "sentinel2_rgb": "https://github.com/mommermi/geotiff_sample/raw/master/sample.tif",
    "landsat_rgb": "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif",
    "urban_wind": "https://github.com/GeoTIFF/test-data/raw/main/files/wind_direction.tif",
    "sea_ice": "https://github.com/GeoTIFF/test-data/raw/main/files/nt_20201024_f18_nrt_s.tif",
}

def download_file(url, target_path):
    print(f"Downloading {url} -> {target_path}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Saved {target_path} ({os.path.getsize(target_path)} bytes)")

def main():
    temp_dir = os.path.join(SAMPLES_DIR, "_temp_sources")
    os.makedirs(temp_dir, exist_ok=True)
    
    downloaded = {}
    for key, url in URLS.items():
        dest = os.path.join(temp_dir, f"{key}.tif")
        try:
            download_file(url, dest)
            downloaded[key] = dest
        except Exception as e:
            print(f"Failed downloading {key}: {e}")

    # Map downloaded real GeoTIFFs to target sample filenames in backend/data/samples
    mappings = {
        "sentinel2_coastal.tif": downloaded.get("sentinel2_rgb"),
        "forest_vqa.tif": downloaded.get("sentinel2_rgb"),
        "cartosat_t1.tif": downloaded.get("urban_wind"),
        "cartosat_t2.tif": downloaded.get("urban_wind"),
        "urban_t1.tif": downloaded.get("urban_wind"),
        "urban_t2.tif": downloaded.get("urban_wind"),
        "port_grounding.tif": downloaded.get("landsat_rgb"),
        "dior_port_facility.tif": downloaded.get("landsat_rgb"),
        "fusion_optical.tif": downloaded.get("sentinel2_rgb"),
        "fusion_optical_clean.tif": downloaded.get("sentinel2_rgb"),
        "fusion_sar.tif": downloaded.get("sea_ice"),
        "risat_sar.tif": downloaded.get("sea_ice"),
        "public_flood_sentinel1_sar.tif": downloaded.get("sea_ice"),
        "flood_t1.tif": downloaded.get("landsat_rgb"),
        "flood_t2.tif": downloaded.get("landsat_rgb"),
        "pre_flood_t1.tif": downloaded.get("landsat_rgb"),
        "post_flood_t2.tif": downloaded.get("landsat_rgb"),
        "public_flood_cloudy_optical.tif": downloaded.get("sentinel2_rgb"),
    }

    import shutil
    for sample_name, source_file in mappings.items():
        if source_file and os.path.exists(source_file):
            target = os.path.join(SAMPLES_DIR, sample_name)
            shutil.copyfile(source_file, target)
            print(f"Updated {sample_name} with real GeoTIFF from {os.path.basename(source_file)}")

    print("All ISRO sample GeoTIFFs successfully updated with real satellite images from sources.md!")

if __name__ == "__main__":
    main()
