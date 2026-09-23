import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np

def fetch_urban_sar():
    token_resp = requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc', timeout=10)
    token = token_resp.json()['token']
    
    stac_resp = requests.post('https://planetarycomputer.microsoft.com/api/stac/v1/search', 
                              json={'collections': ['sentinel-1-rtc'], 'bbox': [77.6, 12.9, 77.7, 12.98], 'limit': 1}, 
                              timeout=10)
    feat = stac_resp.json()['features'][0]
    vv_url = f"{feat['assets']['vv']['href']}?{token}"
    vh_url = f"{feat['assets']['vh']['href']}?{token}"
    
    print("Opening authentic Sentinel-1 VV...")
    with rasterio.open(vv_url) as src_vv:
        print("Sentinel-1 CRS:", src_vv.crs)
        xmin, ymin, xmax, ymax = transform_bounds('EPSG:4326', src_vv.crs, 77.6, 12.9, 77.7, 12.98)
        print("Transformed bounds:", xmin, ymin, xmax, ymax)
        win_vv = from_bounds(xmin, ymin, xmax, ymax, src_vv.transform)
        vv_data = src_vv.read(1, window=win_vv, out_shape=(768, 768))
        
    print("Opening authentic Sentinel-1 VH...")
    with rasterio.open(vh_url) as src_vh:
        win_vh = from_bounds(xmin, ymin, xmax, ymax, src_vh.transform)
        vh_data = src_vh.read(1, window=win_vh, out_shape=(768, 768))
        
    sar_2band = np.stack([vv_data, vh_data], axis=0).astype(np.float32)
    sar_2band = np.nan_to_num(sar_2band, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Linear to normalized [0, 1] range for dual-pol neural processing
    p99 = np.percentile(sar_2band, 99.5)
    if p99 > 0:
        sar_2band = np.clip(sar_2band / p99, 0.0, 1.0)
        
    print("Saving urban_sar.tif Shape:", sar_2band.shape, "Min:", sar_2band.min(), "Max:", sar_2band.max())
    
    from pathlib import Path
    for dest_dir in [Path("backend/data/samples"), Path("data/samples"), Path("frontend/public/samples")]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_tif = dest_dir / "urban_sar.tif"
        with rasterio.open(
            str(out_tif),
            'w',
            driver='GTiff',
            height=768,
            width=768,
            count=2,
            dtype='float32',
            crs='EPSG:4326',
            transform=rasterio.transform.from_bounds(77.6, 12.9, 77.7, 12.98, 768, 768)
        ) as dst:
            dst.write(sar_2band[0], 1)
            dst.write(sar_2band[1], 2)
            
    print("SUCCESS! Authentic Sentinel-1 SAR pass saved to urban_sar.tif")

if __name__ == "__main__":
    fetch_urban_sar()
