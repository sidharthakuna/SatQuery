import os
import urllib.parse
import requests
import io
import time
from PIL import Image

samples_dir = r"c:\Users\0000\OneDrive\Desktop\SatQuery-Ai\SatQuery\backend\data\samples"
os.makedirs(samples_dir, exist_ok=True)

prompts = {
    'urban_t1.tif': 'Realistic satellite top-down view of a developing city, bare land, sparse old roads, optical true color, earth observation',
    'urban_t2.tif': 'Realistic satellite top-down view of a highly developed city, new modern highways, dense urban buildings, optical true color',
    'flood_t1.tif': 'Realistic satellite top-down view of a green river valley and agricultural fields, normal blue river, optical true color',
    'flood_t2.tif': 'Realistic satellite top-down view of a severely flooded river valley, widespread muddy brown water covering green fields, disaster',
    'fusion_optical.tif': 'Realistic satellite top-down optical capture of a port city severely obscured by thick white cumulus clouds, true color',
    'fusion_sar.tif': 'Realistic SAR radar backscatter grayscale image of a maritime port, high contrast black and white, bright metallic structures, dark water, top down',
    'forest_vqa.tif': 'Realistic high resolution optical satellite top-down view of a dense Amazon forest canopy, true color',
    'sentinel2_coastal.tif': 'Realistic optical satellite top-down view of a coastal strait, deep blue ocean water meeting green land, true color',
    'risat_sar.tif': 'Realistic SAR radar backscatter grayscale image of icebergs in the dark ocean, top down, black and white',
    'port_grounding.tif': 'Very high resolution optical satellite top-down view of a complex naval harbor facility with distinct commercial vessels, dry docks, and industrial cranes, true color'
}

aliases = {
    'cartosat_t1.tif': 'urban_t1.tif',
    'cartosat_t2.tif': 'urban_t2.tif',
    'pre_flood_t1.tif': 'flood_t1.tif',
    'post_flood_t2.tif': 'flood_t2.tif',
    'public_flood_cloudy_optical.tif': 'fusion_optical.tif',
    'public_flood_sentinel1_sar.tif': 'fusion_sar.tif',
    'sentinel2_forest_canopy.tif': 'forest_vqa.tif',
    'dior_port_facility.tif': 'port_grounding.tif'
}

for name, prompt in prompts.items():
    if os.path.exists(os.path.join(samples_dir, name)) and name != 'urban_t2.tif':
        pass # we already have some, but let's just redownload to be safe. Actually, let's skip the one we got successfully.
    
    if name == 'urban_t1.tif':
        # we got this one successfully in the previous run
        continue

    print(f"Generating {name}...")
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&nologo=true"
    
    success = False
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                print("429 Too Many Requests, sleeping 10s...")
                time.sleep(10)
                continue
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert('RGB')
            img = img.resize((768, 768))
            img.save(os.path.join(samples_dir, name), format='TIFF')
            print(f"Saved {name}")
            success = True
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed to generate {name}: {e}")
            time.sleep(5)
    
    time.sleep(5) # Rate limit avoidance between successful images

# Handle aliases
for alias, source in aliases.items():
    print(f"Creating alias {alias} -> {source}")
    try:
        import shutil
        shutil.copyfile(os.path.join(samples_dir, source), os.path.join(samples_dir, alias))
    except Exception as e:
        print(f"Failed to copy alias {alias}: {e}")

print("All done!")
