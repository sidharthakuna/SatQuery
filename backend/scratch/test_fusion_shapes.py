import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from app.models.fusion_net import OpticalSARCrossAttentionNetV2

m = OpticalSARCrossAttentionNetV2()
ckpt_path = Path("data/checkpoints/fusion_net.pt")
st = torch.load(ckpt_path, map_location="cpu")
m.load_state_dict(st["model_state_dict"])
m.eval()
print("Loaded V2 successfully!")

try:
    out512, conf512 = m(torch.randn(1, 3, 512, 512), torch.randn(1, 2, 512, 512))
    print("512x512 passed! out:", out512.shape, "conf:", conf512.shape)
except Exception as e:
    print("512x512 failed:", e)

try:
    out768, conf768 = m(torch.randn(1, 3, 768, 768), torch.randn(1, 2, 768, 768))
    print("768x768 passed! out:", out768.shape, "conf:", conf768.shape)
except Exception as e:
    print("768x768 failed:", e)
