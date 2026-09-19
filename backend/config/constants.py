"""
SatQuery AI — Domain Constants & Sensor Band Definitions
These are the anti-hallucination ground-truth values used throughout the system.
"""

from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════
#  Coordinate Reference Systems
# ═══════════════════════════════════════════════════════════════
DEFAULT_CRS = "EPSG:4326"
WEB_MERCATOR_CRS = "EPSG:3857"

# ═══════════════════════════════════════════════════════════════
#  Sensor Band Maps
# ═══════════════════════════════════════════════════════════════
CARTOSAT_2S_BANDS: Dict[int, str] = {
    1: "Blue (B2) — 0.45–0.52 µm",
    2: "Green (B3) — 0.52–0.59 µm",
    3: "Red (B4) — 0.62–0.68 µm",
    4: "NIR (B5) — 0.77–0.86 µm",
}

SENTINEL_2_BANDS: Dict[int, str] = {
    1: "Blue (B02) — 490 nm, 10 m",
    2: "Green (B03) — 560 nm, 10 m",
    3: "Red (B04) — 665 nm, 10 m",
    4: "NIR (B08) — 842 nm, 10 m",
    5: "SWIR-1 (B11) — 1610 nm, 20 m",
    6: "SWIR-2 (B12) — 2190 nm, 20 m",
}

SENTINEL_1_BANDS: Dict[int, str] = {
    1: "VV — Vertical Tx / Vertical Rx",
    2: "VH — Vertical Tx / Horizontal Rx",
}

RISAT_BANDS: Dict[int, str] = {
    1: "HH or VV — Co-polarization (C-band 5.35 GHz)",
    2: "HV or VH — Cross-polarization (C-band 5.35 GHz)",
}

# ═══════════════════════════════════════════════════════════════
#  Radiometric Calibration Constants
# ═══════════════════════════════════════════════════════════════
OPTICAL_PERCENTILE_LOW: float = 2.0
OPTICAL_PERCENTILE_HIGH: float = 98.0
OPTICAL_TOA_SCALE: float = 10000.0

SAR_DB_MIN: float = -25.0
SAR_DB_MAX: float = 0.0
SAR_EPSILON: float = 1e-6

# ═══════════════════════════════════════════════════════════════
#  Sliding Window Tiler Defaults
# ═══════════════════════════════════════════════════════════════
DEFAULT_CHIP_SIZE: int = 512
DEFAULT_STRIDE: int = 384

# ═══════════════════════════════════════════════════════════════
#  Supported File Formats
# ═══════════════════════════════════════════════════════════════
SUPPORTED_EXTENSIONS: List[str] = [".tif", ".tiff", ".geotiff"]
MAX_BANDS_OPTICAL: int = 12
MAX_BANDS_SAR: int = 4

# ═══════════════════════════════════════════════════════════════
#  Modality Detection Heuristics
# ═══════════════════════════════════════════════════════════════
SAR_BAND_COUNTS: List[int] = [1, 2]
OPTICAL_BAND_COUNTS: List[int] = [3, 4, 6, 8, 10, 12]

# ═══════════════════════════════════════════════════════════════
#  Visualization Color Palettes
# ═══════════════════════════════════════════════════════════════
CHANGE_MASK_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "no_change": (0, 0, 0, 0),
    "added": (0, 255, 80, 110),
    "removed": (255, 50, 50, 110),
    "modified": (255, 200, 0, 110),
}

GROUNDING_BOX_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Cyan
GROUNDING_BOX_THICKNESS: int = 3

# ═══════════════════════════════════════════════════════════════
#  Thumbnail Dimensions
# ═══════════════════════════════════════════════════════════════
THUMBNAIL_MAX_SIZE: Tuple[int, int] = (512, 512)
THUMBNAIL_FORMAT: str = "PNG"

# ═══════════════════════════════════════════════════════════════
#  Mock Inference Latency (ms) for realistic demo timing
# ═══════════════════════════════════════════════════════════════
MOCK_LATENCY_RANGE_MS: Tuple[int, int] = (200, 800)
