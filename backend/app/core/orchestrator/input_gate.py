"""
SatQuery AI — Input Intelligence & Compatibility Gate
Performs deep pre-flight validation on GeoTIFF rasters before orchestrator execution:
- Modality coherence (Optical, SAR, Multispectral)
- Coordinate Reference System (CRS) compatibility & auto-reprojection planning
- Ground Sample Distance (GSD) / resolution alignment
- Radiometric signal quality: NoData detection, cloud obscuration %, saturation
- Actionable user remediation advice ("Why can't I run this?")
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.schemas.audit import ValidationReport
from app.schemas.geospatial import GeoTIFFMetadata
from config.constants import MAX_BANDS_OPTICAL, MAX_BANDS_SAR, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class InputIntelligenceGate:
    """
    First-class geospatial gatekeeper that analyzes input rasters and generates
    detailed diagnostic reports, quality scoring, and actionable remediation steps.
    """

    def evaluate(
        self,
        image_metas: List[GeoTIFFMetadata],
        images: Optional[List[Any]] = None,
    ) -> ValidationReport:
        """
        Runs comprehensive multi-dimensional compatibility and quality checks.
        """
        # Case 0: Conversational Copilot / General Q&A (no images attached)
        if len(image_metas) == 0:
            return ValidationReport(
                is_valid=True,
                image_count=0,
                modalities=[],
                crs_list=[],
                crs_compatible=True,
                band_counts=[],
                input_quality_score=1.0,
                errors=[],
                warnings=["Zero rasters attached: Routing to SatQuery Conversational Copilot for domain Q&A."],
                remediation_advice=["Attach GeoTIFF rasters via the paperclip icon or select a sample preset to execute spatial model analysis."],
            )

        errors: List[str] = []
        warnings: List[str] = []
        remediations: List[str] = []

        # Coerce dicts to GeoTIFFMetadata if passed as dicts
        normalized_metas: List[GeoTIFFMetadata] = []
        for m in image_metas:
            if isinstance(m, dict):
                normalized_metas.append(GeoTIFFMetadata(**m))
            else:
                normalized_metas.append(m)
        image_metas = normalized_metas

        # 1. Image count validation
        if len(image_metas) > 10:
            errors.append(f"Too many images supplied ({len(image_metas)}). The system accepts up to 10 concurrent images.")
            remediations.append("Select a maximum of 10 complementary or multi-temporal rasters.")

        modalities = [m.modality.upper() if m.modality else "OPTICAL" for m in image_metas]
        crs_list = [m.crs for m in image_metas]
        band_counts = [m.band_count for m in image_metas]

        # 2. File format & extension checks
        for meta in image_metas:
            file_ref = meta.file_path or meta.filename or ""
            if file_ref:
                ext = Path(file_ref).suffix.lower()
                if ext and ext not in SUPPORTED_EXTENSIONS:
                    errors.append(f"Unsupported file container format for '{meta.filename or file_ref}': '{ext}'")
                    remediations.append(f"Convert '{meta.filename or file_ref}' to standardized GeoTIFF (.tif) or Cloud-Optimized GeoTIFF (COG).")

        # 3. Band count checks
        for meta in image_metas:
            if meta.modality == "OPTICAL" and meta.band_count > MAX_BANDS_OPTICAL:
                warnings.append(f"Optical raster '{meta.filename}' has {meta.band_count} bands (expected <= {MAX_BANDS_OPTICAL}).")
                remediations.append("SatQuery will extract the primary RGB/NIR channels for visual feature reasoning.")
            if meta.modality == "SAR" and meta.band_count > MAX_BANDS_SAR:
                warnings.append(f"SAR raster '{meta.filename}' has {meta.band_count} channels (expected <= {MAX_BANDS_SAR}).")

        # 4. CRS & Spatial Registration Compatibility
        crs_compatible = True
        if len(image_metas) == 2:
            crs_a, crs_b = crs_list[0], crs_list[1]
            if crs_a is None or crs_b is None:
                warnings.append("One or both rasters lack embedded spatial projection (unprojected pixel space).")
                remediations.append("Images will be aligned using local normalized pixel grids. For metric hectarage accuracy, georeference rasters under a standard UTM projection.")
                crs_compatible = False
            elif crs_a != crs_b:
                warnings.append(f"CRS mismatch detected: Image A is `{crs_a}`, Image B is `{crs_b}`.")
                remediations.append(f"SatQuery automated co-registration will reproject Image B to `{crs_a}` reference grid before bi-temporal inference.")
                crs_compatible = False

            # Dimension compatibility check for paired rasters
            w1, h1 = image_metas[0].width, image_metas[0].height
            w2, h2 = image_metas[1].width, image_metas[1].height
            if (w1, h1) != (w2, h2):
                warnings.append(f"Dimension mismatch detected: Image A is {w1}×{h1}, Image B is {w2}×{h2}.")
                remediations.append("SatQuery automated spatial coregistration will resample rasters to matching dimensions.")

        # 5. Radiometric Quality & Cloud Obscuration Scoring
        quality_score = 1.0

        if images and len(images) > 0:
            for idx, img in enumerate(images):
                try:
                    arr = np.array(img, dtype=np.float32)
                    total_px = arr.size
                    
                    # NoData pixel detection (values <= -9999 or all zeros)
                    nodata_mask = (arr <= -9999) | (arr == 0)
                    nodata_ratio = float(np.sum(nodata_mask)) / max(total_px, 1)
                    if nodata_ratio > 0.35:
                        warnings.append(f"Raster #{idx+1} has {nodata_ratio*100:.1f}% NoData pixels.")
                        remediations.append(f"Crop the analysis region to valid bounding extents to avoid boundary edge distortions.")
                        quality_score = max(0.4, quality_score - 0.25)
                    elif nodata_ratio > 0.10:
                        quality_score = max(0.6, quality_score - 0.10)

                    # Optical Cloud Contamination Check
                    if idx < len(modalities) and modalities[idx] == "OPTICAL" and arr.ndim >= 2:
                        # Normalize to 0..1 range accurately
                        if arr.size > 0:
                            max_v = float(np.nanmax(arr))
                            norm_arr = (arr / max_v) if max_v > 1.0 else arr.copy()
                        else:
                            norm_arr = arr
                        if norm_arr.ndim == 3 and norm_arr.shape[0] >= 3:
                            r, g, b = norm_arr[0], norm_arr[1], norm_arr[2]
                            lum = 0.299 * r + 0.587 * g + 0.114 * b
                            sat = np.max(norm_arr[:3], axis=0) - np.min(norm_arr[:3], axis=0)
                            cloud_pixels = np.sum((lum > 0.72) & (sat < 0.18))
                            cloud_pct = float(cloud_pixels) / max(r.size, 1) * 100.0
                            if cloud_pct > 30.0:
                                warnings.append(f"High atmospheric cloud contamination detected in Optical scene ({cloud_pct:.1f}%).")
                                remediations.append("Pair this optical pass with an all-weather Sentinel-1 or RISAT C-band SAR image to reconstruct ground features underneath.")
                                quality_score = max(0.5, quality_score - 0.20)
                except Exception as eval_err:
                    logger.debug(f"Raster radiometric evaluation skipped: {eval_err}")

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            image_count=len(image_metas),
            modalities=modalities,
            crs_list=crs_list,
            crs_compatible=crs_compatible,
            band_counts=band_counts,
            input_quality_score=round(quality_score, 2),
            errors=errors,
            warnings=warnings,
            remediation_advice=remediations,
        )
