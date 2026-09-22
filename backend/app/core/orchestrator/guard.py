"""
SatQuery AI — Input Compatibility Guard
Validates CRS compatibility, band counts, modality, and file format
before routing to specialist tools.
"""

import logging
from pathlib import Path
from typing import List

from app.schemas.audit import ValidationReport
from app.schemas.geospatial import GeoTIFFMetadata
from config.constants import MAX_BANDS_OPTICAL, MAX_BANDS_SAR, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class InputCompatibilityGuard:
    """
    Validates all input images before they enter the orchestrator pipeline.
    Checks:
    1. File format (must be GeoTIFF)
    2. Band count within expected ranges
    3. CRS compatibility between multi-image inputs
    4. Modality coherence (no invalid combinations)
    """

    def validate(self, image_metas: List[GeoTIFFMetadata]) -> ValidationReport:
        """
        Run all validation checks and return a comprehensive report.

        Args:
            image_metas: List of extracted GeoTIFF metadata (1 or 2 images)

        Returns:
            ValidationReport with pass/fail status and diagnostics
        """
        errors: List[str] = []
        warnings: List[str] = []

        # --- Check image count ---
        if len(image_metas) == 0:
            # Valid scenario: Personal AI Agent Q&A / conversational remote sensing copilot
            return ValidationReport(
                is_valid=True,
                image_count=0,
                errors=[],
                warnings=["No rasters attached: Routing to SatQuery Personal AI Agent for domain Q&A and guidance."],
            )


        if len(image_metas) > 2:
            errors.append(f"Too many images ({len(image_metas)}). Maximum is 2.")

        modalities = [m.modality.upper() if m.modality else "OPTICAL" for m in image_metas]
        crs_list = [m.crs for m in image_metas]
        band_counts = [m.band_count for m in image_metas]

        # --- Check file format ---
        for meta in image_metas:
            file_ref = meta.file_path or getattr(meta, "filename", "") or ""
            if file_ref:
                ext = Path(file_ref).suffix.lower()
                if ext and ext not in SUPPORTED_EXTENSIONS:
                    errors.append(f"Unsupported format for {file_ref}: '{ext}'")

        # --- Check band counts ---
        for meta in image_metas:
            if meta.modality == "OPTICAL" and meta.band_count > MAX_BANDS_OPTICAL:
                warnings.append(
                    f"Optical image {meta.file_id} has {meta.band_count} bands "
                    f"(max expected: {MAX_BANDS_OPTICAL})"
                )
            if meta.modality == "SAR" and meta.band_count > MAX_BANDS_SAR:
                warnings.append(
                    f"SAR image {meta.file_id} has {meta.band_count} bands "
                    f"(max expected: {MAX_BANDS_SAR})"
                )

        # --- Check CRS compatibility ---
        crs_compatible = True
        if len(image_metas) == 2:
            crs_a, crs_b = crs_list[0], crs_list[1]
            # Handle None CRS (unprojected or unreadable images)
            if crs_a is None or crs_b is None:
                warnings.append(
                    "One or both images have no CRS (unprojected). "
                    "Auto-reprojection will be skipped; results may be misaligned."
                )
                crs_compatible = False
            elif crs_a != crs_b:
                # Not necessarily an error — we can reproject
                warnings.append(
                    f"CRS mismatch: {crs_a} vs {crs_b}. "
                    "Auto-reprojection will be applied."
                )
                crs_compatible = False

        # --- Check modality coherence ---
        if len(image_metas) == 2:
            sorted_mods = sorted(modalities)
            if sorted_mods == ["OPTICAL", "OPTICAL"]:
                # Bi-temporal change detection (valid)
                pass
            elif sorted_mods == ["OPTICAL", "SAR"]:
                # Cross-modal fusion (valid)
                pass
            elif sorted_mods == ["SAR", "SAR"]:
                # SAR-SAR change detection (valid but warn)
                warnings.append(
                    "Both images are SAR. Bi-temporal SAR change detection will be used."
                )
            else:
                errors.append(f"Invalid modality combination: {modalities}")

        # --- Check dimensions match for paired inputs ---
        if len(image_metas) == 2:
            w1, h1 = image_metas[0].width, image_metas[0].height
            w2, h2 = image_metas[1].width, image_metas[1].height
            if (w1, h1) != (w2, h2):
                warnings.append(
                    f"Dimension mismatch: {w1}×{h1} vs {w2}×{h2}. "
                    "Resampling will be applied."
                )

        is_valid = len(errors) == 0

        report = ValidationReport(
            is_valid=is_valid,
            image_count=len(image_metas),
            modalities=modalities,
            crs_list=crs_list,
            crs_compatible=crs_compatible,
            band_counts=band_counts,
            errors=errors,
            warnings=warnings,
        )

        if is_valid:
            logger.info(f"Input validation PASSED ({len(image_metas)} images, modalities={modalities})")
        else:
            logger.warning(f"Input validation FAILED: {errors}")

        return report
