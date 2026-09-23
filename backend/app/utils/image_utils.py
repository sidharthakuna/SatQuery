"""
SatQuery AI — Image Utilities
Thumbnail generation, mask colorization, and bounding box rendering.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config.constants import (
    CHANGE_MASK_COLORS,
    GROUNDING_BOX_COLOR,
    GROUNDING_BOX_THICKNESS,
    THUMBNAIL_MAX_SIZE,
)

logger = logging.getLogger(__name__)


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Helper to load a scalable font across platforms (Linux, Docker, Windows)."""
    font_candidates = ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]
    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def create_thumbnail(
    image_array: np.ndarray,
    output_path: str,
    max_size: Tuple[int, int] = THUMBNAIL_MAX_SIZE,
) -> str:
    """
    Create a TIFF thumbnail from a multi-band array.

    Args:
        image_array: Float32 array of shape (C, H, W) normalized to [0, 1]
        output_path: Where to save the thumbnail
        max_size: Maximum (width, height) for thumbnail

    Returns:
        Path to the saved thumbnail
    """
    c, h, w = image_array.shape

    if c >= 3:
        rgb = image_array[:3]
    elif c == 2:
        rgb = np.stack([image_array[0], image_array[1], image_array[0]], axis=0)
    else:
        rgb = np.stack([image_array[0]] * 3, axis=0)

    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(np.transpose(rgb_uint8, (1, 2, 0)), mode="RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    if output_path.lower().endswith(".png"):
        output_path = output_path[:-4] + ".tif"
    img.save(output_path, format="TIFF")

    logger.debug(f"Created thumbnail: {output_path} ({img.size[0]}x{img.size[1]})")
    return output_path


def colorize_change_mask(
    mask: np.ndarray,
    output_path: str,
    color: Tuple[int, int, int, int] = None,
) -> str:
    """
    Convert a binary change mask to a colorized RGBA TIFF overlay.

    Args:
        mask: 2D uint8 array where 1 = changed region
        output_path: Where to save the colorized mask
        color: RGBA color for changed pixels (default: cyan overlay)

    Returns:
        Path to the saved colorized mask
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    change_color = color or CHANGE_MASK_COLORS["added"]
    rgba[mask > 0] = change_color

    img = Image.fromarray(rgba, mode="RGBA")
    if output_path.lower().endswith(".png"):
        output_path = output_path[:-4] + ".tif"
    img.save(output_path, format="TIFF")

    logger.debug(f"Created colorized change mask: {output_path}")
    return output_path


def draw_bounding_boxes(
    image_path: str,
    boxes: List[Tuple[float, float, float, float]],
    output_path: str,
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (239, 68, 68),
) -> str:
    """
    Draw bounding boxes on a satellite image chip.

    Args:
        image_path: Path to the base image (TIFF/GeoTIFF)
        boxes: List of (ymin, xmin, ymax, xmax) in normalized coordinates [0, 1]
        output_path: Where to save the annotated image
        labels: Optional text labels for each box
        color: RGB outline color (default: red)

    Returns:
        Path to the annotated image
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _get_font(12)

    for i, (ymin, xmin, ymax, xmax) in enumerate(boxes):
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        # Draw rectangle with thickness
        for offset in range(2):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        if labels and i < len(labels):
            label = labels[i]
            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 18), label, fill=(0, 0, 0), font=font)

    if output_path.lower().endswith(".png"):
        output_path = output_path[:-4] + ".tif"
    img.save(output_path, format="TIFF")
    logger.debug(f"Drew {len(boxes)} bounding boxes on {output_path}")
    return output_path


def create_comparison_image(
    image_a_path: str,
    image_b_path: str,
    output_path: str,
    labels: Tuple[str, str] = ("Before (t₁)", "After (t₂)"),
) -> str:
    """
    Create a side-by-side comparison image from two input images.

    Args:
        image_a_path: Path to first image
        image_b_path: Path to second image
        output_path: Where to save the comparison
        labels: Labels for each side

    Returns:
        Path to the saved comparison image
    """
    img_a = Image.open(image_a_path).convert("RGB")
    img_b = Image.open(image_b_path).convert("RGB")

    # Resize to same height
    target_h = min(img_a.height, img_b.height, 512)
    aspect_a = img_a.width / img_a.height
    aspect_b = img_b.width / img_b.height
    img_a = img_a.resize((int(target_h * aspect_a), target_h), Image.Resampling.LANCZOS)
    img_b = img_b.resize((int(target_h * aspect_b), target_h), Image.Resampling.LANCZOS)

    # Create combined canvas
    gap = 4
    total_w = img_a.width + gap + img_b.width
    canvas = Image.new("RGB", (total_w, target_h + 24), color=(15, 23, 42))
    canvas.paste(img_a, (0, 24))
    canvas.paste(img_b, (img_a.width + gap, 24))

    # Draw labels
    draw = ImageDraw.Draw(canvas)
    font = _get_font(14)

    draw.text((4, 4), labels[0], fill=(34, 211, 238), font=font)
    draw.text((img_a.width + gap + 4, 4), labels[1], fill=(34, 211, 238), font=font)

    if output_path.lower().endswith(".png"):
        output_path = output_path[:-4] + ".tif"
    canvas.save(output_path, format="TIFF")
    logger.debug(f"Created comparison image: {output_path}")
    return output_path


def compute_ssim_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0,
) -> Tuple[float, float]:
    """
    Computes authentic Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR)
    between two rasters/images in pure NumPy / SciPy without external skimage dependency.

    Args:
        img1: First image array (H, W, C) or (H, W) or (C, H, W) in [0, max_val]
        img2: Second image array of matching shape
        max_val: Maximum dynamic range (typically 1.0 for float32 or 255.0 for uint8)

    Returns:
        (ssim_score, psnr_db)
    """
    from scipy.ndimage import uniform_filter

    arr1 = np.asarray(img1, dtype=np.float64)
    arr2 = np.asarray(img2, dtype=np.float64)

    # Convert CHW to HWC if needed
    if arr1.ndim == 3 and arr1.shape[0] in [1, 2, 3, 4] and arr1.shape[0] < arr1.shape[1]:
        arr1 = np.transpose(arr1, (1, 2, 0))
    if arr2.ndim == 3 and arr2.shape[0] in [1, 2, 3, 4] and arr2.shape[0] < arr2.shape[1]:
        arr2 = np.transpose(arr2, (1, 2, 0))

    # Match shapes if slight crop discrepancy
    min_h = min(arr1.shape[0], arr2.shape[0])
    min_w = min(arr1.shape[1], arr2.shape[1])
    arr1 = arr1[:min_h, :min_w]
    arr2 = arr2[:min_h, :min_w]

    mse = float(np.mean((arr1 - arr2) ** 2))
    if mse < 1e-10:
        psnr = 50.0
    else:
        psnr = round(float(10.0 * np.log10((max_val ** 2) / mse)), 2)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    if arr1.ndim == 3:
        ssims = []
        channels = min(arr1.shape[2], arr2.shape[2])
        for ch in range(channels):
            x = arr1[:, :, ch]
            y = arr2[:, :, ch]
            ux = uniform_filter(x, size=11)
            uy = uniform_filter(y, size=11)
            uxx = uniform_filter(x * x, size=11)
            uyy = uniform_filter(y * y, size=11)
            uxy = uniform_filter(x * y, size=11)

            vx = uxx - ux * ux
            vy = uyy - uy * uy
            vxy = uxy - ux * uy

            ssim_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12)
            ssims.append(float(np.mean(ssim_map)))
        ssim = round(float(np.mean(ssims)), 4)
    else:
        ux = uniform_filter(arr1, size=11)
        uy = uniform_filter(arr2, size=11)
        uxx = uniform_filter(arr1 * arr1, size=11)
        uyy = uniform_filter(arr2 * arr2, size=11)
        uxy = uniform_filter(arr1 * arr2, size=11)

        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        ssim_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12)
        ssim = round(float(np.mean(ssim_map)), 4)

    return float(np.clip(ssim, -1.0, 1.0)), float(psnr)

