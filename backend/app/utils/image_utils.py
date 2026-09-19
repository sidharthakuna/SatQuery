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
