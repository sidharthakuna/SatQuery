"""
SatQuery AI — Shared File Lookup Utility
Provides a single, canonical implementation of `find_uploaded_file` so that
the query, stream, and tile endpoints all use identical extension handling.

Bug fixed: endpoints_stream and endpoints_tiles previously omitted '.geotiff'
from their local copies, causing files uploaded with that extension to be
invisible to WebSocket queries and the tile server.
"""

from pathlib import Path
from typing import Optional

from config.constants import SUPPORTED_EXTENSIONS
from config.settings import settings


def find_uploaded_file(file_id: str) -> Optional[str]:
    """
    Locate an image raster file by its file_id or filename in the upload directory
    or the sample rasters directory.

    Searches for:
    1. Direct filesystem paths if a valid path was passed.
    2. Exact matches or prefix matches in upload_dir or samples_dir.

    Args:
        file_id: The file identifier, filename, or stem.

    Returns:
        Absolute path string if found, otherwise None.
    """
    if not file_id:
        return None

    # Helper to verify path is safely inside allowed directories
    def _is_safe_path(p: Path) -> bool:
        resolved = p.resolve()
        allowed = [settings.upload_dir.resolve(), settings.samples_dir.resolve(), settings.data_dir.resolve()]
        return any(resolved == a or a in resolved.parents for a in allowed)

    # Disallow path traversal sequences
    if ".." in file_id:
        return None

    # Check direct path only if it stays within allowed directories
    direct_path = Path(file_id)
    if direct_path.exists() and direct_path.is_file():
        if _is_safe_path(direct_path):
            return str(direct_path.resolve())
        return None

    candidate_dirs = [settings.upload_dir, settings.samples_dir, Path("data/samples")]

    for d in candidate_dirs:
        if not d.exists():
            continue

        # Exact filename match
        exact = d / file_id
        if exact.exists() and exact.is_file():
            return str(exact.resolve())

        # Exact match with supported extensions appended
        for ext in SUPPORTED_EXTENSIONS:
            cand = d / f"{file_id}{ext}"
            if cand.exists() and cand.is_file():
                return str(cand.resolve())

        # Search directory entries
        for fp in d.iterdir():
            if not fp.is_file():
                continue
            ext = fp.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                if fp.name.startswith(file_id) or fp.stem == file_id:
                    return str(fp.resolve())

    return None
