"""
SatQuery AI — Visual Grounding Domain Reasoner
Synthesizes object detection, spatial bounding reticles, coordinate tables,
and quadrant clustering into articulate intelligence briefing text.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class GroundingDomainReasoner:
    """
    Synthesizes precision visual grounding evidence into structured,
    cartographically referenced answers.
    """

    @staticmethod
    def reason_grounding(
        query: str,
        boxes: List[List[float]],
        image_metas: List[Any],
        extra: Dict[str, Any],
        images: Optional[List[Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[str]]:
        q_clean = re.sub(
            r'^(please\s+)?(locate|find|detect|outline|show me|highlight|where are|identify|pinpoint)(\s+(and|or)\s+(outline|locate|find|detect|isolate|highlight))?(\s+all|\s+the|\s+any)?\s*',
            '',
            query,
            flags=re.IGNORECASE,
        ).strip('?. ')
        q_target = q_clean if q_clean else "target features"
        count = len(boxes)

        meta = image_metas[0] if image_metas else {}
        fname = getattr(meta, "filename", getattr(meta, "file_id", "scene.tif"))
        crs = getattr(meta, "crs", "EPSG:4326")
        width = getattr(meta, "width", 512)
        height = getattr(meta, "height", 512)

        # Categorize spatial distribution across quadrants
        quadrants = {"North-West": 0, "North-East": 0, "South-West": 0, "South-East": 0, "Central sector": 0}
        for b in boxes:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            x_ratio = cx / max(width, 1)
            y_ratio = cy / max(height, 1)

            if 0.35 <= x_ratio <= 0.65 and 0.35 <= y_ratio <= 0.65:
                quadrants["Central sector"] += 1
            elif x_ratio < 0.5 and y_ratio < 0.5:
                quadrants["North-West"] += 1
            elif x_ratio >= 0.5 and y_ratio < 0.5:
                quadrants["North-East"] += 1
            elif x_ratio < 0.5 and y_ratio >= 0.5:
                quadrants["South-West"] += 1
            else:
                quadrants["South-East"] += 1

        active_quads = [f"**{k}** ({v})" for k, v in quadrants.items() if v > 0]
        quad_summary = ", ".join(active_quads) if active_quads else "dispersed across the footprint"

        lines = []
        if count == 0:
            lines.append(f"I inspected the satellite scene **{fname}** for **\"{q_target}\"**, but no candidate features met the grounding threshold.")
            lines.append("\n> **Observation**: Feature dimensions may fall below sensor Ground Sample Distance (GSD), or spectral contrast against adjacent substrate is low.")
            lines.append("\n**Suggested Next Steps:**")
            lines.append("- Broaden your query (e.g. *\"Locate all structures or buildings\"*).")
            lines.append("- Toggle the Map layer or check the multispectral bands.")
            return "\n".join(lines), ["Broaden detection prompt", "Switch to Map view", "Check sensor resolution"]

        lines.append(f"I analyzed the scene and isolated **{count} target instance(s)** matching **\"{q_target}\"** across the raster footprint.")
        lines.append(f"The detected targets are clustered primarily in the {quad_summary}, referenced under `{crs}` coordinates.")

        lines.append("\n### Identified Target Coordinates")
        lines.append("| Target | Bounding Box [X1, Y1, X2, Y2] | Centroid (X, Y) | Dimensions (W × H) | Verification |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        clusters = extra.get("clusters", []) if isinstance(extra, dict) else []
        for i, b in enumerate(boxes, 1):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            cx, cy = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
            bw, bh = x2 - x1, y2 - y1
            c_name = None
            if i - 1 < len(clusters) and isinstance(clusters[i - 1], dict):
                c_name = clusters[i - 1].get("zone")
            label_text = c_name if c_name else f"Target #{i}"
            lines.append(f"| **{label_text}** | `[{x1}, {y1}, {x2}, {y2}]` | `({cx}, {cy})` | {bw} × {bh} px | Grounded candidate |")

        lines.append("\n### Spatial & Operational Context")
        lines.append(f"- **Distribution**: Grounded features show distinct clustering in {quad_summary}.")
        lines.append(f"- **Spatial Footprint**: Bounding boxes range up to {max((b[2]-b[0]) for b in boxes):.0f}px in width, typical for organized civil or industrial installations.")

        suggestions = [
            f"Tell me more about Target #1",
            "Calculate total bounding area",
            "Compare with cadastral map",
            "Export PDF Mission Briefing",
        ]
        return "\n".join(lines), suggestions
