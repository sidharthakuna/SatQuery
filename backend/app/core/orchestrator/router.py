"""
SatQuery AI — Query Intent Classifier & Tool Router
Deterministic rule-based routing from natural language queries to TaskType
and required tools. No LLM hallucination in the routing layer.
"""

import logging
import re
from typing import Dict, List, Tuple, Any, Optional

from app.schemas.audit import TaskType
from app.schemas.geospatial import GeoTIFFMetadata

logger = logging.getLogger(__name__)

# Keyword sets for intent classification
_GROUNDING_KEYWORDS = frozenset([
    "highlight", "locate", "find", "detect", "box", "bounding",
    "where is", "show me", "identify", "mark", "segment",
    "point out", "outline", "circle", "pinpoint",
])

_CHANGE_KEYWORDS = frozenset([
    "change", "changed", "changes", "changing", "different", "difference", "before",
    "after", "temporal", "compare", "comparison", "evolution",
    "growth", "expansion", "shrink", "deforestation", "construction",
    "flood", "floods", "flooding", "flooded", "inundat", "inundation", "inundated",
    "submerg", "submerged", "submergence", "overflow", "overflowing", "deluge", "retreat",
    "erosion", "urbanization", "safe zone", "safe zones", "safety zone", "evacuation",
    "evacuate", "fallback", "next safe",
])

_FUSION_KEYWORDS = frozenset([
    "cloud", "penetrate", "radar", "sar", "microwave",
    "fuse", "fusion", "combine", "complement", "cross-modal",
    "through cloud", "under cloud", "weather",
])

_VQA_KEYWORDS = frozenset([
    "land cover", "vegetation", "ndvi", "terrain", "urban", "density", "water", "turbidity",
    "forest", "canopy", "crop", "agriculture", "soil", "albedo", "reflectance", "spectral",
    "elevation", "features", "scene", "image", "visible", "appearance", "condition",
    "infrastructure", "building", "river", "road", "lake", "harbor", "runway", "coastal",
    "desert", "mountain", "structures", "area", "region", "what is", "describe", "analyze",
    "explain", "evaluate", "inspect", "assess", "tell me about", "what does", "is there",
    "are there", "how much", "percentage", "type of", "environment", "quality", "resolution",
])

_CONVERSATIONAL_KEYWORDS = frozenset([
    "hi", "hello", "hey", "who are you", "what can you do", "help", "how to use",
    "capabilities", "guide", "welcome", "good morning", "good evening", "good afternoon",
    "thank you", "thanks", "what is satquery", "how are you", "how are u", "feeling",
    "how do you feel", "whats up", "what's up", "joke", "tell me a joke", "who made you",
    "bye", "goodbye",
])

_COMPOUND_INDICATORS = [
    r"\b(and\s+also|as\s+well\s+as|along\s+with|in\s+addition\s+to|together\s+with)\b",
    r"\b(changes?\s+and\s+(highlight|locate|find|detect|outline|box|segment|identify))\b",
    r"\b((highlight|locate|find|detect|outline|box|segment)\s+(.+?)\s+and\s+(compare|detect\s+changes?|changes?|assess\s+change))\b",
    r"\b(both\s+(detect\s+changes?|compare)\s+and\s+(highlight|locate|find|detect|outline|box))\b",
    r"\b((describe|analyze|what\s+is|classify)\s+(.+?)\s+and\s+(highlight|locate|find|detect|outline|box))\b",
    r"\b((highlight|locate|find|detect|outline|box)\s+(.+?)\s+and\s+(describe|analyze|explain|classify))\b",
    r"\b((remove|pierce|penetrate|fuse|de-cloud|eliminate|filter\s+out)\s+(.+?)\s*clouds?\s+(.+?)\s+and\s+(tell|detect|locate|measure|calculate|assess|find|map))\b",
    r"\b((remove|pierce|penetrate|fuse|de-cloud|eliminate)\s+clouds?\s+and\s+(tell|detect|locate|measure|calculate|assess|find|map))\b",
    r"\b(clouds?\s+(obstruct|cover|block)\s+(.+?)\s+and\s+(detect|tell|measure|locate|map|find))\b",
    r"\b(safe\s+zones?|fallback\s+zones?|evacuation\s+zones?)\b",
    r"\b(where\s+are\s+(the\s+)?safe\s+zones?|mark\s+them|next\s+safe\s+zone)\b",
]

# Educational / theoretical question patterns (strictly for zero-image conceptual learning)
_EDUCATIONAL_PATTERNS = [
    r"\b(conceptually|theoretically|in theory|in principle|fundamentally)\b",
    r"\b(what does .+ mean|definition of|meaning of)\b",
    r"^(what would you recommend|what are the limitations)\b",
    r"\b(electromagnetic spectrum|photogrammetry|lidar|hyperspectral)\b",
    r"\b(history of|significance of|importance of|applications of)\b",
    r"\b(sun-synchronous|orbit type|geostationary)\b",
    r"\b(copernicus|landsat program|sentinel constellation)\b",
    r"\b(precision agriculture|climate change|wildfire detection|drought monitoring)\b",
    r"\b(machine learning|deep learning|neural network|artificial intelligence)\b",
    r"\b(gis|geographic information system|map projection|coordinate system)\b",
]



class QueryIntentClassifier:
    """
    Hybrid neural & deterministic intent classifier.
    Combines deep sequence prediction from AgentIntentNet v2.0
    with physical raster sensor validation (image count, modality compatibility).
    """

    def __init__(self, provider=None):
        self._provider = provider

    def _get_provider(self):
        if self._provider is None:
            try:
                from app.inference.factory import InferenceFactory
                self._provider = InferenceFactory.get_provider()
            except Exception as e:
                logger.debug(f"Could not load inference provider for intent routing: {e}")
        return self._provider

    def classify(
        self,
        query: str,
        image_metas: List[GeoTIFFMetadata],
    ) -> Tuple[TaskType, List[str], Dict[str, Any]]:
        """
        Classify a query into a TaskType and return the tools/params needed.

        Args:
            query: Natural-language user query
            image_metas: Metadata for 0, 1, or 2 uploaded images

        Returns:
            Tuple of (TaskType, tool_ids_list, default_parameters)
        """
        q_lower = query.strip().lower()
        num_images = len(image_metas)
        modalities = [m.modality for m in image_metas]

        # 1. Neural Intent Prediction from AgentIntentNet v2.0
        provider = self._get_provider()
        neural_pred: Optional[Dict[str, Any]] = None
        neural_task: Optional[str] = None
        neural_conf: float = 0.0

        if provider and hasattr(provider, "predict_intent"):
            try:
                neural_pred = provider.predict_intent(query)
                if neural_pred:
                    neural_task = neural_pred.get("task_type")
                    neural_conf = float(neural_pred.get("confidence", 0.0))
                    logger.info(
                        f"AgentIntentNet neural prediction: {neural_task} ({neural_conf*100:.1f}% confidence)"
                    )
            except Exception as e:
                logger.warning(f"AgentIntentNet neural inference error: {e}")

        # Common intent parameters including neural telemetry
        intent_extra = {}
        if neural_pred:
            intent_extra["neural_intent"] = {
                "task_type": neural_task,
                "confidence": neural_conf,
                "model": neural_pred.get("model"),
                "probabilities": neural_pred.get("probabilities", {}),
            }

        # ═══════════════════════════════════════════════════════
        #  Zero-image scenarios: Personal AI Agent Copilot & Q&A
        # ═══════════════════════════════════════════════════════
        if num_images == 0:
            logger.info("Classified as AGENT_ASSISTANT (zero images provided: General Q&A / Copilot)")
            params = {"mode": "conversational_qna"}
            params.update(intent_extra)
            return (
                TaskType.AGENT_ASSISTANT,
                ["tool_agent_qna"],
                params,
            )

        # Check for conversational greeting / general Q&A / educational queries
        conversational_patterns = [
            r"^(hi|hello|hey|greetings|howdy)\b",
            r"^how\s+(are|r)\s+(u|you|things)",
            r"^how\s+(do|are)\s+(u|you)\s+feel",
            r"^how('?s|\s+is)\s+it\s+going",
            r"^what('?s|\s+is)\s+up",
            r"^(thanks|thank\s+you)\b",
            r"^(bye|goodbye|see\s+ya|see\s+you)\b",
            r"^who\s+(are\s+you|made\s+you|created\s+you)",
            r"^what\s+can\s+you\s+do",
            r"^(help|tell\s+me\s+a\s+joke)\b",
        ]
        is_greeting = any(re.search(pattern, q_lower) for pattern in conversational_patterns)

        has_task_keywords = (
            self._has_keywords(q_lower, _GROUNDING_KEYWORDS)
            or self._has_keywords(q_lower, _CHANGE_KEYWORDS)
            or self._has_keywords(q_lower, _FUSION_KEYWORDS)
            or self._has_keywords(q_lower, _VQA_KEYWORDS)
        )

        image_context_words = {"image", "scene", "raster", "tile", "here", "this", "area", "photo", "view", "satellite", "ground"}
        has_image_ref = any(w in q_lower for w in image_context_words)

        is_educational = any(re.search(pattern, q_lower) for pattern in _EDUCATIONAL_PATTERNS)
        is_pure_educational = is_educational and num_images == 0 and not has_image_ref

        # Route to AGENT_ASSISTANT only if:
        # 1. Zero images uploaded (General Q&A / Copilot mode), OR
        # 2. Pure greeting/farewell ("hi", "bye", etc.), OR
        # 3. Explicit theoretical/educational question with NO image context and NO task keywords
        if is_greeting or (is_pure_educational and num_images == 0) or (
            (self._has_keywords(q_lower, _CONVERSATIONAL_KEYWORDS) or (neural_task == "AGENT_ASSISTANT" and neural_conf > 0.80))
            and not has_task_keywords
            and not has_image_ref
            and num_images == 0
        ):
            logger.info("Classified as AGENT_ASSISTANT (conversational/educational Q&A)")
            params = {"mode": "conversational_with_context"}
            params.update(intent_extra)
            return (
                TaskType.AGENT_ASSISTANT,
                ["tool_agent_qna"],
                params,
            )

        # Check for explicit multi-model keywords
        _EXPLICIT_MULTI_MODEL_KEYWORDS = {
            "multi-model", "multimodel", "multi model", "all models", "run all models",
            "integrated multi-model", "pipeline synthesis", "combine all 4 models",
            "combine all models", "every model", "full pipeline", "multi-model pipeline",
            "multi model analysis", "all 4 models",
        }
        is_explicit_multi_model = any(k in q_lower for k in _EXPLICIT_MULTI_MODEL_KEYWORDS)

        has_grounding = (neural_task == "SINGLE_GROUNDING" and neural_conf > 0.65) or self._has_keywords(q_lower, _GROUNDING_KEYWORDS)
        has_change = (neural_task == "BITEMPORAL_CHANGE" and neural_conf > 0.50) or self._has_keywords(q_lower, _CHANGE_KEYWORDS)
        has_fusion = (neural_task == "CROSS_MODAL_FUSION" and neural_conf > 0.50) or self._has_keywords(q_lower, _FUSION_KEYWORDS)

        # ═══════════════════════════════════════════════════════
        #  Two-image scenarios
        # ═══════════════════════════════════════════════════════
        if num_images == 2:
            sorted_mods = sorted(modalities)

            # Multi-model compound request with 2 images
            explicit_box_or_localize = any(k in q_lower for k in ["box", "bounding", "outline", "pinpoint", "highlight", "show the affected", "show affected", "locate the affected"])
            has_multiple_objectives = sum([
                any(k in q_lower for k in ["flood", "flooded", "change", "expansion", "growth", "damage"]),
                any(k in q_lower for k in ["building", "buildings", "structures", "how many"]),
                any(k in q_lower for k in ["road", "roads", "infrastructure", "bridges", "routes"]),
            ]) >= 2
            has_compound_connectors = any(re.search(p, q_lower) for p in _COMPOUND_INDICATORS)
            is_compound_multi = (
                is_explicit_multi_model
                or has_multiple_objectives
                or has_compound_connectors
                or (has_change and explicit_box_or_localize and neural_task != "BITEMPORAL_CHANGE")
                or any(k in q_lower for k in ["and highlight", "and box", "and outline", "and pinpoint", "and show", "show the affected"])
                or ((has_fusion or "sar" in q_lower) and (explicit_box_or_localize or has_change))
                or (neural_task == "MULTI_MODEL" and neural_conf > 0.60 and (explicit_box_or_localize or has_change or has_grounding))
            )
            if is_compound_multi:
                tools = []
                if sorted_mods == ["OPTICAL", "SAR"] or any(k in q_lower for k in ["sar", "cloud", "radar", "penetrat"]):
                    tools.append("tool_optical_sar_fusion")
                if sorted_mods == ["OPTICAL", "OPTICAL"] or has_change:
                    tools.extend(["tool_change_detection", "tool_change_vqa"])
                if has_grounding or any(k in q_lower for k in ["box", "locate", "highlight", "boundaries", "pinpoint"]):
                    tools.append("tool_grounding")
                if is_explicit_multi_model:
                    tools.extend(["tool_optical_sar_fusion", "tool_change_detection", "tool_change_vqa", "tool_grounding"])
                tools = list(dict.fromkeys(tools))

                # Route single-domain results cleanly if compound criteria are not truly multi-tool
                if not is_explicit_multi_model:
                    if tools == ["tool_optical_sar_fusion"]:
                        logger.info("Routing to CROSS_MODAL_FUSION (Single fusion domain selected)")
                        params = {"fusion_mode": "cross_attention", "cloud_threshold": 0.3}
                        params.update(intent_extra)
                        return (
                            TaskType.CROSS_MODAL_FUSION,
                            ["tool_optical_sar_fusion"],
                            params,
                        )
                    elif set(tools).issubset({"tool_change_detection", "tool_change_vqa"}):
                        logger.info("Routing to BITEMPORAL_CHANGE (Only change detection domain selected)")
                        params = {"threshold": 0.5, "backbone": "ChangeFormer-V6"}
                        params.update(intent_extra)
                        return (
                            TaskType.BITEMPORAL_CHANGE,
                            ["tool_change_detection", "tool_change_vqa"],
                            params,
                        )

                logger.info("Classified as MULTI_MODEL (Compound multi-model / pipeline request with 2 images)")
                params = {
                    "threshold": 0.5,
                    "backbone": "ChangeFormer-V6",
                    "box_threshold": 0.35,
                    "text_threshold": 0.25,
                    "is_compound": True,
                }
                params.update(intent_extra)
                return (
                    TaskType.MULTI_MODEL,
                    tools,
                    params,
                )

            # Cross-modal fusion: one OPTICAL + one SAR, or neural fusion intent, or cloud-penetration/fusion keywords
            is_cloud_or_fusion_query = (
                (neural_task == "CROSS_MODAL_FUSION" and neural_conf > 0.50)
                or any(w in q_lower for w in [
                    "optical and sar", "sar and optical", "optical + sar", "sar + optical",
                    "fuse", "fusion", "cross-modal", "cross modal", "through cloud",
                    "penetrate cloud", "pierce cloud", "remove cloud", "remove clouds",
                    "de-cloud", "under cloud", "microwave", "radar penetration", "overcast",
                ])
            )
            if sorted_mods == ["OPTICAL", "SAR"] or is_cloud_or_fusion_query:
                logger.info("Classified as CROSS_MODAL_FUSION (OPTICAL+SAR pair or cloud/fusion intent)")
                params = {"fusion_mode": "cross_attention", "cloud_threshold": 0.3}
                params.update(intent_extra)
                return (
                    TaskType.CROSS_MODAL_FUSION,
                    ["tool_optical_sar_fusion"],
                    params,
                )

            # Two optical images (or same modality): Temporal comparison / Change detection / Flood Assessment
            # All disaster, flood, inundation, urban change, and displacement queries are standard BITEMPORAL_CHANGE
            logger.info(f"Classified as BITEMPORAL_CHANGE ({modalities} temporal comparison)")
            params = {"threshold": 0.5, "backbone": "ChangeFormer-V6"}
            params.update(intent_extra)
            return (
                TaskType.BITEMPORAL_CHANGE,
                ["tool_change_detection", "tool_change_vqa"],
                params,
            )

        # ═══════════════════════════════════════════════════════
        #  Single-image scenarios
        # ═══════════════════════════════════════════════════════
        if num_images == 1:
            if is_explicit_multi_model:
                logger.info("Classified as MULTI_MODEL (Explicit multi-model request on single image)")
                params = {
                    "temperature": 0.2,
                    "box_threshold": 0.35,
                    "text_threshold": 0.25,
                    "is_compound": True,
                }
                params.update(intent_extra)
                return (
                    TaskType.MULTI_MODEL,
                    ["tool_rs_vqa", "tool_grounding"],
                    params,
                )

            has_explicit_grounding = any(
                k in q_lower for k in [
                    "locate", "find", "box", "bounding", "highlight", "outline", "pinpoint",
                    "where is", "where are", "segment", "point out", "detect", "detection",
                    "grounding", "dino", "grounding dino", "show me the", "how many ships",
                    "how many buildings", "ships in", "buildings in", "tanks in", "aircraft in",
                ]
            )
            has_descriptive_vqa = any(
                w in q_lower for w in [
                    "describe", "what is the land", "tell me about the scene", "analyze the terrain",
                    "explain the land", "land cover", "landcover", "vegetation condition", "overview of the area",
                ]
            ) or (neural_task == "SINGLE_VQA" and neural_conf > 0.60)

            # Grounding intent: queries asking to detect/locate/find objects
            is_grounding = (
                has_explicit_grounding
                or (neural_task == "SINGLE_GROUNDING" and neural_conf > 0.60)
            )
            if is_grounding and not has_descriptive_vqa:
                logger.info(f"Classified as SINGLE_GROUNDING (grounding query: {query})")
                params = {"box_threshold": 0.35, "text_threshold": 0.25}
                params.update(intent_extra)
                return (
                    TaskType.SINGLE_GROUNDING,
                    ["tool_grounding"],
                    params,
                )

            # Default single image: VQA
            logger.info(f"Classified as SINGLE_VQA (default single-image task, neural={neural_task})")
            params = {"temperature": 0.2}
            params.update(intent_extra)
            return (
                TaskType.SINGLE_VQA,
                ["tool_rs_vqa"],
                params,
            )

        # ═══════════════════════════════════════════════════════
        #  Fallback: Route to Agent Assistant
        # ═══════════════════════════════════════════════════════
        logger.info(f"Fallback: Routing query with {num_images} images to AGENT_ASSISTANT")
        params = {"mode": "fallback_assistance"}
        params.update(intent_extra)
        return (
            TaskType.AGENT_ASSISTANT,
            ["tool_agent_qna"],
            params,
        )

    @staticmethod
    def _has_keywords(text: str, keywords: frozenset) -> bool:
        """Check if any keyword appears in the text using word boundaries."""
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

