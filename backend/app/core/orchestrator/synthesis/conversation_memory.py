"""
SatQuery AI — Multi-Turn Conversational Memory & Context Engine
Maintains conversation state, tracks entity references, resolves follow-up questions,
and ensures fluent, context-aware dialogues without repeating canned introductions.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class ConversationMemoryTracker:
    """
    Analyzes multi-turn dialogue history to detect follow-up queries,
    resolve anaphoric references (e.g. 'it', 'that area', 'the sector', 'the flood'),
    and maintain continuous conversational flow.
    """

    @staticmethod
    def extract_prior_context(history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extracts key entities, topics, quadrants, and metrics discussed in prior dialogue turns.
        """
        if not history:
            return {
                "turns_count": 0,
                "topics": [],
                "quadrants": [],
                "last_user_query": None,
                "last_assistant_answer": None,
                "is_followup": False,
                "referenced_entities": [],
            }

        clean_history = [
            turn for turn in history
            if isinstance(turn, dict) and turn.get("content")
        ]

        turns_count = len(clean_history)
        last_user_query = None
        last_assistant_answer = None

        for turn in reversed(clean_history):
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user" and not last_user_query:
                last_user_query = content
            elif role in ("assistant", "model") and not last_assistant_answer:
                last_assistant_answer = content
            if last_user_query and last_assistant_answer:
                break

        topics = []
        combined_prior = " ".join([t.get("content", "").lower() for t in clean_history[-4:]])
        if any(w in combined_prior for w in ["flood", "submerged", "inundat", "overflow"]):
            topics.append("flood_inundation")
        if any(w in combined_prior for w in ["vegetation", "canopy", "crop", "forest", "agriculture"]):
            topics.append("vegetation")
        if any(w in combined_prior for w in ["building", "structure", "urban", "housing", "settlement"]):
            topics.append("urban_structures")
        if any(w in combined_prior for w in ["water", "river", "lake", "reservoir", "hydrology"]):
            topics.append("hydrology")
        if any(w in combined_prior for w in ["road", "highway", "transport", "route"]):
            topics.append("transportation")
        if any(w in combined_prior for w in ["port", "ship", "vessel", "maritime", "berth"]):
            topics.append("maritime_port")

        # Spatial quadrant detection in prior context with word boundaries
        quadrants = []
        if re.search(r'\b(north-west|northwest|nw)\b', combined_prior):
            quadrants.append("North-West")
        if re.search(r'\b(north-east|northeast|ne)\b', combined_prior):
            quadrants.append("North-East")
        if re.search(r'\b(south-west|southwest|sw)\b', combined_prior):
            quadrants.append("South-West")
        if re.search(r'\b(south-east|southeast|se)\b', combined_prior):
            quadrants.append("South-East")

        # Target entities mentioned with word boundaries
        entities = []
        for ent in ["building", "reservoir", "lake", "forest", "crop", "runway", "river", "road", "dock", "port", "ship", "tank"]:
            if re.search(rf'\b{ent}s?\b', combined_prior):
                entities.append(ent)

        # Extract prior bounding boxes / ROI from previous analytical results
        prior_boxes = []
        for turn in reversed(clean_history):
            res = turn.get("result") or turn.get("extra")
            if isinstance(res, dict):
                b = res.get("bounding_boxes")
                if not b and "spatial_evidence" in res and isinstance(res["spatial_evidence"], dict):
                    b = res["spatial_evidence"].get("bounding_boxes")
                if b and isinstance(b, list) and len(b) > 0:
                    prior_boxes = b
                    break

        return {
            "turns_count": turns_count,
            "topics": topics,
            "quadrants": quadrants,
            "last_user_query": last_user_query,
            "last_assistant_answer": last_assistant_answer,
            "is_followup": turns_count >= 1,
            "referenced_entities": entities,
            "prior_boxes": prior_boxes,
        }

    @staticmethod
    def resolve_spatial_coreference(query: str, history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Resolves spatial pronouns ('that area', 'there', 'it', 'inside that', 'the other sector')
        by binding them to the previous turn's active entities, sectors, and bounding box ROIs.
        """
        context = ConversationMemoryTracker.extract_prior_context(history)
        q_lower = query.lower()
        resolved_sector = None
        resolved_entity = None
        resolved_roi_box = None

        if any(w in q_lower for w in ["that area", "that sector", "that quadrant", "there", "it", "inside that", "within that", "in that"]):
            if context["quadrants"]:
                resolved_sector = context["quadrants"][-1]
            if context["referenced_entities"]:
                resolved_entity = context["referenced_entities"][-1]
            if context["prior_boxes"]:
                resolved_roi_box = context["prior_boxes"][0]

        # Check for complementary/opposite sector requests
        if any(w in q_lower for w in ["other sector", "opposite side", "other quadrant"]):
            if "North-West" in context["quadrants"]:
                resolved_sector = "South-East"
            elif "North-East" in context["quadrants"]:
                resolved_sector = "South-West"
            elif "South-West" in context["quadrants"]:
                resolved_sector = "North-East"
            elif "South-East" in context["quadrants"]:
                resolved_sector = "North-West"

        return {
            "resolved_sector": resolved_sector,
            "resolved_entity": resolved_entity,
            "resolved_roi_box": resolved_roi_box,
            "prior_topics": context["topics"],
            "is_coreference": (resolved_sector is not None or resolved_entity is not None or resolved_roi_box is not None),
        }

    @staticmethod
    def is_followup_question(query: str, history: Optional[List[Dict[str, Any]]]) -> bool:
        """
        Determines if current query is continuing a prior discussion.
        """
        if not history or len(history) == 0:
            return False

        q_lower = query.lower().strip()
        followup_starters = [
            "what about", "how about", "and the", "can you tell me more", "tell me more",
            "is there any", "are there any", "why", "how come", "what else", "where is it",
            "is it safe", "how many", "does it have", "compare", "near that", "in that area",
            "explain further", "specifically", "what of", "what kind of", "show me", "chart", "plot",
        ]

        if any(q_lower.startswith(starter) for starter in followup_starters):
            return True

        pronouns = ["that", "it", "them", "these", "those", "there", "the other", "this sector", "that sector"]
        words = q_lower.split()
        if any(p in words for p in pronouns) and len(words) <= 10:
            return True

        return False

    @staticmethod
    def formulate_contextual_opening(query: str, history: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        """
        Produces a conversational opening acknowledging prior discussion
        so the assistant sounds continuous rather than repetitive.
        """
        context = ConversationMemoryTracker.extract_prior_context(history)
        if not context["is_followup"]:
            return None

        last_q = context["last_user_query"]
        topics = context["topics"]
        quads = context["quadrants"]
        coref = ConversationMemoryTracker.resolve_spatial_coreference(query, history)

        if coref["resolved_sector"]:
            return f"Zeroing in on the **{coref['resolved_sector']}** sector discussed in our previous turn:"

        if "flood_inundation" in topics and any(w in query.lower() for w in ["safe", "dry", "road", "access", "where"]):
            return "Following up on our flood assessment of the low-lying sectors:"
        elif "vegetation" in topics and any(w in query.lower() for w in ["more", "crop", "health", "tree", "species", "chart", "distribution"]):
            return "Expanding on the vegetative canopy characteristics and spectral distribution analyzed earlier:"
        elif "urban_structures" in topics and any(w in query.lower() for w in ["road", "connectivity", "density", "more", "structure"]):
            return "Continuing our examination of the structural and urban layout:"
        elif last_q:
            clean_q = re.sub(r'^[Ww]hat (is|are|does) (the )?', '', last_q).rstrip('?')
            if len(clean_q) > 4:
                return f"Building upon our previous analysis regarding {clean_q}:"

        return "Continuing from our earlier observations across this satellite pass:"
