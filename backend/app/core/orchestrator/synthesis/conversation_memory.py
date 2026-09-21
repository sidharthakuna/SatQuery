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
    resolve anaphoric references (e.g. 'it', 'that area', 'the flood'),
    and maintain continuous conversational flow.
    """

    @staticmethod
    def extract_prior_context(history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extracts key entities, topics, and metrics discussed in prior dialogue turns.
        """
        if not history:
            return {
                "turns_count": 0,
                "topics": [],
                "last_user_query": None,
                "last_assistant_answer": None,
                "is_followup": False,
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

        return {
            "turns_count": turns_count,
            "topics": topics,
            "last_user_query": last_user_query,
            "last_assistant_answer": last_assistant_answer,
            "is_followup": turns_count >= 1,
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
            "explain further", "specifically", "what of", "what kind of",
        ]

        if any(q_lower.startswith(starter) for starter in followup_starters):
            return True

        pronouns = ["that", "it", "them", "these", "those", "there", "the other"]
        words = q_lower.split()
        if any(p in words for p in pronouns) and len(words) <= 7:
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

        if "flood_inundation" in topics and any(w in query.lower() for w in ["safe", "dry", "road", "access", "where"]):
            return "Following up on our flood assessment of the low-lying sectors:"
        elif "vegetation" in topics and any(w in query.lower() for w in ["more", "crop", "health", "tree", "species"]):
            return "Expanding on the vegetative canopy characteristics observed earlier:"
        elif "urban_structures" in topics and any(w in query.lower() for w in ["road", "connectivity", "density", "more"]):
            return "Continuing our examination of the settlement infrastructure:"
        elif last_q:
            clean_q = re.sub(r'^[Ww]hat (is|are|does) (the )?', '', last_q).rstrip('?')
            return f"Regarding your question in relation to {clean_q}:"

        return "Continuing from our earlier discussion on this scene:"
