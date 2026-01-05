"""Captioner implementations used by the reward interface."""
from __future__ import annotations

from typing import Any, List, Optional

from omni_reward.VLM_utils.VLM_api import get_vlm
from omni_reward.VLM_utils.VLM_local import VLMBase

class VLMCaptioner:
    """Captioner that delegates to any configured VLM from VLM_utils."""

    def __init__(
        self,
        vlm_type: str = "openai",
        caption_template: str = "structured_v1",
        goal_context: Optional[str] = None,
        vlm: Optional[VLMBase] = None,
        **vlm_kwargs,
    ):
        self.vlm_type = vlm_type
        self.caption_template = caption_template
        self.goal_context = goal_context
        self.vlm = vlm or get_vlm(vlm_type, **vlm_kwargs)

    def caption(self, image: Any, goal_text: Optional[str] = None) -> str:
        goal = goal_text or self.goal_context
        caption_text = self.vlm.generate_caption(image, template=self.caption_template, goal=goal)
        print("[VLMCaptioner] caption:", caption_text)
        return caption_text

    def caption_multi(self, image: Any, n: int = 1, goal_text: Optional[str] = None):
        return [self.caption(image, goal_text=goal_text) for _ in range(n)]

    def set_goal_context(self, goal_text: Optional[str]) -> None:
        self.goal_context = goal_text

    def enrich_goal(self, goal_text: Optional[str]) -> Optional[str]:
        """Hook for goal text enrichment for accelerating the learning. so that it is more informative for embedding and similarity comparison.
        This function uses the VLM itself to generate additional context about the goal."""
        
        if not goal_text:
            return goal_text
            
        # Define questions to extract rich information about the goal
        # TODO: discuss these questions with team and refine
        enrichment_questions = [
            f"what is the goal state of robot in the end to achieve: '{goal_text}'?",
            f"What are the key visual elements or objects involved in the goal: '{goal_text}'?",
            f"What specific actions or movements are required to achieve: '{goal_text}'?",
            f"What would be the success criteria or indicators for completing: '{goal_text}'?",
            f"What are potential intermediate steps or milestones for: '{goal_text}'?",
            f"What spatial relationships or positions are important for: '{goal_text}'?"
        ]
        
        # Collect enriched information from VLM
        enriched_parts = [f"Original Goal: {goal_text}"]
        
        for question in enrichment_questions:
            try:
                # Use VLM to answer each question
                response = self.vlm.generate_text(question)
                if response and response.strip():
                    enriched_parts.append(response.strip())
            except Exception as e:
                print(f"[VLMCaptioner] Warning: Failed to enrich goal with question '{question}': {e}")
                continue
        
        # Combine all information into enriched goal text
        # TODO : using a better formatting strategy by LLM instead of simple joining
        enriched_goal = " | ".join(enriched_parts)
        
        print(f"[VLMCaptioner] Enriched goal from '{goal_text}' to: {enriched_goal}")
        
        return enriched_goal
