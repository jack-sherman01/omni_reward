"""Captioner implementations used by the reward interface."""
from __future__ import annotations

from typing import Any, List, Optional
import hashlib
import json

from omni_reward.VLM_utils.VLM_api import get_vlm
from omni_reward.VLM_utils.VLM_local import VLMBase

class VLMCaptioner:
    """Captioner that delegates to any configured VLM from VLM_utils."""

    def __init__(
        self,
        vlm_type: str = "openai",
        caption_template: str = "detailed_state",  # Changed default to detailed_state
        goal_context: Optional[str] = None,
        vlm: Optional[VLMBase] = None,
        max_caption_tokens: int = 1024,  # New parameter for longer captions
        **vlm_kwargs,
    ):
        self.vlm_type = vlm_type
        self.caption_template = caption_template
        self.goal_context = goal_context
        self.vlm = vlm or get_vlm(vlm_type, **vlm_kwargs)
        self.max_caption_tokens = max_caption_tokens

        # In-memory cache for expensive goal enrichment results
        self._enrichment_cache = {}

    def caption(
        self, 
        image: Any, 
        goal_text: Optional[str] = None,
        template: Optional[str] = None,
        detailed: bool = True,  # New flag for detailed captions
    ) -> str:
        """Generate a caption for an image.
        
        Args:
            image: Image to caption
            goal_text: Optional goal context
            template: Override template (uses self.caption_template if None)
            detailed: If True, use detailed template; if False, use simple template
        """
        goal = goal_text or self.goal_context
        
        # Select template based on detailed flag
        if template is not None:
            use_template = template
        elif detailed:
            use_template = "detailed_state" if goal is None else "goal_oriented"
        else:
            use_template = self.caption_template
        
        caption_text = self.vlm.generate_caption(
            image, 
            template=use_template, 
            goal=goal,
            max_tokens=self.max_caption_tokens,
        )
        print(f"[VLMCaptioner] caption ({use_template}):\n{caption_text}")
        return caption_text

    def caption_detailed(self, image: Any, goal_text: Optional[str] = None) -> str:
        """Generate an extremely detailed caption for state representation."""
        return self.caption(image, goal_text=goal_text, template="detailed_state", detailed=True)
    
    def caption_for_goal(self, image: Any, goal_text: str) -> str:
        """Generate a goal-oriented caption focusing on progress toward the goal."""
        return self.caption(image, goal_text=goal_text, template="goal_oriented", detailed=True)

    def caption_multi(self, image: Any, n: int = 1, goal_text: Optional[str] = None):
        return [self.caption(image, goal_text=goal_text) for _ in range(n)]

    def set_goal_context(self, goal_text: Optional[str]) -> None:
        self.goal_context = goal_text

    def enrich_state(self, state_description: Optional[str], initial_image: Optional[Any] = None) -> Optional[str]:
        """Hook for state description enrichment for accelerating the learning. so that it is more informative for embedding and similarity comparison.
        This function uses the VLM itself to generate additional context about the state."""
        
        if not state_description:
            return state_description
            
        # Define questions to extract rich information about the state
        enrichment_questions = [
            f"What are the key visual elements or objects present in the state: '{state_description} and {initial_image}'?",
            f"Are there any notable spatial relationships or arrangements of objects in the state: '{state_description} and {initial_image}'?",
            f"what is the current state of robot and environment in the state: '{state_description} and {initial_image}'?, like positions, orientations, and interactions. or interactions between robot and objects.",
            f"Are there any potential obstacles or challenges visible in the state: '{state_description} and {initial_image}'?",
        ]   
        # Collect enriched information from VLM
        enriched_parts = [f"Original State: {state_description}"]
        for question in enrichment_questions:
            try:
                # Use VLM to answer each question
                response = self.vlm.generate_caption(initial_image, template=question)
                if response and response.strip():
                    enriched_parts.append(response.strip())
            except Exception as e:
                print(f"[VLMCaptioner] Warning: Failed to enrich state with question '{question}': {e}")
                continue
        # Combine all information into enriched state description
        # TODO : using a better formatting strategy by LLM instead of simple joining. : NOTE: done in LLM_utils.py
        enriched_state = " | ".join(enriched_parts)
        return enriched_state

    def enrich_goal(self, goal_text: Optional[str], goal_image: Optional[Any] = None) -> Optional[str]:
        """Hook for goal text enrichment for accelerating the learning. so that it is more informative for embedding and similarity comparison.
        This function uses the VLM itself to generate additional context about the goal."""
        
        if not goal_text:
            return goal_text

        # In-memory cache for expensive goal enrichment results
        # Cache key: stable hash of goal text + (presence/absence) of goal image
        # Note, avoid hashing goal_image contents here (could be large)
        cache_key_obj = {
            "goal_text": goal_text,
            "has_goal_image": goal_image is not None,
            "caption_template": self.caption_template,
            "vlm_type": self.vlm_type,
        }
        cache_key = hashlib.sha256(json.dumps(cache_key_obj, sort_keys=True).encode("utf-8")).hexdigest()

        cached = self._enrichment_cache.get(cache_key, None)
        if cached is not None:
            print(f"[VLMCaptioner] Using cached enriched goal for key={cache_key}")
            return cached
            
        # Define questions to extract rich information about the goal
        # TODO: discuss these questions with team and refine
        enrichment_questions = [
            f"what is the goal state of robot and environment in the end (final state, described with a detailed description): '{goal_text}'?",
            f"would the goal: '{goal_text} and {goal_image}' involve specific interaction or contact with objects or environment?",
            f"What are the key visual elements or objects involved in the goal: '{goal_text} and {goal_image}'?",
            # f"What specific actions or movements are required to achieve: '{goal_text}'?",
            f"What would be the success criteria or indicators for completing: '{goal_text} and {goal_image}'?",
            # f"What are potential intermediate steps or milestones for: '{goal_text}'?",
            f"What spatial relationships or positions are important for: '{goal_text} and {goal_image}'?"
        ]
        
        # Collect enriched information from VLM
        enriched_parts = [f"Original Goal: {goal_text}"]
        
        for question in enrichment_questions:
            try:
                # Use VLM to answer each question
                # TODO: consider using a more advanced LLM for better enrichment
                response = self.vlm.generate_caption(goal_image, template=question, goal=goal_text)
                if response and response.strip():
                    enriched_parts.append(response.strip())
            except Exception as e:
                print(f"[VLMCaptioner] Warning: Failed to enrich goal with question '{question}': {e}")
                continue
        
        # Combine all information into enriched goal text
        # Now uses JSON formatting, better for LLM instead of | joining.
        enriched_goal_obj = {
            "original_goal": goal_text,
            "enrichment": [
                {"question": q, "answer": a}
                for q, a in zip(enrichment_questions, enriched_parts[1:])
            ],
        }
        enriched_goal = json.dumps(enriched_goal_obj, ensure_ascii=False, sort_keys=True)

        # Update cache
        self._enrichment_cache[cache_key] = enriched_goal

        print(f"[VLMCaptioner] Enriched goal from '{goal_text}' to: {enriched_goal}")
        
        return enriched_goal
