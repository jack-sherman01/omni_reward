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
