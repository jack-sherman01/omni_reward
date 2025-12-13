from omni_reward.reward.interface import OmniRewardInterface
from omni_reward.vision.captioner import VLMCaptioner
from omni_reward.vision.text_encoder import TextEncoder

import argparse
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image

def omni_reward_interface(
    *,
    goal: str,
    vlm_type: str = "openai",
    vision_model: str = "gpt-4o",
    caption_template: str = "structured_v1",
    embedding_model: str = "all-MiniLM-L6-v2",
    encoder_device: str = "cpu",
    alpha: float = 0.6,
    lambda_: float = 1.0,
) -> OmniRewardInterface:
    
    captioner = VLMCaptioner(
        vlm_type=vlm_type,
        caption_template=caption_template,
        vision_model=vision_model
    )

    encoder = TextEncoder(model_name=embedding_model, device=encoder_device)

    reward_fn = OmniRewardInterface(
        captioner=captioner,
        text_encoder=encoder,
        alpha=alpha,
        lambda_=lambda_,
    )
    reward_fn.start_episode(goal_text=goal)

    return reward_fn