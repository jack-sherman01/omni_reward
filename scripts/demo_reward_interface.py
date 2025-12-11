#!/usr/bin/env python3
"""Demonstration script for the OmniRewardInterface using OpenAI's VLM."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image

from omni_reward.reward.interface import OmniRewardInterface
from omni_reward.vision.captioner import VLMCaptioner
from omni_reward.vision.text_encoder import TextEncoder


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Sequence of image files representing timesteps")
    parser.add_argument("--goal", required=True, help="Textual goal for the scene")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha used inside the potential function")
    parser.add_argument("--caption-template", default="structured_v1", help="Caption template key defined in VLM_utils")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--encoder-device", default="cpu", help="Device for the text encoder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    captioner = VLMCaptioner(
        vlm_type="openai", # using OpenAI for this test
        caption_template=args.caption_template,
        vision_model="gpt-4o",
    )

    encoder = TextEncoder(model_name=args.embedding_model, device=args.encoder_device)

    reward_fn = OmniRewardInterface(
        captioner=captioner,
        text_encoder=encoder,
        alpha=args.alpha,
        lambda_=1.0,
    )

    reward_fn.start_episode(goal_text=args.goal)

    for image_path in args.images:
        image = load_image(image_path)
        reward = reward_fn.step(image)
        print(
            f"t={reward_fn.timestep:02d} path={image_path} reward={reward:.5f}"
        )

if __name__ == "__main__":
    main()
