"""Demonstration script for the OmniRewardInterface using VLM captioning."""

import argparse
from pathlib import Path

# Required imports:
from omni_reward.reward.use import omni_reward_interface
from PIL import Image
import numpy as np

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

# just for sanity checking and testing in this repository:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Sequence of image files representing timesteps")
    parser.add_argument("--goal", required=True, help="Textual goal for the scene")
    # parser.add_argument("--vlm-type", default="openai", help="Which VLM implementation to use (see VLM_utils)")
    # parser.add_argument("--vision-model", default="gpt-4o", help="Provider-specific vision model identifier")
    # parser.add_argument("--alpha", type=float, default=0.6, help="Alpha used inside the potential function")
    # parser.add_argument("--lambda_", type=float, default=1.0, dest="lambda_", help="Lambda used to blend tactile and vision modalities (not implemented yet)")
    # parser.add_argument("--caption-template", default="structured_v1", help="Caption template key defined in VLM_utils")
    # parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    # parser.add_argument("--encoder-device", default="cpu", help="Device for the text encoder")
    return parser.parse_args()

# example
def main():
    args = parse_args()
    reward_fn = omni_reward_interface(
        goal=args.goal
        # pass in other optional input parameters here.
    )

    # TODO: get images from your env in your benchmark
    # image = 
    # reward = reward_fn.get_current_reward(image)
    for image_path in args.images:
        image = load_image(image_path)
        reward = reward_fn.get_current_reward(image)
        print(f"t={reward_fn.timestep:02d} path={image_path} reward={reward:.5f}")

if __name__ == "__main__":
    main()
