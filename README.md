# Omni Reward

This repository provides a lightweight, importable Omni Reward function that turns camera observations into a semantic reward signal. 

- `scripts/demo_example.py` gives an example on how to import and use the reward function in any benchmark.

## Install Dependencies

Create a virtual environment if you haven't already:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the repo and its dependencies in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

## API keys

Set the environment variables required by your VLM backend (for example, `export OPENAI_API_KEY=sk...` for GPT-4o).

## How to run a quick sanity check within this repository:

```bash
python scripts/demo_example.py --goal "Move the red Cheez-Its box directly on top of the red mug." ./test_images/f0.png ./test_images/f1.png 
```

## How to import into a benchmark (follow the example in `scripts/demo_example.py`): 

Step 1. Copy over the "omni_reward/omni_reward" folder from inside this repo.

Step 2. Make sure the benchmark's virtual environment contains the same or compatible dependency installations (as shown in requirements.txt)

Step 3. Import the omni_reward_interface using an example like this:

```python
from omni_reward.reward.use import omni_reward_interface
from PIL import Image
import numpy as np

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

goal_text = "Move the red Cheez-Its box directly on top of the red mug."

# Create the reward function
reward_fn = omni_reward_interface(
    goal = goal_text
    # pass in other optional input parameters here, such as the following:
    # vlm_type: str = "openai",
    # vision_model: str = "gpt-4o",
    # embedding_model: str = "all-MiniLM-L6-v2",
    # alpha: float = 0.6,
)

# TODO: get images from your env in your benchmark
image = load_image(image_path)
reward = reward_fn.get_current_reward(image)
```

The first caption observed in an episode becomes the baseline description. On
later steps the reward is the potential difference between the current caption
and that baseline relative to the goal description.

## if you use subgoals:
you just need to create a new wraped env below the original env (env = gym.make(ENV_ID)):

```python
wrapped_env = OmniRewardWrapper(env, captioner, text_encoder, goal_text, use_subgoals=True)

``` 
then you should use the wrapped_env as normal gym env. The reward will be calculated based on subgoals: for example,

```python
obs, reward, done, info = wrapped_env.step(action)
```

for your reference, here below is an example useage of the wrapped env:

```python
import gym

from omni_reward.examples.env_wrapper import OmniRewardWrapper
from omni_reward.vision.captioner import VLMCaptioner
# NOTE: replace TextEncoderImpl with the actual encoder class in omni_reward/vision/text_encoder.py
from omni_reward.vision.text_encoder import TextEncoderImpl  # TODO: check file for the right class

ENV_ID = "CartPole-v1"  # or your own env id

def main():
    # 1) original env
    env = gym.make(ENV_ID)

    # 2) build captioner and text encoder (same config as you would use for omni_reward_interface)
    captioner = VLMCaptioner(
        vlm_type="openai",          # or your backend
        caption_template="structured_v1",
    )
    text_encoder = TextEncoderImpl(
        model_name="all-MiniLM-L6-v2",  # for example; check your implementation
    )

    goal_text = "Move the red Cheez-Its box directly on top of the red mug."

    # 3) wrap env with subgoals enabled
    wrapped_env = OmniRewardWrapper(
        env=env,
        captioner=captioner,
        text_encoder=text_encoder,
        goal_text=goal_text,
        use_subgoals=True,  # this flag turns on subgoal decomposition & subgoal-based reward
    )

    # 4) use wrapped_env as a normal Gym env in your rollout
    obs = wrapped_env.reset()
    done = False
    while not done:
        action = wrapped_env.action_space.sample()
        obs, reward, done, info = wrapped_env.step(action)

        # optional: inspect subgoal info
        if "omni_reward_info" in info:
            rinfo = info["omni_reward_info"]
            print(
                f"reward={reward:.3f}, "
                f"subgoal={rinfo['current_subgoal']!r}, "
                f"progress={rinfo['progress']:.2f}"
            )

if __name__ == "__main__":
    main()
```


## Implementing your own VLM provider

- To swap VLM providers, pass the matching `vlm_type` and/or `vision_model` as function arguments in reward_fn = omni_reward_interface(). 
- If your VLM provider is not available, register them through `omni_reward/VLM_utils/VLM_api.py`.
- To change caption prompts, add/edit templates in `omni_reward/VLM_utils/templates.py` and reference them via the `caption_template` argument.

--

## Repo file details

- `omni_reward/vision/captioner.py` wraps any configured VLM and produces captions.
- `omni_reward/vision/text_encoder.py` wraps SentenceTransformers for embeddings.
- `omni_reward/reward/multimodal.py` implements the potential function.
- `omni_reward/reward/interface.py` exposes a small stateful class that returns shaped rewards.
- `omni_reward/reward/use.py` provides the omni_reward_interface ready to import.
- `omni_reward/VLM_utils/*` holds the VLM factory plus caption templates.

## Legacy code

Legacy code is preserved under `archive_unused` for reference. They are no longer used as part of the lightweight workflow described above.
