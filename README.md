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
import os
import gym

from omni_reward.examples.env_wrapper import OmniRewardWrapper
from omni_reward.vision.captioner import VLMCaptioner
from omni_reward.vision.text_encoder import TextEncoder

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create base environment
env = gym.make("YourEnv-v1", render_mode="rgb_array")

# Initialize captioner with detailed captions
captioner = VLMCaptioner(
    vlm_type="openai",
    caption_template="detailed_state",  # Options: "simple", "structured_v1", "goal_oriented", "detailed_state"
    max_caption_tokens=1024,  # Allow longer, more detailed captions
)
text_encoder = TextEncoder()

goal_text = "Push the puck to the red goal position."

# Wrap with OmniRewardWrapper
wrapped_env = OmniRewardWrapper(
    env=env,
    captioner=captioner,
    text_encoder=text_encoder,
    goal_text=goal_text,
    use_subgoals=True,  # Enable subgoal decomposition
    openai_api_key=os.getenv("OPENAI_API_KEY"),  # Optional: pass API key directly
    # Image saving options
    save_images=True,
    image_save_dir="./images/my_task",
    task_name="my_task",
    # Camera settings (for MuJoCo/MetaWorld environments)
    camera_name="corner2",  # Options: "corner", "corner2", "corner3", "topview", "behindGripper", "frontview"
    # VLM call frequency (to reduce API costs)
    vlm_call_interval=10,  # Call VLM every 10 steps instead of every step
    use_interpolated_reward=True,  # Use cached reward between VLM calls
)
```

### Using the Wrapped Environment

```python
obs, info = wrapped_env.reset()
done = False

while not done:
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    
    # Check if VLM was called this step
    if info.get('vlm_called', False):
        print("VLM was called this step")
    
    # Inspect subgoal info
    if "omni_reward_info" in info:
        rinfo = info["omni_reward_info"]
        print(
            f"reward={reward:.3f}, "
            f"subgoal_idx={rinfo.get('current_subgoal_index', 'N/A')}, "
            f"subgoal={rinfo.get('current_subgoal', 'N/A')!r}, "
            f"completed={rinfo.get('subgoal_completed', False)}, "
            f"all_done={rinfo.get('all_completed', False)}"
        )
    
    done = terminated or truncated

wrapped_env.close()
```

## OmniRewardWrapper Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | gym.Env | required | Base environment to wrap |
| `captioner` | VLMCaptioner | required | Captioner for generating image descriptions |
| `text_encoder` | TextEncoder | required | Text encoder for semantic embeddings |
| `goal_text` | str | required | High-level goal description |
| `use_subgoals` | bool | `True` | Enable automatic subgoal decomposition |
| `openai_api_key` | str | `None` | OpenAI API key (or set via environment variable) |
| `save_images` | bool | `True` | Save rendered images to disk |
| `image_save_dir` | str | `None` | Directory for saved images (default: `./images/{task_name}`) |
| `task_name` | str | `"default_task"` | Task name for organizing saved images |
| `camera_name` | str | `"corner2"` | Camera view for rendering (MuJoCo environments) |
| `vlm_call_interval` | int | `10` | Call VLM every N steps (reduces API costs) |
| `use_interpolated_reward` | bool | `True` | Use cached reward between VLM calls |

## Caption Templates

The captioner supports multiple templates for different levels of detail:

| Template | Description | Use Case |
|----------|-------------|----------|
| `simple` | Brief one-line description | Fast, low-cost captioning |
| `structured_v1` | Structured multi-section description | General purpose |
| `goal_oriented` | Focuses on goal progress | Goal-conditioned tasks |
| `detailed_state` | Maximum detail state description | High-fidelity reward computation |

Example with detailed captions:

```python
captioner = VLMCaptioner(
    caption_template="detailed_state",
    max_caption_tokens=1024,
)
```

## Camera Views (MuJoCo/MetaWorld)

For MuJoCo-based environments like MetaWorld, you can specify different camera views:

| Camera | Description |
|--------|-------------|
| `corner` | Default corner view (rear-right) |
| `corner2` | Side view (left side) |
| `corner3` | Side view (right side) |
| `topview` | Top-down view |
| `behindGripper` | Behind the gripper |
| `frontview` | Custom front view (if configured) |

## Reducing VLM API Costs

Calling VLM at every step can be expensive. Use `vlm_call_interval` to reduce costs:

```python
wrapped_env = OmniRewardWrapper(
    # ... other params ...
    vlm_call_interval=10,  # Call VLM every 10 steps
    use_interpolated_reward=True,  # Use last reward between calls
)
```

Reward interpolation strategies (configurable in `env_wrapper.py`):
- **Last reward**: Use the most recent VLM reward (default)
- **Decay**: Reward decays over time between VLM calls
- **Zero**: Return 0 between VLM calls (sparse reward)

## Implementing your own VLM provider

- To swap VLM providers, pass the matching `vlm_type` and/or `vision_model` as function arguments in reward_fn = omni_reward_interface(). 
- If your VLM provider is not available, register them through `omni_reward/VLM_utils/VLM_api.py`.
- To change caption prompts, add/edit templates in `omni_reward/VLM_utils/templates.py` and reference them via the `caption_template` argument.

---

## Repo file details

- `examples/env_wrapper.py` provides the `OmniRewardWrapper` for easy environment integration.
- `omni_reward/vision/captioner.py` wraps any configured VLM and produces captions.
- `omni_reward/vision/text_encoder.py` wraps SentenceTransformers for embeddings.
- `omni_reward/reward/multimodal.py` implements the potential function.
- `omni_reward/reward/interface.py` exposes a small stateful class that returns shaped rewards.
- `omni_reward/reward/use.py` provides the omni_reward_interface ready to import.
- `omni_reward/VLM_utils/templates.py` contains caption prompt templates.
- `omni_reward/VLM_utils/VLM_api.py` holds the VLM factory.

## Legacy code

Legacy code is preserved under `archive_unused` for reference. They are no longer used as part of the lightweight workflow described above.
