# omni_reward

This repo is an initial prototype for exploring a generalizable reward function for robotic RL using semantic information derived from vision and tactile inputs.
The method converts images and tactile inputs into text captions, embeds them, and computes reward based on semantic similarity to goal and baseline descriptions. The reward signal combines two adaptive weights: alpha (α) for shaping long-horizon vs short-horizon progress, and lambda (λ) for balancing vision-based potential with tactile-based potential. Both parameters can be dynamically tuned by an LLM.

The current code in this repo is just the first working pseudocode structure to iterate on.

## Features

- A sandbox for testing reward functions built from vision-to-text and language embeddings.  
- A place to plug in simple RL agents (PPO, SAC, etc) and verify whether these semantic rewards behave better than raw image-based ones.  
- A modular layout so we can swap in better models, different inputs, and different RL algorithms  

## Folder structure

### `configs/`
Top-level YAML config files defining:
- which environment to run  
- which reward setup to use  
- training parameters  

### `scripts/`
- `run_omni_rl.py`  
Entry point that ties together configs, env, reward, and agent training.

### `omni_reward/` :

#### `vision/`
Converting images into text or embeddings.  
- `captioner.py`, wrapper for VLM captioning  
- `text_encoder.py`, wrapper for text embedding models  

#### `reward/`
Implements the actual reward mechanism:
- `multimodal.py`, computes unified semantic potential using α and λ  
- `reward_wrapper.py`, wraps an env and turns potential differences into rewards  
- `builders.py`, creates the potential function objects used by the wrapper  

#### `agents/`
Lightweight PPO and SAC implementations.  
PPO is more complete for now because it’s easier to prototype with.

#### `envs/`
Toy environments + wrappers for testing reward functions.  
- `toy_gridworld.py` is just a quick sanity-check environment.  
- `wrappers.py` has observation and reward wrappers for augmenting env outputs.

#### `llm/`
- `tuning.py`, helper for LLM-selected alpha (α) and lambda (λ) tuning, with optional fixed-value mode.

#### `utils/`
Small utilities (config loading, seeding)

---

## Installation

Create a virtual environment and install the repo in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

You may also run ```pip install -r requirements.txt``` .


## How to run

Right now, everything runs through:

```bash
python scripts/run_omni_rl.py --train-config configs/train/gridworld.yaml
```
