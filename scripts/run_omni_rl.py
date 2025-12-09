import argparse
import gymnasium
import numpy as np

from omni_reward.utils.config import load_config
from omni_reward.utils.seed import set_seed

from omni_reward.envs.toy_gridworld import ToyGridWorld
from omni_reward.envs.wrappers import ObsWrapper

from omni_reward.vision.captioner import SimpleCaptioner
from omni_reward.vision.text_encoder import TextEncoder

from omni_reward.reward.builders import build_potential_fn
from omni_reward.reward.reward_wrapper import OmniRewardWrapper
from omni_reward.llm.tuning import AlphaLambdaTuner, FixedAlphaLambda

from omni_reward.agents.ppo import PPOAgent


def make_env(env_cfg):
    return ToyGridWorld(
        size=env_cfg["size"],
        goal=env_cfg["goal_position"],
        start=env_cfg["start_position"],
        render_size=env_cfg["render"]["resolution"][0]
    )


def main(train_cfg_path):
    cfg = load_config(train_cfg_path)
    set_seed(cfg["seed"])

    env_cfg = load_config(cfg["env_config"])
    reward_cfg = load_config(cfg["reward_config"])

    env = make_env(env_cfg)
    env = ObsWrapper(env,
                        goal_text=reward_cfg["goal_text"],
                        baseline_text=reward_cfg["baseline_text"])

    captioner = SimpleCaptioner()
    encoder = TextEncoder()

    potential_fn = build_potential_fn(encoder, reward_cfg)

    # Build tuner
    if reward_cfg["use_llm_tuner"]:
        tuner = AlphaLambdaTuner(
            alpha_default=reward_cfg["alpha_default"],
            lambda_default=reward_cfg["lambda_default"]
        )
    else:
        tuner = FixedAlphaLambda(
            alpha=reward_cfg["alpha_default"],
            lambda_=reward_cfg["lambda_default"]
        )

    # Wrap environment with reward
    env = OmniRewardWrapper(
        env=env,
        captioner=captioner,
        potential_fn=potential_fn,
        tuner=tuner,
        task_desc=reward_cfg.get("task_description", "robot task")
    )

    obs = env.reset()
    obs_shape = obs["image"].shape
    action_dim = env.action_space.n

    agent = PPOAgent(obs_shape, action_dim)

    for step in range(cfg["total_steps"]):
        action = agent.act(obs)
        next_obs, rew, done, info = env.step(action)

        # TODO: store transition & update agent

        obs = next_obs
        if done:
            obs = env.reset()

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=str, required=True)
    args = parser.parse_args()
    main(args.train_config)
