from omni_reward.reward.multimodal import UnifiedMultimodalPotential

def build_potential_fn(text_encoder, reward_cfg):
    return UnifiedMultimodalPotential(
        text_encoder=text_encoder,
        vision_goal_text=reward_cfg["goal_text"],
        baseline_text=reward_cfg["baseline_text"],
        tactile_goal_text=reward_cfg.get("tactile_goal_text"),
    )
