import gym
from omni_reward.reward.interface import OmniRewardInterface

class OmniRewardWrapper(gym.Wrapper):
    """Gym environment wrapper that replaces rewards with OmniReward."""
    
    def __init__(self, env, captioner, text_encoder, goal_text, use_subgoals=True):
        super().__init__(env)
        self.reward_interface = OmniRewardInterface(captioner, text_encoder)
        self.goal_text = goal_text
        self.use_subgoals = use_subgoals
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        if self.use_subgoals:
            # Start the episode using decomposed subgoals
            self.reward_interface.start_episode_with_subgoals(
                self.goal_text, 
                initial_image=obs,
                auto_decompose=True
            )
        else:
            # Start the episode with a single final goal
            self.reward_interface.start_episode(self.goal_text, initial_image=obs)
        
        return obs
    
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        
        if self.use_subgoals:
            # Compute reward using subgoal-based progression
            result = self.reward_interface.compute_reward_with_subgoals(
                scene_image=obs,
                auto_advance=True,
                completion_bonus=10.0  # Extra reward for completing a subgoal
            )
            
            reward = result['reward']
            info['omni_reward_info'] = result
            
            # Terminate the episode when all subgoals are completed
            if result.get('all_completed', False):
                done = True
        else:
            # Compute reward directly from the final goal
            reward = self.reward_interface.compute_reward(obs)
            
        return obs, reward, done, info