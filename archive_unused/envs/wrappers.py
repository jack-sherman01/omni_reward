import gymnasium

class ObsWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env, goal_text, baseline_text):
        super().__init__(env)
        self.goal_text = goal_text
        self.baseline_text = baseline_text

    def observation(self, obs):
        return {
            "image": obs,
            "goal_text": self.goal_text,
            "baseline_text": self.baseline_text
        }
