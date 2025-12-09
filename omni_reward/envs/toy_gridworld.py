import gymnasium
import numpy as np
from gymnasium import spaces

class ToyGridWorld(gymnasium.Env):
    # Sanity-check environment. 2D grid-world example. Quick debug/validation of RL loop before upgrading to robotic sim.
    # Observation is an RGB image of the state.

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size=16, goal=(10, 10), start=(1, 1), render_size=84):
        super().__init__()

        self.size = size
        self.goal = np.array(goal, dtype=np.int32)
        self.start = np.array(start, dtype=np.int32)

        self.agent_pos = self.start.copy()
        self.render_size = render_size

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, render_size, render_size),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)  # up, down, left, right

    def reset(self):
        self.agent_pos = self.start.copy()
        return self._get_observation()

    def step(self, action):
        if action == 0:   # up
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1: # down
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.size - 1)
        elif action == 2: # left
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3: # right
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.size - 1)
        # TODO: use reward? like OmniReward, baseline could be distance-based?
        done = np.array_equal(self.agent_pos, self.goal)
        return self._get_observation(), 0.0, done, {}

    def _get_observation(self):
        img = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)

        # Draw agent (blue)
        ax = int((self.agent_pos[0] / self.size) * self.render_size)
        ay = int((self.agent_pos[1] / self.size) * self.render_size)
        img[ay:ay+4, ax:ax+4] = [50, 100, 255]

        # Draw goal (red)
        gx = int((self.goal[0] / self.size) * self.render_size)
        gy = int((self.goal[1] / self.size) * self.render_size)
        img[gy:gy+4, gx:gx+4] = [255, 80, 80]

        return np.transpose(img, (2, 0, 1))  # channel-first
