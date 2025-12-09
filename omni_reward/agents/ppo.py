import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

# basic ppo example with random parameters just to compile

class PolicyNet(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        C, H, W = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * ((H//4-1) * (W//4-1)), 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, action_dim)
        self.v = nn.Linear(256, 1)

    def forward(self, obs_image):
        x = self.cnn(obs_image / 255.0)
        x = self.fc(x)
        return self.pi(x), self.v(x)


class PPOAgent:
    def __init__(self, obs_shape, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.model = PolicyNet(obs_shape, action_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def act(self, obs):
        obs_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0)
        logits, _ = self.model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item())
