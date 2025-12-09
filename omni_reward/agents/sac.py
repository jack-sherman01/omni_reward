class SACAgent:
    def __init__(self, obs_shape, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        # TODO: implement actor + critic networks
        # or plug into SB3 for real training.
        pass

    def act(self, obs):
        # TODO: deterministic/ stochastic action selection
        return np.random.randint(0, action_dim)
