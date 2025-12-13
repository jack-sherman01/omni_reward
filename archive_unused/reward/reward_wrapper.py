import gymnasium
import numpy as np
from collections import deque

class OmniRewardWrapper(gymnasium.Wrapper):

    def __init__(
        self,
        env,
        captioner,
        potential_fn,
        tuner=None,
        task_desc="robot task",
        student_model=None,
        n_captions=3,
        smooth_window=5,
        cache_size=5000,
        confidence_threshold=0.25
    ):
        super().__init__(env)
        self.captioner = captioner
        self.potential_fn = potential_fn
        self.tuner = tuner
        self.task_desc = task_desc

        self.student = student_model
        self.n_captions = n_captions
        self.smooth_window = smooth_window
        self.confidence_threshold = confidence_threshold
        self.cache_size = cache_size

        self.potential_buffer = deque(maxlen=smooth_window)
        self.cache = {}
        self.prev_smoothed = None
        self.step_count = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        pot = self._compute_smoothed_potential(obs)
        self.prev_smoothed = pot
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        smoothed = self._compute_smoothed_potential(obs)
        reward = smoothed - self.prev_smoothed
        self.prev_smoothed = smoothed

        self.step_count += 1

        return obs, float(reward), done, info

    def _compute_smoothed_potential(self, obs):
        pot = self._compute_potential(obs)
        self.potential_buffer.append(pot)
        return float(np.mean(self.potential_buffer))


    def _compute_potential(self, obs):
        # Student model option
        if self.student is not None:
            return float(self.student.predict(obs["image"]))

        return self._compute_teacher_potential(obs)


    def _compute_teacher_potential(self, obs):
        image = obs["image"]

        key = hash(image.tobytes())
        if key in self.cache:
            return self.cache[key]

        captions = self.captioner.caption_multi(image, n=self.n_captions)
        captions = [
            c for c in captions
            if self._caption_confidence(c) >= self.confidence_threshold
        ]

        # Dynamic α and λ (or fixed fallback)
        if self.tuner:
            params = self.tuner.suggest(obs, self.step_count, self.task_desc)
            curr_alpha = params["alpha"]
            curr_lambda = params["lambda"]
        else:
            curr_alpha = 0.6
            curr_lambda = 0.7

        potentials = [
            self.potential_fn.compute(
                obs,
                c,
                alpha=curr_alpha,
                lambda_=curr_lambda
            )
            for c in captions
        ]

        pot = float(np.mean(potentials))

        if len(self.cache) < self.cache_size:
            self.cache[key] = pot

        return pot


    def _caption_confidence(self, caption):
        length = len(caption.split())
        if length < 3: return 0.1
        if length < 6: return 0.3
        return 0.8
