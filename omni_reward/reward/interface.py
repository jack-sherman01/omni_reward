"""Simple importable interface for computing semantic rewards.

The interface hides the potential-based reward shaping logic behind a stateful
class so RL agents can call it each timestep with just the rendered image, the
current timestep, and the textual goal description.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, Any, Dict

import numpy as np

from omni_reward.reward.multimodal import UnifiedMultimodalPotential


class Captioner(Protocol):
    """Minimal protocol required for captioning models. (delete later)"""

    def caption(self, image: Any, goal_text: Optional[str] = None) -> str:  # pragma: no cover
        ...


class TextEncoder(Protocol):
    """Minimal protocol required for text encoders used by the potential. (delete later)"""

    def encode_one(self, text: str) -> np.ndarray:  # pragma: no cover - protocol definition
        ...

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - protocol definition
        ...


@dataclass
class OmniRewardInterface:
    """High-level wrapper for the semantic potential reward.

    Parameters
    ----------
    captioner:
        Object capable of turning an image tensor/array into a text caption.
    text_encoder:
        Encoder used by :class:`UnifiedMultimodalPotential`.
    alpha:
        Weight applied inside the vision potential (default 0.6 as per design).
    lambda_:
        Blend between vision and tactile potentials. Set to 1.0 to ignore tactile inputs.
    """

    captioner: Captioner
    text_encoder: TextEncoder
    alpha: float = 0.6
    lambda_: float = 1.0

    goal_text: Optional[str] = field(default=None, init=False)
    baseline_text: Optional[str] = field(default=None, init=False)
    _potential: Optional[UnifiedMultimodalPotential] = field(default=None, init=False, repr=False)
    _prev_potential: Optional[float] = field(default=None, init=False, repr=False)
    _prev_timestep: Optional[int] = field(default=None, init=False, repr=False)

    def reset_episode(self) -> None:
        """Clears baseline/potential information between episodes."""
        self.goal_text = None
        self.baseline_text = None
        self._potential = None
        self._prev_potential = None
        self._prev_timestep = None

    def _ensure_goal(self, goal_text: str) -> None:
        if self.goal_text is None:
            self.goal_text = goal_text
        elif goal_text != self.goal_text:
            raise ValueError(
                "Goal text changed during an episode. Call reset_episode() before switching goals."
            )

    def _compute_potential(self, scene_image: Any, caption: str) -> float:
        if self._potential is None:
            if self.goal_text is None or self.baseline_text is None:
                raise RuntimeError("Potential requested before goal/baseline were initialized.")
            self._potential = UnifiedMultimodalPotential(
                text_encoder=self.text_encoder,
                vision_goal_text=self.goal_text,
                baseline_text=self.baseline_text,
            )

        obs: Dict[str, Any] = {"image": scene_image, "tactile": None}
        potential = self._potential.compute(
            obs=obs,
            caption=caption,
            alpha=self.alpha,
            lambda_=self.lambda_,
        )
        return float(potential)

    def compute_reward(self, scene_image: Any, timestep: int, goal_text: str) -> float:
        """Return shaped reward for the provided scene observation.

        Parameters
        ----------
        scene_image: Any
            2D (HxW or CxHxW) image tensor/array used by the captioner.
        timestep: int
            Current environment timestep. When it resets to zero we treat it
            as the start of a new episode and clear the stored baseline.
        goal_text: str
            Natural language description of the desired outcome.
        """
        if self._prev_timestep is not None and timestep <= self._prev_timestep:
            # Episode restarted (either at 0 or manual reset)
            self.reset_episode()

        self._ensure_goal(goal_text)

        caption = self.captioner.caption(scene_image, goal_text=goal_text)

        if self.baseline_text is None:
            # Use the very first caption as the baseline reference.
            self.baseline_text = caption
            potential = self._compute_potential(scene_image, caption)
            reward = 0.0
        else:
            potential = self._compute_potential(scene_image, caption)
            if self._prev_potential is None:
                reward = 0.0
            else:
                reward = potential - self._prev_potential

        self._prev_potential = potential
        self._prev_timestep = timestep
        return reward

    __call__ = compute_reward
