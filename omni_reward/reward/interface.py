"""Simple importable interface for computing semantic rewards.

The interface hides the potential-based reward shaping logic behind a stateful
class so RL agents can call it each timestep with just the rendered image, the
current timestep, and the textual goal description.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, Any, Dict, List

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
class StepRecord:
    """Container for per-timestep bookkeeping."""

    timestep: int
    caption: str
    potential: float
    reward: float


@dataclass
class EpisodeState:
    """Mutable state tracked across RL episodes."""

    timestep: int = -1
    goal_text: Optional[str] = None
    baseline_text: Optional[str] = None
    prev_potential: Optional[float] = None
    history: List[StepRecord] = field(default_factory=list)


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

    _potential: Optional[UnifiedMultimodalPotential] = field(default=None, init=False, repr=False)
    _state: EpisodeState = field(default_factory=EpisodeState, init=False, repr=False)

    @property
    def goal_text(self) -> Optional[str]:  # pragma: no cover - trivial accessor
        return self._state.goal_text

    @property
    def baseline_text(self) -> Optional[str]:  # pragma: no cover - trivial accessor
        return self._state.baseline_text

    @property
    def prev_potential(self) -> Optional[float]:  # pragma: no cover - trivial accessor
        return self._state.prev_potential

    @property
    def timestep(self) -> int:  # pragma: no cover - trivial accessor
        return self._state.timestep

    @property
    def history(self) -> List[StepRecord]:  # pragma: no cover - trivial accessor
        return list(self._state.history)

    def start_episode(
        self,
        goal_text: str,
        baseline_image: Optional[Any] = None,
        baseline_caption: Optional[str] = None,
    ) -> None:
        """Reset the interface and optionally prime the baseline.

        Parameters
        ----------
        goal_text:
            Text goal that will remain fixed for the episode.
        baseline_image:
            Optional first observation used to bootstrap the baseline potential.
        baseline_caption:
            Skip re-captioning when the baseline caption is already known.
        """

        self.reset_episode()
        self._state.goal_text = goal_text

        if baseline_image is None and baseline_caption is None:
            return

        caption = baseline_caption
        if caption is None:
            caption = self.captioner.caption(baseline_image, goal_text=goal_text)

        self._state.baseline_text = caption

        if baseline_image is None:
            # Without an image we cannot compute the initial potential yet.
            return

        potential = self._compute_potential(baseline_image, caption)
        self._state.prev_potential = potential
        self._state.timestep = 0
        self._state.history.append(
            StepRecord(timestep=0, caption=caption, potential=potential, reward=0.0)
        )

    def reset_episode(self) -> None:
        """Clears baseline/potential information between episodes."""
        self._potential = None
        self._state = EpisodeState()

    def _ensure_goal(self, goal_text: str) -> None:
        if self.goal_text is None:
            self._state.goal_text = goal_text
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

    def compute_reward(
        self,
        scene_image: Any,
        timestep: Optional[int] = None,
        goal_text: Optional[str] = None,
    ) -> float:
        """Return shaped reward for the provided scene observation.

        Parameters
        ----------
        scene_image: Any
            2D (HxW or CxHxW) image tensor/array used by the captioner.
        timestep: Optional[int]
            Current environment timestep. When omitted the interface tracks it
            internally and simply assumes the next sequential step.
        goal_text: Optional[str]
            Natural language description of the desired outcome. Provide it once
            (either via :meth:`start_episode` or the first :meth:`step`) and it
            will remain locked in for the episode.
        """
        if timestep is None:
            timestep = 0 if self._state.timestep < 0 else self._state.timestep + 1
        elif self._state.timestep >= 0 and timestep <= self._state.timestep:
            # Episode restarted (either at 0 or manual reset). Preserve caller goal if provided.
            cached_goal = goal_text or self.goal_text
            self.reset_episode()
            if cached_goal is not None:
                self._state.goal_text = cached_goal

        if goal_text is not None:
            self._ensure_goal(goal_text)
        elif self.goal_text is None:
            raise ValueError("Goal text must be provided at least once before computing rewards.")

        caption = self.captioner.caption(scene_image, goal_text=goal_text)

        if self.baseline_text is None:
            # Use the very first caption as the baseline reference.
            self._state.baseline_text = caption

        potential = self._compute_potential(scene_image, caption)
        prev_potential = self._state.prev_potential
        reward = 0.0 if prev_potential is None else potential - prev_potential

        self._state.prev_potential = potential
        self._state.timestep = timestep
        self._state.history.append(
            StepRecord(timestep=timestep, caption=caption, potential=potential, reward=reward)
        )
        return reward

    __call__ = compute_reward

    def step(self, scene_image: Any, goal_text: Optional[str] = None) -> float:
        """Streaming-friendly alias that only requires the latest image.

        Parameters
        ----------
        scene_image:
            Observation image for the current timestep.
        goal_text:
            Provide once to initialize the goal for the episode.
        """

        return self.compute_reward(scene_image=scene_image, timestep=None, goal_text=goal_text)
