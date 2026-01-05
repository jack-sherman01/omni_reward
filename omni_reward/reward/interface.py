"""Small, stateful interface for turning images into semantic rewards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from omni_reward.reward.multimodal import UnifiedMultimodalPotential
from omni_reward.vision.captioner import VLMCaptioner


class Captioner(Protocol):
    """Minimal protocol required for captioning models."""

    def caption(self, image: Any, goal_text: Optional[str] = None) -> str:  # pragma: no cover
        ...


class TextEncoder(Protocol):
    """Minimal protocol required for text encoders used by the potential."""

    def encode_one(self, text: str) -> np.ndarray:  # pragma: no cover - protocol definition
        ...

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - protocol definition
        ...


@dataclass
class RewardStep:
    """Container for per-timestep bookkeeping."""

    timestep: int
    image_caption: str
    potential: float
    reward: float


class OmniRewardInterface:
    """High-level wrapper for the semantic potential reward.

    The class hides all potential-based shaping logic so the caller only needs to provide
         an image, 
         an optional timestep, 
         and the goal text. 
    Internally, the first generated caption becomes the baseline description and the reward is the difference between consecutive potentials.
    """

    def __init__(
        self,
        captioner: Captioner,
        text_encoder: TextEncoder,
        *,
        alpha: float = 0.6,
        lambda_: float = 1.0,
        store_history: bool = True,
    ) -> None:
        self.captioner = captioner
        self.text_encoder = VLMCaptioner.enrich_goal(text_encoder)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.store_history = store_history

        self.reset_episode()

    @property
    def history(self) -> List[RewardStep]:  # pragma: no cover - trivial accessor
        """Return a shallow copy of the per-episode history."""

        return list(self._history)

    def reset_episode(self, goal_text: Optional[str] = None) -> None:
        """Clear cached state so a new episode can start."""

        self._potential = None
        self.goal_text = self.enrich_goal(goal_text)
        self.baseline_caption = None
        self.prev_potential = None
        self.timestep = -1
        self._history = []

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

        self.reset_episode(goal_text=goal_text)

        caption = baseline_caption
        if caption is None and baseline_image is not None:
            caption = self.captioner.caption(baseline_image, goal_text=goal_text)

        if caption is None:
            return

        self.baseline_caption = caption
        if baseline_image is None:
            return

        potential = self._compute_potential(baseline_image, caption)
        self.prev_potential = potential
        self.timestep = 0
        self._record_step(timestep=0, caption=caption, potential=potential, reward=0.0)

    def _record_step(self, *, timestep: int, image_caption: str, potential: float, reward: float) -> None:
        """ Save to history for logging purposes """ 
        if not self.store_history:
            return
        self._history.append(
            RewardStep(timestep=timestep, image_caption=image_caption, potential=potential, reward=reward)
        )

    def _resolve_goal(self, goal_text: Optional[str]) -> str:
        if goal_text is not None:
            if self.goal_text is not None and goal_text != self.goal_text:
                raise ValueError("Goal text changed during an episode. Call reset_episode() before switching goals.")
            self.goal_text = goal_text

        if self.goal_text is None:
            raise ValueError("Goal text must be provided before computing rewards.")

        return self.goal_text

    def _resolve_timestep(self, supplied_timestep: Optional[int], incoming_goal: Optional[str]) -> int:
        if supplied_timestep is None:
            return 0 if self.timestep < 0 else self.timestep + 1

        if self.timestep >= 0 and supplied_timestep <= self.timestep:
            cached_goal = incoming_goal or self.goal_text
            self.reset_episode(goal_text=cached_goal)

        return supplied_timestep

    def _prime_baseline(self, caption: str) -> None:
        if self.baseline_caption is None:
            self.baseline_caption = caption
            self._potential = None  # Force re-instantiation with the new baseline.

    def _ensure_potential_ready(self) -> None:
        if self._potential is not None:
            return
        if self.goal_text is None or self.baseline_caption is None:
            raise RuntimeError("Potential requested before goal/baseline were initialized.")
        self._potential = UnifiedMultimodalPotential(
            text_encoder=self.text_encoder,
            vision_goal_text=self.goal_text,
            baseline_text=self.baseline_caption,
        )

    def _compute_potential(self, image_caption: str) -> float:
        self._ensure_potential_ready()
        potential = self._potential.compute(
            image_caption=image_caption,
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
        """Return the shaped reward for the provided scene observation."""

        timestep = self._resolve_timestep(timestep, goal_text)
        resolved_goal = self._resolve_goal(goal_text)

        image_caption = self.captioner.caption(scene_image, goal_text=resolved_goal)
        self._prime_baseline(image_caption)

        potential = self._compute_potential(image_caption)
        prev_potential = self.prev_potential
        reward = 0.0 if prev_potential is None else potential - prev_potential

        self.prev_potential = potential
        self.timestep = timestep
        self._record_step(timestep=timestep, image_caption=image_caption, potential=potential, reward=reward)
        return reward

    def get_current_reward(self, scene_image: Any, goal_text: Optional[str] = None) -> float:
        """Streaming-friendly alias that only requires the latest image.
        # NOTE: Heng changed the method name from step to get_current_reward to avoid confusion from step() in common RL libraries.
        Parameters
        ----------
        scene_image:
            Observation image for the current timestep.
        goal_text:
            Provide once to initialize the goal for the episode.
        """

        return self.compute_reward(scene_image=scene_image, timestep=None, goal_text=goal_text)

    __call__ = compute_reward