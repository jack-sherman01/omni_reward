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
        self.text_encoder = text_encoder
        self.alpha = alpha
        self.lambda_ = lambda_
        self.store_history = store_history
        
        # Add subgoal-related attributes
        self.subgoals = []
        self.current_subgoal_idx = 0
        self.subgoal_completion_threshold = 0.8  # Configurable threshold for subgoal completion
        
        self.reset_episode()

    def decompose_goal(self, goal_text: str) -> List[str]:
        """Decompose a final goal into ordered subgoals using VLM.
        
        Parameters
        ----------
        goal_text:
            The final goal to decompose.
            
        Returns
        -------
        List of subgoals in order of execution.
        """
        if not hasattr(self.captioner, 'vlm'):
            # Fallback if VLM is not available
            return [goal_text]
            
        decomposition_prompt = f"""
        Please decompose the following robot task goal into a sequence of simpler subgoals.
        Each subgoal should be achievable and lead progressively toward the final goal.
        
        Final Goal: {goal_text}
        
        Provide the subgoals as a numbered list, one per line.
        """
        
        try:
            response = self.captioner.vlm.generate_text(decomposition_prompt)
            # Parse the response to extract subgoals
            lines = response.strip().split('\n')
            subgoals = []
            for line in lines:
                # Remove numbering and clean up
                cleaned = line.strip()
                if cleaned and not cleaned.startswith('#'):
                    # Remove common numbering patterns like "1.", "1)", etc.
                    import re
                    cleaned = re.sub(r'^[\d]+[.\)]\s*', '', cleaned)
                    if cleaned:
                        subgoals.append(cleaned)
            
            if not subgoals:
                return [goal_text]
                
            print(f"[OmniRewardInterface] Decomposed goal into {len(subgoals)} subgoals:")
            for i, sg in enumerate(subgoals, 1):
                print(f"  {i}. {sg}")
                
            return subgoals
            
        except Exception as e:
            print(f"[OmniRewardInterface] Failed to decompose goal: {e}")
            return [goal_text]

    def start_episode_with_subgoals(
        self,
        goal_text: str,
        baseline_image: Optional[Any] = None,
        baseline_caption: Optional[str] = None,
        auto_decompose: bool = True,
    ) -> None:
        """Start an episode with automatic goal decomposition.
        
        Parameters
        ----------
        goal_text:
            The final goal to achieve.
        baseline_image:
            Optional first observation.
        baseline_caption:
            Optional baseline caption.
        auto_decompose:
            Whether to automatically decompose the goal into subgoals.
        """
        if auto_decompose:
            self.subgoals = self.decompose_goal(goal_text)
        else:
            self.subgoals = [goal_text]
            
        self.current_subgoal_idx = 0
        
        # Start with the first subgoal
        current_goal = self.subgoals[0] if self.subgoals else goal_text
        self.start_episode(current_goal, baseline_image, baseline_caption)

    def check_subgoal_completion(self, current_potential: float) -> bool:
        """Check if current subgoal is completed based on potential.
        
        Parameters
        ----------
        current_potential:
            The current potential value.
            
        Returns
        -------
        Whether the subgoal is considered complete.
        """
        return current_potential >= self.subgoal_completion_threshold

    def advance_to_next_subgoal(self) -> bool:
        """Advance to the next subgoal if available.
        
        Returns
        -------
        True if advanced to next subgoal, False if all subgoals completed.
        """
        if self.current_subgoal_idx < len(self.subgoals) - 1:
            self.current_subgoal_idx += 1
            next_subgoal = self.subgoals[self.current_subgoal_idx]
            
            print(f"[OmniRewardInterface] Advancing to subgoal {self.current_subgoal_idx + 1}/{len(self.subgoals)}: {next_subgoal}")
            
            # Keep the baseline but update the goal
            prev_baseline = self.baseline_caption
            self.reset_episode(goal_text=next_subgoal)
            self.baseline_caption = prev_baseline
            
            return True
        return False

    def compute_reward_with_subgoals(
        self,
        scene_image: Any,
        timestep: Optional[int] = None,
        auto_advance: bool = True,
        completion_bonus: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute reward with subgoal progression.
        
        Parameters
        ----------
        scene_image:
            Current observation.
        timestep:
            Optional timestep.
        auto_advance:
            Whether to automatically advance to next subgoal when current is completed.
        completion_bonus:
            Bonus reward for completing a subgoal.
            
        Returns
        -------
        Dictionary containing:
            - reward: The computed reward
            - subgoal_completed: Whether a subgoal was completed
            - current_subgoal: Current subgoal text
            - progress: Progress through subgoals (fraction)
        """
        # Use current subgoal as the goal
        current_goal = self.subgoals[self.current_subgoal_idx] if self.subgoals else self.goal_text
        
        # Compute regular reward
        reward = self.compute_reward(scene_image, timestep, goal_text=current_goal)
        
        result = {
            'reward': reward,
            'subgoal_completed': False,
            'current_subgoal': current_goal,
            'progress': (self.current_subgoal_idx + 1) / len(self.subgoals) if self.subgoals else 1.0,
            'subgoal_index': self.current_subgoal_idx,
            'total_subgoals': len(self.subgoals)
        }
        
        # Check for subgoal completion
        if self.prev_potential is not None and self.check_subgoal_completion(self.prev_potential):
            result['subgoal_completed'] = True
            result['reward'] += completion_bonus
            
            if auto_advance:
                if not self.advance_to_next_subgoal():
                    result['all_completed'] = True
                    print("[OmniRewardInterface] All subgoals completed!")
        
        return result

    @property
    def history(self) -> List[RewardStep]:  # pragma: no cover - trivial accessor
        """Return a shallow copy of the per-episode history."""

        return list(self._history)

    def reset_episode(self, goal_text: Optional[str] = None) -> None:
        """Clear cached state so a new episode can start."""

        self._potential = None
        self.goal_text = VLMCaptioner.enrich_goal(goal_text)
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