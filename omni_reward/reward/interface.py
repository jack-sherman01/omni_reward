"""Small, stateful interface for turning images into semantic rewards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from omni_reward.reward.multimodal import UnifiedMultimodalPotential
from omni_reward.vision.captioner import VLMCaptioner
from omni_reward.LLM_utils import (
    get_llm_client,
    enrich_state_description,
    enrich_goal_description,
    decompose_goal_to_subgoals,
)


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

    def __init__(self, captioner, text_encoder, allow_goal_change: bool = False):
        """
        Initialize the OmniReward interface.
        
        Args:
            captioner: Vision-language model for generating captions
            text_encoder: Model for encoding text to embeddings
            allow_goal_change: If True, allows goal text to change during episode (for subgoals)
        """
        self.captioner = captioner
        self.text_encoder = text_encoder
        self.allow_goal_change = allow_goal_change
        
        # Episode state
        self._current_goal = None
        self._initial_image = None
        self._subgoals = None
        self.current_subgoal_index = 0
        self.subgoal_completion_threshold = 0.8  # Configurable threshold for subgoal completion
        
        # Reward computation parameters
        self.alpha = 0.5  # Weight for potential computation
        self.lambda_ = 0.9  # Discount factor for potential computation
        
        # History tracking
        self.store_history = True  # Whether to store history
        self._history = []
        
        # Potential and baseline
        self._potential = None
        self.goal_text = None
        self.baseline_caption = None
        self.prev_potential = None
        self.timestep = -1
        
        # Subgoals tracking
        self.subgoals = None
        
        self.reset_episode()

    # not used currently
    # def decompose_goal_vlm(self, goal_text: str) -> List[str]:
    #     """Decompose a final goal into ordered subgoals using VLM.
        
    #     Parameters
    #     ----------
    #     goal_text:
    #         The final goal to decompose.
            
    #     Returns
    #     -------
    #     List of subgoals (enriched not simple) in order of execution.
    #     """
    #     if not hasattr(self.captioner, 'vlm'):
    #         # Fallback if VLM is not available
    #         return [goal_text]
    #     # NOTE:I think here below we dont need to indicate the size of the subgoals, just decompose into simpler steps.
    #     decomposition_prompt = f"""
    #     Please decompose the following robot task goal into a sequence of simpler subgoals.
    #     Each subgoal should be achievable and lead progressively toward the final goal.
        
    #     Final Goal: {goal_text}
        
    #     Provide the subgoals as a numbered list, one per line.
    #     """
        
    #     try:
    #         response = self.captioner.vlm.generate_text(decomposition_prompt)
    #         # Parse the response to extract subgoals
    #         lines = response.strip().split('\n')
    #         subgoals = []
    #         for line in lines:
    #             # Remove numbering and clean up
    #             cleaned = line.strip()
    #             if cleaned and not cleaned.startswith('#'):
    #                 # Remove common numbering patterns like "1.", "1)", etc.
    #                 import re
    #                 cleaned = re.sub(r'^[\d]+[.\)]\s*', '', cleaned)
    #                 if cleaned:
    #                     # enrich the goal description after decomposition
    #                     rich_cleaned = enrich_goal_description(cleaned, domain="robotics") # using LLM 
    #                     # rich_cleaned = VLMCaptioner.enrich_goal(cleaned) # using VLM
    #                     subgoals.append(rich_cleaned)
            
    #         if not subgoals:
    #             return [goal_text]
                
    #         print(f"[OmniRewardInterface] Decomposed goal into {len(subgoals)} subgoals:")
    #         for i, sg in enumerate(subgoals, 1):
    #             print(f"  {i}. {sg}")
                
    #         return subgoals
            
    #     except Exception as e:
    #         print(f"[OmniRewardInterface] Failed to decompose goal: {e}")
    #         return [goal_text]

    def start_episode_with_subgoals(
        self,
        goal_text: str,
        initial_image: Any = None,
        auto_decompose: bool = True,
    ) -> None:
        """Start a new episode with subgoal decomposition."""
        # Store the original goal
        self._original_goal = goal_text
        self.current_subgoal_index = 0
        
        if auto_decompose:
            try:
                # Decompose goal into subgoals using LLM
                self.subgoals = decompose_goal_to_subgoals(goal_text)
                if not self.subgoals or len(self.subgoals) == 0:
                    print(f"[OmniRewardInterface] Warning: Failed to decompose goal, using original")
                    self.subgoals = [goal_text]
            except Exception as e:
                print(f"[OmniRewardInterface] Warning: Error decomposing goal: {e}")
                self.subgoals = [goal_text]
        else:
            self.subgoals = [goal_text]
        
        print(f"[OmniRewardInterface] Subgoals initialized: {len(self.subgoals)} subgoals")
        for i, sg in enumerate(self.subgoals):
            print(f"  {i+1}. {sg}")
        
        # Start with first subgoal - this sets self.goal_text
        current_goal = self.subgoals[0]
        self.start_episode(current_goal, initial_image)

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
        if self.current_subgoal_index < len(self.subgoals) - 1: 
            self.current_subgoal_index += 1  
            next_subgoal = self.subgoals[self.current_subgoal_index]  
            
            print(f"[OmniRewardInterface] Advancing to subgoal {self.current_subgoal_index + 1}/{len(self.subgoals)}: {next_subgoal}")
            
            # Keep the baseline but update the goal
            prev_baseline = self.baseline_caption
            self.reset_episode(goal_text=next_subgoal)
            self.baseline_caption = prev_baseline
            
            return True
        return False

    def compute_reward_with_subgoals(
        self,
        scene_image: Any,
        auto_advance: bool = True,
        completion_bonus: float = 10.0,
    ) -> dict:
        """Compute reward with subgoal tracking.
    
        Returns:
            dict with keys:
                - reward: float
                - current_subgoal_index: int
                - current_subgoal: str
                - subgoal_completed: bool
                - all_completed: bool
                - all_subgoals: List[str]
        """
        # Get current subgoal info with safe defaults
        current_index = getattr(self, 'current_subgoal_index', 0)
        subgoals = getattr(self, 'subgoals', None)
    
        # Handle case where subgoals is None or empty
        if subgoals is None or len(subgoals) == 0:
            # Fallback to using goal_text as the only subgoal
            subgoals = [self.goal_text] if self.goal_text else ["Complete the task"]
            self.subgoals = subgoals
            self.current_subgoal_index = 0
            current_index = 0
    
        if current_index >= len(subgoals):
            # All subgoals completed
            return {
                'reward': completion_bonus,
                'current_subgoal_index': current_index,
                'current_subgoal': 'All completed',
                'subgoal_completed': False,
                'all_completed': True,
                'all_subgoals': subgoals,
            }
    
        current_subgoal = subgoals[current_index]
    
        # Compute reward for current subgoal
        reward = self.compute_reward(scene_image)
    
        # Check if subgoal is completed based on threshold
        threshold = getattr(self, 'subgoal_completion_threshold', 0.8)
        subgoal_completed = reward > threshold
    
        if subgoal_completed and auto_advance:
            self.current_subgoal_index = current_index + 1
            reward += completion_bonus
    
        all_completed = (current_index + 1 >= len(subgoals)) and subgoal_completed
    
        return {
            'reward': reward,
            'current_subgoal_index': current_index,
            'current_subgoal': current_subgoal,
            'subgoal_completed': subgoal_completed,
            'all_completed': all_completed,
            'all_subgoals': subgoals,
        }

    @property
    def history(self) -> List[RewardStep]:  # pragma: no cover - trivial accessor
        """Return a shallow copy of the per-episode history."""

        return list(self._history)

    def reset_episode(self, goal_text: Optional[str] = None) -> None:
        """Clear cached state so a new episode can start.
        
        Note: This does NOT reset goal_text or subgoals - those are set by
        start_episode() or start_episode_with_subgoals().
        """
        self._potential = None
        self.baseline_caption = None
        self.prev_potential = None
        self.timestep = -1
        self._history = []
        self.current_subgoal_index = 0
        # Note: goal_text is not reset here

    def start_episode(
        self,
        goal_text: str,
        initial_image: Any = None,
    ) -> None:
        """Start a new episode with the given goal.
        
        Parameters
        ----------
        goal_text:
            The goal description for this episode.
        initial_image:
            Optional initial image to establish baseline.
        """
        # Enrich the goal text
        try:
            self.goal_text = enrich_goal_description(goal_text, domain="robotics")
        except Exception as e:
            print(f"[OmniRewardInterface] Warning: Failed to enrich goal: {e}")
            self.goal_text = goal_text  # Use original if enrichment fails
    
        # Reset state but keep goal_text
        self.baseline_caption = None
        self.prev_potential = None
        self.timestep = -1
        self._history = []
    
        print(f"[OmniRewardInterface] Episode started with goal: {self.goal_text[:100]}...")
    
        # If initial image provided, generate baseline caption
        if initial_image is not None:
            try:
                caption = self.captioner.caption(initial_image, goal_text=self.goal_text)
                self.baseline_caption = caption
                print(f"[OmniRewardInterface] Baseline caption: {caption[:100]}...")
            except Exception as e:
                print(f"[OmniRewardInterface] Warning: Failed to generate baseline caption: {e}")

    def _record_step(self, *, timestep: int, image_caption: str, potential: float, reward: float) -> None:
        """ Save to history for logging purposes """ 
        if not self.store_history:
            return
        self._history.append(
            RewardStep(timestep=timestep, image_caption=image_caption, potential=potential, reward=reward)
        )

    def _resolve_goal(self, goal_text: Optional[str] = None) -> str:
        """Resolve the goal text to use.
    
        Parameters
        ----------
        goal_text:
            Optional override goal text.
        
        Returns
        -------
        The goal text to use.
        """
        if goal_text is not None:
            return goal_text
    
        if self.goal_text is not None and self.goal_text != "":
            return self.goal_text
    
        # Try to get from subgoals
        if hasattr(self, 'subgoals') and self.subgoals and len(self.subgoals) > 0:
            idx = getattr(self, 'current_subgoal_index', 0)
            if idx < len(self.subgoals):
                return self.subgoals[idx]
    
        # Try original goal
        if hasattr(self, '_original_goal') and self._original_goal:
            return self._original_goal
    
        raise ValueError("No goal text available. Call start_episode() first.")

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
        # TODO: can be optimized to avoid double captioning when priming baseline
        image_caption = self.captioner.caption(scene_image, goal_text=resolved_goal)
        self._prime_baseline(image_caption)

        potential = self._compute_potential(image_caption)
        prev_potential = self.prev_potential
        # NOTE: only use potential as reward for test now, need to add difference as a reward term later.
        # reward = 0.0 if prev_potential is None else potential - prev_potential
        reward = potential # use potential as reward for test

        self.prev_potential = potential
        self.timestep = timestep
        self._record_step(timestep=timestep, image_caption=image_caption, potential=potential, reward=reward)
        
        # Compute completion sense reward
        # R_completion = r_base + β · sigmoid(k · (Φ - τ)) · (1 + Δ_progress)
        # where:
        #   - r_base: base potential reward
        #   - Φ: current potential (similarity to goal)
        #   - τ: completion threshold
        #   - k: sigmoid steepness factor
        #   - Δ_progress: improvement from previous step
        #   - β: completion bonus weight
        completion_sense_reward = self._compute_completion_sense_reward(
            base_reward=reward,
            current_potential=potential,
            prev_potential=prev_potential,
        )
        
        return completion_sense_reward

    def _compute_completion_sense_reward(
        self,
        base_reward: float,
        current_potential: float,
        prev_potential: Optional[float],
        completion_threshold: float = 0.7,
        sigmoid_steepness: float = 10.0,
        completion_bonus_weight: float = 0.5,
    ) -> float:
        """Compute completion-aware reward with smooth transition near goal.

        The completion sense reward combines base reward with a sigmoid-based
        completion bonus that activates as the agent approaches the goal:

            R_completion = r_base + β · σ(k · (Φ - τ)) · (1 + max(0, ΔΦ))

        where:
            - r_base: base potential reward from current state
            - Φ: current potential (normalized similarity to goal, in [-1, 1])
            - τ: completion threshold (typically 0.7-0.9)
            - k: sigmoid steepness factor (controls sharpness of transition)
            - σ(x) = 1 / (1 + exp(-x)): sigmoid function for smooth activation
            - ΔΦ = Φ_t - Φ_{t-1}: progress from previous step
            - β: completion bonus weight

        This formulation provides: 
            1. Smooth reward increase as agent approaches completion
            2. Extra bonus for continued progress near the goal
            3. No discontinuous jumps at threshold boundaries

        Args:
            base_reward: The base potential reward.
            current_potential: Current state's potential value Φ(s).
            prev_potential: Previous state's potential value Φ(s').
            completion_threshold: Threshold τ where completion bonus activates.
            sigmoid_steepness: Steepness k of sigmoid transition.
            completion_bonus_weight: Weight β for completion bonus.

        Returns:
            The completion-aware shaped reward.
        """
        # Sigmoid activation: σ(k · (Φ - τ))
        # Maps potential to [0, 1] with smooth transition around threshold
        sigmoid_input = sigmoid_steepness * (current_potential - completion_threshold)
        completion_activation = 1.0 / (1.0 + np.exp(-sigmoid_input))
        
        # Progress term: ΔΦ = Φ_t - Φ_{t-1}
        # Rewards continued improvement, especially near goal
        progress = 0.0
        if prev_potential is not None:
            progress = max(0.0, current_potential - prev_potential)
        
        # Completion bonus: β · σ(k · (Φ - τ)) · (1 + ΔΦ)
        completion_bonus = completion_bonus_weight * completion_activation * (1.0 + progress)
        
        # Final reward: r_base + completion_bonus
        completion_sense_reward = base_reward + completion_bonus
        
        return completion_sense_reward

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