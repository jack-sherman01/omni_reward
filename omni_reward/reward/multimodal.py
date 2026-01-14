from typing import Any, Mapping, Optional, Sequence
import numpy as np

def cosine_sim(a, b):
    return float(np.dot(a, b))

class UnifiedMultimodalPotential:
    """Φ(s) = λ Φ_V(s, α) + (1 - λ) Φ_T(s).

    Vision Φ_V uses both the goal and baseline caption along with α. 
    Tactile Φ_T is optional for now and only uses similarity to a goal description when present.
    """

    def __init__(
        self,
        text_encoder,
        vision_goal_text: str,
        baseline_text: str,
        tactile_goal_text: Optional[str] = None,
    ):
        self.encoder = text_encoder
        self.goal_emb = self.encoder.encode_one(vision_goal_text)
        self.base_emb = self.encoder.encode_one(baseline_text)
        self.tactile_goal_emb = None
        self.tactile_captioner = None

    def _vision_potential(self, caption: str, alpha: float) -> float:
        """Compute the vision-based potential function.

        The vision potential combines goal-seeking and baseline-avoiding behaviors:

            Φ_V(s, α) = α · sim(e_s, e_goal) - (1 - α) · sim(e_s, e_base)

        where:
            - e_s: embedding of the current state caption
            - e_goal: embedding of the goal description
            - e_base: embedding of the baseline description
            - sim(·, ·): cosine similarity between two embeddings
            - α ∈ [0, 1]: weighting factor that balances goal attraction vs baseline repulsion

        When α = 1: pure goal-seeking (maximize similarity to goal)
        When α = 0: pure baseline-avoiding (minimize similarity to baseline)
        When 0 < α < 1: combined behavior encouraging progress toward goal while leaving baseline

        Args:
            caption: Text description of the current visual state.
            alpha: Weighting factor between goal similarity and baseline dissimilarity.

        Returns:
            The scalar potential value for the current state.
        """
        caption_emb = self.encoder.encode_one(caption)
        p_goal = cosine_sim(caption_emb, self.goal_emb)
        p_base = -cosine_sim(caption_emb, self.base_emb)
        
        return alpha * p_goal + (1 - alpha) * p_base

    def _tactile_potential(self, tactile_vec: Optional[np.ndarray]) -> float:
        if tactile_vec is None or self.tactile_goal_emb is None:
            return 0.0
        caption = self.tactile_captioner.to_caption(tactile_vec)
        emb = self.encoder.encode_one(caption)
        return cosine_sim(emb, self.tactile_goal_emb)

    def compute(self, image_caption: str, alpha: float, lambda_: float) -> float:
        # TODO: should we do normalization here for the potentials?
        phi_v = self._vision_potential(image_caption, alpha)
        #TODO: enable tactile potential when available and add force potential as well
        # phi_t = self._tactile_potential(None)
        return lambda_ * phi_v + (1 - lambda_) #* phi_t
