import numpy as np

def cosine_sim(a, b):
    return float(np.dot(a, b))

class TactileCaptioner:
    def __init__(self):
        pass

    def to_caption(self, tactile_vec):
        # TODO: these thresholds are not properly set, just for illustration
        if tactile_vec is None:
            return "no tactile information"
        if tactile_vec.max() < 0.1:
            return "no contact detected"
        if tactile_vec.mean() < 0.3:
            return "light contact with the object"
        if tactile_vec.mean() < 0.7:
            return "firm grasp on the object"
        return "very strong pressure, possibly unstable"


class UnifiedMultimodalPotential:
    # Φ(s) = λ Φ_V(s, α) + (1 - λ) Φ_T(s)
    # Φ_V uses α (vision baseline & goal)
    # Φ_T uses only goal similarity (no tactile baseline)

    def __init__(
        self,
        text_encoder,
        vision_goal_text,
        baseline_text,
        tactile_goal_text=None,
        paraphraser=None
    ):
        self.encoder = text_encoder

        # Vision embeddings
        self.goal_emb = self.encoder.encode_one(vision_goal_text)
        self.base_emb = self.encoder.encode_one(baseline_text)

        # Tactile goal embedding
        self.tactile_goal_emb = (
            self.encoder.encode_one(tactile_goal_text)
            if tactile_goal_text is not None else None
        )

        self.tactile_captioner = TactileCaptioner()
        self.paraphraser = paraphraser


    def _vision_potential(self, caption, alpha):
        captions = [caption]

        if self.paraphraser:
            captions.extend(self.paraphraser.paraphrase(caption, n=2))

        embeddings = self.encoder.encode_many(captions)

        vals = []
        for eC in embeddings:
            p_goal = cosine_sim(eC, self.goal_emb)
            p_base = -cosine_sim(eC, self.base_emb)

            # Φ_V = α cos(eC, e_goal) + (1 - α)(-cos(eC, e_base))
            vals.append(alpha * p_goal + (1 - alpha) * p_base)

        return float(np.mean(vals))


    def _tactile_potential(self, tactile_vec):
        if tactile_vec is None or self.tactile_goal_emb is None:
            return 0.0

        caption = self.tactile_captioner.to_caption(tactile_vec)
        emb = self.encoder.encode_one(caption)
        return cosine_sim(emb, self.tactile_goal_emb)


    def compute(self, obs, caption, alpha, lambda_):
        tactile_vec = obs.get("tactile", None)

        phi_v = self._vision_potential(caption, alpha)
        phi_t = self._tactile_potential(tactile_vec)

        return lambda_ * phi_v + (1 - lambda_) * phi_t
