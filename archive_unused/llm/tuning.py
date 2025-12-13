class AlphaLambdaTuner:
    # α tunes for long-horizon vs. short-distance task
    # λ tunes for vision vs. tactile modality balance

    def __init__(self, llm=None, alpha_default=0.6, lambda_default=0.7):
        self.llm = llm
        self.alpha_default = alpha_default
        self.lambda_default = lambda_default

    def suggest(self, obs, step, task_desc):
        # TODO: Replace with LLM call
        tactile = obs.get("tactile")

        # Heuristic for α (shaping)
        if "long" in task_desc.lower():
            alpha = max(0.2, self.alpha_default - 0.002 * step)
        else:
            alpha = self.alpha_default

        # Heuristic for λ (modality)
        if tactile is not None and tactile.mean() > 0.2:
            lambda_ = 0.4  # rely more on tactile
        else:
            lambda_ = self.lambda_default

        return {"alpha": float(alpha), "lambda": float(lambda_)}


class FixedAlphaLambda:
    # use fixed values from config, for sanity-checking / debugging
    def __init__(self, alpha, lambda_):
        self.alpha = alpha
        self.lambda_ = lambda_

    def suggest(self, obs, step, task_desc):
        return {"alpha": self.alpha, "lambda": self.lambda_}
