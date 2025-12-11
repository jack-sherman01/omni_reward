from __future__ import annotations

from typing import Dict, Optional

CAPTION_TEMPLATES: Dict[str, str] = { # you can add and edit prompts here by adding a "structured_v2", or editing this existing structured_v1
    "structured_v1": (
        "You are a robotics perception module. "
        "Describe the scene shown in the image with a focus on manipulable objects, "
        "and their spatial relationships."
        "Focus on how the scene relates to this goal: {goal}. "
        "Respond ONLY in JSON with the following structure:\n"
        "{{\n"
        "  \"caption\": \"one sentence summary\",\n"
        # "  \"confidence\": number between 0 and 1\n"            we could use this later for the VLM to self-filter out egregiously bad captions / image inputs
        "}}\n"
        "Make the caption concrete enough that it can be embedded for similarity "
        "comparison with the goal text."
    ),
}


def get_caption_template(name: str, goal: Optional[str] = None) -> str:
    """Return the prompt template matching *name* with goal substitution."""
    if name not in CAPTION_TEMPLATES:
        raise ValueError(
            f"Unknown caption template '{name}'. Available templates: {list(CAPTION_TEMPLATES.keys())}"
        )
    return CAPTION_TEMPLATES[name].format(goal=goal or "the instructed goal")
