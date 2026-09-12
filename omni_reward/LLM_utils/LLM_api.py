"""
LLM utility functions for text generation, prompt enhancement, and goal processing.
"""
from __future__ import annotations

import os
import re
import openai
from typing import Any, Dict, List, Optional, Union

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Anthropic client
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Try to import Google Generative AI client
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMClient:
    """Unified client for interacting with various LLM providers."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """Initialize the LLM client.
        
        Parameters
        ----------
        provider:
            LLM provider to use. Options: "openai", "anthropic", "gemini".
        model:
            Model name to use. If None, uses default for provider.
        api_key:
            API key. If None, reads from environment variable.
        temperature:
            Sampling temperature for generation.
        max_tokens:
            Maximum tokens to generate.
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4o"
            self.client = OpenAI(api_key=self.api_key)
            
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.model = model or "claude-3-sonnet-20240229"
            self.client = anthropic.Anthropic(api_key=self.api_key)
            
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
            genai.configure(api_key=self.api_key)
            self.model = model or "gemini-1.5-flash"  # Free tier model
            self.client = genai.GenerativeModel(self.model)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from the LLM.
        
        Parameters
        ----------
        prompt:
            User prompt/query.
        system_prompt:
            Optional system prompt to set context.
        temperature:
            Override default temperature.
        max_tokens:
            Override default max tokens.
            
        Returns
        -------
        Generated text response.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content.strip()
            
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        
        elif self.provider == "gemini":
            # Combine system prompt and user prompt for Gemini
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            generation_config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=tokens,
            )
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            return response.text.strip()
        
        return ""


# Default LLM client instance (lazily initialized)
_default_client: Optional[LLMClient] = None

# Default model configuration (can be overridden via environment variables)
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def get_llm_client(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Get or create an LLM client instance.
    
    Parameters
    ----------
    provider:
        LLM provider to use. Options: "openai", "anthropic", "gemini".
    model:
        Model name to use. If None, uses default from environment or fallback.
    **kwargs:
        Additional arguments passed to LLMClient.
        
    Returns
    -------
    LLMClient instance.
    """
    global _default_client
    
    # Use environment-based defaults if model not specified
    if model is None:
        if provider.lower() == "openai":
            model = DEFAULT_OPENAI_MODEL
        elif provider.lower() == "anthropic":
            model = DEFAULT_ANTHROPIC_MODEL
        elif provider.lower() == "gemini":
            model = DEFAULT_GEMINI_MODEL
    
    if _default_client is None or _default_client.provider != provider:
        _default_client = LLMClient(provider=provider, model=model, **kwargs)
    return _default_client


# ============================================================================
# Prompt Enhancement Utilities
# ============================================================================

def enhance_prompt(
    prompt: str,
    context: Optional[str] = None,
    style: str = "detailed",
    client: Optional[LLMClient] = None,
) -> str:
    """Enhance a prompt to be more detailed and specific.
    
    Parameters
    ----------
    prompt:
        Original prompt to enhance.
    context:
        Optional context about the task/domain.
    style:
        Enhancement style. Options: "detailed", "concise", "technical".
    client:
        LLM client to use. If None, uses default client.
        
    Returns
    -------
    Enhanced prompt text.
    """
    llm = client or get_llm_client()
    
    style_instructions = {
        "detailed": "Make it more detailed and descriptive while keeping the core meaning.",
        "concise": "Make it more precise and concise while preserving key information.",
        "technical": "Make it more technical and specific with measurable criteria.",
    }
    
    system_prompt = """You are a prompt enhancement assistant. Your task is to improve 
    the given prompt to be clearer and more effective for downstream tasks."""
    
    user_prompt = f"""
    Original prompt: "{prompt}"
    {f'Context: {context}' if context else ''}
    
    Instructions: {style_instructions.get(style, style_instructions['detailed'])}
    
    Provide only the enhanced prompt without any explanation.
    """
    
    try:
        return llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.3)
    except Exception as e:
        print(f"[LLM_api] Warning: Failed to enhance prompt: {e}")
        return prompt

def enrich_state_description(
    current_state_text: str,
    domain: str = "robotics",
    client: Optional[LLMClient] = None,
) -> str:
    """LLM Enrich a state description with additional context and details.
    
    Parameters
    ----------
    current_state_text:
        Original state description.
    domain:
        Task domain for context. Options: "robotics", "navigation", "manipulation".
    client:
        LLM client to use.
        
    Returns
    -------
    Enriched state description with additional context.
    """
    llm = client or get_llm_client()
    
    system_prompt = f"""You are a {domain} state description expert. Your task is to 
    enrich state descriptions to make them more informative for learning algorithms."""
    
    user_prompt = f"""
    Original state description: "{current_state_text}"
    
    Please enrich this description by adding:
    1. Key visual elements or objects involved
    2. the current spatial relationships of objects
    3. the current robot state, e.g., positions, orientations, configurations
    4. Relevant state information (Important physical interactions occurring, forces, contacts, tactile info etc.)
    
    Provide a single enriched description that combines all this information naturally.
    Keep it concise but informative.
    """
    
    try:
        enriched = llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.3)
        return f"{current_state_text} | {enriched}"
    except Exception as e:
        print(f"[LLM_api] Warning: Failed to enrich state description: {e}")
        return current_state_text
    
def enrich_goal_description(
    goal_text: str,
    domain: str = "robotics",
    client: Optional[LLMClient] = None,
) -> str:
    """Enrich a goal description with additional context and details.
    
    Parameters
    ----------
    goal_text:
        Original goal description.
    domain:
        Task domain for context. Options: "robotics", "navigation", "manipulation".
    client:
        LLM client to use.
        
    Returns
    -------
    Enriched goal description with additional context.
    """
    llm = client or get_llm_client()
    
    system_prompt = f"""You are a {domain} task specification expert. Your task is to 
    enrich goal descriptions to make them more informative for learning algorithms."""
    
    user_prompt = f"""
    Original goal: "{goal_text}"
    
    Please enrich this goal description by adding:
    1. Key visual elements or objects involved
    2. Expected spatial relationships
    3. Success criteria or indicators
    4. Important physical interactions
    
    Provide a single enriched description that combines all this information naturally.
    Keep it concise but informative.
    """
    
    try:
        enriched = llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.3)
        return f"{goal_text} | {enriched}"
    except Exception as e:
        print(f"[LLM_api] Warning: Failed to enrich goal: {e}")
        return goal_text


# ============================================================================
# Goal Decomposition Utilities
# ============================================================================

def decompose_goal_to_subgoals(
    goal_text: str,
    max_subgoals: int = 5,
    domain: str = "robotics",
    client: Optional[LLMClient] = None,
) -> List[str]:
    """Decompose a complex goal into simpler subgoals.
    
    Parameters
    ----------
    goal_text:
        The final goal to decompose.
    max_subgoals:
        Maximum number of subgoals to generate.
    domain:
        Task domain for context.
    client:
        LLM client to use.
        
    Returns
    -------
    List of subgoals in execution order.
    """
    llm = client or get_llm_client()
    
    system_prompt = f"""You are a {domain} task planning expert. Your task is to 
    decompose complex goals into simpler, achievable subgoals for agent or robot execution."""
    
    user_prompt = f"""
    Decompose the following goal into a sequence of simpler subgoals:
    
    Goal: "{goal_text}"
    
    Requirements:
    - Generate at most {max_subgoals} subgoals
    - Each subgoal should be clearly achievable
    - Subgoals should be in logical execution order
    - Each subgoal should lead progressively toward the final goal
    
    Provide the subgoals as a numbered list, one per line.
    """
    
    try:
        response = llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.3)
        return _parse_numbered_list(response)
    except Exception as e:
        print(f"[LLM_api] Warning: Failed to decompose goal: {e}")
        return [goal_text]


def _parse_numbered_list(text: str) -> List[str]:
    """Parse a numbered list from LLM response.
    
    Parameters
    ----------
    text:
        Raw text containing numbered items.
        
    Returns
    -------
    List of parsed items.
    """
    lines = text.strip().split('\n')
    items = []
    
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        # Remove common numbering patterns: "1.", "1)", "1:", "- ", "* "
        cleaned = re.sub(r'^[\d]+[.\):\s]+', '', cleaned)
        cleaned = re.sub(r'^[-*]\s+', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            items.append(cleaned)
    
    return items


# ============================================================================
# Caption and Description Utilities
# ============================================================================

def refine_caption(
    caption: str,
    goal_context: Optional[str] = None,
    client: Optional[LLMClient] = None,
) -> str:
    """Refine a scene caption to be more relevant to the goal.
    
    Parameters
    ----------
    caption:
        Original scene caption.
    goal_context:
        Optional goal context to make caption more relevant.
    client:
        LLM client to use.
        
    Returns
    -------
    Refined caption.
    """
    llm = client or get_llm_client()
    
    system_prompt = """You are a scene description specialist. Refine captions to be 
    more precise and relevant for visual understanding tasks."""
    
    goal_part = f"\nGoal context: {goal_context}" if goal_context else ""
    
    user_prompt = f"""
    Original caption: "{caption}"{goal_part}
    
    Refine this caption to be more precise about:
    - Object positions and spatial relationships
    - Relevant state information
    - Key visual details
    
    Provide only the refined caption.
    """
    
    try:
        return llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.2)
    except Exception as e:
        print(f"[LLM_api] Warning: Failed to refine caption: {e}")
        return caption


def generate_success_criteria(
    goal_text: str,
    client: Optional[LLMClient] = None,
) -> List[str]:
    """Generate specific success criteria for a goal.
    
    Parameters
    ----------
    goal_text:
        The goal to generate criteria for.
    client:
        LLM client to use.
        
    Returns
    -------
    List of success criteria.
    """
    llm = client or get_llm_client()
    
    system_prompt = """You are a task evaluation expert. Generate clear, 
    verifiable success criteria for robotics tasks."""
    
    user_prompt = f"""
    Goal: "{goal_text}"
    
    Generate 3-5 specific success criteria that indicate this goal has been achieved.
    Each criterion should be:
    - Visually verifiable
    - Specific and measurable
    - Focused on the end state
    
    Provide as a numbered list.
    """
    
    try:
        response = llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.3)
        return _parse_numbered_list(response)
    except Exception as e:
        print(f"[LLM_api] Warning: Failed to generate success criteria: {e}")
        return [f"Goal '{goal_text}' is achieved"]


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def batch_generate(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    client: Optional[LLMClient] = None,
) -> List[str]:
    """Generate responses for multiple prompts.
    
    Parameters
    ----------
    prompts:
        List of prompts to process.
    system_prompt:
        Optional system prompt applied to all.
    client:
        LLM client to use.
        
    Returns
    -------
    List of generated responses.
    """
    llm = client or get_llm_client()
    results = []
    
    for prompt in prompts:
        try:
            response = llm.generate(prompt, system_prompt=system_prompt)
            results.append(response)
        except Exception as e:
            print(f"[LLM_api] Warning: Failed to generate for prompt: {e}")
            results.append("")
    
    return results


def combine_enriched_information(
    parts: List[str],
    style: str = "natural",
    client: Optional[LLMClient] = None,
) -> str:
    """Combine multiple pieces of enriched information into coherent text.
    
    Parameters
    ----------
    parts:
        List of text pieces to combine.
    style:
        Combination style. Options: "natural", "bullet", "concatenate".
    client:
        LLM client to use.
        
    Returns
    -------
    Combined text.
    """
    if style == "concatenate":
        return " | ".join(parts)
    
    if style == "bullet":
        return "\n- " + "\n- ".join(parts)
    
    # Natural combination using LLM
    llm = client or get_llm_client()
    
    user_prompt = f"""
    Combine the following pieces of information into a single coherent description:
    
    {chr(10).join(f'- {p}' for p in parts)}
    
    Create a natural, flowing description that incorporates all key information.
    Keep it concise.
    """
    
    try:
        return llm.generate(user_prompt, temperature=0.3)
    except Exception as e:
        print(f"[LLM_api] Warning: Failed to combine information: {e}")
        return " | ".join(parts)