"""
VLM_api.py - API-based Vision-Language Model Implementations and Factory

================================================================================
Description:
================================================================================
    This module provides API-based implementations of Vision-Language Models
    (VLMs) from multiple cloud providers including OpenAI, Google Gemini, 
    Alibaba Qwen, and Anthropic Claude. It also provides a unified factory 
    interface for creating VLM instances from both API and local sources.

================================================================================
Supported API Providers:
================================================================================
    - OpenAI: GPT-4o, GPT-4V (gpt-4-vision-preview)
    - Google: Gemini Pro Vision, Gemini 1.5 Pro/Flash
    - Alibaba: Qwen-VL-Plus, Qwen-VL-Max (via DashScope)
    - Anthropic: Claude 3 Opus/Sonnet/Haiku

================================================================================
Features:
================================================================================
    - Multi-provider API support with consistent interface
    - Unified VLMFactory for creating any VLM type (API or local)
    - Automatic base64 encoding for image transmission
    - JSON-based structured output parsing for task evaluation
    - Shared caption templates (see templates.py) for consistent scene descriptions
    - Fallback handling for malformed API responses
    - Convenient get_vlm() function for quick VLM instantiation
    - Support for both numpy arrays and PIL Images as input
    - Automatic channel format conversion (C,H,W) <-> (H,W,C)

================================================================================
Classes:
================================================================================
    API-based VLMs (defined in this file):
        - OpenAIVLM: OpenAI API-based VLM (GPT-4o, GPT-4V)
        - GeminiVLM: Google Gemini API-based VLM
        - QwenVLM: Alibaba Qwen API-based VLM (DashScope)
        - ClaudeVLM: Anthropic Claude API-based VLM
    
    Factory:
        - VLMFactory: Factory class for creating VLM instances
            - register(name, vlm_class): Register a new VLM type
            - create(vlm_type, **kwargs): Create a VLM instance by type name
            - available_types(): List all registered VLM types
            - get_info(): Get descriptions of all registered types

================================================================================
Registered VLM Types:
================================================================================
    API Models (defined here):
        - "openai": OpenAIVLM (GPT-4o/GPT-4V)
        - "gemini": GeminiVLM (Gemini Pro Vision / 1.5)
        - "qwen": QwenVLM (Qwen-VL-Plus/Max)
        - "claude": ClaudeVLM (Claude 3 Opus/Sonnet/Haiku)
    
    Local Models (imported from VLM_local.py):
        - "clip": CLIPLocalVLM (fast, embedding-based, ~400MB)
        - "llava": LocalLLaVAVLM (detailed reasoning, GPU-intensive, ~7GB+)

================================================================================
Usage Examples:
================================================================================
    from omni_reward.VLM_utils.VLM_api import get_vlm, VLMFactory
    
    # API-based VLMs
    vlm = get_vlm("openai", api_key="sk-...")
    vlm = get_vlm("gemini", api_key="AIza...")
    vlm = get_vlm("qwen", api_key="sk-...")
    vlm = get_vlm("claude", api_key="sk-ant-...")
    
    # Using the factory directly
    vlm = VLMFactory.create("gemini", api_key="AIza...", model="gemini-1.5-pro")
    
    # Computing embeddings
    text_emb = vlm.get_text_embedding("robot picking up object")
    image_emb = vlm.get_image_embedding(observation)
    
    # Evaluating task progress
    result = vlm.evaluate_task_progress(
        image=current_frame,
        task_description="Stack the red block on the blue block",
        history_images=[prev_frame1, prev_frame2]
    )
    print(f"Progress: {result['estimated_progress']}")
    print(f"Completed: {result['completed']}")
    print(f"Explanation: {result['explanation']}")

================================================================================
Environment Variables:
================================================================================
    - OPENAI_API_KEY: OpenAI API key
    - GOOGLE_API_KEY: Google Gemini API key
    - DASHSCOPE_API_KEY: Alibaba DashScope API key (for Qwen)
    - ANTHROPIC_API_KEY: Anthropic API key (for Claude)

================================================================================
Notes:
================================================================================
    - API calls incur costs; use local models (from VLM_local.py) for dev/testing
    - Some providers don't have native embedding APIs; embeddings are computed
      indirectly via image description followed by text embedding
    - JSON parsing includes fallback handling for malformed responses
    - All VLM types share the same interface defined in VLMBase
    - For Qwen, you need to use DashScope API (China region) or compatible endpoint

================================================================================
Dependencies:
================================================================================
    - openai: pip install openai (for OpenAI)
    - google-generativeai: pip install google-generativeai (for Gemini)
    - dashscope: pip install dashscope (for Qwen)
    - anthropic: pip install anthropic (for Claude)
    - PIL: pip install Pillow
    - numpy: pip install numpy

================================================================================
Author: OmniReward Team
================================================================================
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    dashscope = None
    MultiModalConversation = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Import base class and local VLMs for factory registration
from .VLM_local import VLMBase, CLIPLocalVLM, LocalLLaVAVLM
from .templates import get_caption_template


class APIVLMBase(VLMBase):
    """Base class for API-based VLMs with common utility methods"""

    def _to_pil_image(self, image: Union[np.ndarray, "Image.Image"]) -> "Image.Image":
        """Convert numpy array to PIL Image"""
        if isinstance(image, np.ndarray):
            # Handle channel-first format (C, H, W) -> (H, W, C)
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
            return Image.fromarray(image.astype(np.uint8))
        return image

    def _image_to_base64(self, image: Union[np.ndarray, "Image.Image"]) -> str:
        """Convert image to base64 encoded string"""
        pil_image = self._to_pil_image(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _image_to_bytes(self, image: Union[np.ndarray, "Image.Image"]) -> bytes:
        """Convert image to bytes"""
        pil_image = self._to_pil_image(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from response text with fallback handling"""
        try:
            # Try to extract JSON from the response
            # Handle cases where JSON is wrapped in markdown code blocks
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())
            if isinstance(parsed, str):
                return {"caption": parsed}
            return parsed
        except json.JSONDecodeError:
            return {
                "progress": 0.0,
                "explanation": response_text,
                "completed": False,
                "raw_text": response_text,
            }

    def _get_caption_prompt(self, template: str, goal: Optional[str] = None) -> str:
        return get_caption_template(template, goal)

    def _get_progress_prompt(self, task_description: str) -> str:
        """Generate standard prompt for task progress evaluation"""
        return f"""Evaluate the progress of the following task based on the current image.

Task: {task_description}

Please provide:
1. A progress score from 0.0 to 1.0 (0 = not started, 1 = completed)
2. A brief explanation of the current state
3. Whether the task is completed (true/false)

Respond ONLY in JSON format:
{{"progress": 0.5, "explanation": "...", "completed": false}}"""


class OpenAIVLM(APIVLMBase):
    """VLM implementation using OpenAI API (GPT-4o, GPT-4V)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        vision_model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        if openai is None:
            raise ImportError("Please install openai: pip install openai")

        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.embedding_model = embedding_model
        self.vision_model = vision_model

    def get_text_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def get_image_embedding(self, image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        # OpenAI does not have direct image embedding API
        # Use vision model to describe image, then get text embedding
        base64_image = self._image_to_base64(image)
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail in one paragraph."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return self.get_text_embedding(description)

    def evaluate_task_progress(
        self,
        image: Union[np.ndarray, "Image.Image"],
        task_description: str,
        history_images: Optional[List[Union[np.ndarray, "Image.Image"]]] = None,
    ) -> Dict[str, Any]:
        base64_image = self._image_to_base64(image)
        prompt = self._get_progress_prompt(task_description)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
            max_tokens=500,
        )

        result = self._parse_json_response(response.choices[0].message.content)

        return {
            "estimated_progress": result.get("progress", 0.0),
            "explanation": result.get("explanation", ""),
            "completed": result.get("completed", False),
            "raw_response": response.choices[0].message.content,
        }

    def generate_caption(
        self,
        image: Union[np.ndarray, "Image.Image"],
        template: str = "structured_v1",
        goal: Optional[str] = None,
    ) -> str:
        base64_image = self._image_to_base64(image)
        prompt = self._get_caption_prompt(template, goal)

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=400,
        )

        parsed = self._parse_json_response(response.choices[0].message.content)
        caption = parsed.get("caption") or parsed.get("explanation") or parsed.get("raw_text")
        if not caption:
            raise RuntimeError("OpenAI caption produced no text")
        return str(caption).strip()


class GeminiVLM(APIVLMBase):
    """VLM implementation using Google Gemini API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        embedding_model: str = "models/text-embedding-004",
    ):
        if genai is None:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
        self.embedding_model = embedding_model

    def get_text_embedding(self, text: str) -> np.ndarray:
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document",
        )
        return np.array(result["embedding"], dtype=np.float32)

    def get_image_embedding(self, image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        # Gemini doesn't have direct image embedding API
        # Use vision model to describe image, then get text embedding
        pil_image = self._to_pil_image(image)
        response = self.model.generate_content(
            ["Describe this image in detail in one paragraph.", pil_image]
        )
        description = response.text
        return self.get_text_embedding(description)

    def evaluate_task_progress(
        self,
        image: Union[np.ndarray, "Image.Image"],
        task_description: str,
        history_images: Optional[List[Union[np.ndarray, "Image.Image"]]] = None,
    ) -> Dict[str, Any]:
        pil_image = self._to_pil_image(image)
        prompt = self._get_progress_prompt(task_description)

        response = self.model.generate_content([prompt, pil_image])
        result = self._parse_json_response(response.text)

        return {
            "estimated_progress": result.get("progress", 0.0),
            "explanation": result.get("explanation", ""),
            "completed": result.get("completed", False),
            "raw_response": response.text,
        }

    def generate_caption(
        self,
        image: Union[np.ndarray, "Image.Image"],
        template: str = "structured_v1",
        goal: Optional[str] = None,
    ) -> str:
        pil_image = self._to_pil_image(image)
        prompt = self._get_caption_prompt(template, goal)

        response = self.model.generate_content([prompt, pil_image])
        parsed = self._parse_json_response(response.text)
        caption = parsed.get("caption") or parsed.get("explanation") or parsed.get("raw_text")
        if not caption:
            raise RuntimeError("Gemini caption produced no text")
        return str(caption).strip()


class QwenVLM(APIVLMBase):
    """VLM implementation using Alibaba Qwen API (via DashScope)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus",
        embedding_model: str = "text-embedding-v2",
    ):
        if dashscope is None:
            raise ImportError("Please install dashscope: pip install dashscope")

        dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.embedding_model = embedding_model

    def get_text_embedding(self, text: str) -> np.ndarray:
        from dashscope import TextEmbedding

        response = TextEmbedding.call(
            model=self.embedding_model,
            input=text,
        )
        if response.status_code == 200:
            return np.array(response.output["embeddings"][0]["embedding"], dtype=np.float32)
        else:
            raise RuntimeError(f"Qwen embedding failed: {response.message}")

    def get_image_embedding(self, image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        # Use vision model to describe image, then get text embedding
        pil_image = self._to_pil_image(image)

        # Convert image to base64 for DashScope
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{image_data}"},
                    {"text": "Describe this image in detail in one paragraph."},
                ],
            }
        ]

        response = MultiModalConversation.call(
            model=self.model,
            messages=messages,
        )

        if response.status_code == 200:
            description = response.output.choices[0].message.content[0]["text"]
            return self.get_text_embedding(description)
        else:
            raise RuntimeError(f"Qwen vision call failed: {response.message}")

    def evaluate_task_progress(
        self,
        image: Union[np.ndarray, "Image.Image"],
        task_description: str,
        history_images: Optional[List[Union[np.ndarray, "Image.Image"]]] = None,
    ) -> Dict[str, Any]:
        pil_image = self._to_pil_image(image)
        prompt = self._get_progress_prompt(task_description)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{image_data}"},
                    {"text": prompt},
                ],
            }
        ]

        response = MultiModalConversation.call(
            model=self.model,
            messages=messages,
        )

        if response.status_code == 200:
            response_text = response.output.choices[0].message.content[0]["text"]
            result = self._parse_json_response(response_text)
            return {
                "estimated_progress": result.get("progress", 0.0),
                "explanation": result.get("explanation", ""),
                "completed": result.get("completed", False),
                "raw_response": response_text,
            }
        else:
            raise RuntimeError(f"Qwen vision call failed: {response.message}")

    def generate_caption(
        self,
        image: Union[np.ndarray, "Image.Image"],
        template: str = "structured_v1",
        goal: Optional[str] = None,
    ) -> str:
        pil_image = self._to_pil_image(image)
        prompt = self._get_caption_prompt(template, goal)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{image_data}"},
                    {"text": prompt},
                ],
            }
        ]

        response = MultiModalConversation.call(
            model=self.model,
            messages=messages,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Qwen vision call failed: {response.message}")

        response_text = response.output.choices[0].message.content[0]["text"]
        parsed = self._parse_json_response(response_text)
        caption = parsed.get("caption") or parsed.get("explanation") or parsed.get("raw_text")
        if not caption:
            raise RuntimeError("Qwen caption produced no text")
        return str(caption).strip()


class ClaudeVLM(APIVLMBase):
    """VLM implementation using Anthropic Claude API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize Claude VLM.

        Note: Claude doesn't have a native embedding API, so we use OpenAI's
        embedding API as a fallback. You can provide a separate OpenAI API key
        for embeddings, or it will use OPENAI_API_KEY environment variable.
        """
        if anthropic is None:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model

        # Use OpenAI for embeddings since Claude doesn't have embedding API
        self.embedding_model = embedding_model
        self._openai_client = None
        if openai is not None:
            openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if openai_key:
                self._openai_client = openai.OpenAI(api_key=openai_key)

    def get_text_embedding(self, text: str) -> np.ndarray:
        if self._openai_client is None:
            raise RuntimeError(
                "Claude doesn't have native embedding API. "
                "Please provide OpenAI API key for embeddings."
            )
        response = self._openai_client.embeddings.create(
            input=text,
            model=self.embedding_model,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def get_image_embedding(self, image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        # Use Claude to describe image, then get text embedding via OpenAI
        base64_image = self._image_to_base64(image)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in detail in one paragraph.",
                        },
                    ],
                }
            ],
        )

        description = response.content[0].text
        return self.get_text_embedding(description)

    def evaluate_task_progress(
        self,
        image: Union[np.ndarray, "Image.Image"],
        task_description: str,
        history_images: Optional[List[Union[np.ndarray, "Image.Image"]]] = None,
    ) -> Dict[str, Any]:
        base64_image = self._image_to_base64(image)
        prompt = self._get_progress_prompt(task_description)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        response_text = response.content[0].text
        result = self._parse_json_response(response_text)

        return {
            "estimated_progress": result.get("progress", 0.0),
            "explanation": result.get("explanation", ""),
            "completed": result.get("completed", False),
            "raw_response": response_text,
        }

    def generate_caption(
        self,
        image: Union[np.ndarray, "Image.Image"],
        template: str = "structured_v1",
        goal: Optional[str] = None,
    ) -> str:
        base64_image = self._image_to_base64(image)
        prompt = self._get_caption_prompt(template, goal)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        response_text = response.content[0].text
        parsed = self._parse_json_response(response_text)
        caption = parsed.get("caption") or parsed.get("explanation") or parsed.get("raw_text")
        if not caption:
            raise RuntimeError("Claude caption produced no text")
        return str(caption).strip()


class VLMFactory:
    """
    Factory class for creating different VLM instances.

    Provides a unified interface for instantiating any registered VLM type
    (both API-based and local) with a single method call.
    """

    _registry: Dict[str, type] = {
        # Local models (from VLM_local.py)
        "clip": CLIPLocalVLM,
        "llava": LocalLLaVAVLM,
        # API models (defined in this file)
        "openai": OpenAIVLM,
        "gemini": GeminiVLM,
        "qwen": QwenVLM,
        "claude": ClaudeVLM,
    }

    @classmethod
    def register(cls, name: str, vlm_class: type) -> None:
        """
        Register a new VLM type.

        Args:
            name: Short name for the VLM type
            vlm_class: VLM class (must inherit from VLMBase)
        """
        cls._registry[name] = vlm_class

    @classmethod
    def create(cls, vlm_type: str, **kwargs) -> VLMBase:
        """
        Create a VLM instance.

        Args:
            vlm_type: Type of VLM to create (e.g., "clip", "openai", "gemini")
            **kwargs: Arguments to pass to the VLM constructor

        Returns:
            VLMBase: Instantiated VLM object

        Raises:
            ValueError: If vlm_type is not registered
        """
        if vlm_type not in cls._registry:
            raise ValueError(
                f"Unknown VLM type: {vlm_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )
        return cls._registry[vlm_type](**kwargs)

    @classmethod
    def available_types(cls) -> List[str]:
        """Get all available VLM types"""
        return list(cls._registry.keys())

    @classmethod
    def get_info(cls) -> Dict[str, str]:
        """Get information about all registered VLM types"""
        return {
            # Local models
            "clip": "Local CLIP model (fast, embedding-based, ~400MB)",
            "llava": "Local LLaVA model (detailed reasoning, GPU-intensive, ~7GB+)",
            # API models
            "openai": "OpenAI GPT-4o/GPT-4V (API, powerful, paid)",
            "gemini": "Google Gemini Pro Vision / 1.5 (API, free tier available)",
            "qwen": "Alibaba Qwen-VL (API via DashScope, China region)",
            "claude": "Anthropic Claude 3 Opus/Sonnet/Haiku (API, paid)",
        }


def get_vlm(vlm_type: str = "clip", **kwargs) -> VLMBase:
    """
    Convenience function to get a VLM instance.

    Args:
        vlm_type: Type of VLM to create. Options:
            Local models:
                - "clip": Local CLIP model (default, fast)
                - "llava": Local LLaVA model (GPU-intensive)
            API models:
                - "openai": OpenAI GPT-4o/GPT-4V
                - "gemini": Google Gemini
                - "qwen": Alibaba Qwen-VL
                - "claude": Anthropic Claude 3
        **kwargs: Arguments to pass to the VLM constructor

    Returns:
        VLMBase: Instantiated VLM object

    Example:
        >>> vlm = get_vlm("openai", api_key="sk-...")
        >>> embedding = vlm.get_text_embedding("hello world")
    """
    return VLMFactory.create(vlm_type, **kwargs)
