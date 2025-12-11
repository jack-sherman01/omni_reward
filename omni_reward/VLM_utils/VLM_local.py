"""
VLM_local.py - Local Vision-Language Model Implementations

Description:
    This module provides local (on-device) implementations of Vision-Language Models
    for computing text embeddings, image embeddings, and evaluating task progress.
    These models run entirely on the local machine without requiring API calls.

Features:
    - Abstract base class (VLMBase) defining the interface for all VLM implementations
    - CLIP-based local VLM for fast embedding computation
    - LLaVA-based local VLM for more sophisticated task evaluation
    - Automatic device selection (CPU/CUDA)
    - Support for both numpy arrays and PIL images as input
    - Channel-first (C, H, W) to channel-last (H, W, C) automatic conversion

Functions/Classes:
    - VLMBase: Abstract base class defining the VLM interface
        - get_text_embedding(text) -> np.ndarray
        - get_image_embedding(image) -> np.ndarray
        - evaluate_task_progress(image, task_description, history_images) -> Dict
    
    - CLIPLocalVLM: CLIP model implementation for embedding computation
        - Lightweight and fast
        - Good for similarity-based reward computation
        - Uses cosine similarity for progress evaluation
    
    - LocalLLaVAVLM: LLaVA model implementation for visual reasoning
        - More sophisticated language understanding
        - Better for complex task evaluation
        - Uses CLIP internally for embeddings

Usage:
    from omni_reward.VLM_utils.VLM_local import CLIPLocalVLM, LocalLLaVAVLM
    
    # Using CLIP for fast embeddings
    clip_vlm = CLIPLocalVLM(model_name="openai/clip-vit-base-patch32", device="cuda")
    text_emb = clip_vlm.get_text_embedding("a robot arm picking up a red cube")
    image_emb = clip_vlm.get_image_embedding(observation_image)
    similarity = np.dot(text_emb, image_emb)
    
    # Using LLaVA for detailed task evaluation
    llava_vlm = LocalLLaVAVLM(model_name="llava-hf/llava-1.5-7b-hf", device="cuda")
    result = llava_vlm.evaluate_task_progress(
        image=current_observation,
        task_description="Pick up the red cube and place it on the blue plate"
    )
    print(f"Progress: {result['estimated_progress']}")

Notes:
    - Requires torch and transformers packages: pip install torch transformers
    - CLIP models are relatively small (~400MB) and fast
    - LLaVA models are larger (~7GB+) and require more GPU memory
    - For LLaVA, fp16 is used on CUDA for memory efficiency
    - Image inputs can be numpy arrays (H,W,C or C,H,W) or PIL Images
    - All embeddings are L2-normalized for cosine similarity computation

Author: OmniReward Team
"""

import abc
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    torch = None
    CLIPModel = None
    CLIPProcessor = None

from omni_reward.VLM_utils.templates import get_caption_template


class VLMBase(abc.ABC):
    """
    Abstract base class for Vision-Language Models.
    
    This class defines the interface that all VLM implementations must follow.
    Subclasses must implement get_text_embedding, get_image_embedding, and
    evaluate_task_progress methods.
    
    Methods:
        get_text_embedding(text): Compute normalized text embedding vector
        get_image_embedding(image): Compute normalized image embedding vector
        evaluate_task_progress(image, task_description, history_images): 
            Evaluate task completion progress
    
    Note:
        - All embeddings should be L2-normalized for cosine similarity computation
        - Image inputs should support both numpy arrays and PIL Images
    """

    @abc.abstractmethod
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Compute text embedding"""
        pass

    @abc.abstractmethod
    def get_image_embedding(self, image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        """Compute image embedding"""
        pass

    @abc.abstractmethod
    def evaluate_task_progress(
        self,
        image: Union[np.ndarray, "Image.Image"],
        task_description: str,
        history_images: Optional[List[Union[np.ndarray, "Image.Image"]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate task progress based on current image and task description"""
        pass

    @abc.abstractmethod
    def generate_caption(
        self,
        image: Union[np.ndarray, "Image.Image"],
        template: str = "structured_v1",
        goal: Optional[str] = None,
    ) -> str:
        """Generate a textual caption following the requested template."""
        pass


class CLIPLocalVLM(VLMBase):
    """
    VLM implementation using local CLIP model.
    
    CLIP (Contrastive Language-Image Pre-training) provides fast and efficient
    text-image embedding computation suitable for similarity-based rewards.
    
    Args:
        model_name (str): Hugging Face model name. Default: "openai/clip-vit-base-patch32"
            Available models: "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"
        device (str): Device to run model on. "auto", "cuda", or "cpu". Default: "auto"
    
    Requirements:
        pip install torch transformers pillow
    
    Model Info:
        - Size: ~400MB (ViT-B/32), ~600MB (ViT-B/16), ~1.7GB (ViT-L/14)
        - First run will download model from Hugging Face Hub
        - Embedding dimension: 512 (ViT-B) or 768 (ViT-L)
    
    Example:
        >>> vlm = CLIPLocalVLM(device="cuda")
        >>> text_emb = vlm.get_text_embedding("a red cube on a table")
        >>> image_emb = vlm.get_image_embedding(observation)
        >>> similarity = np.dot(text_emb, image_emb)  # cosine similarity
    
    Note:
        - Embeddings are L2-normalized, so dot product equals cosine similarity
        - Supports channel-first (C,H,W) and channel-last (H,W,C) image formats
        - GPU recommended for faster inference but CPU works fine
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        if torch is None or CLIPModel is None:
            raise ImportError("Please install torch and transformers: pip install torch transformers")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def _to_pil_image(self, image: Union[np.ndarray, "Image.Image"]) -> "Image.Image":
        if isinstance(image, np.ndarray):
            # Handle channel-first format (C, H, W) -> (H, W, C)
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
            return Image.fromarray(image.astype(np.uint8))
        return image

    def get_text_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten()

    def get_image_embedding(self, image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        pil_image = self._to_pil_image(image)
        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten()

    def evaluate_task_progress(
        self,
        image: Union[np.ndarray, "Image.Image"],
        task_description: str,
        history_images: Optional[List[Union[np.ndarray, "Image.Image"]]] = None,
    ) -> Dict[str, Any]:
        image_emb = self.get_image_embedding(image)
        task_emb = self.get_text_embedding(task_description)

        # Compute similarity between current image and task description
        current_similarity = float(np.dot(image_emb, task_emb))

        progress_delta = 0.0
        if history_images and len(history_images) > 0:
            prev_image = history_images[-1]
            prev_emb = self.get_image_embedding(prev_image)
            prev_similarity = float(np.dot(prev_emb, task_emb))
            progress_delta = current_similarity - prev_similarity

        return {
            "similarity": current_similarity,
            "progress_delta": progress_delta,
            "estimated_progress": (current_similarity + 1) / 2,  # Normalize to [0, 1]
        }

    def generate_caption(
        self,
        image: Union[np.ndarray, "Image.Image"],
        template: str = "structured_v1",
        goal: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "CLIPLocalVLM cannot produce textual captions. Use a generative VLM such as LLaVA."
        )


class LocalLLaVAVLM(VLMBase):
    """
    VLM implementation using local LLaVA model.
    
    LLaVA (Large Language and Vision Assistant) provides sophisticated visual
    reasoning capabilities for detailed task evaluation. Uses CLIP internally
    for embedding computation.
    
    Args:
        model_name (str): Hugging Face model name. Default: "llava-hf/llava-1.5-7b-hf"
            Available: "llava-hf/llava-1.5-13b-hf", "llava-hf/llava-v1.6-mistral-7b-hf"
        device (str): Device to run model on. "auto", "cuda", or "cpu". Default: "auto"
        clip_model (str): CLIP model for embeddings. Default: "openai/clip-vit-base-patch32"
    
    Requirements:
        pip install torch transformers pillow accelerate
    
    Model Info:
        - Size: ~7GB (7B model), ~13GB (13B model)
        - Requires significant GPU memory (16GB+ recommended for 7B)
        - Uses fp16 on CUDA for memory efficiency
        - First run will download model from Hugging Face Hub
    
    Example:
        >>> vlm = LocalLLaVAVLM(device="cuda")
        >>> result = vlm.evaluate_task_progress(
        ...     image=observation,
        ...     task_description="Pick up the red block"
        ... )
        >>> print(f"Progress: {result['estimated_progress']}")
    
    Note:
        - GPU with 16GB+ VRAM strongly recommended
        - CPU inference is very slow (minutes per evaluation)
        - Embeddings are computed via internal CLIP model
        - Task evaluation uses LLaVA's language generation capability
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "auto",
        clip_model: str = "openai/clip-vit-base-patch32",
    ):
        if torch is None:
            raise ImportError("Please install torch and transformers: pip install torch transformers")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Use CLIP for embeddings
        self.clip_vlm = CLIPLocalVLM(model_name=clip_model, device=self.device)

        # LLaVA for task evaluation
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _to_pil_image(self, image: Union[np.ndarray, "Image.Image"]) -> "Image.Image":
        if isinstance(image, np.ndarray):
            # Handle channel-first format (C, H, W) -> (H, W, C)
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
            return Image.fromarray(image.astype(np.uint8))
        return image

    def get_text_embedding(self, text: str) -> np.ndarray:
        return self.clip_vlm.get_text_embedding(text)

    def get_image_embedding(self, image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        return self.clip_vlm.get_image_embedding(image)

    def evaluate_task_progress(
        self,
        image: Union[np.ndarray, "Image.Image"],
        task_description: str,
        history_images: Optional[List[Union[np.ndarray, "Image.Image"]]] = None,
    ) -> Dict[str, Any]:
        pil_image = self._to_pil_image(image)

        prompt = f"""<image>
Evaluate the progress of this task: {task_description}

Provide a progress score from 0.0 to 1.0 and explain the current state.
Format: Progress: X.X | Explanation: ...
"""

        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)

        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Parse response to extract progress score
        progress = 0.0
        try:
            if "Progress:" in response:
                progress_str = response.split("Progress:")[1].split("|")[0].strip()
                progress = float(progress_str)
        except (ValueError, IndexError):
            pass

        return {
            "estimated_progress": progress,
            "explanation": response,
            "completed": progress >= 0.95,
        }

    def generate_caption(
        self,
        image: Union[np.ndarray, "Image.Image"],
        template: str = "structured_v1",
        goal: Optional[str] = None,
    ) -> str:
        pil_image = self._to_pil_image(image)
        prompt = get_caption_template(template, goal)
        formatted_prompt = f"<image>\n{prompt}\n"

        inputs = self.processor(text=formatted_prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)

        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        try:
            data = json.loads(response)
            if isinstance(data, dict) and "caption" in data:
                return str(data["caption"])
        except json.JSONDecodeError:
            pass

        return response.strip()
