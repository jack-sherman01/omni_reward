import torch
from PIL import Image
from torchvision import transforms

class SimpleCaptioner:
    # Wrapper for any VLM capable of image captioning
    # Replace the mock caption() with a real VLM call.
    def __init__(self, model=None, processor=None, device="cuda"):
        self.model = model 
        self.processor = processor
        self.device = device

        self.to_pil = transforms.ToPILImage()

    # image_tensor: (C,H,W) uint8
    # return: str caption
    def caption(self, image_tensor):
        # Real version would call VLM here. For structure:
        pil = self.to_pil(image_tensor)

        # Mock caption as an example:
        return "A robot arm grasping a red block"

        # Example with basic VLM:
        # inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        # out = self.model.generate(**inputs)
        # return self.processor.decode(out[0], skip_special_tokens=True)
