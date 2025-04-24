from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class FrameCaptioner:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", device_map="cuda")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", device_map="cuda")

    def invoke(self, frame_paths: str):
        results = []
        for path, ts in frame_paths:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to("cuda")
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            results.append({
                "frame_path": path,
                "timestamp": ts,
                "caption": caption
            })
        return results