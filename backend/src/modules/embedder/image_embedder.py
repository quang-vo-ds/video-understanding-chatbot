import logging
from PIL import ImageFile, Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection


class ImageEmbedder:
    def __init__(self) -> None:
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", device_map="cuda")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map="cuda")

    @property
    def dim(self):
        return self.model.config.projection_dim

    def encode(self, inp: list[str]) -> list[list[float]]:
        images = self._get_img_from_urls(inp)

        dense_vectors = []
        for img in images:
            if not img:
                dense_vectors.append([0.0 for _ in range(self.dim)])
            else:
                inputs = self.processor(images=img, return_tensors="pt").to("cuda")
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                dense_vectors.append(image_embeds.tolist()[0])

        return dense_vectors

    @staticmethod
    def _get_img_from_urls(img_paths: list[str]) -> list[ImageFile.ImageFile]:
        images = []
        for path in img_paths:
            images.append(Image.open(path) if path else None)
        return images