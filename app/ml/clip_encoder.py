from PIL import Image
import numpy as np
import open_clip
import torch


class CLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained

        print(f"Loading CLIP model on device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )

        print("Model loaded")

        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("Tokenizer loaded")

        self.model.eval()

    def _normalize_embedding(self, embedding: torch.Tensor) -> np.ndarray:
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0].astype(np.float32)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embedding = self.model.encode_image(image_tensor)

        return self._normalize_embedding(embedding)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)

        with torch.inference_mode():
            embedding = self.model.encode_text(tokens)

        return self._normalize_embedding(embedding)
