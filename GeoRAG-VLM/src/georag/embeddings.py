"""Optional neural encoders; imports and downloads happen only on construction."""


class MiniLMEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def texts(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class CLIPEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def _encode(self, inputs, method):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            features = method(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy()

    def texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        return self._encode(inputs, self.model.get_text_features)

    def images(self, paths):
        from PIL import Image
        images = []
        for path in paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        return self._encode(self.processor(images=images, return_tensors="pt"), self.model.get_image_features)
