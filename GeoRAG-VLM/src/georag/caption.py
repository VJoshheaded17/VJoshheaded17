"""Kosmos-2 image inference; prompts and raw outputs are retained for auditing."""
from .prompts import caption_prompt
from .data import validate_metadata


class KosmosCaptioner:
    def __init__(self, model_name="microsoft/kosmos-2-patch14-224", device=None, seed=17):
        import torch
        from transformers import AutoProcessor, Kosmos2ForConditionalGeneration, set_seed
        set_seed(seed)
        self.torch, self.model_name = torch, model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Kosmos2ForConditionalGeneration.from_pretrained(model_name).to(self.device).eval()
        self.seed = seed

    def describe(self, image_path, metadata, mode="metadata", question="Describe the visible land-cover features.",
                 max_new_tokens=128, num_beams=5):
        from PIL import Image
        validate_metadata(metadata)
        prompt = caption_prompt(metadata, question, mode)
        with Image.open(image_path) as image:
            inputs = self.processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                            num_beams=num_beams, do_sample=False)
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        caption, entities = self.processor.post_process_generation(raw)
        # Generation may include the text prompt; remove only an exact matching prefix.
        if caption.startswith(prompt):
            caption = caption[len(prompt):].strip()
        return {"caption": caption.strip(), "raw_output": raw, "entities": entities,
                "prompt": prompt, "caption_source": self.model_name, "verified": False,
                "config": {"mode": mode, "seed": self.seed, "max_new_tokens": max_new_tokens,
                           "num_beams": num_beams, "do_sample": False, "device": self.device}}
