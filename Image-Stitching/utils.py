"""Portfolio image I/O. Geometry remains entirely in stitching.py."""
from pathlib import Path
import torch
from PIL import Image


def read_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Input folder does not exist: {directory}")
    images = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}:
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                # Avoid torchvision binary/version dependencies in the portfolio runner.
                storage = bytearray(rgb.tobytes())
                images[path.name] = torch.frombuffer(storage, dtype=torch.uint8).reshape(rgb.height, rgb.width, 3).permute(2, 0, 1).clone()
    if not images:
        raise ValueError(f"No supported images found in {directory}")
    return images


def write_image(image, output_path):
    if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
        raise ValueError("Output must be an RGB uint8 tensor")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[-2:]
    data = bytes(image.cpu().permute(1, 2, 0).contiguous().flatten().tolist())
    Image.frombytes('RGB', (width, height), data).save(path)
