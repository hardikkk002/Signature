from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance

IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def preprocess_image(source, augment=False):
    # Close file-backed image handles immediately; training may preprocess
    # thousands of pairs in one process on Windows.
    with Image.open(source) as opened:
        image = opened.convert("L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    if augment:
        image = image.rotate(float(np.random.uniform(-5, 5)), fillcolor=255)
        image = ImageEnhance.Brightness(image).enhance(float(np.random.uniform(.9, 1.1)))
    return np.asarray(image, dtype=np.float32)[..., None] / 255.0

def is_image(path):
    return Path(path).suffix.lower() in ALLOWED_EXTENSIONS
