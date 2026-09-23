import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from create_pairs import create_pairs
from preprocessing import preprocess_image
from threshold import calibrate, save_config
from app.services.model_service import model_service


def main():
    roots = ",".join([
        str(ROOT / "archive" / "BHSig260-Bengali" / "BHSig260-Bengali"),
        str(ROOT / "BHSig260-Hindi" / "BHSig260-Hindi"),
        str(ROOT / "CEDAR" / "CEDAR"),
    ])
    pairs = create_pairs(roots, pairs_per_class=50, seed=43)
    model_service.load()
    distances, labels = [], []
    for left, right, label in pairs:
        images = np.asarray([preprocess_image(left), preprocess_image(right)])
        embeddings = model_service.encoder_body.predict(images, batch_size=1, verbose=0)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        distances.append(float(np.sqrt(np.sum(np.square(embeddings[0] - embeddings[1])) + np.finfo(np.float32).eps)))
        labels.append(label)
    selected = calibrate(np.asarray(distances), np.asarray(labels))
    save_config(ROOT / "backend" / "models" / "model_config.json", selected["threshold"])
    print(selected)


if __name__ == "__main__":
    main()
