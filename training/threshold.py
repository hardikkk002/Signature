import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def metrics_at_threshold(distances, labels, threshold):
    predictions = (np.asarray(distances) <= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    genuine, forged = labels == 1, labels == 0
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(labels, predictions)), "precision": float(precision_score(labels, predictions, zero_division=0)), "recall": float(recall_score(labels, predictions, zero_division=0)), "f1": float(f1_score(labels, predictions, zero_division=0)), "far": float(np.mean(predictions[forged] == 1)) if forged.any() else 0.0, "frr": float(np.mean(predictions[genuine] == 0)) if genuine.any() else 0.0}


def calibrate(distances, labels):
    values = np.unique(np.asarray(distances))
    results = [metrics_at_threshold(distances, labels, t) for t in values]
    return max(results, key=lambda x: (x["f1"], x["accuracy"]))


def save_config(path, threshold):
    config = {"model_name": "Siamese Signature Verification CNN", "input_size": [224, 224, 1], "embedding_size": 128, "distance_metric": "euclidean", "threshold": float(threshold), "genuine_label": 1, "forged_label": 0, "preprocessing": {"grayscale": True, "resize": [224, 224], "normalize": True}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
