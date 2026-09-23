import time
import numpy as np
from app.services.model_service import model_service
from app.services.preprocessing import preprocess_image

def verify_signatures(reference_image, test_image):
    started = time.perf_counter()
    a = np.expand_dims(preprocess_image(reference_image), 0)
    b = np.expand_dims(preprocess_image(test_image), 0)
    distance = model_service.predict_distance(a, b)
    threshold = model_service.threshold
    similarity = max(0.0, min(100.0, (1 - distance / max(threshold * 2, 1e-7)) * 100))
    genuine = distance <= threshold
    return {"result": "Genuine" if genuine else "Forged", "is_genuine": genuine, "distance": distance, "threshold": threshold, "similarity_score": similarity, "processing_time_ms": (time.perf_counter() - started) * 1000}
