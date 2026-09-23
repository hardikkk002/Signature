# Offline Signature Forgery Detection

A deployment-ready full-stack signature verification system using a TensorFlow/Keras Siamese CNN. The backend performs inference only: it loads a previously trained model and calibrated threshold once at startup. It never retrains and never stores uploaded signatures.

## Architecture

React/Vite uploads two images to FastAPI. Shared preprocessing converts each image to grayscale, resizes it to `224x224`, normalizes it to `[0,1]`, and adds a channel dimension. A shared-weight Siamese encoder creates 128-dimensional L2-normalized embeddings. Euclidean distance is compared with the validation-calibrated threshold: `distance <= threshold` is Genuine, otherwise Forged.

## Structure

```text
frontend/          React UI
backend/app/       FastAPI API, model service, preprocessing, verification
backend/models/    trained .keras artifact and model_config.json
training/          pair creation, training, calibration, evaluation
shared/            single preprocessing implementation used by training/inference
sample_data/       optional real sample images
```

## Install and run

### Backend (Python 3.10+)

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API is documented at `http://localhost:8000/docs`. Copy `.env.example` to `.env` to configure model paths, frontend origin, file size, or an optional `MODEL_THRESHOLD` override.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` if the backend is not at `http://localhost:8000`.

## Training (separate process)

Training data is expected in `training/data/<writer>/{genuine,forged}/` with PNG/JPG/JPEG files. Keep writer identities disjoint between splits. For example, use writer folders 1–44 for training and 45–49 for validation:

```bash
cd training
python train.py --data data --train-writers 1,2,3 --val-writers 4,5 --pairs-per-class 2000
```

The script uses contrastive loss, Adam at `0.0001`, checkpointing, early stopping, and CPU-compatible TensorFlow. It writes the best model to `backend/models/siamese_signature_model.keras`, a final model beside it, and writes the validation-calibrated threshold to `backend/models/model_config.json`. No threshold is selected using test data.

Evaluate unseen writers with:

```bash
python evaluate.py --data data/test --model ../backend/models/siamese_signature_model.keras --threshold <threshold-from-model_config.json>
```

Metrics and plots are written to `training/results/` including `metrics.json`, `distance_distribution.png`, `threshold_analysis.png`, and `roc_curve.png`. Pair generation avoids self-pairs, balances labels, uses a seed, and reports reductions/replacement when data cannot supply the requested count.

## API

- `GET /api/health` — reports API and model status.
- `POST /api/verify` — multipart fields `reference_signature` and `test_signature`.

Example:

```bash
curl -X POST http://localhost:8000/api/verify \
  -F reference_signature=@reference.png -F test_signature=@test.png
```

```json
{"result":"Genuine","is_genuine":true,"distance":0.42,"threshold":0.85,"similarity_score":75.3,"processing_time_ms":145.0}
```

Similarity Score is a configurable UI-oriented normalization of distance, not a probability. Invalid MIME types, extensions, corrupt images, missing files, files over 5 MB, missing models, and inference errors receive appropriate HTTP errors. Uploaded content is processed in memory.

## Troubleshooting

If health reports `model_loaded: false`, run training with real data and ensure both files exist in `backend/models/`. The application intentionally never returns fake predictions. TensorFlow can run on CPU; a GPU is optional. If CORS blocks the browser, set `FRONTEND_URL` in the backend environment.
