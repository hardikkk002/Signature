from io import BytesIO
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.config import settings
from app.schemas import VerificationResponse
from app.services.model_service import model_service
from app.services.preprocessing import is_image
from app.services.verification import verify_signatures
from app.utils.logger import logger

router = APIRouter(prefix="/api")
MAX_BYTES = settings.max_file_size_mb * 1024 * 1024

@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_service.loaded, "model_name": "Siamese Signature Verification CNN"}

async def read_image(upload: UploadFile):
    if not upload.filename or not is_image(upload.filename) or (upload.content_type or "").lower() not in {"image/png", "image/jpeg"}:
        raise HTTPException(400, "Unsupported image. Please upload a PNG, JPG, or JPEG file.")
    content = await upload.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES: raise HTTPException(413, f"File size is too large. Maximum allowed size is {settings.max_file_size_mb} MB.")
    if not content: raise HTTPException(400, "Uploaded image is empty.")
    return BytesIO(content)

@router.post("/verify", response_model=VerificationResponse)
async def verify(reference_signature: UploadFile | None = File(None), test_signature: UploadFile | None = File(None)):
    if reference_signature is None or test_signature is None: raise HTTPException(400, "Please upload two valid signature images.")
    if not model_service.loaded: raise HTTPException(500, "Model not loaded. Please place the trained model in backend/models/.")
    try:
        reference, test = await read_image(reference_signature), await read_image(test_signature)
        return verify_signatures(reference, test)
    except HTTPException: raise
    except (OSError, ValueError) as exc:
        logger.warning("Invalid image: %s", exc)
        raise HTTPException(400, "The uploaded file is not a valid image.")
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(500, "Signature verification failed. Please try again.")
