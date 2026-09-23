from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

class VerificationResponse(BaseModel):
    result: str
    is_genuine: bool
    distance: float
    threshold: float
    similarity_score: float
    processing_time_ms: float
