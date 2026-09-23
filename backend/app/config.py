from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_path: str = "backend/models/siamese_signature_model.keras"
    model_config_path: str = "backend/models/model_config.json"
    model_threshold: float | None = None
    frontend_url: str = "http://localhost:5173"
    max_file_size_mb: int = 5
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / value

settings = Settings()
