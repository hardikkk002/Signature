from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/ directory locally and /var/task on Vercel
ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_path: str = "models/siamese_signature_model.keras"
    model_config_path: str = "models/model_config.json"
    model_threshold: float | None = None

    frontend_url: str = "http://localhost:5173"
    max_file_size_mb: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    def resolve(self, value):
        path = Path(value)

        if path.is_absolute():
            return path

        return ROOT / value


settings = Settings()
