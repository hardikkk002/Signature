from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import settings
from app.services.model_service import model_service
from app.utils.logger import logger

@asynccontextmanager
async def lifespan(app):
    try: model_service.load()
    except Exception as exc: logger.error("Model unavailable: %s", exc)
    yield

app = FastAPI(title="Offline Signature Forgery Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_url], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
