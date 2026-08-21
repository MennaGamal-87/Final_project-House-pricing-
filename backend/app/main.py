import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.prediction import router as prediction_router
from app.core.config import get_settings
from app.services.inference import model_service
from app.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("app.main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup (not on every request).
    model_service.load()
    logger.info("Model loaded. %s is ready.", settings.app_name)
    yield
    logger.info("Shutting down %s.", settings.app_name)


app = FastAPI(title=settings.app_name, version=settings.api_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)


@app.get("/", tags=["root"])
def root():
    return {"message": f"{settings.app_name} v{settings.api_version} - see /docs for the API."}
