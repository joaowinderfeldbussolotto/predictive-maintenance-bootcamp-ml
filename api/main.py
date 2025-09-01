"""
Main FastAPI application for Predictive Maintenance API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from app.routes import health, predictions, models
from app.services.model_service import ModelService
from app.utils.config import settings


# Global model service instance
model_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global model_service
    logger.info("Starting Predictive Maintenance API...")

    # Startup: load models
    model_service = ModelService(settings=settings)
    await model_service.load_models()
    app.state.model_service = model_service
    logger.info("Models loaded successfully")

    yield

    # Shutdown
    logger.info("Shutting down Predictive Maintenance API...")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predictive Maintenance System API\n\n"
        "Features:\n"
        "- Failure Prediction: predict if a machine will fail\n"
        "- Failure Type Classification: probabilities for FDF, FDC, FP, FTE, FA\n"
        "- Probability Scores and Batch Processing\n"
        "- Health Monitoring and Model status\n\n"
        "Machine Types: L (Low), M (Medium), H (High)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(models.router, prefix="/models", tags=["Models"])


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to docs"""
    return RedirectResponse(url="/docs")


@app.get("/info")
async def info():
    """Get API information"""
    return {
        "name": "Predictive Maintenance API",
        "version": "1.0.0",
        "description": "Machine learning-based predictive maintenance system",
        "docs_url": "/docs",
        "health_check": "/health",
        "prediction_endpoint": "/predictions/predict",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )
