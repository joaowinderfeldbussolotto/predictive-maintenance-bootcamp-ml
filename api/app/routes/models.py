from fastapi import APIRouter, Request
from app.schemas.model import ModelStatus, FeatureSpec

router = APIRouter()


@router.get("/status", response_model=ModelStatus)
async def model_status(request: Request):
    ms = getattr(request.app.state, "model_service", None)
    return ModelStatus(
        loaded=bool(ms and ms.is_loaded),
        version=getattr(ms, "version", "unknown"),
        trained_on=getattr(ms, "trained_on", None),
        threshold=getattr(ms, "threshold", 0.5),
    )


@router.get("/features", response_model=list[FeatureSpec])
async def model_features(request: Request):
    ms = getattr(request.app.state, "model_service", None)
    return ms.get_feature_specs() if ms else []


@router.post("/reload")
async def model_reload(request: Request):
    ms = getattr(request.app.state, "model_service", None)
    if not ms:
        return {"reloaded": False}
    await ms.load_models()
    return {"reloaded": True, "version": ms.version}
