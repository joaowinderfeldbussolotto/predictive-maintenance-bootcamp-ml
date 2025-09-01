from fastapi import APIRouter, Request
import time

router = APIRouter()

_start_time = time.time()


@router.get("/")
async def health_root(request: Request):
    ms = getattr(request.app.state, "model_service", None)
    uptime = time.time() - _start_time
    return {
        "status": "ok",
        "uptime_s": round(uptime, 2),
        "model_loaded": bool(ms and ms.is_loaded),
        "version": "1.0.0",
    }


@router.get("/liveness")
async def liveness():
    return {"status": "alive"}


@router.get("/readiness")
async def readiness(request: Request):
    ms = getattr(request.app.state, "model_service", None)
    ready = bool(ms and ms.is_loaded)
    return {"ready": ready}
