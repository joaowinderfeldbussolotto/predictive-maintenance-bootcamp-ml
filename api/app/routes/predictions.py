from fastapi import APIRouter, Request, HTTPException
from typing import List

from app.schemas.prediction import (
    Measurement,
    BatchMeasurement,
    Prediction,
    BatchPrediction,
    BinaryClassificationResponse,
)
from app.utils.config import settings

router = APIRouter()


@router.post("/binary-classification", response_model=BinaryClassificationResponse)
async def predict_binary_classification(measurement: Measurement, request: Request):
    """
    Endpoint para classificação binária de falha de máquina.
    
    Carrega o pipeline binary_classification.pkl e retorna:
    - falha_maquina: True/False
    - probabilidade_falha: 0.0 a 1.0
    """
    ms = getattr(request.app.state, "model_service", None)
    if not ms or not ms.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = await ms.predict_binary_classification(measurement)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/predict", response_model=Prediction)
async def predict(measurement: Measurement, request: Request):
    """
    Realiza a predição de falha de máquina e tipos de falha (multi-label).
    """
    ms = getattr(request.app.state, "model_service", None)
    if not ms or not ms.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = await ms.predict_one(measurement)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/predict/batch")
async def predict_batch(payload: BatchMeasurement, request: Request):
    """
    Endpoint batch multi-label em construção.
    """
    return {
        "message": "Endpoint batch em construção",
        "detail": "Este endpoint está sendo desenvolvido.",
        "status": "under_construction",
        "received_measurements": len(payload.measurements)
    }

@router.get("/example")
async def example_payload():
    return {
        "tipo": "M",
        "temperatura_ar": 298.1,
        "temperatura_processo": 308.6,
        "umidade_relativa": 65.0,
        "velocidade_rotacional": 1551.0,
        "torque": 42.8,
        "desgaste_da_ferramenta": 108.0,
        "id": "sample-1",
        "id_produto": "M-1001",
    }
