from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from app.schemas.common import Tipo, FailureType, RiskLevel


class Measurement(BaseModel):
    tipo: Tipo = Field(description="Tipo da máquina: L/M/H")
    temperatura_ar: float
    temperatura_processo: float
    umidade_relativa: float
    velocidade_rotacional: float
    torque: float
    desgaste_da_ferramenta: float
    # Optional passthrough identifiers
    id: Optional[str | int] = None
    id_produto: Optional[str] = None


class BinaryClassificationResponse(BaseModel):
    """Response for binary classification endpoint"""
    falha_maquina: bool = Field(description="Predição de falha da máquina (True/False)")
    probabilidade_falha: float = Field(description="Probabilidade de falha (0.0 a 1.0)")
    probabilidade_sem_falha: float = Field(description="Probabilidade de não falha (0.0 a 1.0)")
    id: Optional[str | int] = None
    id_produto: Optional[str] = None


class BatchMeasurement(BaseModel):
    measurements: List[Measurement] = Field(min_length=1)


class Prediction(BaseModel):
    will_fail: bool
    machine_failure_probability: float
    failure_type_probs: Dict[FailureType, float]
    most_likely_failure: Optional[FailureType]
    risk_level: RiskLevel
    id: Optional[str | int] = None
    id_produto: Optional[str] = None


class BatchSummary(BaseModel):
    count: int
    avg_failure_prob: float
    top_failure_type: Optional[FailureType]


class BatchPrediction(BaseModel):
    predictions: List[Prediction]
    summary: BatchSummary
