from __future__ import annotations

from typing import List, Dict, Optional
import asyncio
from datetime import datetime
import joblib
import pandas as pd
import os
from loguru import logger

from app.schemas.prediction import (
    Measurement,
    BatchMeasurement,
    Prediction,
    BatchPrediction,
    BatchSummary,
    BinaryClassificationResponse,
)
from app.schemas.common import FailureType, RiskLevel
from app.schemas.model import FeatureSpec
from app.utils.config import Settings


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_loaded: bool = False
        self.version: str = "0.1.0"
        self.trained_on: Optional[str] = None
        self.threshold: float = settings.PREDICTION_THRESHOLD

        # Placeholder for actual ML artifacts (pipeline, encoder, model, etc.)
        self._model = None
        self._binary_model = None

    async def load_models(self):
        """Load the binary classification pipeline from disk"""
        try:
            # Load binary classification pipeline
            model_path = os.path.join(self.settings.MODEL_DIR, "xgboost_undersample_pipeline.pkl")
            logger.info(f"Loading binary classification model from: {model_path}")
            
            if os.path.exists(model_path):
                self._binary_model = joblib.load(model_path)

                logger.info("Binary classification pipeline loaded successfully")
            else:
                logger.warning(f"Model file not found: {model_path}")
                # Keep placeholder for development
                self._binary_model = None
                
            self.is_loaded = True
            self.trained_on = datetime.utcnow().strftime("%Y-%m-%d")
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            raise

    async def predict_binary_classification(self, m: Measurement) -> BinaryClassificationResponse:
        """Predict machine failure using binary classification model"""
        if not self.is_loaded or self._binary_model is None:
            raise ValueError("Binary classification model not loaded")
        
        # Calculate sensor_ok based on temperature values
        sensor_ok = m.temperatura_ar > 0 and m.temperatura_processo > 0
        
        # Prepare data for the model
        data = pd.DataFrame({
            'tipo': [m.tipo],
            'temperatura_ar': [m.temperatura_ar],
            'temperatura_processo': [m.temperatura_processo],
            'umidade_relativa': [m.umidade_relativa],
            'velocidade_rotacional': [m.velocidade_rotacional],
            'torque': [m.torque],
            'desgaste_da_ferramenta': [m.desgaste_da_ferramenta],
            'sensor_ok': [sensor_ok]
        })
        
        # Get prediction and probabilities
        classifier = self._binary_model.get('pipeline')
        prediction = classifier.predict(data)[0]
        probabilities = classifier.predict_proba(data)[0]

        return BinaryClassificationResponse(
            falha_maquina=bool(prediction),
            probabilidade_falha=float(probabilities[1]),  # Probability of failure (class 1)
            probabilidade_sem_falha=float(probabilities[0]),  # Probability of no failure (class 0)
            id=m.id,
            id_produto=m.id_produto
        )

    async def predict_one(self, m: Measurement) -> Prediction:
        # Simple heuristic placeholder; replace with model.predict_proba
        # Normalize features roughly and compute a toy score
        score = 0.0
        score += (m.temperatura_processo - m.temperatura_ar) / 100.0
        score += (m.torque / 100.0)
        score += (m.desgaste_da_ferramenta / 300.0)
        score = max(0.0, min(1.0, score))

        # Distribute per-failure-type probabilities (toy)
        probs: Dict[FailureType, float] = {
            FailureType.FDF: score * 0.30,
            FailureType.FDC: score * 0.20,
            FailureType.FP: score * 0.20,
            FailureType.FTE: score * 0.20,
            FailureType.FA: max(0.0, 1.0 - score) * 0.10,
        }

        most_likely = max(probs, key=probs.get)

        # Risk level
        risk = (
            RiskLevel.high if score >= 0.7 else RiskLevel.medium if score >= 0.4 else RiskLevel.low
        )

        return Prediction(
            will_fail=bool(score >= self.threshold),
            machine_failure_probability=score,
            failure_type_probs=probs,
            most_likely_failure=most_likely,
            risk_level=risk,
            id=m.id,
            id_produto=m.id_produto,
        )

    async def predict_batch(self, payload: BatchMeasurement) -> BatchPrediction:
        if len(payload.measurements) > self.settings.MAX_BATCH_SIZE:
            # Raising exceptions is handled at route level; here we ensure sane behavior too
            raise ValueError("Batch too large")

        preds: List[Prediction] = []
        for m in payload.measurements:
            preds.append(await self.predict_one(m))

        # Summary
        if preds:
            avg_prob = sum(p.machine_failure_probability for p in preds) / len(preds)
            # Aggregate most likely failure type across predictions
            agg: Dict[FailureType, float] = {}
            for p in preds:
                if p.most_likely_failure is not None:
                    agg[p.most_likely_failure] = agg.get(p.most_likely_failure, 0.0) + 1.0
            top_ft = max(agg, key=agg.get) if agg else None
        else:
            avg_prob = 0.0
            top_ft = None

        return BatchPrediction(
            predictions=preds,
            summary=BatchSummary(
                count=len(preds), avg_failure_prob=avg_prob, top_failure_type=top_ft
            ),
        )

    def get_feature_specs(self) -> list[FeatureSpec]:
        return [
            FeatureSpec(name="tipo", dtype="enum", required=True, allowed_values=["L", "M", "H"]),
            FeatureSpec(name="temperatura_ar", dtype="float", required=True, min=200, max=400),
            FeatureSpec(name="temperatura_processo", dtype="float", required=True, min=200, max=500),
            FeatureSpec(name="umidade_relativa", dtype="float", required=True, min=0, max=100),
            FeatureSpec(name="velocidade_rotacional", dtype="float", required=True, min=0, max=5000),
            FeatureSpec(name="torque", dtype="float", required=True, min=0, max=100),
            FeatureSpec(name="desgaste_da_ferramenta", dtype="float", required=True, min=0, max=10000),
        ]
