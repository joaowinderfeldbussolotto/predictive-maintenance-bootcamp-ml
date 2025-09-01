from typing import Optional, Literal
from pydantic import BaseModel


class ModelStatus(BaseModel):
    loaded: bool
    version: str
    trained_on: Optional[str] = None
    threshold: float


class FeatureSpec(BaseModel):
    name: str
    dtype: Literal["float", "int", "string", "enum"]
    required: bool = True
    min: Optional[float] = None
    max: Optional[float] = None
    allowed_values: Optional[list[str]] = None
