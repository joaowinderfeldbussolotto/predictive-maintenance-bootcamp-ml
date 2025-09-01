from enum import Enum


class Tipo(str, Enum):
    L = "L"
    M = "M"
    H = "H"


class FailureType(str, Enum):
    FDF = "FDF"
    FDC = "FDC"
    FP = "FP"
    FTE = "FTE"
    FA = "FA"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
