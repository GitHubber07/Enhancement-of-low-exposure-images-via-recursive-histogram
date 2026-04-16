from pydantic import BaseModel
from typing import List

class MetricsData(BaseModel):
    algorithm_used: str
    original_exposure: float
    enhanced_exposure: float
    original_entropy: float
    enhanced_entropy: float

class HistogramsData(BaseModel):
    original: List[int]
    enhanced: List[int]

class EnhancerResponse(BaseModel):
    image_base64: str
    metrics: MetricsData
    histograms: HistogramsData
