# schemas/predict_schema.py
from pydantic import BaseModel
from typing import List

class Point(BaseModel):
    x: float
    y: float

class PredictResponse(BaseModel):
    filename: str
    points: List[Point]
    status: str