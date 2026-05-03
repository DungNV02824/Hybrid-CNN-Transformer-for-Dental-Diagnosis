# schemas/dental_cv_schema.py
"""
Pydantic models for the /predict-dental endpoint.
"""

from typing import List, Tuple

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detected dental pathology."""

    label: str = Field(
        ...,
        description="Vietnamese display name of the detected condition",
        examples=["Sâu răng"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0–1)",
        examples=[0.92],
    )
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [x1, y1, x2, y2] in pixels",
        examples=[[120.5, 45.0, 300.0, 210.0]],
    )


class DentalCVResponse(BaseModel):
    """Response payload returned by POST /predict-dental."""

    detections: List[Detection] = Field(
        default_factory=list,
        description="List of detected dental pathologies (sorted by confidence desc)",
    )
    total_objects: int = Field(
        ...,
        ge=0,
        description="Total number of detections returned",
    )
    image_result: str = Field(
        ...,
        description="Base64-encoded PNG of the original image with bounding boxes drawn",
    )
    inference_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time taken for model inference (milliseconds)",
    )


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str = Field(default="ok", description="Server status")
    model_loaded: bool = Field(
        ..., description="Whether the Dental CV model singleton is already initialised"
    )
    device: str = Field(..., description="Torch device used for inference (cpu / cuda)")
