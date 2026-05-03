# routers/dental_cv_router.py
"""
FastAPI router for the Dental CV pathology detection endpoint.

Endpoints
---------
POST /predict-dental
    Upload a dental X-ray (JPG / PNG) and receive detected pathologies
    with bounding boxes drawn on the annotated image.

GET  /health
    Lightweight liveness check; also reports model load state and device.
"""

import logging

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from models import cv_model as _cv_model_module
from schemas.cv_schema import DentalCVResponse, HealthResponse
from services.cv_service import predict_dental, predict_dental_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["CV"])

_ALLOWED_MIME   = {"image/jpeg", "image/png", "image/jpg"}
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# POST /predict-dental
# ---------------------------------------------------------------------------

@router.post(
    "/predict-dental",
    response_model=DentalCVResponse,
    summary="Detect dental pathologies in an X-ray image",
    response_description=(
        "JSON containing detected pathologies (label, confidence, bbox), "
        "total object count, inference time, and a base64 annotated PNG."
    ),
)
async def predict_dental_endpoint(
    file: UploadFile = File(
        ...,
        description="Dental X-ray image (JPG or PNG, max 20 MB)",
    ),
    conf: float = Query(
        default=0.3,
        ge=0.01,
        le=1.0,
        description=(
            "Minimum confidence threshold for detections (0.01–1.0). "
            "Lower values return more (possibly false-positive) detections."
        ),
    ),
):
    """
    Upload a **dental X-ray** (JPG or PNG) and receive detected pathologies:

    | Label (VI)           | Label (EN)      |
    |----------------------|-----------------|
    | Sâu răng             | Cavity          |
    | Trám răng            | Fillings        |
    | Răng mọc ngầm        | Impacted Tooth  |
    | Cấy ghép implant     | Implant         |

    **Response fields**
    - `detections` – sorted by confidence (highest first)
    - `total_objects` – count of returned detections
    - `image_result` – base64 PNG with bounding boxes
    - `inference_ms` – model inference wall-clock time
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted: {', '.join(sorted(_ALLOWED_MIME))}"
            ),
        )

    image_bytes: bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_FILE_BYTES // (1024 * 1024)} MB.",
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    try:
        result = predict_dental(image_bytes, conf_threshold=conf)
    except FileNotFoundError as exc:
        logger.error("Model file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        logger.warning("Invalid image from client: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Inference runtime error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error during dental CV inference: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /predict-dental/image
# ---------------------------------------------------------------------------

@router.post(
    "/predict-dental/image",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Annotated JPEG image with pathology bounding boxes",
        }
    },
    summary="Detect dental pathologies and return annotated image",
)
async def predict_dental_image_endpoint(
    file: UploadFile = File(
        ...,
        description="Dental X-ray image (JPG or PNG, max 20 MB)",
    ),
    conf: float = Query(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold for detections (0.01–1.0)",
    ),
):
    """
    Upload a **dental X-ray** (JPG or PNG) and receive back a **JPEG image**
    with bounding boxes drawn around detected pathologies:

    | Label (VI)           | Label (EN)      |
    |----------------------|-----------------|
    | Sâu răng             | Cavity          |
    | Trám răng            | Fillings        |
    | Răng mọc ngầm        | Impacted Tooth  |
    | Cấy ghép implant     | Implant         |

    - **file**: image file (JPG or PNG) — max 20 MB
    - **conf**: minimum confidence threshold (0.01 – 1.0, default 0.3)
    """
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted: {', '.join(sorted(_ALLOWED_MIME))}"
            ),
        )

    image_bytes: bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_FILE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        result_bytes = predict_dental_image(image_bytes, conf_threshold=conf)
    except FileNotFoundError as exc:
        logger.error("Model file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        logger.warning("Invalid image from client: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Inference runtime error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error during dental CV image inference: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    return Response(
        content=result_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="result_{file.filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness check",
    tags=["Health"],
)
def health_check():
    """
    Returns server status and whether the Dental CV model is already loaded
    (avoids first-request latency if you warm up the model on startup).
    """
    import torch  # already imported above, but explicit here for clarity

    model_loaded = _cv_model_module._model is not None
    device_str   = str(_cv_model_module.DEVICE)

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device=device_str,
    )
