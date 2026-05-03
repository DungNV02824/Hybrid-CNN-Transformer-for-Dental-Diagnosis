from fastapi import APIRouter, File, Query, UploadFile, HTTPException
from fastapi.responses import Response

from services.fdi_service import predict, predict_image

router = APIRouter(prefix="/fdi", tags=["FDI"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/predict")
async def predict_tooth(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """
    Upload a dental X-ray or photo and detect teeth using the FDI model.

    - **file**: image file (JPEG, PNG, WebP, BMP) — max 10 MB
    - **conf**: minimum confidence threshold (0.01 – 1.0, default 0.25)
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10 MB.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict(image_bytes, conf_threshold=conf)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}",
        )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        **result,
    }


@router.post(
    "/predict/image",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}, "description": "Annotated image with detection boxes"}},
)
async def predict_tooth_image(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """
    Upload a dental image and receive back a JPEG with bounding boxes drawn
    around each detected tooth (FDI notation labels + confidence score).

    - **file**: image file (JPEG, PNG, WebP, BMP) — max 10 MB
    - **conf**: minimum confidence threshold (0.01 – 1.0, default 0.25)
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result_bytes = predict_image(image_bytes, conf_threshold=conf)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return Response(
        content=result_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="result_{file.filename}"'},
    )

