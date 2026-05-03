# routers/dental_issues_router.py
"""
FastAPI router cho endpoint kết hợp FDI + CV.

Endpoints
---------
POST /predict/dental-issues
    Tải lên ảnh X-quang răng → nhận diện từng răng (FDI) + phát hiện bệnh lý (CV)
    → trả về danh sách các răng có vấn đề cùng tọa độ và loại bệnh.
"""

import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from services.dental_issues_service import predict_dental_issues, predict_dental_stats

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dental Issues – Combined FDI + CV"])

# Các định dạng ảnh được chấp nhận
_ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}

# Giới hạn kích thước file: 20 MB
_MAX_FILE_BYTES = 20 * 1024 * 1024


@router.post(
    "/predict/dental-issues",
    summary="Nhận diện răng và bệnh lý trên ảnh X-quang",
    response_description=(
        "JSON chứa danh sách các răng có bệnh lý, "
        "kèm bounding box và độ tin cậy."
    ),
)
async def predict_dental_issues_endpoint(
    file: UploadFile = File(
        ...,
        description="Ảnh X-quang răng (JPG, PNG, WebP, BMP — tối đa 20 MB)",
    ),
    fdi_conf: float = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Ngưỡng confidence cho mô hình FDI (nhận diện răng).",
    ),
    cv_conf: float = Query(
        default=0.30,
        ge=0.01,
        le=1.0,
        description="Ngưỡng confidence cho mô hình CV (phát hiện bệnh lý).",
    ),
):
    """
    **Quy trình xử lý:**

    1. Mô hình **FDI** phát hiện các răng → bounding box + số hiệu FDI.
    2. Mô hình **CV** phát hiện bệnh lý → bounding box + loại bệnh
       (`Cavity`, `Fillings`, `Impacted Tooth`, `Implant`).
    3. Ghép cặp: tâm của box bệnh lý nằm trong box răng nào thì gán cho răng đó;
       nếu không khớp, dùng IoU làm tiêu chí dự phòng.
    4. Trả về danh sách các **răng có bệnh lý** kèm chi tiết từng bệnh.

    **Ví dụ response:**
    ```json
    {
      "status": "success",
      "data": [
        {
          "tooth_number": "46",
          "tooth_bbox": [120, 200, 310, 390],
          "issues": [
            {
              "issue_name": "Cavity",
              "confidence": 0.89,
              "issue_bbox": [145, 220, 280, 360]
            }
          ]
        }
      ],
      "summary": {
        "total_teeth_detected": 28,
        "teeth_with_issues": 3,
        "total_issues_found": 4
      }
    }
    ```
    """
    # ------------------------------------------------------------------
    # Kiểm tra định dạng file
    # ------------------------------------------------------------------
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Định dạng file '{file.content_type}' không được hỗ trợ. "
                f"Chấp nhận: {', '.join(sorted(_ALLOWED_MIME))}."
            ),
        )

    # ------------------------------------------------------------------
    # Đọc dữ liệu file
    # ------------------------------------------------------------------
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Không thể đọc file tải lên: {exc}",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File tải lên rỗng.")

    if len(image_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File quá lớn ({len(image_bytes) / 1024 / 1024:.1f} MB). "
                "Kích thước tối đa cho phép là 20 MB."
            ),
        )

    # ------------------------------------------------------------------
    # Chạy pipeline phân tích
    # ------------------------------------------------------------------
    try:
        result = predict_dental_issues(
            image_bytes=image_bytes,
            fdi_conf=fdi_conf,
            cv_conf=cv_conf,
        )
    except FileNotFoundError as exc:
        # File model không tồn tại → lỗi cấu hình server
        logger.error("Model file không tìm thấy: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi cấu hình server – model không tìm thấy: {exc}",
        )
    except ValueError as exc:
        # Ảnh không hợp lệ
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        # Lỗi inference
        logger.exception("Lỗi trong quá trình inference: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi inference: {exc}",
        )
    except Exception as exc:
        # Lỗi không mong đợi
        logger.exception("Lỗi không xác định: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Đã xảy ra lỗi không xác định trong quá trình xử lý.",
        )

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Hàm dùng chung: validate và đọc file ảnh
# ---------------------------------------------------------------------------

async def _read_image_bytes(file: UploadFile) -> bytes:
    """Validate định dạng + kích thước, đọc và trả về bytes của file ảnh."""
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Định dạng file '{file.content_type}' không được hỗ trợ. "
                f"Chấp nhận: {', '.join(sorted(_ALLOWED_MIME))}."
            ),
        )
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file tải lên: {exc}")

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="File tải lên rỗng.")
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File quá lớn ({len(data) / 1024 / 1024:.1f} MB). "
                "Kích thước tối đa cho phép là 20 MB."
            ),
        )
    return data


def _handle_service_errors(exc: Exception) -> None:
    """Chuyển đổi exception từ service layer sang HTTPException phù hợp."""
    if isinstance(exc, FileNotFoundError):
        logger.error("Model file không tìm thấy: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi cấu hình server – model không tìm thấy: {exc}")
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, RuntimeError):
        logger.exception("Lỗi inference: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi inference: {exc}")
    logger.exception("Lỗi không xác định: %s", exc)
    raise HTTPException(status_code=500, detail="Đã xảy ra lỗi không xác định trong quá trình xử lý.")


# ---------------------------------------------------------------------------
# POST /predict/dental-stats
# ---------------------------------------------------------------------------

@router.post(
    "/predict/dental-stats",
    summary="Thống kê tổng quan tình trạng sức khỏe răng miệng",
    response_description=(
        "JSON thống kê tổng số răng, số răng khỏe mạnh, "
        "số răng có vấn đề, số răng cần điều trị."
    ),
)
async def predict_dental_stats_endpoint(
    file: UploadFile = File(
        ...,
        description="Ảnh X-quang răng (JPG, PNG, WebP, BMP — tối đa 20 MB)",
    ),
    fdi_conf: float = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Ngưỡng confidence cho mô hình FDI (nhận diện răng).",
    ),
    cv_conf: float = Query(
        default=0.30,
        ge=0.01,
        le=1.0,
        description="Ngưỡng confidence cho mô hình CV (phát hiện bệnh lý).",
    ),
):
    """
    Phân tích ảnh X-quang và phân loại **toàn bộ** các răng:

    | Nhóm | Tiêu chí |
    |---|---|
    | **Khỏe mạnh** | FDI phát hiện nhưng CV không tìm thấy bệnh lý nào |
    | **Có vấn đề** | Có ít nhất 1 bệnh lý (Cavity / Fillings / Impacted Tooth / Implant) |
    | **Cần điều trị** | Có `Cavity` hoặc `Impacted Tooth` (chưa được xử lý) |

    > `Fillings` (đã trám) và `Implant` (đã cấy ghép) **không** được tính vào nhóm "cần điều trị"
    > vì đó là tình trạng đã được can thiệp.

    **Ví dụ response:**
    ```json
    {
      "status": "success",
      "statistics": {
        "total_teeth": 28,
        "healthy_teeth": 23,
        "teeth_with_issues": 5,
        "teeth_need_treatment": 3
      },
      "data": {
        "healthy": [
          { "tooth_number": "11", "tooth_bbox": [10, 20, 80, 100] }
        ],
        "with_issues": [
          {
            "tooth_number": "46",
            "tooth_bbox": [120, 200, 310, 390],
            "needs_treatment": true,
            "issues": [
              { "issue_name": "Cavity", "confidence": 0.89, "issue_bbox": [145, 220, 280, 360] }
            ]
          },
          {
            "tooth_number": "25",
            "tooth_bbox": [50, 60, 150, 180],
            "needs_treatment": false,
            "issues": [
              { "issue_name": "Fillings", "confidence": 0.92, "issue_bbox": [60, 70, 140, 170] }
            ]
          }
        ]
      }
    }
    ```
    """
    image_bytes = await _read_image_bytes(file)

    try:
        result = predict_dental_stats(
            image_bytes=image_bytes,
            fdi_conf=fdi_conf,
            cv_conf=cv_conf,
        )
    except Exception as exc:
        _handle_service_errors(exc)

    return JSONResponse(content=result)

