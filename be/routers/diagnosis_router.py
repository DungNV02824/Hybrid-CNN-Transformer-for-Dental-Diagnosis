# routers/diagnosis_router.py
"""
FastAPI router cho Dental Full Report.

Endpoints
---------
POST /api/v1/diagnosis/full-report
    Tải lên ảnh Panoramic + ảnh Cephalometric (tùy chọn).
    Trả về báo cáo tổng hợp: phân tích răng, góc SNA/SNB/ANB, tư vấn AI.
"""

import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from schemas.diagnosis_schema import FullReportResponse
from services.diagnosis_service import build_full_report

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/diagnosis",
    tags=["Full Diagnosis Report"],
)

_ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB


async def _read_validated_image(file: UploadFile, field_name: str) -> bytes:
    """Đọc và validate file ảnh tải lên; raise HTTPException nếu không hợp lệ."""
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{field_name}' phải là ảnh (JPEG/PNG/WebP/BMP). "
                f"Nhận được: {file.content_type}"
            ),
        )
    data = await file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{field_name}' vượt quá giới hạn 20 MB.",
        )
    if not data:
        raise HTTPException(status_code=422, detail=f"'{field_name}' rỗng.")
    return data


@router.post(
    "/full-report",
    response_model=FullReportResponse,
    summary="Báo cáo chẩn đoán nha khoa toàn diện",
    response_description=(
        "JSON tổng hợp gồm: summary, phân tích Panoramic, "
        "phân tích Cephalometric, tư vấn AI và chi tiết điều trị từng răng."
    ),
)
async def full_report(
    panoramic_file: UploadFile = File(
        ...,
        description="Ảnh X-quang Panoramic (JPG/PNG/WebP/BMP, tối đa 20 MB)",
    ),
    ceph_file: UploadFile = File(
        default=None,
        description="Ảnh Cephalometric / Sọ nghiêng (tùy chọn, tối đa 20 MB)",
    ),
    fdi_conf: float = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Confidence threshold cho mô hình FDI (nhận diện răng)",
    ),
    cv_conf: float = Query(
        default=0.30,
        ge=0.01,
        le=1.0,
        description="Confidence threshold cho mô hình CV (phát hiện bệnh lý)",
    ),
):
    """
    ### Quy trình xử lý

    **Ảnh Panoramic** (bắt buộc):
    1. Mô hình **FDI** (`fdi.pt`) detect từng răng → số hiệu FDI + bounding box.
    2. Mô hình **CV** (`CV.pt`) detect bệnh lý → `Cavity`, `Fillings`,
       `Impacted Tooth`, `Implant`.
    3. Ghép cặp bệnh lý vào răng (ưu tiên tâm-trong-box, dự phòng IoU ≥ 0.10).
    4. Phân loại: `NEEDS_TREATMENT` (Cavity, Impacted Tooth) vs
       `ALREADY_TREATED` (Fillings, Implant).

    **Ảnh Cephalometric** (tùy chọn):
    1. Mô hình **LRC** (`checkpoint_ep70_LRC.pth`) predict 29 điểm mốc.
    2. Tính góc **SNA**, **SNB** (công thức vector cosin tại đỉnh N),
       **ANB = SNA − SNB**.
    3. Kết luận: Class I (0–4°) / Class II Hô (> 4°) / Class III Móm (< 0°).

    **Tư vấn AI** (GPT-4o):
    - Tổng hợp findings → prompt tiếng Việt → JSON tư vấn chuyên nghiệp.
    - Yêu cầu `OPENAI_API_KEY` trong `.env`.
    """
    # --- Đọc & validate ảnh ---
    panoramic_bytes = await _read_validated_image(panoramic_file, "panoramic_file")

    ceph_bytes: bytes | None = None
    if ceph_file and ceph_file.filename:
        ceph_bytes = await _read_validated_image(ceph_file, "ceph_file")

    # --- Chạy pipeline ---
    try:
        report = await build_full_report(
            panoramic_bytes=panoramic_bytes,
            ceph_bytes=ceph_bytes,
            fdi_conf=fdi_conf,
            cv_conf=cv_conf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("Lỗi runtime trong pipeline full-report")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Lỗi không xác định trong pipeline full-report")
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {exc}")

    return report
