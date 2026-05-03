from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from schemas.lrc_schema import PredictResponse, Point
# Import toàn bộ 3 hàm từ service
from services.lrc_services import process_and_predict, process_and_draw, process_and_draw_analysis, process_and_get_analysis_data, get_ceph_ai_analysis_from_image

router = APIRouter(
    prefix="/api",
    tags=["LRC"]
)

@router.post("/predict_landmarks", response_model=PredictResponse, summary="Trả về tọa độ JSON 29 điểm")
async def predict_landmarks(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPG, PNG)")
    
    try:
        image_bytes = await file.read()
        predicted_points = process_and_predict(image_bytes)
        return PredictResponse(
            filename=file.filename,
            points=predicted_points,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_landmarks_image", summary="Trả về ảnh vẽ tất cả 29 điểm (Chấm đỏ + Số thứ tự)")
async def predict_landmarks_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPG, PNG)")

    try:
        image_bytes = await file.read()
        result_image_bytes = process_and_draw(image_bytes)
        return Response(content=result_image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# API MỚI DÀNH RIÊNG CHO YÊU CẦU CỦA BẠN (VẼ ĐƯỜNG NỐI, GÓC & CHẨN ĐOÁN)
# =========================================================================
@router.post("/predict_analysis_image", summary="Trả về ảnh X-Quang đã phân tích Hô/Móm (Đường nối, Góc SNA, SNB, ANB)")
async def predict_analysis_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPG, PNG)")

    try:
        image_bytes = await file.read()
        # Gọi hàm AI xử lý và phân tích
        result_image_bytes = process_and_draw_analysis(image_bytes)
        
        # Trả về trực tiếp ảnh để xem trên Postman/Swagger
        return Response(content=result_image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict_analysis_data",
    summary="Trả về tọa độ điểm mốc, góc SNA/SNB/ANB và chẩn đoán Hô/Móm dạng JSON",
)
async def predict_analysis_data(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPG, PNG)")

    try:
        image_bytes = await file.read()
        result = process_and_get_analysis_data(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ceph_ai_analysis",
    summary="Phân tích Cephalometric AI chuyên sâu: chạy mô hình LRC → tính góc → GPT-4o diễn giải",
)
async def ceph_ai_analysis(file: UploadFile = File(...)):
    """
    Upload ảnh Cephalometric (phim sọ nghiêng).
    Pipeline:
      1. Mô hình HRNetW32 detect 29 landmark → tính SNA, SNB, ANB.
      2. GPT-4o (gpt-4o) phân tích chuyên sâu và đưa ra kế hoạch điều trị.
    Trả về JSON CephAiAnalysis.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPG, PNG)")

    try:
        image_bytes = await file.read()
        result = await get_ceph_ai_analysis_from_image(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))