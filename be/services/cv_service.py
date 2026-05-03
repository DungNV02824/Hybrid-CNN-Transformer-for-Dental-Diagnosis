# services/dental_cv_service.py
"""
Dental CV inference pipeline.

Responsibilities:
  1. Preprocess uploaded image bytes into a format accepted by YOLO.
  2. Run model inference via the singleton loaded in models/dental_cv_model.py.
  3. Post-process raw predictions (threshold filter, Vietnamese label mapping).
  4. Visualise results on the original image using OpenCV.
  5. Return structured detection data + base64-encoded annotated image.
"""

import base64
import io
import logging
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from models.cv_model import get_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping – English class name → Vietnamese display name
# The keys must match exactly the class names stored in CV.pt (model.names).
# If the model uses numeric indices only, the FALLBACK_LABELS dict is used.
# ---------------------------------------------------------------------------
LABEL_VI: Dict[str, str] = {
    "Cavity":         "Sâu răng",
    "Fillings":       "Trám răng",
    "Impacted Tooth": "Răng mọc ngầm",
    "Implant":        "Cấy ghép implant",
}

# Fallback by index if the model's class names don't match the keys above
FALLBACK_LABELS: Dict[int, str] = {
    0: "Sâu răng",
    1: "Trám răng",
    2: "Răng mọc ngầm",
    3: "Cấy ghép implant",
}

# ---------------------------------------------------------------------------
# Colour palette (BGR) – one distinct colour per class index
# ---------------------------------------------------------------------------
_COLOURS: List[Tuple[int, int, int]] = [
    (56,  56,  255),   # class 0 – Cavity         – blue-red
    (31, 112,  255),   # class 1 – Fillings       – orange-blue
    (29, 178,  255),   # class 2 – Impacted Tooth – yellow-blue
    (10, 249,  72),    # class 3 – Implant        – green
    (134, 219, 61),
    (255, 157, 151),
    (255, 178, 29),
    (147, 210, 204),
]

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1
_BOX_THICK  = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_label(class_name: str, class_idx: int) -> str:
    """
    Return the Vietnamese label for a detection.
    Try the English name first; fall back to numeric index.
    """
    vi = LABEL_VI.get(class_name)
    if vi:
        return vi
    vi = FALLBACK_LABELS.get(class_idx)
    return vi if vi else class_name   # last resort: raw name from model


def _colour_for(class_idx: int) -> Tuple[int, int, int]:
    return _COLOURS[class_idx % len(_COLOURS)]


def _draw_detections(
    bgr_image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw bounding boxes and label+confidence text onto a copy of bgr_image.

    Args:
        bgr_image:  OpenCV image in BGR format (H×W×3, uint8).
        detections: list of detection dicts (must include label_vi, confidence,
                    bbox as [x1,y1,x2,y2], class_idx).

    Returns:
        Annotated BGR image (same size, new array).
    """
    canvas = bgr_image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        colour = _colour_for(det["class_idx"])
        label_text = f"{det['label_vi']}  {det['confidence']:.0%}"

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, _BOX_THICK)

        # Label background pill
        (tw, th), baseline = cv2.getTextSize(
            label_text, _FONT, _FONT_SCALE, _FONT_THICK
        )
        bg_y1 = max(y1 - th - baseline - 6, 0)
        bg_y2 = max(y1, th + baseline + 6)
        cv2.rectangle(canvas, (x1, bg_y1), (x1 + tw + 6, bg_y2), colour, cv2.FILLED)

        # Label text (white, readable on any colour background)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 3, bg_y2 - baseline - 2),
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            _FONT_THICK,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_dental(
    image_bytes: bytes,
    conf_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Full inference pipeline for dental pathology detection.

    Args:
        image_bytes:     Raw bytes of the uploaded image (JPG / PNG).
        conf_threshold:  Minimum confidence score to keep a detection (0–1).

    Returns:
        {
          "detections":    [ { label, confidence, bbox, label_vi } ],
          "total_objects": int,
          "image_result":  str  (base64-encoded annotated PNG),
          "inference_ms":  float (wall-clock inference time in ms),
        }

    Raises:
        ValueError:  if the image bytes cannot be decoded.
        RuntimeError: if the model raises an unexpected error.
    """
    # ------------------------------------------------------------------
    # 1. Decode image → PIL (for YOLO) and OpenCV (for drawing)
    # ------------------------------------------------------------------
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Convert PIL → BGR numpy array (OpenCV native format)
    bgr_image: np.ndarray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # 2. Inference
    # ------------------------------------------------------------------
    model = get_model()

    t_start = time.perf_counter()
    # YOLO handles all internal preprocessing (resize, normalise, batch)
    results = model.predict(
        source=pil_image,
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "Dental CV inference completed in %.1f ms (conf_threshold=%.2f)",
        inference_ms,
        conf_threshold,
    )

    # ------------------------------------------------------------------
    # 3. Parse raw YOLO output
    # ------------------------------------------------------------------
    detections: List[Dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx   = int(box.cls.item())
            cls_name  = result.names.get(cls_idx, str(cls_idx))
            conf_val  = float(box.conf.item())
            label_vi  = _resolve_label(cls_name, cls_idx)

            detections.append(
                {
                    "label":       cls_name,
                    "label_vi":    label_vi,
                    "confidence":  round(conf_val, 4),
                    "bbox":        [
                        round(x1, 2), round(y1, 2),
                        round(x2, 2), round(y2, 2),
                    ],
                    "class_idx":   cls_idx,
                }
            )

    # Sort highest confidence first
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ------------------------------------------------------------------
    # 4. Visualise – draw boxes on original image
    # ------------------------------------------------------------------
    annotated_bgr = _draw_detections(bgr_image, detections)

    # Encode annotated image to PNG → base64
    success, png_buf = cv2.imencode(".png", annotated_bgr)
    if not success:
        raise RuntimeError("Failed to encode annotated image to PNG.")

    image_b64 = base64.b64encode(png_buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    # 5. Build clean response (drop internal class_idx from public output)
    # ------------------------------------------------------------------
    public_detections = [
        {
            "label":      d["label_vi"],          # Vietnamese label
            "confidence": d["confidence"],
            "bbox":       d["bbox"],              # [x1, y1, x2, y2]
        }
        for d in detections
    ]

    return {
        "detections":    public_detections,
        "total_objects": len(public_detections),
        "image_result":  image_b64,
        "inference_ms":  round(inference_ms, 2),
    }


def predict_dental_image(
    image_bytes: bytes,
    conf_threshold: float = 0.3,
) -> bytes:
    """
    Full inference pipeline for dental pathology detection.
    Returns a JPEG image with bounding boxes drawn (as raw bytes).

    Args:
        image_bytes:     Raw bytes of the uploaded image (JPG / PNG).
        conf_threshold:  Minimum confidence score to keep a detection (0–1).

    Returns:
        JPEG image bytes with annotated bounding boxes.

    Raises:
        ValueError:   if the image bytes cannot be decoded.
        RuntimeError: if the model raises an unexpected error.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    bgr_image: np.ndarray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    model = get_model()

    results = model.predict(
        source=pil_image,
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
    )

    detections: list = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx  = int(box.cls.item())
            cls_name = result.names.get(cls_idx, str(cls_idx))
            conf_val = float(box.conf.item())
            label_vi = _resolve_label(cls_name, cls_idx)
            detections.append(
                {
                    "label":      cls_name,
                    "label_vi":   label_vi,
                    "confidence": round(conf_val, 4),
                    "bbox":       [x1, y1, x2, y2],
                    "class_idx":  cls_idx,
                }
            )

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    annotated_bgr = _draw_detections(bgr_image, detections)

    success, jpg_buf = cv2.imencode(
        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not success:
        raise RuntimeError("Failed to encode annotated image to JPEG.")

    return jpg_buf.tobytes()
