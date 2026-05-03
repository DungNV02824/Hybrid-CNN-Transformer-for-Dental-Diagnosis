import io
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).resolve().parent.parent / "Model" / "fdi.pt"

_model = None


def get_model():
    """Lazy-load the YOLO model once and cache it."""
    global _model
    if _model is None:
        from ultralytics import YOLO  # import here so startup is fast

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def predict(image_bytes: bytes, conf_threshold: float = 0.25) -> Dict[str, Any]:
    """
    Run YOLO inference on raw image bytes.

    Returns a dict with:
      - detections: list of detected teeth with box, class name, and confidence
      - num_detections: total number of teeth detected
    """
    model = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(source=image, conf=conf_threshold, verbose=False)

    detections: List[Dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "tooth": result.names[int(box.cls.item())],
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "num_detections": len(detections),
        "detections": detections,
    }


# Colour palette — one colour per class index
_COLOURS = [
    (255, 56,  56),  (255, 157,  151), (255, 112, 31),  (255, 178,  29),
    (207, 210,  49), (72,  249,  10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,  212, 187),   (44,  153, 168), (0,  194, 255),
    (52,   69, 147), (100, 115, 255),  (0,   24, 236),  (132,  56, 255),
    (82,   0, 133),  (203,  56, 255),  (255,  99, 153), (255,   0, 113),
    (0,   18, 255),  (0,  159, 255),   (0,  255, 255),  (0,  255, 144),
    (255, 255,   0), (255, 127,   0),  (255,   0,   0), (255,  64, 255),
    (128,   0, 255), (0,  128, 255),   (0,  255, 128),  (128, 255,   0),
]


def predict_image(image_bytes: bytes, conf_threshold: float = 0.25) -> bytes:
    """
    Run YOLO inference and return a JPEG image with bounding boxes drawn.
    """
    model = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(source=image, conf=conf_threshold, verbose=False)

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, image.width // 50))
    except OSError:
        font = ImageFont.load_default()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx = int(box.cls.item())
            label = result.names[cls_idx]
            conf = float(box.conf.item())
            colour = _COLOURS[cls_idx % len(_COLOURS)]

            # Draw bounding box (2 px border)
            for offset in range(2):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=colour,
                )

            # Draw label background + text
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill=colour,
            )
            draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

