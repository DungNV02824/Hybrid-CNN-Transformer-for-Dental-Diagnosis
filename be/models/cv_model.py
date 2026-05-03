# models/dental_cv_model.py
"""
Singleton loader for the Dental CV detection model (CV.pt).

CV.pt is a YOLOv8/v5 model trained to detect:
  - caries    (Sâu răng)
  - pulpitis  (Viêm tủy)
  - bone_loss (Tiêu xương)
  - lesion    (Tổn thương)

The model is loaded ONCE on first request and reused for every subsequent
inference call – no repeated I/O or GPU warmup.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model path – resolves to <project_root>/Model/CV.pt
# ---------------------------------------------------------------------------
MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "Model" / "CV.pt"
)

# Prefer GPU when available; fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level singleton – populated on first call to get_model()
_model = None


def get_model():
    """
    Lazily load the YOLO model and cache it for the lifetime of the process.

    Raises:
        FileNotFoundError: if the model file does not exist at MODEL_PATH.
        RuntimeError:      if the model cannot be loaded by ultralytics.
    """
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Dental CV model not found at '{MODEL_PATH}'. "
            "Ensure CV.pt is placed inside the Model/ directory."
        )

    logger.info("Loading Dental CV model from '%s' on device '%s' …", MODEL_PATH, DEVICE)

    try:
        # ultralytics handles both YOLOv5 and YOLOv8 .pt checkpoints
        from ultralytics import YOLO  # delayed import for fast startup

        _model = YOLO(str(MODEL_PATH))
        # Move underlying PyTorch model to the target device
        _model.to(DEVICE)
        logger.info("Dental CV model loaded successfully. Classes: %s", _model.names)
    except Exception as exc:
        logger.exception("Failed to load Dental CV model: %s", exc)
        raise RuntimeError(f"Model load error: {exc}") from exc

    return _model
