from ultralytics import YOLO
import os


# Define constants
MODELS_DIR = "models"


def load_yolo_model(model_path: str):
    """Load a YOLO model from models directory"""
    # Simply join with models directory
    full_path = os.path.join(MODELS_DIR, model_path)

    # Check if model exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model not found: {full_path}")

    return YOLO(full_path)


def get_model_info(model_path: str):
    """Get model size in MB and number of parameters"""
    full_path = os.path.join(MODELS_DIR, model_path)

    try:
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        model = YOLO(full_path)
        params = sum(p.numel() for p in model.parameters())
        return size_mb, params
    except Exception:
        return 0, 0
