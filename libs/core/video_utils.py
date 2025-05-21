import cv2
import numpy as np


def get_video_orientation(frame):
    """Detect video orientation based on frame dimensions"""
    if frame is None:
        return "landscape"
    h, w = frame.shape[:2]
    return "portrait" if h > w else "landscape"


def ensure_frame_dims(frame):
    """Ensure frame dimensions are valid"""
    try:
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return frame if h > 0 and w > 0 else None
    except:
        return None


def resize_frame(frame, target_size=(640, 640)):
    """Resize frame while maintaining aspect ratio"""
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    height, width = frame.shape[:2]

    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    # Ensure minimum dimensions
    new_width = max(new_width, 32)
    new_height = max(new_height, 32)

    # Resize frame using INTER_LINEAR for better quality
    try:
        resized = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    return resized
