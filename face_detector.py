"""
DNN-based face detector using OpenCV's YuNet model.
Replaces the old SSD Caffe model with a faster, more accurate ONNX model
that also returns 5 facial landmarks (needed by SFace).
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Any

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dnn_face_detector"
)
_YUNET_PATH = os.path.join(_MODEL_DIR, "face_detection_yunet_2023mar.onnx")

_detector = None
_current_input_size = (0, 0)


def _get_detector(w: int, h: int) -> cv2.FaceDetectorYN:
    global _detector, _current_input_size
    if not os.path.exists(_YUNET_PATH):
        raise FileNotFoundError(f"YuNet model missing: {_YUNET_PATH}")

    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(
            model=_YUNET_PATH,
            config="",
            input_size=(w, h),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )
        _current_input_size = (w, h)
    elif _current_input_size != (w, h):
        _detector.setInputSize((w, h))
        _current_input_size = (w, h)
    return _detector


def detect_faces_yunet(frame: np.ndarray, confidence_threshold: float = 0.5) -> Any:
    """
    Returns the raw YuNet output array: shape (N, 15)
    Each row: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
    """
    h, w = frame.shape[:2]
    detector = _get_detector(w, h)
    detector.setScoreThreshold(confidence_threshold)

    # YuNet expects BGR format (default in OpenCV)
    _, faces = detector.detect(frame)
    return faces if faces is not None else []


def detect_faces(frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    Backward-compatible function returning (x, y, w, h) boxes.
    """
    faces = detect_faces_yunet(frame, confidence_threshold)
    boxes = []
    (h, w) = frame.shape[:2]
    for face in faces:
        box = face[:4].astype(int)
        x, y, bw, bh = box

        # Clip
        x, y = max(0, x), max(0, y)
        end_x, end_y = min(w - 1, x + bw), min(h - 1, y + bh)
        bw, bh = end_x - x, end_y - y

        if bw > 0 and bh > 0:
            boxes.append((x, y, bw, bh))
    return boxes
