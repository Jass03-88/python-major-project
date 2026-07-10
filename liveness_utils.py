import os
import cv2
import math
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Any

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "dnn_face_detector")
_TASK_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")

# Auto-download the model if missing
if not os.path.exists(_TASK_PATH):
    print("Downloading MediaPipe Face Landmarker model...")
    os.makedirs(_MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 
        _TASK_PATH
    )

_landmarker = None

def get_landmarker():
    global _landmarker
    if _landmarker is None:
        base_options = python.BaseOptions(model_asset_path=_TASK_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker

class LivenessTracker:
    """
    Tracks Eye Aspect Ratio (EAR) using MediaPipe FaceLandmarker to detect physical blinks.
    """

    def __init__(self, ear_threshold: float = 0.21, consecutive_frames: int = 2) -> None:
        self.landmarker = get_landmarker()
        
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        self.blink_counter = 0
        self.blink_detected = False

        # Right eye (image left for mirrored camera)
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Left eye (image right for mirrored camera)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]

    def _distance(self, p1: Any, p2: Any) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _calculate_ear(self, landmarks: List[Any], eye_indices: List[int]) -> float:
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        # Vertical distances
        v1 = self._distance(p2, p6)
        v2 = self._distance(p3, p5)
        # Horizontal distance
        h = self._distance(p1, p4)

        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def process_frame(self, frame: np.ndarray) -> None:
        """Processes a BGR frame and updates internal blink_detected state."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.landmarker.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.ear_threshold:
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.consecutive_frames:
                    self.blink_detected = True
                self.blink_counter = 0

    def is_liveness_successful(self) -> bool:
        return self.blink_detected

    def release(self) -> None:
        # We no longer close the landmarker here so it can be reused in future sessions
        pass
