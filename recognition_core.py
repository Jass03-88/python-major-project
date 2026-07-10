"""
Shared face registration / recognition engine. Uses SFace Deep Learning
for embedding extraction and YuNet for face detection/alignment.
"""

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import config
import security_utils
import telegram_utils
import attendance_manager
import face_detector
import liveness_utils

DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings.json"
SFACE_PATH = os.path.join("dnn_face_detector", "face_recognition_sface_2021dec.onnx")

_sface = None


def get_sface() -> Any:
    global _sface
    if _sface is None:
        if not os.path.exists(SFACE_PATH):
            raise FileNotFoundError(f"SFace model missing: {SFACE_PATH}")
        _sface = cv2.FaceRecognizerSF_create(SFACE_PATH, "")
    return _sface


class RegistrationSession:
    """Captures face crops for a new user, one frame at a time."""

    def __init__(self, user_id: str, target_count: int = 15) -> None:
        self.user_id = user_id
        self.target_count = target_count
        self.count = 0
        self.user_folder = os.path.join(DATASET_PATH, user_id)
        os.makedirs(self.user_folder, exist_ok=True)

    @property
    def is_complete(self) -> bool:
        return self.count >= self.target_count

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # We save raw RGB/BGR frames for SFace retraining
        faces = face_detector.detect_faces(frame)
        annotated = frame.copy()
        for x, y, w, h in faces:
            if self.is_complete:
                break
            self.count += 1
            # Save the full face crop. We could save the aligned face here,
            # but saving raw crop allows retraining with different models later.
            face_img = frame[y : y + h, x : x + w]
            if face_img.size > 0:
                cv2.imwrite(
                    os.path.join(self.user_folder, f"face_{self.count}.jpg"), face_img
                )
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return annotated


class LoginSession:
    """Collects predictions/liveness signal over several frames."""

    def __init__(self, frames_to_collect: Optional[int] = None, deadline_seconds: int = 8) -> None:
        self.frames_to_collect = frames_to_collect or config.FRAMES_TO_COLLECT
        self.deadline = time.time() + deadline_seconds
        self.collected = 0
        self.frame_count = 0
        self.predictions = {}

        self.liveness_tracker = liveness_utils.LivenessTracker()
        self.last_frame = None
        self.last_faces = []

        self.sface = get_sface()
        self.embeddings = {}
        if os.path.exists(EMBEDDINGS_PATH):
            try:
                with open(EMBEDDINGS_PATH, "r") as f:
                    self.embeddings = json.load(f)
            except Exception:
                pass

        self.liveness_available = True

    @property
    def is_complete(self) -> bool:
        return self.collected >= self.frames_to_collect or time.time() > self.deadline

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        annotated = frame.copy()

        faces = face_detector.detect_faces_yunet(frame)

        if len(faces) > 0 and self.frame_count > 5:
            self.last_frame = frame.copy()
            self.last_faces = faces
            self.collected += 1

            for face in faces:
                x, y, w, h = face[:4].astype(int)

                aligned_face = self.sface.alignCrop(frame, face)
                feature = self.sface.feature(aligned_face)

                best_match = "UNKNOWN"
                best_distance = float("inf")

                for user, user_emb in self.embeddings.items():
                    emb_array = np.array([user_emb], dtype=np.float32)
                    dist = self.sface.match(
                        feature, emb_array, cv2.FaceRecognizerSF_FR_COSINE
                    )
                    if dist < best_distance:
                        best_distance = dist
                        best_match = user

                self.predictions.setdefault(best_match, []).append(best_distance)

                if self.liveness_available:
                    self.liveness_tracker.process_frame(frame)

                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 200, 0), 2)

        return annotated

    def finalize(self) -> dict:
        face_detected = self.last_frame is not None and len(self.last_faces) > 0
        authenticated = False
        liveness_failed = False
        name = "UNKNOWN"

        if face_detected:
            best_name, best_conf = None, None
            for candidate, confs in self.predictions.items():
                avg_conf = sum(confs) / len(confs)
                if best_conf is None or avg_conf < best_conf:
                    best_name, best_conf = candidate, avg_conf

            if (
                self.liveness_available
                and config.LIVENESS_ENABLED
                and not self.liveness_tracker.is_liveness_successful()
            ):
                liveness_failed = True

        if self.liveness_available and self.liveness_tracker:
            self.liveness_tracker.release()

        if (
            face_detected
            and not liveness_failed
            and best_name != "UNKNOWN"
            and best_conf < config.CONFIDENCE_THRESHOLD
        ):
            authenticated = True
            name = best_name

        return {
            "face_detected": face_detected,
            "authenticated": authenticated,
            "liveness_failed": liveness_failed,
            "name": name,
            "last_frame": self.last_frame,
        }


def handle_login_result(result: dict) -> dict:
    if not result["face_detected"]:
        return {
            "granted": False,
            "message": "No face detected within the capture window.",
        }

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = result["name"]
    authenticated = result["authenticated"]
    liveness_failed = result["liveness_failed"]

    with open("Logs.txt", "a") as log:
        if authenticated:
            security_utils.reset_failures()
            log.write(f"{time_now} | {name} | ACCESS GRANTED\n")
        else:
            reason = "LIVENESS CHECK FAILED" if liveness_failed else "ACCESS DENIED"
            log.write(f"{time_now} | UNKNOWN | {reason}\n")

            file_safe_time = time_now.replace(":", "-").replace(" ", "_")
            intruder_path = f"intruder_{file_safe_time}.jpg"
            cv2.imwrite(intruder_path, result["last_frame"])

            caption = (
                f"🚨 Liveness check failed (possible photo/spoof attempt) at {time_now}."
                if liveness_failed
                else f"🚨 Intruder Detected!\nUnauthorized access attempt at {time_now}."
            )
            telegram_utils.send_denial_alert(intruder_path, caption)

            triggered = security_utils.record_failure(
                config.LOCKOUT_THRESHOLD,
                config.LOCKOUT_WINDOW_SECONDS,
                config.LOCKOUT_DURATION_SECONDS,
            )
            if triggered:
                telegram_utils.send_message(
                    f"🔒 System locked for {config.LOCKOUT_DURATION_SECONDS}s after repeated failed attempts."
                )

    if authenticated:
        attendance_result = attendance_manager.mark_attendance(name)
        return {"granted": True, "message": f"Welcome, {name}! {attendance_result}"}
    reason_msg = (
        "Liveness check failed — possible spoof attempt."
        if liveness_failed
        else "Access denied — face not recognized."
    )
    return {"granted": False, "message": reason_msg}


def list_registered_users() -> List[str]:
    if not os.path.exists(DATASET_PATH):
        return []
    return sorted(
        [
            u
            for u in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, u))
        ]
    )


def delete_user(user_id: str) -> bool:
    import shutil

    user_folder = os.path.join(DATASET_PATH, user_id)
    if os.path.isdir(user_folder):
        shutil.rmtree(user_folder)
        return True
    return False


def retrain_model() -> Tuple[bool, str]:
    """Extract embeddings using SFace and save to embeddings.json"""
    embeddings = {}
    if not os.path.exists(DATASET_PATH):
        return False, "No dataset folder found."

    sface = get_sface()
    count = 0

    for user in sorted(os.listdir(DATASET_PATH)):
        user_path = os.path.join(DATASET_PATH, user)
        if os.path.isdir(user_path):
            user_embeddings = []
            for image_name in os.listdir(user_path):
                img_path = os.path.join(user_path, image_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = face_detector.detect_faces_yunet(img)
                if len(faces) == 0:
                    continue

                face = max(faces, key=lambda f: f[2] * f[3])
                aligned_face = sface.alignCrop(img, face)
                feature = sface.feature(aligned_face)
                user_embeddings.append(feature[0].tolist())

            if user_embeddings:
                avg_emb = np.mean(user_embeddings, axis=0)
                embeddings[user] = avg_emb.tolist()
                count += 1

    if not embeddings:
        return False, "No faces found in dataset to extract embeddings."

    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(embeddings, f)

    return True, f"Embeddings extracted for {count} identities."
