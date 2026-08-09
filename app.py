"""
SignSense — minimal detection-only backend.

No data collection, no training. Point this at a model you already trained
(svm_model.pkl + scaler.pkl, same format as the training pipeline: an
sklearn SVC with probability=True, and a fitted StandardScaler) sitting
next to app.py, and it will detect hand signs live from your webcam.

Pipeline per frame (same math as training):
  - MediaPipe Hands -> 21 landmarks
  - normalize relative to the wrist, scaled by wrist->middle-MCP distance
  - StandardScaler.transform -> model.predict / predict_proba
  - majority vote over the last 5 frames, 0.8 confidence threshold
"""

import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

CONFIDENCE_THRESHOLD = 0.8
BUFFER_SIZE = 5

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        self.camera_error = None

        self.latest_frame_bytes = None
        self.last_frame_time = 0.0
        self.hand_detected = False

        self.model = None
        self.scaler = None
        self.model_error = None

        self.pred_buffer = deque(maxlen=BUFFER_SIZE)
        self.last_prediction = {"label": None, "confidence": 0.0, "low_confidence": True}


state = AppState()


def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        state.model, state.scaler = None, None
        state.model_error = (
            f"No trained model found. Place 'svm_model.pkl' and 'scaler.pkl' "
            f"next to app.py (expected in: {BASE_DIR})."
        )
        return
    try:
        state.model = joblib.load(MODEL_PATH)
        state.scaler = joblib.load(SCALER_PATH)
        state.model_error = None
    except Exception as e:
        state.model, state.scaler = None, None
        state.model_error = f"Failed to load model: {e}"


load_model()  # try once at startup


# ---------------------------------------------------------------------------
def normalize_landmarks(hand_landmarks) -> list:
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    data = []
    for lm in hand_landmarks.landmark:
        data.extend((np.array([lm.x, lm.y, lm.z]) - wrist).tolist())

    mcp = hand_landmarks.landmark[9]
    scale = np.linalg.norm(np.array([mcp.x, mcp.y, mcp.z]) - wrist)
    if scale > 1e-6:
        data = [v / scale for v in data]
    return data


def handle_predict_frame(hand_landmarks):
    if state.model is None or state.scaler is None:
        return
    data = normalize_landmarks(hand_landmarks)
    if len(data) != 63:
        return

    scaled = state.scaler.transform([data])
    pred = state.model.predict(scaled)[0]
    prob = float(max(state.model.predict_proba(scaled)[0]))
    state.pred_buffer.append((pred, prob))

    if len(state.pred_buffer) == state.pred_buffer.maxlen:
        labels = [p[0] for p in state.pred_buffer]
        majority_label = max(set(labels), key=labels.count)
        matching = [p[1] for p in state.pred_buffer if p[0] == majority_label]
        avg_conf = float(np.mean(matching))
        state.last_prediction = {
            "label": majority_label,
            "confidence": avg_conf,
            "low_confidence": avg_conf < CONFIDENCE_THRESHOLD,
        }


def draw_overlay(frame, hand_detected: bool):
    h, _ = frame.shape[:2]
    if state.model is None:
        cv2.putText(frame, "No model loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif hand_detected:
        p = state.last_prediction
        if p["label"] is not None:
            if p["low_confidence"]:
                cv2.putText(frame, "Low Confidence", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"{p['label']} ({p['confidence']:.2f})", (10, 40),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 0), 2)
    if not hand_detected:
        cv2.putText(frame, "No hand detected", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
def open_camera(index: int = 0):
    """Robust camera open — see notes in the full project; DirectShow first on Windows."""
    import platform
    system = platform.system()
    if system == "Windows":
        candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Darwin":
        candidates = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        candidates = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in candidates:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok = False
            for _ in range(10):
                ok, _ = cap.read()
                if ok:
                    break
                time.sleep(0.1)
            if ok:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
        cap.release()
    return None


def camera_loop():
    cap = open_camera(0)
    state.cap = cap
    if cap is None:
        state.camera_error = (
            "Could not open the webcam. It may be in use by another app, "
            "disabled in your OS privacy settings, or on an index other than 0."
        )
        state.camera_running = False
        return

    state.camera_error = None
    state.pred_buffer.clear()
    state.last_prediction = {"label": None, "confidence": 0.0, "low_confidence": True}

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 150

    while state.camera_running:
        ok, frame = cap.read()
        if not ok:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                state.camera_error = "The camera stopped sending frames. Stop and Start again."
                break
            time.sleep(0.02)
            continue
        consecutive_failures = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_detected = bool(result.multi_hand_landmarks)
        if hand_detected:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handle_predict_frame(hand_landmarks)
        state.hand_detected = hand_detected
        draw_overlay(frame, hand_detected)

        ok2, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok2:
            with state.lock:
                state.latest_frame_bytes = jpeg.tobytes()
                state.last_frame_time = time.time()

    state.camera_running = False
    cap.release()
    state.cap = None


def mjpeg_generator():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while state.camera_running:
        with state.lock:
            frame = state.latest_frame_bytes
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary + frame + b"\r\n"
        time.sleep(0.03)


# ---------------------------------------------------------------------------
app = FastAPI(title="SignSense Detector")


@app.post("/api/start")
def start_detection():
    if state.model is None:
        load_model()  # in case the files just got added/retrained
        if state.model is None:
            raise HTTPException(400, state.model_error)

    if state.camera_running:
        return {"status": "already_running"}

    state.latest_frame_bytes = None
    state.camera_error = None
    state.camera_running = True
    state.camera_thread = threading.Thread(target=camera_loop, daemon=True)
    state.camera_thread.start()

    timeout_at = time.time() + 8.0
    while time.time() < timeout_at:
        if state.camera_error:
            state.camera_running = False
            raise HTTPException(500, state.camera_error)
        if state.latest_frame_bytes is not None:
            return {"status": "started"}
        time.sleep(0.15)

    state.camera_running = False
    raise HTTPException(500, "Camera did not respond in time. Close any other app using it and try again.")


@app.post("/api/stop")
def stop_detection():
    state.camera_running = False
    return {"status": "stopped"}


@app.post("/api/reload_model")
def reload_model():
    load_model()
    if state.model is None:
        raise HTTPException(400, state.model_error)
    return {"status": "loaded"}


@app.get("/api/status")
def status():
    frame_age = (time.time() - state.last_frame_time) if state.last_frame_time else None
    return {
        "running": state.camera_running,
        "model_loaded": state.model is not None,
        "model_error": state.model_error,
        "camera_error": state.camera_error,
        "hand_detected": state.hand_detected,
        "stalled": bool(state.camera_running and frame_age is not None and frame_age > 3.0),
        **state.last_prediction,
    }


@app.get("/video_feed")
def video_feed():
    if not state.camera_running:
        raise HTTPException(400, "Camera is not running. Start it first.")
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- Static frontend --------------------------------------------------------
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(STATIC_DIR):
    raise RuntimeError(
        f"\n\nCould not find the 'static' folder next to app.py (expected at: {STATIC_DIR}).\n"
        "Make sure you extracted the FULL project into the same directory, not just app.py.\n"
    )
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n  SignSense Detector starting -> http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
