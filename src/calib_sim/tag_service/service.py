"""FastAPI wrapper for AprilTag detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

from calib_sim.tag_service.detector import AprilTag36h11Detector

app = FastAPI(title="AprilTag Detection Service", version="0.1.0")
detector = AprilTag36h11Detector()


@app.get("/health")
def health() -> dict:
    """Simple process health endpoint."""
    return {"ok": True, "detector": "opencv_aruco_apriltag36h11"}


@app.post("/v1/detect/frame")
async def detect_frame(
    image: UploadFile = File(...),
    fx: float | None = Form(None),
    fy: float | None = Form(None),
    cx: float | None = Form(None),
    cy: float | None = Form(None),
    tag_size_m: float | None = Form(None),
) -> dict:
    """Detect tags in one uploaded image.

    The API accepts standard multipart form uploads so that it is easy to call
    from curl, notebooks, or a web UI.
    """
    payload = await image.read()
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"frame_index": 0, "timestamp_s": None, "detections": [], "error": "Could not decode image."}

    result = detector.detect_image(frame, frame_index=0, fx=fx, fy=fy, cx=cx, cy=cy, tag_size_m=tag_size_m)
    return detector.to_jsonable(result)


@app.post("/v1/detect/video")
async def detect_video(video: UploadFile = File(...), every_n_frames: int = Form(1)) -> dict:
    """Detect tags in a video upload.

    For large files a production service would stream or page the results. Here
    we return all frame detections for simplicity.
    """
    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    results = detector.detect_video(tmp_path, every_n_frames=max(1, int(every_n_frames)))
    return {"frames": [detector.to_jsonable(frame_result) for frame_result in results]}
