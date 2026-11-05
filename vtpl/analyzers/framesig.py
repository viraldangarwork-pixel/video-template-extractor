from pathlib import Path

import cv2
import numpy as np

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def read_small_frames(video: Path, size: tuple[int, int] = (36, 64)) -> np.ndarray:
    """Every frame, resized to size=(w, h), RGB float32 in 0..1. Shape (N, h, w, 3)."""
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    if not out:
        raise ValueError(f"could not decode frames from {video}")
    return np.stack(out)


def luma(frames: np.ndarray) -> np.ndarray:
    """Mean Rec.601 luma per frame, shape (N,)."""
    return (frames @ _LUMA).mean(axis=(1, 2))
