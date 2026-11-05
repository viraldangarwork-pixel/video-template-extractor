FPS = 30


def frame_to_ms(frame: int, fps: float = FPS) -> int:
    return int(round(frame * 1000 / fps))


def ms_to_frame(ms: int, fps: float = FPS) -> int:
    return int(round(ms * fps / 1000))
