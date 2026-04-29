import threading
import time
from dataclasses import dataclass

import cv2


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 90
    buffer_size: int = 1
    fourcc: str = 'MJPG'
    autofocus: int = 0


class WebcamVideoStream:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = cv2.VideoCapture(int(config.index))
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open camera index {config.index}')

        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(config.fourcc)))
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(config.fps))

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(config.buffer_size))
        except Exception:
            pass

        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(config.autofocus))
        except Exception:
            pass

        try:
            self.cap.set(cv2.CAP_PROP_ZOOM, 0)
        except Exception:
            pass

        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._running = False
        self._thread = None

    def start(self) -> 'WebcamVideoStream':
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame, float(self._frame_ts)

    def set_exposure(self, value: float) -> None:
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        except Exception:
            pass

    def set_gain(self, value: float) -> None:
        try:
            self.cap.set(cv2.CAP_PROP_GAIN, float(value))
        except Exception:
            pass

    def set_auto_exposure(self, value: float) -> None:
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(value))
        except Exception:
            pass

    def _update(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.001)
                continue
            ts = time.time()
            with self._lock:
                self._frame = frame
                self._frame_ts = ts
