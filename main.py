import time
from datetime import datetime
import math
import csv
from pathlib import Path
import threading
import sys

import cv2
import numpy as np

from ultralytics import YOLO


class _FilteredStderr:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s: str) -> int:
        if 'NNPACK.cpp:56' in s or 'Could not initialize NNPACK' in s:
            return len(s)
        return self._wrapped.write(s)

    def flush(self) -> None:
        return self._wrapped.flush()


sys.stderr = _FilteredStderr(sys.stderr)


import torch

from camera_utils import CameraConfig, WebcamVideoStream


BASE_DIR = Path(__file__).resolve().parent
DETECTIONS_DIR = BASE_DIR / 'detections'
PLATES_DIR = DETECTIONS_DIR / 'plates'
FRAMES_DIR = DETECTIONS_DIR / 'frames'
CSV_PATH = DETECTIONS_DIR / 'plates.csv'
CSV_FIELDNAMES = ['plate_text', 'date', 'time', 'confidence', 'plate_image', 'frame_image', 'bbox_xyxy']

PLATE_MODEL_PATH = 'runs/detect/runs/ghana_npr_v3_final3/weights/best.pt'
CHAR_MODEL_PATH = 'runs/detect/runs/char_recognition_v13/weights/best.pt'
VEHICLE_MODEL_PATH = 'yolov8s.pt'
TRACKER_CONFIG = 'bytetrack.yaml'

USE_VEHICLE_GATE = False
VEHICLE_CLASSES = {2, 3, 5, 7}

PLATE_MIN_CONF = 0.35
VEHICLE_MIN_CONF = 0.35
CHAR_MIN_CONF = 0.35
PLATE_PADDING_PX = 20

INFERENCE_FPS = 20.0
VEHICLE_FPS = 10.0
PLATE_IMGSZ = 640
VEHICLE_IMGSZ = 640

CAPTURE_COOLDOWN_SECONDS = 1.0
DEDUP_SECONDS = 3.0

AUTO_EXPOSURE_STEPS = 7
BRIGHT_TARGET_MEAN = 110.0
BRIGHT_LOW_THRESH = 80.0
BRIGHT_HIGH_THRESH = 180.0
EXPOSURE_STEP_SIZE = 1
GAIN_STEP_SIZE = 5

torch.backends.nnpack.enabled = False


VIEW_SCALE = 0.75

SHOW_PLATE_BOX = True


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _auto_exposure_step(stream, mean_brightness: float, current_exp: int, current_gain: int) -> tuple[int, int]:
    """Adjust exposure/gain toward target brightness using small steps."""
    if mean_brightness < BRIGHT_LOW_THRESH:
        # Too dark: increase exposure first, then gain
        if current_exp < 25 - AUTO_EXPOSURE_STEPS:
            new_exp = min(25, current_exp + EXPOSURE_STEP_SIZE)
            new_gain = current_gain
        else:
            new_exp = current_exp
            new_gain = min(255, current_gain + GAIN_STEP_SIZE)
    elif mean_brightness > BRIGHT_HIGH_THRESH:
        # Too bright: decrease gain first, then exposure
        if current_gain > AUTO_EXPOSURE_STEPS * GAIN_STEP_SIZE:
            new_gain = max(0, current_gain - GAIN_STEP_SIZE)
            new_exp = current_exp
        else:
            new_gain = current_gain
            new_exp = max(0, current_exp - EXPOSURE_STEP_SIZE)
    else:
        new_exp = current_exp
        new_gain = current_gain
    if new_exp != current_exp:
        stream.set_exposure(float(new_exp - 13))
    if new_gain != current_gain:
        stream.set_gain(float(new_gain))
    return new_exp, new_gain


def _sanitize_plate_text(text: str) -> str:
    return ''.join(ch for ch in text.upper() if ch.isalnum())


def _append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
 
        out = {k: '' for k in CSV_FIELDNAMES}
        out.update(row)
        writer.writerow(out)


def _read_plate_text_with_char_model(char_model: YOLO, plate_bgr) -> str:
    char_results = char_model.predict(source=plate_bgr, verbose=False)
    if not char_results:
        return ''

    r = char_results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return ''

    boxes = r.boxes
    names = r.names or {}

    chars = []
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf < CHAR_MIN_CONF:
            continue
        cls_id = int(boxes.cls[i].item())
        label = names.get(cls_id, str(cls_id))
        xyxy = boxes.xyxy[i].cpu().numpy().astype(float)
        x_center = float((xyxy[0] + xyxy[2]) / 2.0)
        chars.append((x_center, label))

    if not chars:
        return ''

    chars.sort(key=lambda t: t[0])
    text = ''.join(ch for _, ch in chars)
    return _sanitize_plate_text(text)


def _prep_plate_for_ocr(plate_bgr):
    if plate_bgr is None:
        return None
    h, w = plate_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return plate_bgr

    plate_bgr = cv2.resize(plate_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gamma = _auto_gamma(gray, target_mean=140.0)
    plate_bgr = _apply_gamma(plate_bgr, gamma)
    plate_bgr = _apply_clahe_bgr(plate_bgr, clip_limit=2.0, tile_grid=8)
    return plate_bgr


def _pick_camera_index(preferred: int, fallbacks: list[int]) -> int:
    candidates = [preferred] + [i for i in fallbacks if i != preferred]
    tried = []
    for idx in candidates:
        tried.append(idx)
        cap = cv2.VideoCapture(int(idx))
        ok = cap.isOpened()
        ok_frame = False
        if ok:
            ok_frame, _ = cap.read()
        cap.release()
        if ok and ok_frame:
            return int(idx)
    raise RuntimeError(f'No working camera found. Tried: {tried}')


def _auto_gamma(gray_u8, target_mean: float) -> float:
    mean = float(gray_u8.mean())
    mean = max(1.0, min(254.0, mean))
    target = max(1.0, min(254.0, float(target_mean)))
    g = target / 255.0
    m = mean / 255.0
    if m <= 0.0 or g <= 0.0 or m == 1.0:
        return 1.0
    try:
        gamma = math.log(g) / math.log(m)
    except (ValueError, ZeroDivisionError):
        gamma = 1.0
    return float(max(0.25, min(4.0, gamma)))


def _apply_gamma(bgr, gamma: float):
    inv_gamma = 1.0 / max(1e-6, float(gamma))
    table = (255.0 * (np.power((np.arange(256) / 255.0), inv_gamma))).astype('uint8')
    return cv2.LUT(bgr, table)


def _apply_clahe_bgr(bgr, clip_limit: float, tile_grid: int):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _auto_enhance(bgr):
    small = cv2.resize(bgr, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())

    # Exposure/gain should be handled by the camera. This is a lightweight
    # software assist to improve readability under harsh lighting.
    if mean < 70.0:
        bgr = cv2.convertScaleAbs(bgr, alpha=1.35, beta=0)
    elif mean < 95.0:
        bgr = cv2.convertScaleAbs(bgr, alpha=1.15, beta=0)
    elif mean > 190.0:
        bgr = cv2.convertScaleAbs(bgr, alpha=0.75, beta=0)
    elif mean > 165.0:
        bgr = cv2.convertScaleAbs(bgr, alpha=0.88, beta=0)

    # Improve local contrast when image is flat (e.g., haze, glare, shadows).
    if std < 35.0:
        bgr = _apply_clahe_bgr(bgr, clip_limit=2.0, tile_grid=8)

    # Night noise: denoise only when it is dark AND noisy.
    if mean < 60.0 and std > 55.0:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)

    return bgr


class _AnprWorker:
    def __init__(self, plate_model: YOLO, char_model: YOLO):
        self._plate_model = plate_model
        self._char_model = char_model
        self._vehicle_model = None

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ts = 0.0

        self._running = False
        self._thread = None

        self._last_plate_infer_ts = 0.0
        self._last_vehicle_infer_ts = 0.0
        self._last_capture_ts = 0.0
        self._last_seen = {}

        # Multi-object tracking: map track_id -> (bbox, conf, text, last_ts)
        self._tracked_plates = {}
        self._next_capture_per_track = {}

        self._last_status_ts = 0.0

        self.vehicle_gate = bool(USE_VEHICLE_GATE)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_latest_frame(self, frame, ts: float) -> None:
        with self._lock:
            self._latest_frame = frame
            self._latest_ts = float(ts)

    def set_vehicle_gate(self, enabled: bool) -> None:
        self.vehicle_gate = bool(enabled)

    def get_plate_state(self):
        # return list of (track_id, bbox, conf, text)
        with self._lock:
            return [(tid, bbox, conf, txt) for tid, (bbox, conf, txt, _) in self._tracked_plates.items()]

    def _loop(self) -> None:
        plate_interval = 1.0 / max(1e-6, float(INFERENCE_FPS))
        vehicle_interval = 1.0 / max(1e-6, float(VEHICLE_FPS))

        last_processed_ts = 0.0

        while self._running:
            with self._lock:
                frame = self._latest_frame
                ts = float(self._latest_ts)

            if frame is None or ts <= 0.0:
                time.sleep(0.002)
                continue

            # Drop intermediate frames: only process when we see a new timestamp.
            if ts == last_processed_ts:
                time.sleep(0.001)
                continue
            last_processed_ts = ts

            now_ts = time.time()
            src = _auto_enhance(frame)

            vehicle_present = True
            if self.vehicle_gate:
                if self._vehicle_model is None:
                    self._vehicle_model = YOLO(VEHICLE_MODEL_PATH)

                if (now_ts - self._last_vehicle_infer_ts) >= vehicle_interval:
                    vres = self._vehicle_model.predict(source=src, verbose=False, imgsz=int(VEHICLE_IMGSZ))
                    vr0 = vres[0]
                    vehicle_present = False
                    if vr0.boxes is not None and len(vr0.boxes) > 0:
                        vboxes = vr0.boxes
                        for i in range(len(vboxes)):
                            vconf = float(vboxes.conf[i].item())
                            if vconf < VEHICLE_MIN_CONF:
                                continue
                            vcls = int(vboxes.cls[i].item())
                            if vcls in VEHICLE_CLASSES:
                                vehicle_present = True
                                break
                    self._last_vehicle_infer_ts = now_ts

            # Plate detection with tracking using ByteTrack
            if vehicle_present and (now_ts - self._last_plate_infer_ts) >= plate_interval:
                track_results = self._plate_model.track(source=src, persist=True, tracker=TRACKER_CONFIG, verbose=False, conf=PLATE_MIN_CONF, imgsz=int(PLATE_IMGSZ))
                tr0 = track_results[0]

                # Update tracked plates
                new_tracked = {}
                if tr0.boxes is not None and tr0.boxes.id is not None:
                    for box, obj_id, conf in zip(tr0.boxes.xyxy.cpu().numpy(),
                                                tr0.boxes.id.cpu().numpy(),
                                                tr0.boxes.conf.cpu().numpy()):
                        tid = int(obj_id)
                        x1, y1, x2, y2 = map(int, box)
                        x1 = _clamp(x1 - PLATE_PADDING_PX, 0, src.shape[1] - 1)
                        y1 = _clamp(y1 - PLATE_PADDING_PX, 0, src.shape[0] - 1)
                        x2 = _clamp(x2 + PLATE_PADDING_PX, 0, src.shape[1] - 1)
                        y2 = _clamp(y2 + PLATE_PADDING_PX, 0, src.shape[0] - 1)
                        if x2 > x1 and y2 > y1:
                            bbox = (x1, y1, x2, y2)
                            # Preserve previous text if we have it; otherwise mark for OCR
                            prev = self._tracked_plates.get(tid, (bbox, float(conf), '', now_ts))
                            _, _, prev_text, _ = prev
                            new_tracked[tid] = (bbox, float(conf), prev_text, now_ts)
                self._tracked_plates = new_tracked
                self._last_plate_infer_ts = now_ts

            # Periodic OCR and capture per track
            for tid, (bbox, conf, txt, last_update_ts) in list(self._tracked_plates.items()):
                # Run OCR if we lack text or it's stale
                if not txt or (now_ts - last_update_ts) > 2.0:
                    x1, y1, x2, y2 = bbox
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        ocr_input = _prep_plate_for_ocr(crop)
                        plate_text = _read_plate_text_with_char_model(self._char_model, ocr_input)
                        if not plate_text:
                            plate_text = 'UNREADABLE'
                        # Update tracked entry with new text
                        self._tracked_plates[tid] = (bbox, conf, plate_text, now_ts)
                else:
                    plate_text = txt

                # Capture cooldown per track
                if plate_text and plate_text not in ('UNREADABLE', 'DUPLICATE'):
                    next_capture = self._next_capture_per_track.get(tid, 0.0)
                    if now_ts >= next_capture:
                        # Dedup across tracks
                        prev_global = self._last_seen.get(plate_text, 0.0)
                        if (now_ts - prev_global) < DEDUP_SECONDS:
                            continue
                        dt = datetime.now()
                        stamp = dt.strftime('%Y%m%d_%H%M%S')
                        x1, y1, x2, y2 = bbox
                        plate_crop = frame[y1:y2, x1:x2].copy()
                        plate_img_path = PLATES_DIR / f'plate_{stamp}.png'
                        frame_img_path = FRAMES_DIR / f'frame_{stamp}.png'
                        cv2.imwrite(str(plate_img_path), plate_crop)
                        cv2.imwrite(str(frame_img_path), frame)
                        _append_csv_row(
                            CSV_PATH,
                            {
                                'plate_text': plate_text,
                                'date': dt.strftime('%Y-%m-%d'),
                                'time': dt.strftime('%H:%M:%S'),
                                'confidence': f'{conf:.3f}',
                                'plate_image': str(plate_img_path),
                                'frame_image': str(frame_img_path),
                                'bbox_xyxy': f'{x1},{y1},{x2},{y2}',
                            },
                        )
                        self._last_seen[plate_text] = now_ts
                        self._next_capture_per_track[tid] = now_ts + CAPTURE_COOLDOWN_SECONDS
                        print(f"Captured: {plate_text}  conf={conf:.3f}  plate={plate_img_path.name}", flush=True)

            # Status log
            if now_ts - self._last_status_ts >= 2.0:
                if not self._tracked_plates:
                    print(f"Status: vehicle_gate={int(self.vehicle_gate)} plates=NONE", flush=True)
                else:
                    msgs = []
                    for tid, (bbox, conf, txt, _) in self._tracked_plates.items():
                        x1, y1, x2, y2 = bbox
                        msgs.append(f"id{tid} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2}) txt={txt}")
                    print(f"Status: vehicle_gate={int(self.vehicle_gate)} plates={' | '.join(msgs)}", flush=True)
                self._last_status_ts = now_ts


def main() -> None:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLATES_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    plate_model = YOLO(PLATE_MODEL_PATH)
    char_model = YOLO(CHAR_MODEL_PATH)

    cam_index = _pick_camera_index(preferred=1, fallbacks=[3, 0, 2, 4])
    config = CameraConfig(
        index=cam_index,
        width=1920,
        height=1080,
        fps=90,
        buffer_size=1,
        fourcc='MJPG',
        autofocus=0,
    )

    stream = WebcamVideoStream(config).start()
    print(f'Camera index: {cam_index}')

    worker = _AnprWorker(plate_model=plate_model, char_model=char_model)
    worker.start()

    window = 'ANPR Live'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    w0 = int(config.width * float(VIEW_SCALE)) if VIEW_SCALE != 1.0 else int(config.width)
    h0 = int(config.height * float(VIEW_SCALE)) if VIEW_SCALE != 1.0 else int(config.height)
    if w0 > 0 and h0 > 0:
        cv2.resizeWindow(window, w0, h0)

    track_win = 'Controls'
    cv2.namedWindow(track_win, cv2.WINDOW_NORMAL)

    zoom_win = 'Plate Zoom'
    cv2.namedWindow(zoom_win, cv2.WINDOW_NORMAL)

    def on_exposure(val: int) -> None:
        exp = (val - 13)
        stream.set_exposure(float(exp))

    def on_gain(val: int) -> None:
        stream.set_gain(float(val))

    def on_auto_exposure(val: int) -> None:
        stream.set_auto_exposure(float(val))

    # Keep camera exposure automatic by default.
    on_auto_exposure(1)
    cv2.createTrackbar('AutoExposure', track_win, 1, 1, on_auto_exposure)
    cv2.createTrackbar('Exposure', track_win, 13, 25, on_exposure)
    cv2.createTrackbar('Gain', track_win, 0, 255, on_gain)

    state = {
        'view_scale': float(VIEW_SCALE),
        'vehicle_gate': 1 if USE_VEHICLE_GATE else 0,
    }

    def on_view_scale(val: int) -> None:
        pct = _clamp(int(val), 25, 150)
        scale = pct / 100.0
        state['view_scale'] = float(scale)
        cv2.resizeWindow(window, int(config.width * scale), int(config.height * scale))

    cv2.createTrackbar('View Size (%)', track_win, int(VIEW_SCALE * 100), 150, on_view_scale)

    def on_vehicle_gate(val: int) -> None:
        state['vehicle_gate'] = 1 if int(val) > 0 else 0

    cv2.createTrackbar('Vehicle Gate (0/1)', track_win, int(state['vehicle_gate']), 1, on_vehicle_gate)

    last_label_ts = 0.0
    label_line1 = 'Ghana ANPR'
    label_line2 = ''
    label_margin = 10
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 0.5
    label_thickness = 1

    # Auto-exposure state
    current_exp = 13
    current_gain = 0
    last_auto_exp_ts = 0.0

    try:
        while True:
            ok, frame, ts = stream.read()
            if not ok:
                time.sleep(0.001)
                continue

            worker.set_vehicle_gate(int(state['vehicle_gate']) == 1)
            worker.set_latest_frame(frame, ts)

            processed = frame

            # Auto-exposure stepping every 0.5s based on frame brightness
            now_ts = time.time()
            if now_ts - last_auto_exp_ts >= 0.5:
                gray_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                mean_brightness = float(gray_small.mean())
                current_exp, current_gain = _auto_exposure_step(stream, mean_brightness, current_exp, current_gain)
                last_auto_exp_ts = now_ts

            disp_src = processed
            view_scale = float(state['view_scale'])
            if view_scale != 1.0:
                w = int(disp_src.shape[1] * view_scale)
                h = int(disp_src.shape[0] * view_scale)
                w = max(1, w)
                h = max(1, h)
                disp = cv2.resize(disp_src, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                disp = disp_src.copy()

            # Draw all tracked plates
            tracked = worker.get_plate_state()
            for tid, bbox, conf, txt in tracked:
                x1, y1, x2, y2 = bbox
                x1s = int(x1 * view_scale)
                y1s = int(y1 * view_scale)
                x2s = int(x2 * view_scale)
                y2s = int(y2 * view_scale)
                cv2.rectangle(disp, (x1s, y1s), (x2s, y2s), (0, 255, 0), 2)
                label = f"ID{tid} {txt}"
                (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(disp, (x1s, y1s - h_label - 8), (x1s + w_label, y1s), (0, 255, 0), -1)
                cv2.putText(disp, label, (x1s, y1s - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Update zoom window for the most confident plate
                if SHOW_PLATE_BOX:
                    zoom_crop = frame[y1:y2, x1:x2]
                    zoom = cv2.resize(zoom_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    cv2.imshow(zoom_win, zoom)
                    break  # show only the first tracked plate in zoom for now

            now = time.time()
            if now - last_label_ts >= 1.0:
                label_line2 = f"Live ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
                last_label_ts = now

            x = label_margin
            y2 = disp.shape[0] - label_margin
            y1 = y2 - 18
            cv2.putText(disp, label_line1, (x, y1), label_font, label_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
            cv2.putText(disp, label_line2, (x, y2), label_font, label_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)

            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        worker.stop()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
