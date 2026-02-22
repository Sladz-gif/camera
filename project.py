import os
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DETECTIONS_DIR = BASE_DIR / 'detections'
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
PLATES_DIR = DETECTIONS_DIR / 'plates'
PLATES_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = DETECTIONS_DIR / 'frames'
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DETECTIONS_DIR / 'plates.csv'

MODEL_PATH = 'best.pt'
CAMERA_INDEX = 2
MIN_CONF = 0.35
CROP_PADDING_PX = 25
CAPTURE_COOLDOWN_SECONDS = 2.0
SAVE_FULL_FRAME = True

torch.backends.nnpack.enabled = False


def _clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def _sanitize_plate_text(text: str) -> str:
    cleaned = ''.join(ch for ch in text.upper() if ch.isalnum())
    return cleaned


def _append_csv_row(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Change CAMERA_INDEX at the top of project.py to match your device."
        )

    last_capture_ts = 0.0

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        results = model.predict(source=frame, verbose=False)
        r0 = results[0]
        annotated = r0.plot()

        plate_crop = None
        plate_bbox = None

        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes
            best_i = int(boxes.conf.argmax().item())
            best_conf = float(boxes.conf[best_i].item())

            if best_conf >= MIN_CONF:
                xyxy = boxes.xyxy[best_i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = map(int, xyxy)

                x1 = _clamp(x1 - CROP_PADDING_PX, 0, frame.shape[1] - 1)
                y1 = _clamp(y1 - CROP_PADDING_PX, 0, frame.shape[0] - 1)
                x2 = _clamp(x2 + CROP_PADDING_PX, 0, frame.shape[1] - 1)
                y2 = _clamp(y2 + CROP_PADDING_PX, 0, frame.shape[0] - 1)

                if x2 > x1 and y2 > y1:
                    plate_bbox = (x1, y1, x2, y2, best_conf)
                    plate_crop = frame[y1:y2, x1:x2].copy()

        cv2.imshow('Detections', annotated)

        if plate_crop is not None:
            zoom = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('Plate (Zoom)', zoom)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        now_ts = time.time()
        if plate_crop is not None and (now_ts - last_capture_ts) >= CAPTURE_COOLDOWN_SECONDS:
            dt = datetime.now()
            ts = dt.strftime('%Y%m%d_%H%M%S')

            plate_img_path = PLATES_DIR / f'plate_{ts}.jpg'
            cv2.imwrite(str(plate_img_path), plate_crop)

            frame_img_path = ''
            if SAVE_FULL_FRAME:
                frame_path = FRAMES_DIR / f'frame_{ts}.jpg'
                cv2.imwrite(str(frame_path), annotated)
                frame_img_path = str(frame_path)

            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray)
            raw_text = max(ocr_results, key=lambda x: x[2])[1] if ocr_results else ''
            plate_text = _sanitize_plate_text(raw_text)

            x1, y1, x2, y2, conf = plate_bbox
            _append_csv_row(
                CSV_PATH,
                {
                    'plate_text': plate_text,
                    'date': dt.strftime('%Y-%m-%d'),
                    'time': dt.strftime('%H:%M:%S'),
                    'confidence': f'{conf:.3f}',
                    'plate_image': str(plate_img_path),
                    'frame_image': frame_img_path,
                    'bbox_xyxy': f'{x1},{y1},{x2},{y2}',
                },
            )

            print(f"Captured plate='{plate_text}' saved={plate_img_path}")
            last_capture_ts = now_ts

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
