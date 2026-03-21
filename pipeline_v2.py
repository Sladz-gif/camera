
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Model
model = YOLO('unified_car_ocr_final/weights/best.pt')

# Character Mapping
char_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R',
    29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'
}

def get_enhanced_zoom(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def process_video(video_in, video_out):
    cap = cv2.VideoCapture(video_in)
    width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model.track(frame, persist=True, conf=0.4, verbose=False)[0]
        if results.boxes.id is not None:
            for box, obj_id, cls in zip(results.boxes.xyxy.cpu().numpy(), 
                                        results.boxes.id.cpu().numpy(), 
                                        results.boxes.cls.cpu().numpy()):
                if int(cls) == 21:
                    enhanced = get_enhanced_zoom(frame, box)
                    if enhanced is not None:
                        ocr_res = model(enhanced, conf=0.5, verbose=False)[0]
                        chars = [c for c in ocr_res.boxes.data.cpu().numpy() if int(c[5]) not in [21, 33]]
                        chars.sort(key=lambda x: x[0]) 
                        plate_text = "".join([char_map.get(int(c[5]), '') for c in chars])
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{int(obj_id)} | {plate_text}", (x1, y1-10), 2, 0.7, (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
