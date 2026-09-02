"""
Drone Tracking System — Optimized for Raspberry Pi 5 + PC
==========================================================
Target: 25+ FPS on Pi 5, 30+ FPS on desktop

How we sped things up:
  • Each frame is downscaled to half resolution for processing (320x240),
    but the display stays at full size.
  • Red and blue color masks are computed with a single GaussianBlur
    and a single HSV conversion — no redundant work.
  • YOLO input shrunk from 640 to 320 (roughly 4x faster inference),
    and it only runs every 5th frame.
  • If opencv-contrib is installed, we use CSRT tracking; otherwise
    we fall back to CamShift automatically.
  • Distance comparisons use squared values instead of sqrt — faster math.
  • The draw() function was stripped of any repeated calculations.
  • Pi 5: set CAM_ID to whatever your LCCV / picamera2 device index is.
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue
from typing import Optional, Tuple

# ──────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────
CAMERA_ID       = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
PROC_WIDTH      = 320       # Width used during mask / contour processing
PROC_HEIGHT     = 240       # Height used during mask / contour processing
MODEL_PATH      = "best_slim.onnx"
YOLO_INPUT_SIZE = 320       # Optimized for Pi 5 (320x320)
CONFIDENCE_THR  = 0.93      # Lower slightly on Pi 5 — 0.98 is too strict there
NMS_THR         = 0.45
YOLO_SKIP       = 5         # Run YOLO once every N frames

# HSV skin range (remains constant for filtering)
HSV_SKIN_RANGE = (np.array([0,  25,  80], np.uint8), np.array([20, 150, 255], np.uint8))

# Shape / area thresholds ──────────────────────────────────
MIN_CONTOUR_AREA   = 400      # Lowered to handle targets at a distance
MIN_FILL_RATIO     = 0.40     # Contour must fill at least 40% of its convex hull
MIN_COMPACTNESS    = 0.02     # Lowered for more tolerance
TRACKER_MASK_THR   = 0.20     # Tracker bbox must have ≥20% foreground pixels
MIN_SATURATION_VAL = 40       # Reject near-grey / near-black blobs
MAX_AREA_RATIO     = 3.0      # Target can't suddenly triple or shrink to a third
MAX_JUMP_SQ        = 1250 ** 2  # Squared max pixel jump between frames

# PD controller gains ──────────────────────────────────────
KP       = 0.005
KD       = 0.002
MAX_VEL  = 2.0   # m/s

# Derived scale factors (small-frame coords → full-frame coords)
SCALE_X = FRAME_WIDTH  / PROC_WIDTH
SCALE_Y = FRAME_HEIGHT / PROC_HEIGHT

# Morphology kernels — created once, reused every frame
_ELLIPSE_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_ELLIPSE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Display colors (BGR) ─────────────────────────────────────
COLOR_HYBRID   = (0, 200, 255)
COLOR_YOLO     = (255, 120,   0)
COLOR_CSRT     = (200, 200,   0)
COLOR_HSV      = (  0, 255, 100)
COLOR_RED_YOLO = (  0, 100, 255)
COLOR_BLUE_YOLO= (255, 180,   0)
COLOR_RED_HSV  = (  0, 255,   0)
COLOR_BLUE_HSV = (255, 255,   0)
COLOR_WAITING  = (  0, 165, 255)
COLOR_CROSSHAIR= (  0, 255, 255)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_MODE_COLORS = {"hybrid": COLOR_HYBRID, "yolo": COLOR_YOLO, "csrt": COLOR_CSRT}
_FUSION_ALPHA = 0.7   # How much weight YOLO gets in the hybrid position estimate

# Type aliases for readability
BBox    = Tuple[int, int, int, int]
HsvResult  = Optional[tuple]
YoloResult = Optional[tuple]
Position   = Optional[tuple]


# ──────────────────────────────────────────────────────────
#  THREADED CAMERA
# ──────────────────────────────────────────────────────────
class ThreadedCamera:
    """Captured frames in a background thread to reduce latency."""
    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


# ──────────────────────────────────────────────────────────
#  COLOR MASK COMPUTATION  (biggest speed win)
# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────
#  DYNAMIC HSV MANAGER
# ──────────────────────────────────────────────────────────
class HSVManager:
    """Manages HSV ranges and provides trackbars for real-time tuning."""
    def __init__(self):
        # Red ranges (two bands for hue wrap-around)
        self.red_low1  = [0, 130, 50]
        self.red_high1 = [10, 255, 255]
        self.red_low2  = [165, 130, 50]
        self.red_high2 = [180, 255, 255]
        # Broader blue range for better hexagon detection
        self.blue_low  = [95, 80, 40]
        self.blue_high = [135, 255, 255]
        self.win_name  = "HSV Tuning"
        self.created   = False

    def setup_trackbars(self):
        if self.created: return
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        def n(x): pass
        # Red Band 1
        cv2.createTrackbar("R1_H_min", self.win_name, self.red_low1[0],  180, n)
        cv2.createTrackbar("R1_H_max", self.win_name, self.red_high1[0], 180, n)
        cv2.createTrackbar("R_S_min",  self.win_name, self.red_low1[1],  255, n)
        cv2.createTrackbar("R_V_min",  self.win_name, self.red_low1[2],  255, n)
        # Blue Band
        cv2.createTrackbar("B_H_min", self.win_name, self.blue_low[0],  180, n)
        cv2.createTrackbar("B_H_max", self.win_name, self.blue_high[0], 180, n)
        cv2.createTrackbar("B_S_min", self.win_name, self.blue_low[1],  255, n)
        cv2.createTrackbar("B_V_min", self.win_name, self.blue_low[2],  255, n)
        self.created = True

    def get_ranges(self):
        if not self.created:
            return ([np.array(self.red_low1), np.array(self.red_high1)],
                    [np.array(self.red_low2), np.array(self.red_high2)]), \
                   (np.array(self.blue_low), np.array(self.blue_high))

        r1_h_min = cv2.getTrackbarPos("R1_H_min", self.win_name)
        r1_h_max = cv2.getTrackbarPos("R1_H_max", self.win_name)
        r_s_min  = cv2.getTrackbarPos("R_S_min",  self.win_name)
        r_v_min  = cv2.getTrackbarPos("R_V_min",  self.win_name)
        
        b_h_min = cv2.getTrackbarPos("B_H_min", self.win_name)
        b_h_max = cv2.getTrackbarPos("B_H_max", self.win_name)
        b_s_min = cv2.getTrackbarPos("B_S_min", self.win_name)
        b_v_min = cv2.getTrackbarPos("B_V_min", self.win_name)

        red = [
            (np.array([r1_h_min, r_s_min, r_v_min]), np.array([r1_h_max, 255, 255])),
            (np.array([180 - r1_h_max, r_s_min, r_v_min]), np.array([180, 255, 255])),
        ]
        blue = (np.array([b_h_min, b_s_min, b_v_min]), np.array([b_h_max, 255, 255]))
        return red, blue

hsv_manager = HSVManager()

def compute_masks(small_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce red and blue foreground masks from a single blur + HSV conversion.
    Working on the downscaled frame (PROC_WIDTH × PROC_HEIGHT) keeps this fast.
    """
    # Median blur is better for salt-and-pepper reflections
    denoised = cv2.medianBlur(small_frame, 5)
    blurred  = cv2.GaussianBlur(denoised, (5, 5), 0)
    hsv      = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    red_ranges, blue_range = hsv_manager.get_ranges()

    # Red mask
    red = cv2.bitwise_or(
        cv2.inRange(hsv, *red_ranges[0]),
        cv2.inRange(hsv, *red_ranges[1]),
    )
    red = cv2.bitwise_and(red, cv2.bitwise_not(cv2.inRange(hsv, *HSV_SKIN_RANGE)))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, _ELLIPSE_9)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN,  _ELLIPSE_3)

    # Blue mask
    blue = cv2.inRange(hsv, *blue_range)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, _ELLIPSE_9)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN,  _ELLIPSE_3)

    return red, blue


# ──────────────────────────────────────────────────────────
#  CONTOUR HELPERS
# ──────────────────────────────────────────────────────────
def is_valid_contour(contour: np.ndarray) -> bool:
    """Return True only if the contour is large, solid, and reasonably compact."""
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return False

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area < 1 or area / hull_area < MIN_FILL_RATIO:
        return False

    perimeter = cv2.arcLength(contour, True)
    return perimeter >= 1 and area / (perimeter * perimeter) >= MIN_COMPACTNESS


def classify_shape(contour: np.ndarray) -> str:
    """
    Guess the geometric shape using fuzzy logic (vertices + solidity + compactness).
    More robust than just counting vertices for distant or noisy targets.
    """
    area      = cv2.contourArea(contour)
    hull      = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1: return "Unknown"
    
    solidity  = area / hull_area
    perimeter = cv2.arcLength(hull, True)
    # A/P^2 metric (Circularity/Compactness)
    compactness = area / (perimeter * perimeter + 1e-5)
    
    # Vertex count from simplified polygon
    approx = cv2.approxPolyDP(hull, 0.035 * perimeter, True)
    sides  = len(approx)

    # 1. Square / Rectangle logic
    if 3 <= sides <= 5:
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h
        if solidity > 0.85:
            return "Square" if 0.8 <= aspect_ratio <= 1.2 else "Rectangle"

    # 2. Hexagon logic (Harder to detect vertices, so we rely on solidity + compactness)
    # Regular Hexagon A/P^2 is approx 0.072. We use a range.
    if 5 <= sides <= 8:
        if solidity > 0.90 and 0.06 < compactness < 0.085:
            return "Hexagon"

    # 3. Triangle logic
    if sides == 3:
        return "Triangle"

    # 4. Circle logic
    if sides >= 7:
        if solidity > 0.95 and compactness > 0.075:
            return "Circle"

    return f"{sides}-gon" if sides > 2 else "Unknown"


def find_best_contour(mask: np.ndarray) -> Optional[tuple]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if is_valid_contour(c)]
    if not valid:
        return None

    best = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)

    # Moment yerine bounding box merkezi — 3D nesnelerde daha kararlı
    cx = int((x + w / 2) * SCALE_X)
    cy = int((y + h / 2) * SCALE_Y)

    bbox = (int(x * SCALE_X), int(y * SCALE_Y), int(w * SCALE_X), int(h * SCALE_Y))
    area = cv2.contourArea(best) * SCALE_X * SCALE_Y
    scaled_contour = (best * np.array([SCALE_X, SCALE_Y], np.float32)).astype(np.int32)
    shape = classify_shape(best)

    return cx, cy, area, bbox, scaled_contour, shape


def detect_with_hsv(red_mask: np.ndarray, blue_mask: np.ndarray) -> HsvResult:
    """
    Run contour detection on both color masks and pick the best candidate.
    Preference order: red rectangle > blue hexagon > any red > any blue.
    """
    red_result  = find_best_contour(red_mask)
    blue_result = find_best_contour(blue_mask)

    if red_result  and red_result[5]  in ("Square", "Rectangle"):
        return (*red_result,  "red")
    if blue_result and blue_result[5] == "Hexagon":
        return (*blue_result, "blue")
    if red_result:   return (*red_result,  "red")
    if blue_result:  return (*blue_result, "blue")
    return None


# ──────────────────────────────────────────────────────────
#  YOLOv8n ONNX DETECTOR  (320×320 input)
# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────
#  THREADED YOLOv8n ONNX DETECTOR
# ──────────────────────────────────────────────────────────
class ThreadedYOLO:
    def __init__(self) -> None:
        self.ready   = False
        self._last_result: YoloResult = None
        self._input_frame = None
        self._roi_center  = None
        self._running     = False
        self._lock        = threading.Lock()
        self._new_frame   = threading.Event()

        try:
            self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.ready = True
            print(f"[YOLO] Loaded: {MODEL_PATH}  input: {YOLO_INPUT_SIZE}×{YOLO_INPUT_SIZE}")
        except Exception as err:
            print(f"[YOLO] Could not load model ({err}) — running in HSV-only mode.")

    def start(self):
        if not self.ready or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while self._running:
            if self._new_frame.wait(timeout=0.01):
                self._new_frame.clear()
                with self._lock:
                    frame = self._input_frame.copy()
                    roi_center = self._roi_center
                
                result = self._run_inference(frame, roi_center)
                
                with self._lock:
                    self._last_result = result

    def _run_inference(self, frame: np.ndarray, roi_center: Optional[Tuple[int, int]] = None) -> YoloResult:
        frame_h, frame_w = frame.shape[:2]

        crop_x_offset = 0
        crop_y_offset = 0
        scale = 1.0
        pad_x, pad_y = 0, 0

        if roi_center is not None:
            # Frame Cutter (ROI Cropping) Logic
            cx, cy = roi_center
            half_size = YOLO_INPUT_SIZE // 2
            
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(frame_w, cx + half_size)
            y2 = min(frame_h, cy + half_size)

            if x2 - x1 < YOLO_INPUT_SIZE:
                if x1 == 0: x2 = min(frame_w, YOLO_INPUT_SIZE)
                if x2 == frame_w: x1 = max(0, frame_w - YOLO_INPUT_SIZE)
            if y2 - y1 < YOLO_INPUT_SIZE:
                if y1 == 0: y2 = min(frame_h, YOLO_INPUT_SIZE)
                if y2 == frame_h: y1 = max(0, frame_h - YOLO_INPUT_SIZE)

            crop_w = x2 - x1
            crop_h = y2 - y1
            
            crop_x_offset = x1
            crop_y_offset = y1

            crop = frame[y1:y2, x1:x2]
            
            canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, np.uint8)
            pad_x = (YOLO_INPUT_SIZE - crop_w) // 2
            pad_y = (YOLO_INPUT_SIZE - crop_h) // 2
            canvas[pad_y:pad_y + crop_h, pad_x:pad_x + crop_w] = crop
        else:
            scale = YOLO_INPUT_SIZE / max(frame_w, frame_h)
            new_w, new_h = int(frame_w * scale), int(frame_h * scale)
            pad_x, pad_y = (YOLO_INPUT_SIZE - new_w) // 2, (YOLO_INPUT_SIZE - new_h) // 2

            canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, np.uint8)
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        self.net.setInput(
            cv2.dnn.blobFromImage(canvas, 1 / 255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), swapRB=True)
        )
        raw_output = self.net.forward()[0].T

        boxes, scores, class_ids = [], [], []
        for row in raw_output:
            class_scores = row[4:]
            best_class   = int(np.argmax(class_scores))
            confidence   = float(class_scores[best_class])
            if confidence < CONFIDENCE_THR:
                continue
            cx_r, cy_r, bw, bh = row[:4]
            x1 = int((cx_r - bw / 2 - pad_x) / scale)
            y1 = int((cy_r - bh / 2 - pad_y) / scale)
            boxes.append([x1 + crop_x_offset, y1 + crop_y_offset, int(bw / scale), int(bh / scale)])
            scores.append(confidence)
            class_ids.append(best_class)

        if not boxes:
            return None

        kept = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THR, NMS_THR)
        if len(kept) == 0:
            return None

        idx = kept[0] if isinstance(kept[0], (int, np.integer)) else kept[0][0]
        x, y, w, h = boxes[idx]
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        if w <= 0 or h <= 0:
            return None

        color_label = "red" if class_ids[idx] == 0 else "blue"
        return x + w // 2, y + h // 2, w * h, (x, y, w, h), scores[idx], color_label

    def update(self, frame: np.ndarray, roi_center: Optional[Tuple[int, int]] = None) -> YoloResult:
        if not self.ready:
            return None
        with self._lock:
            self._input_frame = frame
            self._roi_center  = roi_center
        self._new_frame.set()
        return self._last_result

    def stop(self):
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join()


# ──────────────────────────────────────────────────────────
#  TRACKER  (CSRT if available, CamShift as fallback)
# ──────────────────────────────────────────────────────────
class Tracker:
    def __init__(self) -> None:
        self._tracker      = None
        self._camshift_hist = None
        self._camshift_bbox: Optional[BBox] = None
        self.alive         = False
        self._use_csrt     = hasattr(cv2, "TrackerCSRT_create")
        mode_name = "CSRT (opencv-contrib)" if self._use_csrt else "CamShift (fallback)"
        print(f"[Tracker] Using {mode_name}.")

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        x, y, w, h = [int(v) for v in bbox]
        if w < 10 or h < 10:
            return

        if self._use_csrt:
            self._tracker = cv2.TrackerCSRT_create()
            self._tracker.init(frame, (x, y, w, h))
        else:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                return
            hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
            hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            self._camshift_hist = hist
            self._camshift_bbox = (x, y, w, h)

        self.alive = True

    def update(self, frame: np.ndarray, foreground_mask: np.ndarray) -> Optional[tuple]:
        if not self.alive:
            return None

        if self._use_csrt:
            success, raw_bbox = self._tracker.update(frame)
            if not success:
                self.reset()
                return None
            x, y, w, h = (int(v) for v in raw_bbox)
        else:
            if self._camshift_hist is None:
                self.reset()
                return None
            hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            back_proj = cv2.calcBackProject([hsv], [0], self._camshift_hist, [0, 180], 1)
            back_proj &= foreground_mask
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            rot_rect, _ = cv2.CamShift(back_proj, self._camshift_bbox, criteria)
            x, y, w, h  = cv2.boundingRect(cv2.boxPoints(rot_rect).astype(np.int32))
            self._camshift_bbox = (max(0, x), max(0, y), w, h)

        frame_h, frame_w = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        if w <= 0 or h <= 0:
            self.reset()
            return None

        roi = foreground_mask[y:y + h, x:x + w]
        if roi.size == 0 or cv2.countNonZero(roi) / (w * h) < TRACKER_MASK_THR:
            self.reset()
            return None

        return x + w // 2, y + h // 2, w * h, (x, y, w, h)

    def reset(self) -> None:
        self._tracker      = None
        self._camshift_hist = None
        self._camshift_bbox = None
        self.alive         = False


# ──────────────────────────────────────────────────────────
#  SAFETY VALIDATOR
# ──────────────────────────────────────────────────────────
class Validator:
    """Rejects detections that look physically impossible frame-to-frame."""

    def __init__(self) -> None:
        self._prev_area: Optional[float] = None
        self._prev_cx:   Optional[int]   = None
        self._prev_cy:   Optional[int]   = None

    def check(self, frame: np.ndarray, position: Position) -> bool:
        if position is None:
            return False

        cx, cy, area, bbox = position[:4]

        # Reject sudden area explosions or collapses
        if self._prev_area is not None:
            ratio = area / self._prev_area
            if not (1 / MAX_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
                self._clear_history()
                return False

        # Reject teleportation
        if self._prev_cx is not None:
            dx = cx - self._prev_cx
            dy = cy - self._prev_cy
            if dx * dx + dy * dy > MAX_JUMP_SQ:
                self._clear_history()
                return False

        # Reject washed-out or near-black blobs
        x, y, w, h = bbox
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if hsv[:, :, 1].mean() < MIN_SATURATION_VAL or hsv[:, :, 2].mean() < MIN_SATURATION_VAL:
            self._clear_history()
            return False

        self._prev_area = area
        self._prev_cx   = cx
        self._prev_cy   = cy
        return True

    def _clear_history(self) -> None:
        self._prev_area = self._prev_cx = self._prev_cy = None

    def reset(self) -> None:
        self._clear_history()


# ──────────────────────────────────────────────────────────
#  PD CONTROLLER
# ──────────────────────────────────────────────────────────
class PDController:
    """Proportional-Derivative controller for centering the drone on the target."""

    def __init__(self) -> None:
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0

    def compute(self, error_x: float, error_y: float) -> Tuple[float, float]:
        vel_x = float(np.clip(KP * error_y + KD * (error_y - self._prev_error_y), -MAX_VEL, MAX_VEL))
        vel_y = float(np.clip(KP * error_x + KD * (error_x - self._prev_error_x), -MAX_VEL, MAX_VEL))
        self._prev_error_x = error_x
        self._prev_error_y = error_y
        return vel_x, vel_y


# ──────────────────────────────────────────────────────────
#  HYBRID FUSION
# ──────────────────────────────────────────────────────────
def fuse_detections(yolo_result: YoloResult, hsv_result: Optional[tuple]) -> Position:
    """
    Blend YOLO and HSV positions into a single estimate.
    If YOLO result is present, it makes the primary decision and overrides HSV.
    """
    if yolo_result:
        return (*yolo_result[:4], "yolo")
    if hsv_result:
        return (*hsv_result[:4],  "hsv")
    return None


# ──────────────────────────────────────────────────────────
#  VISUALIZATION
# ──────────────────────────────────────────────────────────
def draw_overlay(frame, position, hsv_result, yolo_result, frame_center, vel_x, vel_y,
                 fps, debug_mode, yolo_enabled):
    """Draw all HUD elements onto the frame in-place."""
    cx_frame, cy_frame = frame_center
    frame_width = frame.shape[1]

    # Crosshair at frame center
    cv2.line(frame, (cx_frame - 25, cy_frame), (cx_frame + 25, cy_frame), COLOR_CROSSHAIR, 1)
    cv2.line(frame, (cx_frame, cy_frame - 25), (cx_frame, cy_frame + 25), COLOR_CROSSHAIR, 1)

    # YOLO bounding box
    if yolo_result and yolo_enabled:
        x, y, w, h = yolo_result[3]
        color = COLOR_RED_YOLO if yolo_result[5] == "red" else COLOR_BLUE_YOLO
        label = "Red Square" if yolo_result[5] == "red" else "Blue Hexagon"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        cv2.putText(frame, f"{label} {yolo_result[4]:.2f}", (x, y - 6), _FONT, 0.45, color, 1)

    # HSV contour outline
    if hsv_result:
        color = COLOR_RED_HSV if hsv_result[6] == "red" else COLOR_BLUE_HSV
        cv2.drawContours(frame, [cv2.convexHull(hsv_result[4])], -1, color, 1)

    # Main tracking overlay
    if position:
        cx, cy, area, bbox = position[:4]
        mode = position[4] if len(position) > 4 else "-"
        x, y, w, h = bbox
        color = _MODE_COLORS.get(mode, COLOR_HSV)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
        cv2.line(frame, (cx_frame, cy_frame), (cx, cy), (0, 0, 255), 1)

        error_x = cx - cx_frame
        error_y = cy - cy_frame
        shape_label = hsv_result[5] if hsv_result else "-"
        color_label = hsv_result[6].upper() if hsv_result else "-"

        info_lines = [
            f"Mode: {mode.upper()}   Shape: {shape_label}   [{color_label}]",
            f"Area: {area:.0f}px   Error: ({error_x:+d}, {error_y:+d})",
            f"Vx: {vel_x:+.3f}   Vy: {vel_y:+.3f} m/s",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 28 + i * 22), _FONT, 0.52, color, 1)
    else:
        cv2.putText(frame, "Waiting for target...", (10, 32), _FONT, 0.62, COLOR_WAITING, 2)

    # YOLO on/off badge
    badge_text  = "Y:ON" if yolo_enabled else "Y:OFF"
    badge_color = (40, 180, 40) if yolo_enabled else (40, 40, 180)
    cv2.rectangle(frame, (frame_width - 75, 4), (frame_width - 4, 30), badge_color, -1)
    cv2.putText(frame, badge_text,       (frame_width - 70, 23), _FONT, 0.52, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 75, 52), _FONT, 0.52, COLOR_CROSSHAIR, 1)

    if debug_mode:
        cv2.putText(frame, "[DEBUG]", (5, frame.shape[0] - 8), _FONT, 0.4, COLOR_CROSSHAIR, 1)


# ──────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────
def main() -> None:
    cam = ThreadedCamera(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT).start()
    time.sleep(1.0)  # Allow camera to warm up

    tracker   = Tracker()
    validator = Validator()
    pd        = PDController()
    yolo      = ThreadedYOLO()
    yolo.start()
    yolo_on   = yolo.ready

    fps, frame_count, fps_timer = 0.0, 0, time.time()
    debug_mode = False
    proc_size  = (PROC_WIDTH, PROC_HEIGHT)

    print("\nControls:\n  Q = Quit\n  D = Toggle Debug (Masks & Tuning)\n  R = Reset Tracker\n  Y = Toggle YOLO\n")

    try:
        while True:
            grabbed, frame = cam.read()
            if not grabbed:
                break

            frame_h, frame_w = frame.shape[:2]
            frame_center = (frame_w // 2, frame_h // 2)

            # FPS counter
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps         = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now

            # ── Downscale and build color masks ─────────────
            small_frame            = cv2.resize(frame, proc_size, interpolation=cv2.INTER_LINEAR)
            red_mask, blue_mask    = compute_masks(small_frame)
            combined_mask_small    = cv2.bitwise_or(red_mask, blue_mask)
            combined_mask_fullsize = cv2.resize(combined_mask_small, (frame_w, frame_h),
                                                interpolation=cv2.INTER_LINEAR)

            # ── Detection ───────────────────────────────────
            hsv_result  = detect_with_hsv(red_mask, blue_mask)
            
            roi_center = None
            if tracker.alive and hasattr(tracker, "_camshift_bbox") and tracker._camshift_bbox:
                # If tracker is alive, we use its center
                tx, ty, tw, th = tracker._camshift_bbox
                roi_center = (tx + tw // 2, ty + th // 2)
            elif hsv_result:
                roi_center = (hsv_result[0], hsv_result[1])

            yolo_result = yolo.update(frame, roi_center) if yolo_on else None

            # Build an HSV position tuple compatible with fuse_detections
            hsv_pos = (hsv_result[0], hsv_result[1], hsv_result[2], hsv_result[3]) if hsv_result else None
            fused   = fuse_detections(yolo_result, hsv_pos)

            # ── Tracker ─────────────────────────────────────
            if not tracker.alive and fused:
                tracker.initialize(frame, fused[3])

            tracker_result = tracker.update(frame, combined_mask_fullsize)

            # Main decision logic: YOLO overrides tracker if present
            if fused and fused[4] == "yolo":
                position = fused
                tracker.initialize(frame, fused[3]) # Reset tracker to YOLO's definitive box
            elif tracker_result:
                position = (*tracker_result, "csrt")
            elif fused:
                position = fused
                if not tracker.alive:
                    tracker.initialize(frame, fused[3])
            else:
                position = None

            # ── Validation ──────────────────────────────────
            if not validator.check(frame, position):
                position = None
                tracker.reset()

            # ── PD control output ───────────────────────────
            vel_x = vel_y = 0.0
            if position:
                vel_x, vel_y = pd.compute(
                    position[0] - frame_center[0],
                    position[1] - frame_center[1],
                )

            # ── Debug windows & HUD ─────────────────────────
            if debug_mode:
                hsv_manager.setup_trackbars()
                cv2.imshow("Red Mask",  cv2.resize(red_mask,  (frame_w, frame_h)))
                cv2.imshow("Blue Mask", cv2.resize(blue_mask, (frame_w, frame_h)))

            draw_overlay(frame, position, hsv_result, yolo_result, frame_center,
                         vel_x, vel_y, fps, debug_mode, yolo_on)
            cv2.imshow("Drone Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                if not debug_mode:
                    if cv2.getWindowProperty("Red Mask", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Red Mask")
                    if cv2.getWindowProperty("Blue Mask", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Blue Mask")
                    if cv2.getWindowProperty(hsv_manager.win_name, cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow(hsv_manager.win_name)
                    hsv_manager.created = False
            elif key == ord('r'):
                tracker.reset()
                validator.reset()
                print("[INFO] Tracker and validator reset.")
            elif key == ord('y'):
                yolo_on = not yolo_on
                print(f"[INFO] YOLO {'enabled' if yolo_on else 'disabled'}.")

    finally:
        print("[INFO] Shutting down...")
        cam.stop()
        yolo.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()