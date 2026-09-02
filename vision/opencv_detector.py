"""OpenCV color and shape detector.

This detector is the color authority in the current architecture. It separates
red_square and blue_square using color masks and geometric filters, while YOLO
is used as a general square verifier.
"""

import cv2
import numpy as np

import config
from vision.detection_types import clamp_bbox_xywh, make_detection
from vision.kalman_filter import KalmanFilter2D


DETECTED = "DETECTED"
PREDICTED = "PREDICTED"
LOST = "LOST"


def create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    return None


class OpenCVDetector:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE)
        self.red_lower_1 = np.array(config.RED_LOWER_1, dtype=np.uint8)
        self.red_upper_1 = np.array(config.RED_UPPER_1, dtype=np.uint8)
        self.red_lower_2 = np.array(config.RED_LOWER_2, dtype=np.uint8)
        self.red_upper_2 = np.array(config.RED_UPPER_2, dtype=np.uint8)
        self.blue_lower = np.array(config.BLUE_LOWER, dtype=np.uint8)
        self.blue_upper = np.array(config.BLUE_UPPER, dtype=np.uint8)
        self.clahe = (
            cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT,
                tileGridSize=config.CLAHE_TILE_GRID,
            )
            if config.USE_CLAHE
            else None
        )
        self.target_specs = list(config.TARGET_SPECS)
        self.target_state = {
            spec["name"]: {
                "state": LOST,
                "last_bbox": None,
                "lost_frames": 0,
                "last_centroid": None,
                "tracker": None,
                "tracker_active": False,
                "track_frames": 0,
                "kalman": KalmanFilter2D(dt=1.0 / 30.0, process_noise=1e-2, measurement_noise=1e-1),
                "error": None,
            }
            for spec in self.target_specs
        }

    def preproc(self, frame_bgr):
        blurred = cv2.GaussianBlur(frame_bgr, config.BLUR_KERNEL_SIZE, 0)
        if self.clahe is not None:
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            self.clahe.apply(l_chan, dst=l_chan)
            lab = cv2.merge((l_chan, a_chan, b_chan))
            blurred = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    def create_mask(self, hsv, key):
        if key == "red":
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, self.red_lower_1, self.red_upper_1),
                cv2.inRange(hsv, self.red_lower_2, self.red_upper_2),
            )
        elif key == "blue":
            mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        else:
            raise ValueError(f"Unknown mask key: {key}")

        cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, dst=mask, iterations=1)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, dst=mask, iterations=2)
        return mask

    @staticmethod
    def compute_sides(pts):
        return np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)

    @staticmethod
    def compute_angles(pts):
        arm1 = np.roll(pts, 1, axis=0) - pts
        arm2 = np.roll(pts, -1, axis=0) - pts
        dot = arm1[:, 0] * arm2[:, 0] + arm1[:, 1] * arm2[:, 1]
        n1 = np.linalg.norm(arm1, axis=1)
        n2 = np.linalg.norm(arm2, axis=1)
        denom = np.maximum(n1 * n2, 1e-9)
        cosang = np.clip(dot / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    @staticmethod
    def find_best_approx(cnt, perimeter, expected_vertices):
        for eps in config.POLY_EPS_RANGE:
            approx = cv2.approxPolyDP(cnt, eps * perimeter, True)
            if len(approx) == expected_vertices:
                return approx
        return None

    def shape_check(self, approx, cnt, area, perimeter, ideal_vertices, ideal_angle, ideal_compact):
        if len(approx) != ideal_vertices:
            return False, 0.0, "vertices"
        if perimeter < 1e-6:
            return False, 0.0, "perimeter"

        compactness = 4.0 * np.pi * area / (perimeter * perimeter)
        if abs(ideal_compact - compactness) / ideal_compact > config.COMPACTNESS_TOL:
            return False, 0.0, "compactness"

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area < 1 or area / hull_area < config.MIN_SOLIDITY:
            return False, 0.0, "solidity"

        if not cv2.isContourConvex(approx):
            return False, 0.0, "convexity"

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or abs(w / h - 1.0) > config.SQUARE_AR_TOL:
            return False, 0.0, "aspect"

        pts = approx.reshape(-1, 2).astype(np.float64)
        sides = self.compute_sides(pts)
        if np.any(sides < config.MIN_SIDE_LENGTH):
            return False, 0.0, "side_length"
        mean_side = sides.mean()
        if mean_side < 1e-6 or not np.all(np.abs(sides - mean_side) / mean_side <= config.SIDE_TOLERANCE):
            return False, 0.0, "side_variance"

        angles = self.compute_angles(pts)
        if not np.all(np.abs(angles - ideal_angle) <= config.ANGLE_TOL_DEG):
            return False, 0.0, "angle"

        side_score = max(0.0, 1.0 - float(np.std(sides) / max(mean_side, 1e-6)))
        compact_score = max(0.0, 1.0 - abs(ideal_compact - compactness) / ideal_compact)
        fill_score = min(1.0, area / max(1.0, w * h))
        return True, float(np.clip(0.35 * side_score + 0.35 * compact_score + 0.30 * fill_score, 0.0, 1.0)), "ok"

    def _find_target(self, hsv, spec):
        mask = self.create_mask(hsv, spec["mask_fn"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                break
            if area > config.MAX_CONTOUR_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            approx = self.find_best_approx(cnt, perimeter, spec["vertices"])
            if approx is None:
                continue
            ok, score, reason = self.shape_check(
                approx,
                cnt,
                area,
                perimeter,
                spec["vertices"],
                spec["ideal_angle"],
                spec["ideal_compact"],
            )
            if config.DEBUG:
                print(f"[CV] {spec['name']} shape={reason}")
            if not ok:
                continue
            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue
            bbox = cv2.boundingRect(cnt)
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            return bbox, center, score
        return None

    def detect(self, frame_bgr, frame_id=0):
        vis = frame_bgr.copy()
        hsv = self.preproc(frame_bgr)
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_center = (frame_w // 2, frame_h // 2)
        detections = []

        for spec in self.target_specs:
            name = spec["name"]
            state = self.target_state[name]
            result = self._find_target(hsv, spec)
            tracker = state["tracker"]
            track_ok = False
            track_bbox = None

            if tracker is not None:
                track_ok, tb = tracker.update(frame_bgr)
                if track_ok:
                    track_bbox = clamp_bbox_xywh(tb, frame_bgr.shape)

            if result is not None:
                bbox, center, shape_score = result
                bbox = clamp_bbox_xywh(bbox, frame_bgr.shape)
                error = (center[0] - frame_center[0], center[1] - frame_center[1])
                state.update(
                    {
                        "state": DETECTED,
                        "last_bbox": bbox,
                        "lost_frames": 0,
                        "last_centroid": center,
                        "track_frames": 0,
                        "error": error,
                    }
                )
                state["kalman"].update(center[0], center[1])

                new_tracker = create_csrt_tracker()
                if new_tracker is not None:
                    new_tracker.init(frame_bgr, bbox)
                    state["tracker"] = new_tracker
                    state["tracker_active"] = True

                det = make_detection("opencv", name, bbox, shape_score, DETECTED, frame_id, error=error)
                detections.append(det)
                self._draw_detection(vis, det, spec["color_bgr"], 2)
                continue

            state["lost_frames"] += 1
            if track_ok and state["track_frames"] < config.MAX_TRACK_FRAMES:
                state["track_frames"] += 1
                bbox = track_bbox
                center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                error = (center[0] - frame_center[0], center[1] - frame_center[1])
                state.update(
                    {
                        "state": DETECTED,
                        "last_bbox": bbox,
                        "last_centroid": center,
                        "error": error,
                        "tracker_active": True,
                    }
                )
                state["kalman"].update(center[0], center[1])
                det = make_detection("opencv", name, bbox, 0.55, DETECTED, frame_id, error=error)
                detections.append(det)
                self._draw_detection(vis, det, spec["color_bgr"], 1)
                continue

            state["tracker"] = None
            state["tracker_active"] = False
            state["track_frames"] = 0
            prediction = state["kalman"].predict()
            if (
                prediction is not None
                and state["last_bbox"] is not None
                and state["lost_frames"] <= config.MAX_KALMAN_PREDICT_FRAMES
            ):
                kx, ky = prediction
                _, _, bw, bh = state["last_bbox"]
                bbox = clamp_bbox_xywh((int(kx - bw / 2), int(ky - bh / 2), bw, bh), frame_bgr.shape)
                center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                error = (center[0] - frame_center[0], center[1] - frame_center[1])
                state.update({"state": PREDICTED, "last_centroid": center, "error": error})
                det = make_detection("opencv_kalman", name, bbox, 0.35, PREDICTED, frame_id, error=error)
                detections.append(det)
                self._draw_detection(vis, det, (0, 255, 166), 1)
            else:
                state["state"] = LOST
                if state["lost_frames"] > config.MAX_KALMAN_PREDICT_FRAMES:
                    state["kalman"].reset()

        cv2.drawMarker(vis, frame_center, (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
        return vis, detections

    @staticmethod
    def _draw_detection(frame, det, color, thickness):
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(frame, det["center"], 4, color, -1)
        cv2.putText(
            frame,
            f"{det['target_type']} {det['confidence']:.2f}",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


_DEFAULT_DETECTOR = None


def detect_opencv(frame_bgr, frame_id=0):
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = OpenCVDetector()
    return _DEFAULT_DETECTOR.detect(frame_bgr, frame_id)
