import math


DEBUG = False

ENABLE_REAL_PAYLOAD = False

YOLO_EVERY_SEARCH = 5
YOLO_EVERY_CANDIDATE = 1
YOLO_EVERY_TRACKING = 10
YOLO_EVERY_UNSTABLE = 3
YOLO_EVERY_LOCKED = 10

MAX_YOLO_RESULT_AGE_FRAMES = 15
IOU_VERIFY_THRESH = 0.25
FUSION_CONF_THRESH = 0.64
YOLO_VERIFY_CONF_THRESH = 0.18

CENTER_TOL_X = 40
CENTER_TOL_Y = 40
LOCK_MIN_FRAMES = 10
CANDIDATE_MIN_FRAMES = 2
UNSTABLE_MIN_FRAMES = 2

MIN_CONTOUR_AREA = 800
MAX_CONTOUR_AREA = 80000
MORPH_KERNEL_SIZE = (7, 7)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
BLUR_KERNEL_SIZE = (9, 9)

COMPACTNESS_TOL = 0.25
POLY_EPS_RANGE = (0.020, 0.025, 0.030, 0.035, 0.040, 0.045)
MIN_SOLIDITY = 0.85
SIDE_TOLERANCE = 0.35
MIN_SIDE_LENGTH = 20
ANGLE_TOL_DEG = 22.0
SQUARE_AR_TOL = 0.35
MAX_TRACK_FRAMES = 30
MAX_KALMAN_PREDICT_FRAMES = 20

RED_LOWER_1 = (0, 95, 50)
RED_UPPER_1 = (10, 255, 255)
RED_LOWER_2 = (165, 95, 50)
RED_UPPER_2 = (180, 255, 255)
BLUE_LOWER = (85, 65, 40)
BLUE_UPPER = (135, 255, 255)

TARGET_SPECS = (
    {
        "name": "red_square",
        "label": "red square",
        "vertices": 4,
        "ideal_angle": 90.0,
        "ideal_compact": math.pi / 4.0,
        "color_bgr": (0, 140, 255),
        "mask_fn": "red",
    },
    {
        "name": "blue_square",
        "label": "blue square",
        "vertices": 4,
        "ideal_angle": 90.0,
        "ideal_compact": math.pi / 4.0,
        "color_bgr": (255, 170, 0),
        "mask_fn": "blue",
    },
)

YOLO_CLASS_MAP = {
    0: "square",
    1: "red_square",
    2: "blue_square",
}

YOLO_ROI_SCALE = 2.8
YOLO_MIN_ROI_SIZE = 160
