from vision.fusion import fuse_detections
from vision.yolo_detector import YoloAsyncDetector


def test_yolo_unknown_label_is_displayed_as_square():
    assert YoloAsyncDetector._normalize_target_type("unknown") == "square"
    assert YoloAsyncDetector._normalize_target_type("square_unknown") == "square"


def test_square_yolo_detection_verifies_colored_opencv_target():
    cv_det = {
        "source": "opencv",
        "target_type": "red_square",
        "bbox": (10, 10, 20, 20),
        "center": (20, 20),
        "confidence": 0.9,
        "state": "DETECTED",
        "frame_id": 10,
    }
    yolo_det = {
        "source": "yolo",
        "target_type": "square",
        "bbox": (10, 10, 20, 20),
        "center": (20, 20),
        "confidence": 0.95,
        "state": "DETECTED",
        "frame_id": 10,
    }

    fused = fuse_detections([cv_det], [yolo_det], frame_id=10)

    assert fused is not None
    assert fused["target_type"] == "red_square"
    assert fused["yolo_verified"] is True
    assert fused["matched_yolo"]["target_type"] == "square"
