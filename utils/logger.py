import csv
from pathlib import Path


class CsvLogger:
    FIELDNAMES = (
        "timestamp",
        "frame_id",
        "state",
        "cv_target",
        "yolo_target",
        "fusion_confidence",
        "yolo_verified",
        "error_x",
        "error_y",
        "drop_ready",
        "fps",
        "yolo_infer_ms",
    )

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.file = None
        self.writer = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
            self.writer.writeheader()

    def write(self, row):
        if self.writer is None:
            return
        self.writer.writerow({key: row.get(key, "") for key in self.FIELDNAMES})
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
