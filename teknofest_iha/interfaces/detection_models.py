from __future__ import annotations

"""Detection and fusion message models.

The runtime ROS topics currently carry JSON over `std_msgs/String`. These
dataclasses define that JSON contract in one place so the rest of the code does
not build ad-hoc dictionaries everywhere.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    source: str
    target_type: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float
    state: str
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    error: tuple[float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Detection":
        known = {
            "source",
            "target_type",
            "bbox",
            "center",
            "confidence",
            "state",
            "frame_id",
            "timestamp",
            "error",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            source=str(data["source"]),
            target_type=str(data["target_type"]),
            bbox=tuple(int(v) for v in data["bbox"]),
            center=tuple(int(v) for v in data["center"]),
            confidence=float(data.get("confidence", data.get("fusion_confidence", 0.0))),
            state=str(data.get("state", "UNKNOWN")),
            frame_id=int(data.get("frame_id", 0)),
            timestamp=float(data.get("timestamp", time.time())),
            error=tuple(float(v) for v in data["error"]) if data.get("error") is not None else None,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(self.extra)
        data.pop("extra", None)
        return data


@dataclass
class RawDetectionPacket:
    frame_id: int
    timestamp: float
    opencv: list[dict[str, Any]]
    yolo: list[dict[str, Any]]
    yolo_ran: bool = False
    yolo_frame_id: int | None = None
    yolo_age_frames: int | None = None
    yolo_meta: dict[str, Any] | None = None
    yolo_error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "RawDetectionPacket":
        data = json.loads(text)
        return cls(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            opencv=list(data.get("opencv", [])),
            yolo=list(data.get("yolo", [])),
            yolo_ran=bool(data.get("yolo_ran", bool(data.get("yolo", [])))),
            yolo_frame_id=data.get("yolo_frame_id"),
            yolo_age_frames=data.get("yolo_age_frames"),
            yolo_meta=data.get("yolo_meta"),
            yolo_error=data.get("yolo_error"),
        )


@dataclass
class FusedTargetPacket:
    frame_id: int
    timestamp: float
    primary_target: str
    state: str
    targets: list[dict[str, Any]]
    selected: dict[str, Any] | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "FusedTargetPacket":
        data = json.loads(text)
        return cls(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            primary_target=str(data["primary_target"]),
            state=str(data["state"]),
            targets=list(data.get("targets", [])),
            selected=data.get("selected"),
        )
