# Teknofest IHA ROS 2 Architecture

This package wraps the existing OpenCV, YOLO, and fusion code with ROS 2 nodes.
The first MVP uses JSON payloads over `std_msgs/String` so the system can run
before custom ROS messages are introduced.

## Documentation Map

- `docs/UNIX_MODULAR_ARCHITECTURE.md`
  - Explains the Unix-style process/module boundaries.
  - Use this for jury/system architecture explanation.
- `docs/TOPICS_AND_CONTRACTS.md`
  - Defines every runtime ROS topic contract and JSON payload.
  - Use this when integrating another camera, MAVLink endpoint, or custom ROS messages.
- `docs/STATE_MACHINES.md`
  - Explains the fusion and mission state machines separately.
  - Use this for MATLAB/Stateflow diagrams.
- `docs/CODE_REFERENCE.md`
  - File-by-file explanation of operational code.
  - Use this when handing the code to another teammate or preparing for technical questions.
- `docs/REAL_UAV_READINESS_CHECKLIST.md`
  - Lists the real-aircraft integration checks for camera, MAVLink, payload, geofence, and OBS proof.
  - Use this during the final month before competition.
- `docs/GCS_MAVLINK_INTEGRATION.md`
  - Explains how MAVProxy splits telemetry between ROS 2 and a ground-control station.
  - Use this for QGroundControl, Mission Planner, telemetry radio, and real Pixhawk checks.

## File/Class Plan

- `teknofest_iha/adapters/opencv_adapter.py`
  - `OpenCVAdapter`: calls the existing `vision.opencv_detector.OpenCVDetector`.
- `teknofest_iha/adapters/yolo_adapter.py`
  - `YoloAdapter`: owns the async YOLO worker using `models_archive/iha_best.pt`.
- `teknofest_iha/adapters/fusion_adapter.py`
  - `FusionAdapter`: fuses OpenCV color authority with YOLO shape verification per target.
- `teknofest_iha/adapters/mavlink_adapter.py`
  - `MavlinkAdapter`: only ArduPilot/MAVLink transport and command details.
- `teknofest_iha/core/state_machine.py`
  - `MissionStateMachine`: task-level state transitions.
- `teknofest_iha/core/search_pattern.py`
  - `LawnmowerSearchPattern`: bounded area waypoints.
- `teknofest_iha/core/geofence.py`
  - `Geofence`: warning/violation checks and velocity clamping.
- `teknofest_iha/nodes/*`
  - ROS 2 node wrappers for perception, fusion, mission, safety, MAVLink, and debug viewing.

## MVP Topics

- `/camera`: input `sensor_msgs/msg/Image`
- `/perception/raw_detections`: JSON `std_msgs/msg/String`
- `/fusion/target`: JSON `std_msgs/msg/String`
- `/drone/cmd_takeoff`, `/drone/cmd_velocity`, `/drone/cmd_land`, `/drone/cmd_drop`: JSON/String commands
- `/drone/state`, `/drone/local_position`, `/drone/altitude`: telemetry/status
- `/mission/state`, `/mission/event`: mission state and events
