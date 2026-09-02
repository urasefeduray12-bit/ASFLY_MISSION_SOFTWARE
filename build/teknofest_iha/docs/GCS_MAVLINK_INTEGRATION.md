# GCS and MAVLink Integration

This project does not depend on Mission Planner. Mission Planner, QGroundControl,
or MAVProxy console are ground-control station options that can observe the same
ArduPilot link.

The recommended field architecture is:

```text
Pixhawk / ArduPilot
        |
        v
MAVProxy router
        |------------------> GCS UDP 14550
        |
        +------------------> ROS 2 UDP 14551
```

ROS 2 uses `config/mavlink_gcs_router.yaml`:

```yaml
connection: udpin:127.0.0.1:14551
```

That means the ROS MAVLink bridge waits for MAVProxy telemetry on UDP 14551.

## Why This Is Safer

- ROS does not care whether the GUI is Mission Planner, QGroundControl, or none.
- Linux can use QGroundControl without forcing Mission Planner through Wine.
- If Mission Planner is needed, it can run on a separate Windows laptop and listen
  to the router output.
- The mission code still owns autonomous decisions: perception, fusion, mission
  state, payload release, and RTL/land commands.

## SITL Test

Start ArduPilot SITL normally, then route it:

```bash
cd /home/omer/teknofest_iha_2_gorev
MAVLINK_MASTER=tcp:127.0.0.1:5760 ./scripts/start_mavlink_router.sh
```

Start ROS in a second terminal:

```bash
cd /home/omer/teknofest_iha_2_gorev
./scripts/run_field_ros_stack.sh
```

Connect a GCS to UDP port `14550` if you want a map/parameter interface.

## Real Pixhawk Over USB

Use whichever device appears for the flight controller:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

Then start the router:

```bash
cd /home/omer/teknofest_iha_2_gorev
MAVLINK_MASTER=/dev/ttyACM0,115200 ./scripts/start_mavlink_router.sh
```

Start ROS:

```bash
./scripts/run_field_ros_stack.sh
```

Mission starts only after an explicit command because `AUTOSTART` defaults to
`false`:

```bash
ros2 topic pub --once /mission/cmd_start std_msgs/msg/String "{data: start}"
```

## Real Pixhawk Over Telemetry Radio

Most telemetry radios appear as `/dev/ttyUSB0` and often use `57600` baud:

```bash
MAVLINK_MASTER=/dev/ttyUSB0,57600 ./scripts/start_mavlink_router.sh
```

If the radio uses another baud rate, change only the number after the comma.

## Ground-Control Station Options

QGroundControl on Linux:

- Open QGroundControl.
- It usually auto-detects UDP telemetry on `14550`.

Mission Planner:

- Prefer a Windows laptop if reliability matters.
- Connect with UDP on port `14550`.
- Mission Planner is only for monitoring, parameters, geofence setup, mode
  changes, and emergency intervention. The ROS stack still runs the mission.

## Pre-Competition Check

Before field day, verify these four lines:

```bash
ss -lunp | grep -E '14550|14551'
ros2 topic echo /drone/status --once
ros2 topic echo /drone/state --once
ros2 topic echo /mission/state --once
```

Expected result:

- `/drone/status` is not a heartbeat timeout.
- `/drone/state` shows ArduPilot mode/arming telemetry.
- `/mission/state` waits for start when `AUTOSTART=false`.
