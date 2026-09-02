# Gazebo + ArduPilot + ROS 2 MAVLink Runbook

This stack uses the ArduPilot Gazebo JSON backend. The important detail is that
SITL must be started with `-f gazebo-iris --model JSON`. If `--model JSON` is
omitted, Gazebo and SITL open their ports but MAVLink heartbeat never reaches
MAVProxy or ROS.

## 1. Start Gazebo

```bash
source /opt/ros/jazzy/setup.bash
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
gz sim -r /home/omer/ardupilot_gazebo/worlds/teknofest_bozkir.sdf
```

Quick checks:

```bash
gz topic -l | grep downward_camera
gz topic -e -t /world/teknofest_bozkir/model/iris/model/iris_with_standoffs/link/imu_link/sensor/imu_sensor/imu -n 1
```

## 2. Start ArduPilot SITL

```bash
cd /home/omer/ardupilot
source /home/omer/venv-ardupilot/bin/activate
export PATH=$HOME/venv-ardupilot/bin:$HOME/.local/bin:$PATH
python3 Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --no-rebuild --console --out=127.0.0.1:14551
```

Expected MAVProxy output includes:

```text
Detected vehicle 1:1 on link 0
STABILIZE>
```

## 3. Load SITL Params

The JSON backend may start without the Copter motor/frame params applied. If
`FRAME_CLASS` is `0` or `SERVO1_FUNCTION` is `0`, the vehicle can arm/takeoff
commands but the motors will not produce the correct lift. First load the common
params in the MAVProxy console:

```text
param load /home/omer/ardupilot/Tools/autotest/default_params/copter.parm
param load /home/omer/ardupilot/Tools/autotest/default_params/gazebo-iris.parm
param load /home/omer/ardupilot_gazebo/config/gazebo-iris-gimbal.parm
```

Then set and verify the critical params from the ROS workspace:

```bash
cd /home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package
source .ros_venv/bin/activate
python3 scripts/configure_sitl_params.py
```

The script also sets `PLND_ENABLED=0`, because the Teknofest world does not
currently provide a precision-landing target and ArduPilot can otherwise trigger
precision landing failsafe behavior during LAND.

The script sets `FS_CRASH_CHECK=0` for SITL only. In this Gazebo setup the
vehicle can touch down with enough simulated tilt/acceleration noise to trip the
crash detector after an otherwise completed landing.

Expected critical values:

```text
FRAME_CLASS 1
FRAME_TYPE 1
SERVO1_FUNCTION 33
SERVO2_FUNCTION 34
SERVO3_FUNCTION 35
SERVO4_FUNCTION 36
MOT_PWM_MIN 1100
MOT_PWM_MAX 1900
MNT1_TYPE 0
PLND_ENABLED 0
FS_CRASH_CHECK 0
ARMING_SKIPCHK 65535
```

Do not use `ARMING_SKIPCHK=65535` or `FS_CRASH_CHECK=0` on real aircraft.

Heartbeat checks:

```bash
cd /home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package
source .ros_venv/bin/activate
python3 -c 'from pymavlink import mavutil; m=mavutil.mavlink_connection("udpin:127.0.0.1:14551"); print(m.wait_heartbeat(timeout=5))'
```

## 4. Start ROS Bridge

```bash
cd /home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package
source .ros_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run teknofest_iha mavlink_bridge_node --ros-args --params-file install/teknofest_iha/share/teknofest_iha/config/mavlink.yaml
```

Verify:

```bash
ros2 topic echo /drone/state --once
ros2 topic echo /drone/local_position --once
ros2 topic pub --once /drone/cmd_mode std_msgs/msg/String "{data: '{\"mode\":\"GUIDED\"}'}"
ros2 topic echo /drone/state --once
```

Expected state after the mode command:

```text
"connected": true
"mode": "GUIDED"
```

## 5. Smoke Test Takeoff

Before running the full mission state machine, verify that the Gazebo model can
take off without rolling over:

```bash
cd /home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package
source .ros_venv/bin/activate
python3 scripts/configure_sitl_params.py
python3 scripts/sitl_takeoff_smoke.py --altitude 2.0 --observe-seconds 12
```

Expected result:

```text
summary max_alt=...
```

The script exits with an error if the vehicle does not climb or if roll exceeds
the stability threshold. If this fails, do not start the full mission yet; fix
the Gazebo model / motor mapping / frame params first.

## 6. Start Full Mission

After the smoke test passes, start the complete ROS stack:

```bash
cd /home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package
./scripts/run_full_ros_stack.sh
```

Monitor it from another terminal:

```bash
cd /home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package
./scripts/monitor_mission_topics.sh
```

The important fields are:

```text
/mission/state          state, active_target, released_targets
/drone/local_position   z should decrease in NED as altitude increases
/fusion/target          selected target and target_state
/mission/event          drop_payload events
```
