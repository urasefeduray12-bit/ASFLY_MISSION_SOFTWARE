# Real UAV Readiness Checklist

Bu belge, simden gercek IHA'ya geciste kontrol edilecek maddeleri listeler.
Amaç, calisan yazilim mimarisini bozmadan entegrasyon risklerini ayri ayri
dogrulamaktir.

## Degismemesi Beklenen Kodlar

Asagidaki moduller sim ve gercek IHA arasinda ayni kalmalidir:

- `teknofest_iha/core/state_machine.py`
- `teknofest_iha/core/search_pattern.py`
- `teknofest_iha/core/alignment_controller.py`
- `teknofest_iha/core/geofence.py`
- `teknofest_iha/adapters/fusion_adapter.py`
- `vision/fusion.py`
- `control/state_machine.py`
- `control/payload_logic.py`

Bu dosyalar domain mantigidir. Sahada sorun cikarsa once parametre/topic/adapter
katmani kontrol edilmelidir.

## Degismesi Muhtemel Parametreler

### Kamera

Dosya:

```text
config/perception.yaml
```

Kontroller:

- `camera_topic`
- `model_path`
- `yolo_imgsz`
- `yolo_conf`
- `publish_debug_image`

Gercek Pi camera icin once ROS topic dogrulanir:

```bash
ros2 topic list
ros2 topic info /camera
ros2 topic hz /camera
```

### MAVLink

Dosya:

```text
config/mavlink.yaml
```

Kontroller:

- connection string
- baudrate
- system/component id
- velocity command frame

Gercek Pixhawk'ta once sadece heartbeat test edilir. Mission baslatilmaz.

### Mission

Dosya:

```text
config/mission.yaml
```

Kontroller:

- `takeoff_altitude_m`
- `drop_altitude_m`
- `search_speed_mps`
- `align_max_speed_mps`
- `center_tolerance_px`
- `payload_servo`
- `payload_servo_map_json`
- `payload_pwm`
- `payload_reset_pwm`
- `coordinate_frame`
- `target_specs_json`

### Geofence

Dosya:

```text
config/mission.yaml
```

Kontroller:

- `x_min`
- `x_max`
- `y_min`
- `y_max`
- `warning_margin_m`
- `hard_margin_m`

Gercek saha koordinati netlesmeden failsafe kararlarini agresif yapmamak
gerekir.

## Servo / Payload Dogrulama

Payload mekanizmasi gercek IHA'ya baglanmadan once:

1. Servo kanali Mission Planner ile tek tek test edilir.
2. PWM ac/kapa degerleri not edilir.
3. `payload_dry_run=true` ile yazilim akisi izlenir.
4. IHA pervaneleri sokuluyken `payload_dry_run=false` servo testi yapilir.
5. Ucus testinden once mekanik takilma kontrol edilir.

Gercek servo icin config:

```yaml
payload_dry_run: false
payload_servo: 11
payload_servo_map_json: '{"blue_square":11,"red_square":12}'
payload_pwm: 1900
payload_reset_pwm: 1100
payload_hold_seconds: 0.8
```

## Aşamalı Gercek IHA Test Plani

### 1. Yazilim-only test

- ROS node'lar acilir.
- Kamera goruntusu gelir.
- `/perception/raw_detections` kontrol edilir.
- `/fusion/target` kontrol edilir.
- MAVLink baglanmaz.

### 2. MAVLink heartbeat test

- Pixhawk baglanir.
- Sadece `/drone/state` okunur.
- Arm/takeoff komutu verilmez.

### 3. Servo dry-run test

- Mission akisi `payload_dry_run=true` ile izlenir.
- Terminalde `PAYLOAD_RELEASED` gorulur.
- Fiziksel servo hareket etmez.

### 4. Servo ground test

- Pervaneler sokulur.
- `payload_dry_run=false`.
- Sadece payload komutu denenir.

### 5. Tethered / dusuk irtifa test

- Kisa takeoff.
- Manual override hazir.
- Arama pattern'i kisaltilmis saha ile denenir.

### 6. Tam mission rehearsal

- OBS kaydi.
- Mission console bloklari.
- Kamera overlay.
- Mission Planner telemetry.

## OBS Kaydi Icin Kanit Satirlari

Terminalde su satirlar kayitta gorunmelidir:

```text
[MISSION] state=SEARCH_TARGET  geofence=OK
[VISION ] opencv=DETECTED  yolo=VERIFIED
[FUSION ] target=blue_square  state=LOCKED
[GATE   ] release_gate=TRUE  lock_counter=10/10
[PAYLOAD] released=1/2
[MISSION] state=MISSION_COMPLETE
```

Bu satirlar, state machine'in sadece "uçtu" degil, hangi karar zinciriyle
payload biraktigini gosterir.

## Riskler

### Kamera gecikmesi

Belirti:

```text
YOLO_status=skipped/no fresh result
```

Cozum:

- `imgsz` dusur.
- Kamera FPS/topic frekansini kontrol et.
- ROI davranisini izle.

### Coordinate frame tersligi

Belirti:

IHA hedefe yaklasmak yerine uzaklasir.

Cozum:

- `coordinate_frame` parametresini kontrol et.
- `nav_x/nav_y` ile Gazebo/Mission Planner konumlarini karsilastir.

### Payload servo yanlis kanal

Belirti:

`PAYLOAD_RELEASED` terminalde gorunur ama mekanik hareket etmez.

Cozum:

- Mission Planner servo output test.
- `payload_servo_map_json`, `payload_pwm`, `payload_reset_pwm`.

### Geofence yanlis saha

Belirti:

IHA dogru rotadayken `VIOLATION`.

Cozum:

- Saha koordinatini yeniden olc.
- `coordinate_frame` ve geofence parametrelerini birlikte kontrol et.
