# ROS 2 Topic Contracts

Bu belge, node'lar arasindaki veri sozlesmelerini tanimlar. Bu proje ilk
surumda `std_msgs/msg/String` icinde JSON tasir. Bunun nedeni custom message
dosyalarina gecmeden once mimariyi hizli test edebilmektir.

## Kamera

### `/camera/raw`

Tip:

```text
sensor_msgs/msg/Image
```

Kullanan:

```text
perception_node
```

Anlami:

Karar algoritmasinin kullandigi asil kamera akisi. Gazebo'da downward camera
bridge buraya baglanir.

### `/camera`

Tip:

```text
sensor_msgs/msg/Image
```

Kullanan:

```text
debug_viewer_node
mission_video_recorder_node
```

Anlami:

Kayit ve goruntuleme icin tekrar edilen kamera akisi. Karar icin kullanilmaz.
Frame repeater, kayitta bos goruntu olmasin diye son gecerli frame'i sinirli
sayida tekrar yayinlar.

## Perception

### `/perception/raw_detections`

Ureten:

```text
perception_node
```

Tuketen:

```text
fusion_node
mission_console_node
mission_video_recorder_node
```

Ornek:

```json
{
  "frame_id": 132,
  "timestamp": 1780000000.12,
  "opencv": [
    {
      "source": "opencv",
      "target_type": "red_square",
      "bbox": [220, 160, 44, 42],
      "center": [242, 181],
      "confidence": 0.91,
      "state": "DETECTED",
      "frame_id": 132,
      "error": [-78, -59]
    }
  ],
  "yolo": [
    {
      "source": "yolo",
      "target_type": "square",
      "bbox": [219, 161, 45, 42],
      "center": [241, 182],
      "confidence": 0.86,
      "state": "DETECTED",
      "frame_id": 132
    }
  ],
  "yolo_ran": true,
  "yolo_frame_id": 132,
  "yolo_age_frames": 0,
  "yolo_meta": {
    "mode": "ROI"
  },
  "yolo_error": null
}
```

Alanlar:

- `opencv`: Renk ve sekil temelli aday hedefler. Hedef tipini OpenCV belirler:
  `red_square` veya `blue_square`.
- `yolo`: YOLO'nun genel kare tespitleri. Tek sinifli modelde hedef tipi
  `square` olarak normalize edilir.
- `yolo_ran`: Bu frame icin taze YOLO gozlemi var mi?
- `yolo_frame_id`: Kullanilan YOLO sonucunun frame numarasi.
- `yolo_age_frames`: YOLO sonucunun mevcut frame'e gore yasi.

Neden `yolo_ran` var?

`yolo=[]` tek basina yeterli degildir. Su iki durum farklidir:

```text
YOLO calismadi
YOLO calisti ama kare bulamadi
```

Fusion bu ayrimi kullanir.

## Fusion

### `/fusion/target`

Ureten:

```text
fusion_node
```

Tuketen:

```text
mission_manager_node
mission_console_node
mission_video_recorder_node
perception_node
```

Ornek:

```json
{
  "frame_id": 132,
  "timestamp": 1780000000.24,
  "primary_target": "blue_square",
  "state": "LOCKED",
  "targets": [
    {
      "target_type": "red_square",
      "target_state": "LOCKED",
      "source": "fusion",
      "bbox": [220, 160, 44, 42],
      "center": [242, 181],
      "confidence": 0.91,
      "fusion_confidence": 0.86,
      "yolo_verified": true,
      "yolo_iou": 0.72,
      "yolo_fresh": true,
      "yolo_age_frames": 0,
      "lock_counter": 10,
      "unstable_counter": 0,
      "release_gate": true
    }
  ],
  "selected": {
    "target_type": "red_square",
    "target_state": "LOCKED",
    "release_gate": true
  }
}
```

Alanlar:

- `target_state`: Fusion state machine durumu.
- `fusion_confidence`: OpenCV confidence, YOLO confidence ve IoU ile uretilen
  birlesik skor.
- `yolo_verified`: YOLO kutusu ile OpenCV kutusu yeterince ortusuyor mu?
- `release_gate`: Algilama acisindan payload birakmaya uygun mu?
- `lock_counter`: Kac dogrulanmis locked frame birikti?

`release_gate`, servo komutu degildir. Sadece mission manager'a verilen bir
izin sinyalidir.

## Mission

### `/mission/state`

Ureten:

```text
mission_manager_node
```

Tuketen:

```text
mission_console_node
mission_video_recorder_node
```

Temel alanlar:

- `state`: Mission state.
- `active_target`: Su an hedeflenen yuk hedefi.
- `target_sequence`: Hedef sirasi.
- `released_targets`: Payload birakilmis hedefler.
- `altitude_m`: Goreli irtifa.
- `nav_x`, `nav_y`: Coordinate frame mapper sonrasi navigasyon koordinati.
- `search_index`, `search_axis`, `search_target`, `search_start`: Arama rotasi
  debug bilgisi.

### `/mission/event`

Ureten:

```text
mission_manager_node
```

Ornek payload release:

```json
{
  "command": "drop_payload",
  "target_type": "blue_square",
  "dry_run": true,
  "servo": 9,
  "pwm": 1900,
  "reset_pwm": 1100,
  "hold_seconds": 0.8,
  "drop_estimate": {
    "distance_to_center_m": 0.42,
    "inside_target_footprint": true
  }
}
```

Bu event, OBS terminalinde:

```text
PAYLOAD_RELEASED target=blue_square ...
```

olarak gorunur.

## Drone Commands

Mission manager su komut topic'lerini uretir:

```text
/drone/cmd_mode
/drone/cmd_arm
/drone/cmd_takeoff
/drone/cmd_velocity
/drone/cmd_land
/drone/cmd_drop
```

Bu topic'leri sadece `mavlink_bridge_node` tuketir.

Bu ayrim onemlidir: Mission manager MAVLink protokolunu bilmez; sadece
niyet/komut JSON'u uretir.

## Drone Telemetry

MAVLink bridge su topic'leri uretir:

```text
/drone/state
/drone/local_position
/drone/altitude
/drone/status
```

Mission manager bu telemetry ile:

- GUIDED mode kontrolu,
- arm durumu,
- takeoff irtifasi,
- post-drop climb,
- RTL teyidi

yapar.

## Safety

### `/safety/status`

Ureten:

```text
safety_monitor_node
```

Alan:

```json
{"status":"OK"}
```

veya:

```json
{"status":"WARNING"}
```

veya:

```json
{"status":"VIOLATION"}
```

Mission state machine `VIOLATION` gorurse failsafe'e gecmelidir.
