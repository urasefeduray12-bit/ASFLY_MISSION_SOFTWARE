# Code Reference

Bu belge, operasyonel kod dosyalarinin ne yaptigini ve sistemdeki yerini
aciklar. Amac, yarismaya kadar kodu savunulabilir, devredilebilir ve bakimi
kolay hale getirmektir.

## ROS Nodes

### `teknofest_iha/nodes/perception_node.py`

Gorev:

- Kamera frame'i alir.
- OpenCV ile renkli hedef adaylarini bulur.
- YOLO worker'a periyodik inference isi verir.
- Taze YOLO sonucunu ve OpenCV sonucunu `/perception/raw_detections` olarak
  yayinlar.

Girdi:

- `/camera/raw`
- `/fusion/target` yaklasik state bilgisi. Bu bilgi YOLO frame skipping
  araligini ayarlamak icin kullanilir.

Cikti:

- `/perception/raw_detections`
- `/perception/status`
- `/perception/debug_image`

Neden ayri?

Perception sadece algilama yapar. Mission veya MAVLink komutu uretmez.

### `teknofest_iha/nodes/fusion_node.py`

Gorev:

- Raw OpenCV/YOLO detection paketini alir.
- Fusion adapter ile hedef bazli algisal guven hesaplar.
- `/fusion/target` yayinlar.

Girdi:

- `/perception/raw_detections`

Cikti:

- `/fusion/target`
- `/fusion/status`

Neden ayri?

OpenCV ve YOLO'nun nasil birlestirilecegi perception node'dan ayridir. Bu,
farkli fusion politikalari denemeyi kolaylastirir.

### `teknofest_iha/nodes/mission_manager_node.py`

Gorev:

- Gorev state machine'i calistirir.
- Takeoff, arama, hizalanma, dogrulama, payload release, climb ve RTL kararlarini
  verir.
- Fusion hedeflerini mission hedef sirasi ile birlestirir.

Girdi:

- `/fusion/target`
- `/drone/state`
- `/drone/local_position`
- `/drone/altitude`
- `/safety/status`
- `/mission/cmd_start`

Cikti:

- `/drone/cmd_mode`
- `/drone/cmd_arm`
- `/drone/cmd_takeoff`
- `/drone/cmd_velocity`
- `/drone/cmd_land`
- `/drone/cmd_drop`
- `/mission/state`
- `/mission/event`

Neden ayri?

Mission manager sistemin orkestratorudur. Algilama algoritmasi veya MAVLink
protokol detayi bilmez; sadece topic sozlesmeleriyle konusur.

### `teknofest_iha/nodes/mavlink_bridge_node.py`

Gorev:

- Mission komutlarini MAVLink komutlarina cevirir.
- MAVLink telemetry'i ROS topic'lerine cevirir.

Girdi:

- `/drone/cmd_*`

Cikti:

- `/drone/state`
- `/drone/local_position`
- `/drone/altitude`
- `/drone/status`

Neden ayri?

Pixhawk/SITL baglanti ayrintilari mission manager'a sizmaz.

### `teknofest_iha/nodes/safety_monitor_node.py`

Gorev:

- Local position'u izler.
- Geofence durumunu hesaplar.
- `OK`, `WARNING`, `VIOLATION` yayinlar.

Girdi:

- `/drone/local_position`

Cikti:

- `/safety/status`

### `teknofest_iha/nodes/mission_console_node.py`

Gorev:

- OBS ve terminal icin insan okunur gorev akisi uretir.
- Mission, search, vision, fusion, align, gate ve payload durumlarini blok
  halinde gosterir.

Bu node karar vermez. Sadece gozlem ve kayit icindir.

### `teknofest_iha/nodes/mission_video_recorder_node.py`

Gorev:

- Kamera goruntusunu overlay ile video dosyasina yazar.
- Fusion, mission ve payload event bilgilerini goruntu uzerine basar.

### `teknofest_iha/nodes/debug_viewer_node.py`

Gorev:

- `/perception/debug_image` goruntusunu canli pencerede gosterir.

### `teknofest_iha/nodes/camera_frame_repeater_node.py`

Gorev:

- Karar kamerasi seyrek veya gecikmeli geldiginde display/recording tarafinda
  son frame'i sinirli sayida tekrar yayinlar.

Not:

Bu node karar akisini beslemez. Sadece OBS goruntusunun kopuk gorunmesini
azaltir.

## Adapters

### `teknofest_iha/adapters/opencv_adapter.py`

`vision.opencv_detector.OpenCVDetector` sinifini ROS perception node icin
sarmalar.

Neden var?

Legacy OpenCV kodu dogrudan ROS node'a gomulmez. Adapter, eski kod ile yeni
mimari arasinda sinir olusturur.

### `teknofest_iha/adapters/yolo_adapter.py`

Async YOLO worker'i baslatir, is yollar ve son sonucu okur.

Model path cozumleme sorumlulugu da buradadir:

```text
models_archive/iha_best.pt
```

### `teknofest_iha/adapters/fusion_adapter.py`

Per-target fusion state machine'leri tutar.

Girdisi:

- `RawDetectionPacket`

Ciktisi:

- `FusedTargetPacket`

Bu adapter:

- Taze YOLO sonucunu kontrol eder.
- OpenCV ve YOLO kutularini fuse eder.
- `target_state`, `fusion_confidence`, `release_gate` alanlarini uretir.

### `teknofest_iha/adapters/mavlink_adapter.py`

MAVLink transport adapter'idir.

Sorumluluklari:

- Heartbeat bekleme.
- Mode set.
- Arm/disarm.
- Takeoff.
- Velocity command.
- Land/RTL.
- Servo payload komutu.
- Telemetry okuma.

## Core

### `teknofest_iha/core/state_machine.py`

Mission state transition mantigidir.

ROS bagimsizdir. Testleri:

```text
tests/test_state_machine.py
```

### `teknofest_iha/core/mission_states.py`

Mission state enum tanimidir.

### `teknofest_iha/core/search_pattern.py`

Lawnmower/serpentine arama rotasini uretir.

Ozellikler:

- En yakin saha ucundan baslayabilir.
- Serpentine sirada gereksiz geri donusu engeller.
- Waypoint gecislerinde eksen bazli hiz uretir.

Testleri:

```text
tests/test_search_pattern.py
```

### `teknofest_iha/core/alignment_controller.py`

Goruntu merkez hatasini hiz komutuna cevirir.

Girdi:

- hedef merkezi
- goruntu merkezi

Cikti:

- nav frame hiz komutu

### `teknofest_iha/core/geofence.py`

Arama sahasi sinirlarini ve warning/violation durumunu hesaplar.

### `teknofest_iha/core/coordinate_frame.py`

Gazebo/local frame ile mission navigation frame arasinda donusum yapar.

Simulasyon ve gercek IHA arasinda coordinate farki cikarsa degistirilecek ilk
yer burasidir.

### `teknofest_iha/core/payload_controller.py`

Hangi hedeflere payload birakildigini takip eder.

### `teknofest_iha/core/payload_metrics.py`

Payload birakma noktasinin hedef merkezine tahmini uzakligini hesaplar.

Bu hesap sim/raporlama icindir; servo komutu vermez.

### `teknofest_iha/core/target_selection.py`

Gorunen ve henuz payload almamis hedefler arasindan aktif hedef secimi yapar.

`release_gate` acik hedefe oncelik verir.

## Interfaces

### `teknofest_iha/interfaces/detection_models.py`

Detection ve fusion JSON modellerini tanimlar.

Ana siniflar:

- `Detection`
- `RawDetectionPacket`
- `FusedTargetPacket`

### `teknofest_iha/interfaces/drone_models.py`

MAVLink/vehicle durum modellerini tanimlar.

Ana siniflar:

- `DroneState`
- `LocalPosition`
- `Altitude`

### `teknofest_iha/interfaces/target_models.py`

Hedef adlari ve oncelik modeli.

## Vision

### `vision/opencv_detector.py`

Renk ve sekil tabanli hedef tespiti.

Gorevi:

- HSV maskeleri uretmek.
- Kontur bulmak.
- Karelik/solidity/alan filtreleri uygulamak.
- Kalman tahmini ile kisa sureli takip yapmak.

### `vision/yolo_detector.py`

YOLO modelini async worker icinde calistirir.

Ozellikler:

- ROI crop destekler.
- Tek sinif kare modelini `square` olarak normalize eder.
- Inference sonucunu standart detection formatina cevirir.

### `vision/fusion.py`

OpenCV detection ile YOLO detection arasinda bbox IoU hesaplar.

Ana fonksiyon:

```text
fuse_detections()
```

YOLO dogrulamasi icin:

```text
IoU >= IOU_VERIFY_THRESH
YOLO confidence >= YOLO_VERIFY_CONF_THRESH
```

### `vision/detection_types.py`

Ortak bbox ve detection yardimci fonksiyonlari.

### `vision/kalman_filter.py`

OpenCV tarafindaki kisa sureli takip icin 2D Kalman filtresi.

### `vision/camera.py`

Legacy webcam/camera worker. ROS 2 ana akista dogrudan kullanilmaz.

## Control

### `control/state_machine.py`

Target fusion state machine:

```text
SEARCH
CANDIDATE
TRACKING
UNSTABLE
LOCKED
```

Mission state machine'den farklidir. Sadece hedefin algisal guven durumunu
tutar.

### `control/payload_logic.py`

`release_gate` kosulunu hesaplar.

Bu dosya servo komutu vermez.

## Utils

### `utils/drawing.py`

Legacy OpenCV cizim/HUD fonksiyonlari.

### `utils/logger.py`

CSV log yardimcisi.

## Scripts

### `scripts/prepare_obs_recording.sh`

Gazebo server, Gazebo GUI, ArduPilot SITL, ROS stack ve debug viewer'i OBS
kaydi icin baslatir.

### `scripts/stop_teknofest_stack.sh`

Teknofest demo sureclerini durdurur.

### `scripts/start_mission_for_obs.sh`

`/mission/cmd_start` topic'ine start komutu yollar.

### `scripts/randomize_teknofest_world.py`

Gazebo hedef konumlarini kontrollu rastgelelestirir.

### `scripts/configure_sitl_params.py`

ArduPilot SITL parametrelerini Gazebo JSON backend icin ayarlar.

### `scripts/sitl_takeoff_smoke.py`

Tam gorevden once SITL takeoff saglik testi yapar.

### `scripts/render_yolo_video.py`

Video dosyasi uzerinde YOLO inference calistirir ve kutulari yeni videoya
isler.

### `scripts/test_yolo_phone_camera.py`

Telefon/IP kamera veya webcam uzerinde YOLO modelini canli test eder.

## Legacy / Training

### `main_async_fusion.py`

ROS oncesi tek process demo scriptidir. Ana yarismaya hazir ROS akisi icin
referans degildir, ancak eski algoritmanin nasil calistigini gostermek icin
tutulur.

### `code/`

Egitim, dataset hazirlama ve model analiz araclari.

Bu dosyalar runtime mission stack'in parcasi degildir.
