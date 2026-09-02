# Teknofest IHA Modular Architecture

Bu proje, ROS 2 uzerinde Unix felsefesine yakin bir gorev mimarisi kurar:

- Her program tek bir isi yapar.
- Moduller birbirine dogrudan ic durum uzerinden degil, acik veri sozlesmeleri uzerinden baglanir.
- Dis dunya bagimliliklari adapter katmaninda tutulur.
- Karar mantigi mumkun oldugunca saf Python siniflarinda kalir ve ROS olmadan test edilir.
- Hata ayiklama icin her surec okunabilir durum ciktisi uretir.

## Katmanlar

### `nodes/`

ROS 2 surecleridir. Unix mantigindaki "process" karsiligidir.

Node'larin gorevi:

- Topic okumak.
- Parametre almak.
- Adapter veya core modulu cagirmak.
- Sonucu topic olarak yayinlamak.

Node'larin yapmamasi gerekenler:

- Kamera algilama algoritmasini icinde yazmak.
- Mission state transition mantigini daginik if bloklariyla tutmak.
- MAVLink komut detaylarini gorev mantigina karistirmak.

### `adapters/`

Dis dunya veya legacy kod ile sistem arasindaki sinirdir.

Ornekler:

- `OpenCVAdapter`: OpenCV detector'u ROS perception akisi icin sarmalar.
- `YoloAdapter`: async YOLO worker'i sarmalar.
- `FusionAdapter`: legacy fusion fonksiyonlarini ROS veri modeliyle birlestirir.
- `MavlinkAdapter`: ArduPilot/MAVLink transport detaylarini saklar.

Adapter'larin amaci, geri kalan sistemi kutuphane/API detaylarindan korumaktir.

### `core/`

ROS bagimsiz karar mantigidir. Bu katman, test edilmesi en kolay ve en kritik katmandir.

Ornekler:

- `MissionStateMachine`: gorev state transition'lari.
- `LawnmowerSearchPattern`: arama rotasi.
- `Geofence`: saha sinirlari ve velocity clamp.
- `AlignmentController`: goruntu merkezleme hiz komutu.
- `PayloadController`: hangi hedefe yuk birakildigini takip eder.

Bu siniflar ROS import etmez. Bu nedenle unit test ile hizli dogrulanir.

### `interfaces/`

Topic payload'larinin Python veri sozlesmeleridir.

Bu projede ilk surum JSON tasir:

- `RawDetectionPacket`
- `FusedTargetPacket`
- `DroneState`
- `LocalPosition`
- `Altitude`

Ileride custom ROS message'a gecilirse, once bu katman degisir.

### `vision/` ve `control/`

Ilk tek-sinif YOLO paketinden gelen domain kodudur.

- `vision/`: OpenCV, YOLO, bbox, fusion geometry.
- `control/`: target fusion state machine ve release gate mantigi.

Bu moduller zamanla `teknofest_iha/core` altina tasinabilir; ancak yarismaya yakin donemde calisan sistemi kirmamak icin simdilik adapter ile sarilarak kullanilir.

## Surec Akisi

```text
Gazebo / Pi Camera
      |
      v
perception_node
      |
      | /perception/raw_detections
      v
fusion_node
      |
      | /fusion/target
      v
mission_manager_node
      |
      | /drone/cmd_*
      v
mavlink_bridge_node
      |
      v
ArduPilot / Pixhawk
```

Yan surecler:

```text
safety_monitor_node      -> /safety/status
mission_console_node     -> insan okunur stateflow terminali
mission_video_recorder   -> kayit videosu + overlay
debug_viewer_node        -> canli debug goruntusu
camera_frame_repeater    -> kayit/display icin frame tekrar yayinlama
```

## Unix Felsefesi Kararlari

### 1. Perception payload birakmaz

Perception sadece hedef gorur.

```text
OpenCV = renk + sekil adayi
YOLO   = genel kare dogrulamasi
```

Servo, arm, takeoff, RTL gibi konular perception katmanina ait degildir.

### 2. Fusion aktuatore komut vermez

Fusion sadece algisal guven uretir:

```json
{
  "target_state": "LOCKED",
  "fusion_confidence": 0.86,
  "yolo_verified": true,
  "release_gate": true
}
```

`release_gate`, "algilama acisindan birakmaya uygun" anlamina gelir.

Gercek birakma karari mission manager tarafindadir. Cunku mission manager:

- Irtifayi bilir.
- Aracin modunu bilir.
- Payload daha once birakildi mi bilir.
- Gorev sirasini bilir.
- MAVLink komutlarini bridge'e yollar.

### 3. MAVLink adapter sadece transport bilir

`MavlinkAdapter`, MAVLink komutlarini gonderir ve telemetry okur. Gorev semantigini bilmez.

Bu sayede ileride:

- SITL
- Mission Planner
- Pixhawk
- Farkli MAVLink portlari

degisse bile mission state machine ayni kalir.

### 4. State machine'ler ayridir

Iki farkli state machine vardir:

```text
Target Fusion State Machine
SEARCH -> CANDIDATE -> TRACKING / UNSTABLE / LOCKED

Mission State Machine
INIT -> TAKEOFF -> SEARCH_TARGET -> TARGET_ALIGN -> TARGET_VERIFY -> DROP_TARGET -> POST_DROP_HOVER -> RETURN_HOME
```

Bu ayrim bilincli yapilmistir. Algilama guveni ile ucus/gorev icrasi ayni sey degildir.

## Topic Sozlesmesi

Topic'ler Unix pipe gibi dusunulur. Bir node ciktisi diger node'un girdisidir.

Ornek:

```text
/perception/raw_detections
```

Bu topic:

- OpenCV detection listesini,
- YOLO detection listesini,
- YOLO'nun calisip calismadigini,
- YOLO sonucunun kac frame yasinda oldugunu

tasir.

Fusion node bu topic disinda perception ic durumuna bakmaz.

## Yarismaya Hazirlik Notu

Bu mimari sim ve gercek IHA arasinda su sinirlari korur:

- Kamera kaynagi degisir, perception node ayni kalir.
- MAVLink endpoint degisir, mission manager ayni kalir.
- Model dosyasi degisir, fusion sozlesmesi ayni kalir.
- Servo kanali degisir, mission logic ayni kalir.

Bu nedenle son 1 ayda ana risk, algoritma degil entegrasyon parametreleridir:

- Kamera topic/kalibrasyon.
- MAVLink port/baud.
- Servo kanal/PWM.
- Geofence sinirlari.
- Gercek hedef boyutu ve irtifa.
