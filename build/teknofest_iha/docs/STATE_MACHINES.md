# State Machines

Bu projede iki ana state machine vardir:

1. Target Fusion State Machine
2. Mission State Machine

Bu ayrim bilinclidir. Algilama guveni ile ucus/gorev icrasi ayni sey degildir.

## Target Fusion State Machine

Dosya:

```text
control/state_machine.py
```

Kullanan:

```text
teknofest_iha/adapters/fusion_adapter.py
```

State'ler:

```text
SEARCH
CANDIDATE
TRACKING
UNSTABLE
LOCKED
```

### SEARCH

Anlami:

Hedef yok veya OpenCV hedef adayi uretmedi.

Giris kosulu:

```text
has_opencv_target == false
veya
fused_target is None
```

Etkisi:

```text
cv_seen_counter = 0
lock_counter = 0
unstable_counter = 0
```

### CANDIDATE

Anlami:

OpenCV hedefi yeni gordu. Sistem hemen kilitlenmez; warm-up bekler.

Giris kosulu:

```text
cv_seen_counter <= CANDIDATE_MIN_FRAMES
```

Su an:

```text
CANDIDATE_MIN_FRAMES = 2
```

Bu, tek frame false positive durumlarina karsi koruma saglar.

### TRACKING

Anlami:

OpenCV hedefi takip ediyor, ancak hedef henuz yeterince guvenilir degil.

Giris kosullari:

```text
hedef var
ama lock condition saglanmiyor
ve unstable counter esigi henuz dolmadi
```

### UNSTABLE

Anlami:

OpenCV hedef goruyor ama YOLO taze gozlemde hedefi dogrulayamiyor.

Giris kosulu:

```text
has_recent_yolo == true
yolo_verified == false
unstable_counter >= UNSTABLE_MIN_FRAMES
```

Su an:

```text
UNSTABLE_MIN_FRAMES = 2
```

Bu, tek bir kotu YOLO frame'i ile state'in hemen ziplayip durmasini engeller.

### LOCKED

Anlami:

OpenCV hedefi goruyor, YOLO kareyi dogruluyor, fusion confidence yeterli.

Giris kosulu:

```text
yolo_verified == true
fusion_confidence >= FUSION_CONF_THRESH
```

Su an:

```text
FUSION_CONF_THRESH = 0.64
```

Her locked frame'de:

```text
lock_counter += 1
```

## Release Gate

`release_gate` bir state degildir. Fusion'in mission manager'a verdigi bir izin
sinyalidir.

Dosya:

```text
control/payload_logic.py
```

Kosul:

```text
yolo_verified == true
fusion_confidence >= FUSION_CONF_THRESH
abs(error_x) <= CENTER_TOL_X
abs(error_y) <= CENTER_TOL_Y
lock_counter >= LOCK_MIN_FRAMES
```

Su an:

```text
CENTER_TOL_X = 40 px
CENTER_TOL_Y = 40 px
LOCK_MIN_FRAMES = 10
```

`release_gate=true`, servo komutu degildir. Sadece "algilama acisindan birakma
uygun" anlamina gelir.

## Mission State Machine

Dosya:

```text
teknofest_iha/core/state_machine.py
```

Kullanan:

```text
teknofest_iha/nodes/mission_manager_node.py
```

Mission state'leri:

```text
INIT
WAIT_FOR_CAMERA
CONNECT_MAVLINK
SET_GUIDED
ARM
TAKEOFF
SEARCH_TARGET
TARGET_CANDIDATE
TARGET_ALIGN
TARGET_VERIFY
DROP_TARGET
POST_DROP_HOVER
RETURN_HOME
MISSION_COMPLETE
FAILSAFE
```

### INIT -> WAIT_FOR_CAMERA

Baslangic gecisidir.

### WAIT_FOR_CAMERA -> CONNECT_MAVLINK

Kosul:

```text
camera_ready == true
```

Fusion verisi gelmeden ucus gorevine baslanmaz.

### CONNECT_MAVLINK -> SET_GUIDED

Kosul:

```text
mavlink_connected == true
```

### SET_GUIDED -> ARM

Kosul:

```text
guided == true
```

Mission manager `/drone/cmd_mode` ile GUIDED ister.

### ARM -> TAKEOFF

Kosul:

```text
armed == true
```

### TAKEOFF -> SEARCH_TARGET

Kosul:

```text
abs(altitude_m - takeoff_altitude_m) <= altitude_tolerance_m
```

### SEARCH_TARGET

Bu state'te lawnmower search pattern calisir.

Mission manager:

- Mevcut pozisyonu nav frame'e cevirir.
- `LawnmowerSearchPattern` ile siradaki waypoint velocity'sini alir.
- Geofence velocity clamp uygular.
- `/drone/cmd_velocity` yayinlar.

Hedef gorulurse:

```text
SEARCH_TARGET -> TARGET_CANDIDATE
```

### TARGET_CANDIDATE

Hedef yeni goruldu. Bir sonraki state:

```text
TARGET_ALIGN
```

Hedef kaybolursa:

```text
SEARCH_TARGET
```

### TARGET_ALIGN

Goruntu merkezi ile hedef merkezi arasindaki hata kullanilir.

Dosya:

```text
teknofest_iha/core/alignment_controller.py
```

Hedef merkezlenince:

```text
TARGET_VERIFY
```

### TARGET_VERIFY

Bu state'te:

- hedef merkezde mi,
- fusion lock stabil mi,
- approach enabled ise drop irtifasina inildi mi

kontrol edilir.

Mission manager, fusion'dan gelen release bilgisini ve kendi stabil lock
zamanlayicisini birlikte kullanir.

Kosul saglanirsa:

```text
DROP_TARGET
```

### DROP_TARGET

Mission manager payload event'i uretir:

```text
/mission/event
/drone/cmd_drop
```

MAVLink bridge bu komutu servo/PWM komutuna cevirir.

Payload sadece mission manager tarafindan tetiklenir.

### POST_DROP_HOVER

Payload birakildiktan sonra IHA hemen devam etmez.

Once:

```text
post_drop_hover_s
```

bekler ve sonra eski arama irtifasina geri tirmanir.

Arama irtifasina donmeden sonraki hedefe gecmez.

### RETURN_HOME

Tum hedefler tamamlaninca RTL istenir.

### FAILSAFE

Safety violation varsa state machine failsafe'e gecer.

## OBS Terminal Stateflow

`mission_console_node`, state machine'i insan okunur bloklarla gosterir:

```text
[MISSION] state=SEARCH_TARGET  geofence=OK  lane=03
[SEARCH ] next_point=(100,-10)  target=red_square
[VISION ] opencv=DETECTED  yolo=VERIFIED  iou=MATCHED(0.74)
[FUSION ] target=red_square  state=LOCKED  conf=0.87
[ALIGN  ] error_x=12  error_y=-8  center=OK
[GATE   ] release_gate=TRUE  lock_counter=10/10
[PAYLOAD] released=1/2  target=red_square
[MISSION] state=RESUME_SEARCH  next_target=blue_square
```

Bu format, hem yarismada video kaydi hem de debug icin kullanilir.
