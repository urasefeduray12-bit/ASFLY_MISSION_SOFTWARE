"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         IHA Canlı Kamera Test Scripti — TEKNOFEST                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • Webcam'den canlı görüntü alır                                            ║
║  • YOLOv8 modeli ile nesne tespit eder                                      ║
║  • FPS, confidence, sınıf bilgisini ekranda gösterir                        ║
║  • Ekran görüntüsü almak için S tuşu                                        ║
║  • Çıkış için Q tuşu                                                        ║
║                                                                              ║
║  Kullanım:                                                                   ║
║    python canli_test.py                    (varsayılan kamera)               ║
║    python canli_test.py --cam 1            (harici kamera)                   ║
║    python canli_test.py --conf 0.5         (confidence eşiği)                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  AYARLARFf
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
ULTRALYTICS_CONFIG_DIR = Path(os.environ.get("ASFLY_ULTRALYTICS_DIR", str(Path(tempfile.gettempdir()) / "asfly_ultralytics")))
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
MODEL_PATH = SCRIPT_DIR / "/singleclass_curriculum/models_archive/iha_best.pt"

CLASS_NAMES = ["red_square", "blue_square"]
CLASS_TR = ["Kirmizi Kare", "Mavi Kare"]

# BGR renk paleti
CLASS_COLORS = [
    (0, 60, 220),  # Kırmızı kare  → kırmızı (BGR)
    (220, 100, 0),  # Mavi kare  → mavi (BGR)
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
OUTPUT_DIR = SCRIPT_DIR / "test_ekran_goruntuleri"
OUTPUT_DIR.mkdir(exist_ok=True)


def detect_red_square_hsv(frame: np.ndarray):
    """Find a saturated red square/rectangle as a fallback for live camera."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 115, 75], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 115, 75], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    # Remove most skin-like pixels without killing saturated red targets.
    skin = cv2.inRange(hsv, np.array([0, 20, 65]), np.array([24, 175, 255]))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))

    kernel5 = np.ones((5, 5), np.uint8)
    kernel9 = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel9)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    min_area = max(1300, int(w * h * 0.004))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 28 or bh < 28:
            continue

        aspect = bw / max(1, bh)
        if not 0.65 <= aspect <= 1.45:
            continue

        rect_area = bw * bh
        fill = area / max(1, rect_area)
        if fill < 0.58:
            continue

        roi = frame[y:y + bh, x:x + bw]
        roi_mask = mask[y:y + bh, x:x + bw]
        red_pixels = roi[roi_mask > 0]
        if len(red_pixels) < 250:
            continue

        b_mean, g_mean, r_mean = red_pixels.mean(axis=0)
        red_dominance = r_mean - max(g_mean, b_mean)
        red_ratio = r_mean / max(1.0, g_mean + b_mean)
        if red_dominance < 45 or red_ratio < 1.15:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        corner_score = 1.0 if 4 <= len(approx) <= 7 else 0.75
        aspect_score = max(0.25, 1.0 - abs(1.0 - aspect) * 0.45)
        color_score = float(np.clip(red_dominance / 100.0, 0.35, 1.0))
        score = float(np.clip(fill * aspect_score * corner_score * color_score, 0.0, 0.99))

        if best is None or score > best["conf"]:
            best = {
                "bbox": (x, y, x + bw, y + bh),
                "conf": score,
                "area": area,
            }

    return best


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="IHA Canlı Kamera Test")
    ap.add_argument("--cam", type=int, default=0, help="Kamera indeksi (0=varsayılan)")
    ap.add_argument("--conf", type=float, default=0.20, help="Confidence eşiği")
    ap.add_argument("--low-conf", type=float, default=0.05,
                    help="Debug için en düşük aday confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO giriş boyutu")
    ap.add_argument("--width", type=int, default=640, help="Kamera genişliği")
    ap.add_argument("--height", type=int, default=480, help="Kamera yüksekliği")
    ap.add_argument("--device", default="0",
                    help="cuda için 0, cpu için cpu; boş bırakırsan otomatik")
    ap.add_argument("--augment", action="store_true",
                    help="Daha yavaş ama bazen daha iyi test-time augmentation")
    ap.add_argument("--no-hsv", action="store_true",
                    help="Kirmizi kare icin renk/sekil fallback'ini kapat")
    args = ap.parse_args()

    # ── Model yükle ───────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ pip install ultralytics")
        return

    # Alternatif model yolu ara
    model_path = MODEL_PATH
    if not model_path.exists():
        alts = list(SCRIPT_DIR.glob("runs/**/best.pt"))
        if alts:
            model_path = max(alts, key=lambda p: p.stat().st_mtime)
            print(f"  ℹ Alternatif model: {model_path}")
        else:
            print(f"❌ Model bulunamadı: {MODEL_PATH}")
            return

    print(f"\n🔧 Model yükleniyor: {model_path}")
    model = YOLO(str(model_path))
    print(f"  ✔ Model hazır")

    # ── Kamera aç ────────────────────────────────────────────────────────────
    print(f"\n📷 Kamera {args.cam} açılıyor...")
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Kamera {args.cam} açılamadı!")
        print("   Farklı bir indeks dene: --cam 1")
        return

    print(f"  ✔ Kamera hazır ({args.width}x{args.height})")
    print(f"\n  Kontroller:")
    print(f"    Q = Çıkış")
    print(f"    S = Ekran görüntüsü kaydet")
    print(f"    + = Confidence artır (+0.05)")
    print(f"    - = Confidence azalt (-0.05)\n")

    # ── Değişkenler ───────────────────────────────────────────────────────────
    fps = 0.0
    frame_count = 0
    t0 = time.time()
    conf_thr = args.conf
    shot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame alınamadı, yeniden deneniyor...")
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        # ── FPS hesapla ───────────────────────────────────────────────────────
        frame_count += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frame_count / (now - t0)
            frame_count = 0
            t0 = now

        raw_frame = frame.copy()

        # ── YOLO inference ────────────────────────────────────────────────────
        # Dusuk esikle calistirip asil esigi cizimde uyguluyoruz. Boylece
        # modelin zayif aday gormesi ile hic gormemesi ayriliyor.
        predict_kwargs = dict(
            source=frame,
            conf=min(conf_thr, args.low_conf),
            imgsz=args.imgsz,
            verbose=False,
            augment=args.augment,
        )
        if args.device is not None:
            predict_kwargs["device"] = args.device
        results = model.predict(**predict_kwargs)

        # ── Tespit edilen nesneleri çiz ───────────────────────────────────────
        detected_any = False
        best_box = None  # En yüksek confidence'lı aktif tespit
        best_conf = 0.0
        best_source = "YOLO"
        top_candidate = None  # Eşik altı en iyi aday

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if top_candidate is None or conf > top_candidate[1]:
                    top_candidate = (cls_id, conf, x1, y1, x2, y2)

                is_active = conf >= conf_thr
                color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (0, 255, 0)
                if not is_active:
                    color = (120, 120, 120)
                name = CLASS_TR[cls_id] if cls_id < len(CLASS_TR) else "?"

                if is_active:
                    detected_any = True
                    if conf > best_conf:
                        best_conf = conf
                        best_box = (cls_id, x1, y1, x2, y2)
                        best_source = "YOLO"

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_active else 1)

                # Etiket arka planı
                prefix = "" if is_active else "low "
                label = f"{prefix}{name}  {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
                label_y = max(y1 - 6, th + 4)
                cv2.rectangle(frame,
                              (x1, label_y - th - 4),
                              (x1 + tw + 4, label_y + baseline),
                              color, -1)
                cv2.putText(frame, label,
                            (x1 + 2, label_y - 2),
                            FONT, 0.55, (255, 255, 255), 1)

                # Merkeze çizgi
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2
                if is_active:
                    cv2.line(frame, frame_center, (cx_box, cy_box), color, 1)
                    cv2.circle(frame, (cx_box, cy_box), 4, color, -1)

        hsv_red = None if args.no_hsv else detect_red_square_hsv(raw_frame)
        if hsv_red is not None:
            x1, y1, x2, y2 = hsv_red["bbox"]
            hsv_conf = hsv_red["conf"]
            color = CLASS_COLORS[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"HSV Kirmizi Kare  {hsv_conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, FONT, 0.50, 1)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(frame,
                          (x1, label_y - th - 4),
                          (x1 + tw + 4, label_y + baseline),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 2, label_y - 2),
                        FONT, 0.50, (255, 255, 255), 1)

            should_promote = (
                hsv_conf >= 0.45
                and (
                    not detected_any
                    or best_box is None
                    or best_box[0] != 0
                    or hsv_conf >= best_conf
                )
            )
            if should_promote:
                detected_any = True
                best_box = (0, x1, y1, x2, y2)
                best_conf = hsv_conf
                best_source = "HSV"

        # ── Nişangah (merkez) ─────────────────────────────────────────────────
        cx_f, cy_f = frame_center
        cross_color = (0, 255, 255)
        cv2.line(frame, (cx_f - 20, cy_f), (cx_f + 20, cy_f), cross_color, 1)
        cv2.line(frame, (cx_f, cy_f - 20), (cx_f, cy_f + 20), cross_color, 1)
        cv2.circle(frame, frame_center, 40, cross_color, 1)  # Drop zone dairesi

        # ── Durum mesajı ──────────────────────────────────────────────────────
        if detected_any:
            cls_id = best_box[0]
            status = f"TESPIT: {CLASS_TR[cls_id]}  ({best_source} {best_conf:.2f})"
            s_color = CLASS_COLORS[cls_id]

            # Hedef merkeze yakın mı?
            cx_b = (best_box[1] + best_box[3]) // 2
            cy_b = (best_box[2] + best_box[4]) // 2
            dist = int(((cx_b - cx_f) ** 2 + (cy_b - cy_f) ** 2) ** 0.5)

            if dist < 40:
                status += "  ★ MERKEZ!"
                cv2.circle(frame, frame_center, 40, (0, 255, 0), 3)
        else:
            if top_candidate is not None:
                cls_id, cand_conf, *_ = top_candidate
                status = f"ZAYIF ADAY: {CLASS_TR[cls_id]} ({cand_conf:.2f})"
            else:
                status = "Hedef bekleniyor..."
            s_color = (0, 165, 255)

        cv2.putText(frame, status, (10, 32), FONT, 0.65, s_color, 2)

        # ── Bilgi paneli (sağ üst) ────────────────────────────────────────────
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Conf: {conf_thr:.2f}",
            f"Img: {args.imgsz}",
        ]
        panel_x = w - 150
        cv2.rectangle(frame, (panel_x - 5, 4), (w - 4, 78), (30, 30, 30), -1)
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (panel_x, 22 + i * 20),
                        FONT, 0.48, (200, 200, 200), 1)

        # ── Klavye kontrol bilgisi ────────────────────────────────────────────
        cv2.putText(frame, "Q=Cikis  S=Kaydet  +/-=Conf",
                    (10, h - 10), FONT, 0.42, (150, 150, 150), 1)

        # ── Göster ───────────────────────────────────────────────────────────
        cv2.imshow("IHA Canli Test", frame)

        # ── Klavye ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q veya ESC
            break

        elif key == ord('s'):  # Ekran görüntüsü
            shot_count += 1
            fname = OUTPUT_DIR / f"ekran_{shot_count:04d}.jpg"
            raw_name = OUTPUT_DIR / f"raw_{shot_count:04d}.jpg"
            cv2.imwrite(str(fname), frame)
            cv2.imwrite(str(raw_name), raw_frame)
            print(f"  📸 Kaydedildi → {fname.name} / {raw_name.name}")

        elif key == ord('+') or key == ord('='):
            conf_thr = min(conf_thr + 0.05, 0.95)
            print(f"  Confidence → {conf_thr:.2f}")

        elif key == ord('-'):
            conf_thr = max(conf_thr - 0.05, 0.10)
            print(f"  Confidence → {conf_thr:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test tamamlandı.")
    if shot_count:
        print(f"   {shot_count} ekran görüntüsü → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
