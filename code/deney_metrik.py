

"""
F1-Confidence ve PR eğrisi için düzeltilmiş fonksiyonlar.
Mevcut analiz.py dosyasındaki ilgili fonksiyonları bunlarla değiştir.
Veya bu scripti doğrudan çalıştır — confusion matrix ve F1 bar zaten
kaydedildi, bu script sadece eksik grafikleri tamamlar.
"""
 
from pathlib import Path
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import random
import sys
 
SCRIPT_DIR  = Path(__file__).parent
ULTRALYTICS_CONFIG_DIR = Path(os.environ.get("ASFLY_ULTRALYTICS_DIR", str(Path(tempfile.gettempdir()) / "asfly_ultralytics")))
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
MODEL_PATH  = SCRIPT_DIR /  "iha_yolo.pt"
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset_v6"
if not DEFAULT_DATASET_DIR.is_dir():
    DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DATASET_DIR = Path(os.environ.get("ASFLY_DATASET_DIR", str(DEFAULT_DATASET_DIR)))
YAML_PATH   = DATASET_DIR / "data.yaml"
OUTPUT_DIR  = SCRIPT_DIR / "analiz_sonuclari"
OUTPUT_DIR.mkdir(exist_ok=True)
 
CLASS_NAMES = ["red_square", "blue_square"]
CLASS_TR    = ["Kırmızı Kare", "Mavi Kare"]
COLORS      = ["#e74c3c", "#2980b9"]
CONF_THR    = 0.25
DEVICE      = "0"
 
try:
    from ultralytics import YOLO
except ImportError:
    print("pip install ultralytics")
    sys.exit(1)
 
 
def run_validation(model, split="test"):
    return model.val(
        data=str(YAML_PATH), split=split,
        conf=CONF_THR, iou=0.50,
        device=DEVICE,
        verbose=False, plots=False,
    )
 
 
def plot_f1_confidence_curve(metrics, save_path: Path):
    """F1 vs Confidence eğrisi — curves_results list veya array olabilir."""
    try:
        raw = metrics.box.curves_results
        # List ise numpy'a çevir
        px = np.array(raw[0])
        py = np.array(raw[1])   # precision  shape: (nc, n_thresholds)
        ry = np.array(raw[2])   # recall     shape: (nc, n_thresholds)
 
        # 1D geldiyse (tek sınıf gibi) 2D yap
        if py.ndim == 1:
            py = py[np.newaxis, :]
            ry = ry[np.newaxis, :]
 
    except Exception as e:
        print(f"  ⚠ F1-Confidence verisi alınamadı ({e}), atlanıyor.")
        return
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    all_f1 = []
    for i, (name, tr, color) in enumerate(zip(CLASS_NAMES, CLASS_TR, COLORS)):
        if i >= py.shape[0]:
            break
        p  = py[i]
        r  = ry[i]
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0)
        all_f1.append(f1)
        ax.plot(px, f1, color=color, linewidth=2, label=tr)
 
        best_idx = int(np.argmax(f1))
        ax.scatter(px[best_idx], f1[best_idx], color=color, s=80, zorder=5)
        ax.annotate(f"max {f1[best_idx]:.3f}\n@{px[best_idx]:.2f}",
                    (px[best_idx], f1[best_idx]),
                    textcoords="offset points", xytext=(8, -15),
                    fontsize=9, color=color)
 
    if all_f1:
        mean_f1 = np.mean(all_f1, axis=0)
        ax.plot(px, mean_f1, color='black', linewidth=2.5,
                linestyle='--', label="Ortalama")
 
    ax.set_xlabel("Confidence Eşiği", fontsize=11)
    ax.set_ylabel("F1 Skoru", fontsize=11)
    ax.set_title("Confidence Eşiğine Göre F1 Skoru", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.4);  ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ F1-Confidence eğrisi → {save_path.name}")
 
 
def plot_pr_curve(metrics, save_path: Path):
    """PR eğrisi."""
    try:
        raw = metrics.box.curves_results
        px  = np.array(raw[0])
        py  = np.array(raw[1])
        ry  = np.array(raw[2])
        if py.ndim == 1:
            py = py[np.newaxis, :]
            ry = ry[np.newaxis, :]
    except Exception as e:
        print(f"  ⚠ PR eğrisi verisi alınamadı ({e}), atlanıyor.")
        return
 
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, tr, color) in enumerate(zip(CLASS_NAMES, CLASS_TR, COLORS)):
        if i >= py.shape[0]:
            break
        ap = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
        ax.plot(ry[i], py[i], color=color, linewidth=2,
                label=f"{tr}  (AP@50={ap:.3f})")
 
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Eğrisi", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ PR eğrisi → {save_path.name}")
 
 
def plot_sample_predictions(model, save_path: Path, n: int = 16):
    """16 test görseli + model tahminleri."""
    test_img_dir = DATASET_DIR / "test" / "images"
    test_lbl_dir = DATASET_DIR / "test" / "labels"
    if not test_img_dir.exists():
        test_img_dir = DATASET_DIR / "val" / "images"
        test_lbl_dir = DATASET_DIR / "val" / "labels"
 
    all_imgs = list(test_img_dir.glob("*.jpg"))
    if not all_imgs:
        print("  ⚠ Görüntü bulunamadı.")
        return
 
    selected = random.sample(all_imgs, min(n, len(all_imgs)))
    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Örnek Model Tahminleri (Test Seti)", fontsize=14, fontweight='bold')
 
    for ax, img_path in zip(axes, selected):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ax.axis('off'); continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]
 
        # Ground truth (gri)
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        img_draw = img_rgb.copy()
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx,cy,bw,bh = map(float, parts[1:])
                        x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h)
                        x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)
                        cv2.rectangle(img_draw,(x1,y1),(x2,y2),(180,180,180),1)
 
        # Tahmin
        results = model.predict(str(img_path), conf=CONF_THR,
                                device=DEVICE, verbose=False, imgsz=320)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                color = (46,204,113) if cls_id == 0 else (52,152,219)
                cv2.rectangle(img_draw,(x1,y1),(x2,y2),color,2)
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                cv2.putText(img_draw, label, (x1, max(y1-4,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
 
        ax.imshow(img_draw); ax.axis('off')
        ax.set_title(img_path.stem[-12:], fontsize=7)
 
    for ax in axes[len(selected):]:
        ax.axis('off')
 
    legend_items = [
        mpatches.Patch(color=(180/255,180/255,180/255), label='Ground Truth'),
        mpatches.Patch(color=(46/255,204/255,113/255),  label='red_square tahmini'),
        mpatches.Patch(color=(52/255,152/255,219/255),  label='blue_square tahmini'),
    ]
    fig.legend(handles=legend_items, loc='lower center',
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Örnek tahminler → {save_path.name}")
 
 
def print_report(metrics):
    mp   = float(metrics.box.mp)
    mr   = float(metrics.box.mr)
    mf1  = 2*mp*mr/(mp+mr) if (mp+mr) > 0 else 0
    sep  = "═"*55
    print(f"\n{sep}")
    print("  📊  TEST SETİ SONUÇLARI")
    print(sep)
    print(f"  {'Sınıf':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10}")
    print("  " + "-"*50)
    for i, (cls, tr) in enumerate(zip(CLASS_NAMES, CLASS_TR)):
        p  = float(metrics.box.p[i])  if i < len(metrics.box.p)    else 0
        r  = float(metrics.box.r[i])  if i < len(metrics.box.r)    else 0
        ap = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        print(f"  {tr:<20} {p:>10.4f} {r:>10.4f} {ap:>10.4f}  F1={f1:.4f}")
    print("  " + "-"*50)
    print(f"  {'GENEL':<20} {mp:>10.4f} {mr:>10.4f} "
          f"{metrics.box.map50:>10.4f}  F1={mf1:.4f}")
    print(f"\n  mAP@50:95 : {metrics.box.map:.4f}")
    emoji = "🎉" if metrics.box.map50 >= 0.95 else "✅" if metrics.box.map50 >= 0.90 else "⚠️"
    print(f"  {emoji} Genel F1  : {mf1:.4f}")
    print(sep)
 
 
def main():
    print("\n🔧 Model yükleniyor...")
    model = YOLO(str(MODEL_PATH))
 
    print("📊 Test seti validasyonu...")
    metrics = run_validation(model)
 
    print("\n🎨 Eksik grafikler çiziliyor...")
    plot_f1_confidence_curve(metrics, OUTPUT_DIR / "f1_confidence.png")
    plot_pr_curve(metrics,           OUTPUT_DIR / "pr_curve.png")
    plot_sample_predictions(model,   OUTPUT_DIR / "ornek_tahminler.png")
 
    print_report(metrics)
    print(f"\n✅ Tüm grafikler → {OUTPUT_DIR}/\n")
 
 
if __name__ == "__main__":
    main()
