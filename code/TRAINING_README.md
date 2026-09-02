# ASFLY Single-Class YOLO Training

Bu klasordeki egitim pipeline'i final tek sinifli YOLO datasetini kullanir.

YOLO sinifi:

```txt
0: target_square
```

Kirmizi kare ve mavi kare YOLO icin ayni siniftir. Renk ayrimi egitimde
yapilmaz. Gercek sistemde YOLO bbox verdikten sonra crop uzerinde OpenCV
HSV/LAB renk analizi ile `red_ratio` ve `blue_ratio` hesaplanacak.

## Dosyalar

- `train_singleclass_yolo.py`: dataset kontrolu, egitim, val/test, predict preview, model arsivleme
- `analyze_model_by_metadata.py`: test splitinde stage/renk/blur/isik bazli analiz
- `datasets/ds_final_target_square/data.yaml`: final tek sinifli dataset
- `models_archive/`: egitim bitince `best.pt`, `last.pt`, metrics ve args JSON dosyalari

## On Kontrol

Egitime baslamadan sadece kontrol:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" train_singleclass_yolo.py --dry-check
```

Bu kontrol sunlari denetler:

- `data.yaml` var mi
- `nc=1` mi
- `names[0] == target_square` mi
- train/val/test image ve label klasorleri var mi
- label class id sadece `0` mi
- empty/background/negative orneklerde bos `.txt` label var mi
- `metadata.csv` var mi
- GPU/CPU bilgisi

Class id `1` veya baska bir sinif bulunursa egitim baslamaz.

## 320 Baseline

Raspberry Pi 4 hedefi icin ilk aday:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" train_singleclass_yolo.py --imgsz 320 --epochs 60 --batch 8 --patience 15 --workers 2 --name target_square_yolov8n_320_baseline
```

Windows worker sorunu olursa:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" train_singleclass_yolo.py --imgsz 320 --epochs 60 --batch 8 --patience 15 --workers 0 --name target_square_yolov8n_320_baseline
```

## 416 Karsilastirma

Kucuk hedefler icin daha iyi olabilir, ama Pi FPS maliyeti olculecek:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" train_singleclass_yolo.py --imgsz 416 --epochs 50 --batch 8 --patience 15 --workers 2 --name target_square_yolov8n_416_compare
```

## Augmentasyon Mantigi

Final dataset zaten kontrollu sentetik augmentasyon iceriyor. Bu yuzden YOLO
augmentasyonu sakin tutuldu:

- `mixup=0.0`
- `copy_paste=0.0`
- `mosaic=0.4`
- dusuk HSV jitter
- hafif rotation/translate/scale

Amac datasetin stage mantigini bozacak agresif renk/kolaj etkilerinden
kacinmak.

## Bakilacak Metrikler

Egitimden sonra once sunlara bak:

- Recall
- Precision
- mAP50
- mAP50-95
- F1 curve
- PR curve
- box_loss
- dfl_loss

Tek sinifli modelde `cls_loss` cok belirleyici olmayabilir. Asil oncelik
hedef kacirmamasi, yani recall ve Stage 4 kosullarinda cokmemesidir.

## Metadata Analizi

Egitimden sonra:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" analyze_model_by_metadata.py --model models_archive\target_square_yolov8n_320_baseline_best.pt --imgsz 320 --conf 0.25
```

Bu rapor sunlari verir:

- stage1/stage2/stage3/stage4 recall
- red/blue detection recall
- empty/negative false positive rate
- blur level bazli recall
- lighting/shadow/glare/exposure bazli recall
- kucuk bbox hedeflerde recall

Sentetik mAP cok iyi ciksa bile bu analiz ve webcam testi final karar icin
daha onemli.

## Gercek Test Checklist

Model secimi sadece sentetik skora gore yapilmayacak. Sonra mutlaka:

- webcam hedef testi
- hizli hareket testi
- egimli/perspektif testi
- bos zemin false positive testi
- gunes/parlama testi
- kirmizi tabela / mavi branda negatif testi
- 320 vs 416 FPS karsilastirmasi
- OpenCV renk karari ile red/blue crop testi

## Raspberry Pi Notu

Pi 4 tarafinda YOLO her frame calismayacak. Hedef mimari:

- Her 4 frame'de 1 YOLO
- Aradaki frame'lerde OpenCV takip
- YOLO bbox crop'unda OpenCV renk karari
- 3-5 frame zamansal tutarlilik

Bu nedenle 320 model ana adaydir. 416 model sadece kucuk hedef kazanimi FPS
kaybina degerse secilecek.
