# ASFLY Single-Class Target Square Package

Created: 2026-05-30T12:30:44

Purpose: Raspberry Pi 4 oriented YOLO pipeline where YOLO detects one class only:

```txt
0: target_square
```

Red and blue are not separate YOLO classes. Color must be decided after detection with OpenCV crop analysis (`red_ratio`, `blue_ratio`, HSV/LAB, temporal filtering).

Main model:

```txt
models_archive/iha_best.pt
```

Training source summary:

- Final dataset: 42,000 images total
- train/val/test: 33,601 / 4,200 / 4,199
- Stage mix: Stage1 4k, Stage2 16k, Stage3 10k, Stage4 12k
- Dataset images/labels are intentionally not included in this zip to keep the package portable.
- Dataset metadata, sanity report and preview are included under `dataset_summary/`.

Synthetic test metadata result:

- test total: 4,199
- targets: 3,620
- hits: 3,620
- misses: 0
- false positives: 7
- recall: 1.0
- main weakness: colored_cloth style negatives

Important files:

- `code/train_singleclass_yolo.py`: train/validate/archive pipeline
- `code/analyze_model_by_metadata.py`: stage/color/blur/light metadata analysis
- `code/webcam.py`: webcam/inference side code in this curriculum folder
- `metadata_analysis/summary.txt`: final metadata analysis
- `metadata_analysis/false_positive_grid.jpg`: false positive examples

Recommended competition architecture:

```txt
YOLO target_square bbox
+ OpenCV red/blue crop color ratio
+ squareness/size filter
+ 3-5 frame temporal consistency
```
