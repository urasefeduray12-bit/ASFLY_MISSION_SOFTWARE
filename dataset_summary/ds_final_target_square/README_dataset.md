# ASFLY Final Target Square Dataset

YOLO tek siniflidir: `0: target_square`.

Kirmizi 1x1 m kare ve mavi 2x2 m kare ayni YOLO sinifi olarak egitilir.
Renk karari egitimde YOLO'ya verilmez; gercek sistemde YOLO bbox crop'u uzerinden OpenCV HSV/LAB analiziyle yapilir.

## Label format

```txt
0 x_center y_center width height
```

Bos/background/negative goruntuler icin label dosyasi vardir ama bostur.

## Kaynak stage hedef oranlari

- Stage 1 shape/basic: %10
- Stage 2 field clean: %40
- Stage 3 blur: %25
- Stage 4 sun/shadow/glare: %25

Uretim modu: `full_final`
Hedef stage sayilari: `{'stage1': 4000, 'stage2': 16000, 'stage3': 10000, 'stage4': 12000}`

## Split

Final split hedefi train/val/test = 80/10/10. Split final dataset olusturulurken yeniden yapilir.

## Metadata

metadata.csv icinde source stage, original dosya adi, final dosya adi, renk bilgisi, bbox piksel bilgisi, blur ve lighting bilgileri tutulur.

## Egitim

Ultralytics YOLO egitiminde `data.yaml` dosyasini kullan.
