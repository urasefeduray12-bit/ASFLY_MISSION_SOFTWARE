# ASFLY Tek Sinif Tasarim Notu

Bu klasor eski iki sinifli denemeyi bozmadan tek sinifli yeni mimari icin ayrildi.

## Ana karar

YOLO sadece `target_square` sinifini ogrenir. Kirmizi 1x1 m kare ve mavi 2x2 m kare label dosyalarinda ayni siniftir:

```txt
0 x_center y_center width height
```

Renk karari YOLO tarafindan verilmez. YOLO bbox verdikten sonra crop uzerinde OpenCV HSV/LAB renk analizi yapilir. Final karar:

- YOLO: hedef kare var mi, nerede?
- OpenCV: bbox icindeki hedef kirmizi mi mavi mi?
- Zaman filtresi: karar 3-5 frame tutarli mi?

## Donanim karari

Raspberry Pi 4 icin YOLO her frame calismayacak. Hedef mimari:

- YOLO 4 frame'de 1 calisir.
- Aradaki framelerde OpenCV renk, karelik ve takip kullanilir.
- Dusuk/orta YOLO confidence tek basina yeterli degildir.
- Sadece renk goruldu diye hedef kabul edilmez.

## Dataset karari

Dataset genel dunya degil, yarisma alani gibi davranmalidir:

- zemin: cim, bozkir, kuru ot, toprak, tasli alan
- hedef: yerdeki kirmizi/mavi kare
- negatif: yarisma alaninda gorulebilecek tabela, branda, serit, kiyafet, kask, canta gibi yanilticilar
- felaket augment az oranda kalir; ana dagilimi bozmaz

## Stage'ler

- `ds_stage1_shape_basic`: temel kare formu
- `ds_stage2_field_clean`: temiz yarisma zemini
- `ds_stage3_field_blur`: hareket ve defocus blur
- `ds_stage4_field_sun_shadow`: gunes, golge, glare ve yuzde 5 felaket cep

Her stage tek sinifli `data.yaml` ve analiz icin `metadata.csv` uretir.

## 2026-05-28 kritik uygulama notlari

Bu bolum preview calismalari bitip tam dataset uretimine gecilirken referans alinacak karar hafizasidir.

### Calisma klasoru

Yeni tek sinifli mimari eski iki sinifli dosyalari bozmadan burada tutuluyor:

```txt
C:\Users\byrm\Desktop\python projeleri\asfly_projem\singleclass_curriculum
```

Ana generator:

```txt
iha_dataset_singleclass_curriculum.py
```

Egitim yardimcisi:

```txt
train_singleclass_curriculum.py
```

GPU icin kullanilacak Python ortami:

```txt
C:\Users\byrm\.conda\envs\torch\python.exe
```

Bu ortamda RTX 4050 CUDA goruluyor ve `ultralytics` kurulu.

Arka planlar icin bozuk izinli eski `backgrounds` yerine okunabilir klasor:

```txt
C:\Users\byrm\Desktop\python projeleri\asfly_projem\backgrounds_ok
```

### Tek sinif kurali

Butun stage'lerde YOLO label sinifi ayni kalacak:

```txt
0: target_square
```

Kirmizi ve mavi hedefler label dosyasinda ayrilmayacak. Renk bilgisi sadece `metadata.csv` icinde tutulacak:

- `target_color=red`
- `target_color=blue`
- `target_color=none`

OpenCV renk karari daha sonra bbox crop uzerinden verilecek.

### Stage 1 durumu

Stage 1 hedefi sadece temel kare kavramini ogretmek.

Klasor:

```txt
datasets/ds_stage1_shape_basic
```

Kararlar:

- sade beyaz/gri/matte arka plan
- gercek dataset image icinde sari/yesil bbox yok
- preview bbox sadece `preview_grid.jpg` icinde
- oran: yaklasik `%45 red`, `%45 blue`, `%10 empty`
- hedef boyutu: `%70 orta/buyuk`, `%20 kucuk`, `%10 cok kucuk`
- motion blur, glare, agir golge yok

### Stage 2 durumu

Ilk Stage 2 preview uretildi ama daha sonra daha kontrollu v2 preview hazirlandi.

Kullanilacak preview klasoru:

```txt
datasets/ds_stage2_field_clean_v2_preview
```

Stage 2 hedefi:

```txt
Temiz yarisma zemini uzerinde target_square bulmayi ogretmek.
```

Kararlar:

- arka planlar: cim, kuru ot, bozkir, toprak, acik arazi, tasli zemin
- asfalt/gri zemin cok az veya yok
- cok koyu/siyah/golgeli crop'lar filtrelenir
- agir blur yok
- agir gunes, glare, lens flare, overexposure yok
- negatifler az ve makul: kirmizi panel, mavi serit/branda, beyaz cizgi, renkli bez/parca, yuvarlak obje
- labelsiz kareye cok benzeyen akilli negatifler Stage 2'de kullanilmaz

Preview kontrol sonucu:

- toplam: `120`
- red: `48`
- blue: `48`
- empty/background: `18`
- negative: `6`
- label class: sadece `0`
- empty/negative label dosyalari bos

### Stage 3 revizyon karari

Stage 3 kesinlikle Stage 4'e kaymamali.

Stage 3 hedefi:

```txt
Stage 2 temiz yarisma zemini + kontrollu hafif/orta hareket bulanikligi.
```

Stage 3'te ogretilecekler:

- kamera hareketi
- IHA titresimi
- hafif motion blur
- orta motion blur
- cok az hafif defocus/Gaussian blur

Stage 3'te ogretilmeyecekler:

- sert gunes parlamasi
- agir golge
- lens flare
- asiri overexposure
- asiri karanlik domain shift
- hedef rengini yok eden isik etkileri

Bu etkiler Stage 4'e birakilacak.

### Stage 3 yeni preview hedef klasoru

Eski Stage 3 preview ve Stage 3 v2 preview silinmeden korunabilir. Yeni revizyon icin acik isimli ayri klasor kullanilacak:

```txt
datasets/ds_stage3_field_blur_clean_motion_v2_preview
```

Tam dataset uretimine gecildiginde nihai klasor:

```txt
datasets/ds_stage3_field_blur
```

ancak preview onaylanmadan eski/nihai klasor ezilmeyecek.

### Stage 3 oranlari

Stage 3 tamamen blur dataseti olmayacak. Temiz Stage 2 tarzi pozitifler guclu kalacak.

Hedef oran:

- `%45` temiz/normal Stage 2 tarzi pozitif
- `%25` hafif motion blur pozitif
- `%15` orta motion blur pozitif
- `%5` hafif defocus/Gaussian blur pozitif
- `%5` bos yarisma alani background
- `%5` yarisma alani negatifi

Red/blue pozitifler dengeli tutulacak. Oranlar rastgele kaymasin diye plan bastan deterministik liste olarak kurulacak.

### Stage 3 blur kurallari

Blur hedefi bozmak icin degil, gercekci hareket bulanikligi katmak icindir.

Metadata en az sunlari tutacak:

- `blur_type`: `none`, `motion`, `defocus`
- `blur_level`: `none`, `light`, `medium`
- `motion_blur_length`
- `motion_blur_angle`
- `lighting_level`

Kucuk hedef + blur ozel kurali:

- cok kucuk hedeflerde motion blur yok veya en fazla cok hafif
- kucuk hedeflerde orta motion blur yasak
- orta/buyuk hedeflerde hafif veya orta motion blur olabilir
- agir blur Stage 3'te kullanilmayacak
- hedef kare formu secilemez hale gelirse ornek gecersiz sayilacak

Blur length bbox boyutuna gore clamp edilecek:

```txt
very_small: max 5 px, medium motion yasak
small: max 7 px, medium motion yasak
medium: max 11 px
large: max 15 px
```

Motion blur hedef kenarinin buyuk kismini kaplamayacak.

### Stage 3 isik ve arka plan kurallari

Stage 3 isik degil blur stage'idir.

Izin verilen:

- cok hafif brightness/contrast
- cok hafif color jitter
- dogal zemin varyasyonu

Yasak/Stage 4'e ayrilan:

- agir golge
- sert gunes
- lens flare
- glare
- asiri karanlik goruntu
- asiri parlak patlama

Arka plan filtresi Stage 2 v2'ye benzer kalacak:

- cim
- kuru ot
- bozkir
- toprak
- acik arazi
- tasli zemin

Cok koyu/golgeli crop orani sanity check ile sinirlanacak.

### Stage 3 sanity check kurallari

Dataset uretiminden sonra otomatik kontrol edilmeli:

1. `data.yaml` icinde `nc: 1` ve `0: target_square`
2. Tum label class id degerleri `0`
3. Empty/background/negative label dosyalari bos
4. Preview bbox/yazisi gercek training image icine sizmamis
5. `metadata.csv` satir sayisi image sayisiyla ayni
6. Blur dagilimi hedef oranlara yakin
7. `very_small + medium_motion` varsa hata
8. `small + medium_motion` varsa hata
9. YOLO bbox koordinatlari `0-1` araliginda
10. Cok koyu/siyah veya agir golgeli arka plan orani limit altinda
11. Temiz Stage 2 tarzi pozitif orani korunuyor

### Stage 4 ayrimi

Stage 4 daha sonra ayri ele alinacak.

Stage 4 konulari:

- gunes
- golge
- glare
- lens flare
- exposure/white balance
- blur + isik kombinasyonu
- yaklasik `%5` felaket senaryosu

Stage 3 kodlanirken Stage 4 mantigina gecilmeyecek.

## Stage 4 preview karari

Stage 4 artik gunes/golge/parlama ve sinirli felaket senaryosu icindir. Stage 3'ten farki blur degil, isik kosullarini ogretmesidir.

Preview klasoru:

```txt
datasets/ds_stage4_field_sun_shadow_v2_preview
```

Stage 4 hedefi:

```txt
Yarisma zemini uzerindeki target_square hedefinin gunes, golge, kismi glare, exposure degisimi ve az miktarda blur+isik altinda bulunmasi.
```

Stage 4 pozitif oranlari:

- `%25` clean/normal pozitif
- `%20` shadow pozitif
- `%20` sunny/bright pozitif
- `%10` partial_glare pozitif
- `%10` blur_light pozitif
- `%5` limited disaster pozitif
- `%5` empty/background
- `%5` negative

120 gorsellik preview sonucu:

- toplam: `120`
- red: `54`
- blue: `54`
- empty/background: `6`
- negative: `6`
- clean pozitif: `30`
- sunny pozitif: `24`
- shadow pozitif: `24`
- partial_glare pozitif: `12`
- blur_light pozitif: `12`
- disaster pozitif: `6`

Stage 4 kurallari:

- disaster orani yaklasik `%5` civarinda kalacak
- disaster hedefi tamamen yok etmemeli
- kucuk/cok kucuk hedeflere disaster uygulanmayacak
- partial glare hedefin tamamini beyaza patlatmayacak
- shadow cok karanlik siyah sahneye donusmeyecek
- sunny/bright hedef rengini tamamen soldurmayacak
- label sinifi yine sadece `0: target_square`
- renk karari yine OpenCV tarafinda kalacak

Stage 4 metadata alanlari:

- `blur_type`
- `blur_level`
- `motion_blur_length`
- `motion_blur_angle`
- `sun_level`
- `glare_level`
- `shadow_level`
- `lighting_level`
- `disaster_level`

Stage 4 sanity check:

- `data.yaml` tek sinif mi
- tum label class id degerleri `0` mi
- empty/negative label dosyalari bos mu
- metadata satir sayisi image sayisiyla esit mi
- disaster orani fazla mi
- small/very_small hedefe disaster uygulanmis mi
- asiri karanlik veya asiri parlak gorsel orani limit altinda mi
- preview bbox/yazisi training image icine sizmamis mi

## Stage 4 controlled preview revizyonu

Stage 4 ilk preview dogru yondeydi ama final dataset uretimine gecmeden once daha kontrollu hale getirilecek.

Yeni controlled preview klasoru:

```txt
datasets/ds_stage4_field_sun_shadow_controlled_v2_preview
```

Ana karar:

```txt
Stage 4 zorlastirma asamasi ama hedefi yok etme asamasi degil.
```

Bu revizyonda Stage 4 artik sunlari hedefler:

- normal/temiz Stage 2 tarzi pozitifleri korumak
- hafif/orta golge
- gunesli/parlak ama hedefi patlatmayan sahne
- kontrollu kismi glare/parlama
- hafif overexposure
- hafif color shift
- hafif blur + isik kombinasyonu
- az sayida bos/negative yarisma alani ornegi

Bu revizyonda kullanilmayacak veya cok sinirlanacak seyler:

- hedefin tamamen beyaza patlamasi
- tum goruntunun siyaha gomulmesi
- asiri saturate/yapay renk filtresi
- kucuk hedef + agir glare
- kucuk hedef + agir shadow
- kucuk hedef + blur + agir isik kombinasyonu
- disaster/felaket agirligi

Controlled Stage 4 preview hedef oranlari:

- `%30` clean/normal pozitif
- `%20` shadow pozitif
- `%20` sunny/bright pozitif
- `%10` partial_glare pozitif
- `%10` blur_light pozitif
- `%5` empty/background
- `%5` negative

120 gorsellik preview icin beklenen sayilar:

- clean pozitif: `36`
- shadow pozitif: `24`
- sunny pozitif: `24`
- partial_glare pozitif: `12`
- blur_light pozitif: `12`
- empty/background: `6`
- negative: `6`
- red/blue pozitif dengesi: `54 / 54`

Stage 4 controlled metadata kolonlari:

- `image_name`
- `split`
- `stage`
- `has_target`
- `target_color`
- `target_type`
- `background_type`
- `negative_type`
- `bbox_x`
- `bbox_y`
- `bbox_w`
- `bbox_h`
- `bbox_px_size`
- `rotation_deg`
- `perspective_level`
- `blur_type`
- `blur_level`
- `lighting_type`
- `lighting_level`
- `shadow_level`
- `glare_level`
- `exposure_level`
- `color_shift_level`
- `notes`

Kucuk hedef koruma kurali:

- `very_small`: sadece clean, light shadow veya light sunny; glare ve blur_light yok
- `small`: medium glare yok; blur_light sadece cok hafif; agir shadow yok
- `medium`: shadow/sun/glare kontrollu olabilir
- `large`: daha guclu Stage 4 etkileri uygulanabilir ama hedef yine secilebilir kalmali

Controlled Stage 4 sanity check:

1. `data.yaml` tek sinif `target_square` mi
2. label class id sadece `0` mi
3. empty/negative label dosyalari bos mu
4. metadata satir sayisi image sayisiyla uyumlu mu
5. clean/shadow/sunny/glare/blur_light oranlari hedefe yakin mi
6. cok karanlik goruntu orani limitli mi
7. asiri parlak/patlamis goruntu orani limitli mi
8. asiri color shift orani limitli mi
9. kucuk hedeflere agir glare/shadow/blur kombinasyonu kacmis mi
10. bbox koordinatlari `0-1` araliginda mi
11. preview bbox/yazisi gercek training image icine sizmamis mi

## Final dataset builder

Final YOLO datasetini olusturmak icin ayri script yazildi:

```txt
build_final_dataset.py
```

Bu script stage generator'lara dokunmaz. Sadece onayli stage datasetlerini okur, final klasore kopyalar ve standart final metadata/sanity raporu yazar.

Onayli kaynak stage klasorleri:

- `ds_stage1_shape_basic`
- `ds_stage2_field_clean_v2_preview`
- `ds_stage3_field_blur_clean_motion_v2_preview`
- `ds_stage4_field_sun_shadow_controlled_v2_preview`

Final hedef klasor:

```txt
datasets/ds_final_target_square
```

Debug test klasoru:

```txt
datasets/ds_final_target_square_debug
```

Desteklenen modlar:

- `--mode debug --max-per-stage 50`: pipeline testi, yaklasik 200 image
- `--mode mini_final`: yaklasik 24k image hedefler
- `--mode full_final`: yaklasik 42k image hedefler

Final stage oranlari / hedef sayilar:

- Stage 1: `4000`
- Stage 2: `16000`
- Stage 3: `10000`
- Stage 4: `12000`
- Toplam: `42000`

Not: Ilk oran bazli denemede script `4200/16800/10500/10500` istemisti ve kaynak stage sayilariyla uyusmadigi icin final `40500` kalmisti. `full_final` modu daha sonra bilincli kaynak planina sabitlendi: `4000/16000/10000/12000`.

Final split:

- train: `%80`
- val: `%10`
- test: `%10`

Split her stage icinde ayrica dengelenir. Debug testte 50 image/stage icin sonuc:

- train: `160`
- val: `20`
- test: `20`
- her stage: `50`
- her stage split: `40/5/5`

Final metadata kesin alanlari:

- `sample_id`
- `source_stage`
- `original_image_name`
- `original_split`
- `final_image_name`
- `final_split`
- standart hedef/renk/bbox/blur/lighting alanlari
- `image_hash`
- `label_hash`

Final builder kontrolleri:

- tum label class id degerleri final kopyada `0`
- eski iki sinifli label cikarsa kaynak degistirilmeden final kopyada `0` yapilir ve raporlanir
- empty/background/negative icin bos `.txt` dosyasi garanti edilir
- image hash ve image+label hash duplicate kontrolu yapilir
- preview/debug/grid/annotated/vis isimli dosyalar training kaynagi olarak alinmaz
- `sanity_report.txt` genel ve split bazli dagilimlari yazar
- `cleanup_dry_run.txt` sadece dislanabilecek dosyalari raporlar; hicbir seyi silmez

## 42k final dataset sonucu

Uretilen final dataset:

```txt
datasets/ds_final_target_square
```

Kaynak stage datasetleri ayri tutuldu:

```txt
datasets/final_sources_42k
```

Son sanity sonucu:

- toplam image: `42000`
- train/val/test: `33601 / 4200 / 4199`
- stage sayilari: Stage 1 `4000`, Stage 2 `16000`, Stage 3 `10000`, Stage 4 `12000`
- hedefli/targetsiz: `36200 / 5800`
- renk dagilimi: red `18100`, blue `18100`, none `5800`
- label class donusumu: `0`
- sanity error: `none`
- duplicate uyarisi: `72` image hash grubu ve `72` image+label hash grubu; agirlikla Stage 1 sade/empty benzeri tekrarlar

Split dagilimi targetsiz ornekleri de val/test icerecek sekilde duzeltildi:

- train has_target/none: `28960 / 4641`
- val has_target/none: `3620 / 580`
- test has_target/none: `3620 / 579`

Final preview:

```txt
datasets/ds_final_target_square/preview_grid_final.jpg
```

## Tek sinifli YOLO egitim pipeline

Egitim dosyalari eklendi:

```txt
train_singleclass_yolo.py
analyze_model_by_metadata.py
TRAINING_README.md
```

On kontrol komutu:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" train_singleclass_yolo.py --dry-check
```

Dry-check sonucu:

- CUDA aktif: `NVIDIA GeForce RTX 4050 Laptop GPU`
- torch: `2.6.0+cu124`
- train/val/test images: `33601 / 4200 / 4199`
- train/val/test labels: `33601 / 4200 / 4199`
- class ids: sadece `0`
- empty labels: `5800`
- nonempty labels: `36200`
- sanity errors: `none`

Ultralytics izin sorunu icin script basinda `YOLO_CONFIG_DIR` proje icindeki su klasore yonlendirildi:

```txt
.ultralytics_config
```

Onerilen ilk egitim:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" train_singleclass_yolo.py --imgsz 320 --epochs 60 --batch 8 --patience 15 --workers 2 --name target_square_yolov8n_320_baseline
```

Windows worker sorunu olursa `--workers 0` kullan.

416 karsilastirma:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" train_singleclass_yolo.py --imgsz 416 --epochs 50 --batch 8 --patience 15 --workers 2 --name target_square_yolov8n_416_compare
```

Egitim bitince script:

- val ve test validation calistirir
- test splitinden 50 image predict preview uretir
- `best.pt` ve `last.pt` dosyalarini `models_archive` altina kopyalar
- metrics ve train args JSON olarak kaydeder

Metadata analiz komutu egitimden sonra:

```powershell
& "C:\Users\byrm\.conda\envs\torch\python.exe" analyze_model_by_metadata.py --model models_archive\<MODEL_ADI>_best.pt --imgsz 320 --conf 0.25
```
