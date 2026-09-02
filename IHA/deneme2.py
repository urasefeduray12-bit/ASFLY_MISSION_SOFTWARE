import cv2
import numpy as np
import asyncio

# --- MAVSDK bağımlılıklarını şimdilik devre dışı bıraktık ---
# Bilgisayarda test için Kp değerlerini ve esikleri koruyoruz
Kp_x = 0.002
Kp_y = 0.002
MAX_SPEED = 2.0
INIS_ALAN_ESIK = 15000
MERKEZ_TOLERANS = 30

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

class MaskStack:
    def __init__(self, frame, *arr):
        self.frame = frame
        self.mask = np.zeros(frame.shape[:2], dtype="uint8")
        for arg in arr:
            lower = np.array(arg[0], dtype="uint8")
            upper = np.array(arg[1], dtype="uint8")
            temp_mask = cv2.inRange(self.frame, lower, upper)
            self.mask = cv2.bitwise_or(self.mask, temp_mask)
        _, self.mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

class HedefTakipci:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.onceki_gri = None
        self.onceki_noktalar = None
        self.takip_aktif = False
        self.kayip_sayac = 0
        self.max_kayip = 8

    def kalman_tahmin(self):
        tahmin = self.kalman.predict()
        x = tahmin[0].item()
        y = tahmin[1].item()
        return int(x), int(y)

    def kalman_guncelle(self, cx, cy):
        olcum = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(olcum)
        self.takip_aktif = True
        self.kayip_sayac = 0

    def kalman_guncelle_of(self, cx, cy):
        olcum = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(olcum)

    def optik_akis_guncelle(self, gri_frame):
        if self.onceki_gri is None or self.onceki_noktalar is None:
            self.onceki_gri = gri_frame.copy()
            return None
        yeni_noktalar, durum, _ = cv2.calcOpticalFlowPyrLK(self.onceki_gri, gri_frame, self.onceki_noktalar, None, **LK_PARAMS)
        self.onceki_gri = gri_frame.copy()
        if yeni_noktalar is not None and durum is not None:
            iyi = durum.ravel() == 1
            if np.any(iyi):
                iyi_noktalar = yeni_noktalar[iyi].reshape(-1, 2)
                merkez = iyi_noktalar.mean(axis=0)
                self.onceki_noktalar = iyi_noktalar.reshape(-1, 1, 2)
                return int(merkez[0].item()), int(merkez[1].item())
        return None

    def noktalari_ayarla(self, contour, gri_frame):
        mask = np.zeros(gri_frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        noktalar = cv2.goodFeaturesToTrack(gri_frame, maxCorners=50, qualityLevel=0.1, minDistance=7, mask=mask)
        if noktalar is not None:
            self.onceki_noktalar = noktalar
        self.onceki_gri = gri_frame.copy()

    def kayip_artir(self):
        self.kayip_sayac += 1
        if self.kayip_sayac > self.max_kayip:
            self.takip_aktif = False
            return False
        return True

# --- TEST İÇİN SİMÜLE EDİLMİŞ KONTROLCÜ ---
class MockDroneController:
    def __init__(self):
        self.aktif = True

    async def hedefe_git(self, error_x, error_y, alan):
        vx = np.clip(Kp_y * error_y, -MAX_SPEED, MAX_SPEED)
        vy = np.clip(Kp_x * error_x, -MAX_SPEED, MAX_SPEED)
        # Ekrana drone'un ne yapacağını yazdırıyoruz
        print(f"Sanal Hareket -> İleri Hız: {vx:.2f}, Sağ Hız: {vy:.2f}, Alan: {alan}")

    async def hover(self):
        pass

def renk_tespit(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = [0, 150, 100], [10, 255, 255]
    lower_red2, upper_red2 = [170, 150, 100], [180, 255, 255]
    red_stack = MaskStack(hsv_frame, [lower_red1, upper_red1], [lower_red2, upper_red2])
    contours, _ = cv2.findContours(red_stack.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    en_buyuk, en_buyuk_alan = None, 0
    for contour in contours:
        alan = cv2.contourArea(contour)
        if alan > 2000 and alan > en_buyuk_alan: # Eşik değerini test için biraz düşürdük
            en_buyuk, en_buyuk_alan = contour, alan

    hedef_bilgi = None
    if en_buyuk is not None:
        M = cv2.moments(en_buyuk)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            hedef_bilgi = {"cX": cX, "cY": cY, "alan": en_buyuk_alan, "contour": en_buyuk}
        cv2.drawContours(frame, [en_buyuk], -1, (0, 255, 0), 2)
    return frame, red_stack.mask, hedef_bilgi

async def main():
    drone_ctrl = MockDroneController()
    cap = cv2.VideoCapture(0)
    takipci = HedefTakipci()

    print("SİSTEM BAŞLADI: Kırmızı bir nesne gösterin.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_h, frame_w = frame.shape[:2]
        merkez_x, merkez_y = frame_w // 2, frame_h // 2
        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame, mask, renk_hedef = renk_tespit(frame)

        hedef_cx, hedef_cy, kaynak = None, None, ""

        if renk_hedef:
            hedef_cx, hedef_cy = renk_hedef["cX"], renk_hedef["cY"]
            kaynak = "RENK"
            takipci.kalman_guncelle(hedef_cx, hedef_cy)
            takipci.noktalari_ayarla(renk_hedef["contour"], gri)
        else:
            of_sonuc = takipci.optik_akis_guncelle(gri)
            if of_sonuc and takipci.takip_aktif:
                hedef_cx, hedef_cy = of_sonuc
                takipci.kalman_guncelle_of(hedef_cx, hedef_cy)
                kaynak = "OPTIK AKIS"
                takipci.kayip_artir()
            elif takipci.takip_aktif:
                if takipci.kayip_artir():
                    hedef_cx, hedef_cy = takipci.kalman_tahmin()
                    kaynak = "KALMAN TAHMIN"

        if hedef_cx is not None:
            tahmin_x, tahmin_y = takipci.kalman_tahmin()
            error_x, error_y = tahmin_x - merkez_x, tahmin_y - merkez_y
            
            # Görselleştirmeler
            cv2.circle(frame, (hedef_cx, hedef_cy), 7, (255, 255, 255), -1)
            cv2.circle(frame, (tahmin_x, tahmin_y), 10, (255, 0, 255), 2)
            cv2.line(frame, (merkez_x, merkez_y), (tahmin_x, tahmin_y), (0, 0, 255), 2)
            cv2.putText(frame, f"Mod: {kaynak} ErrX: {error_x}", (10, 30), 1, 1.5, (0, 255, 0), 2)
            
            await drone_ctrl.hedefe_git(error_x, error_y, renk_hedef["alan"] if renk_hedef else 0)

        cv2.circle(frame, (merkez_x, merkez_y), 5, (0, 255, 255), -1)
        cv2.imshow("Test Ekrani", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())