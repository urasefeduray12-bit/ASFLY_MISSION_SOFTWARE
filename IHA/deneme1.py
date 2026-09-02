import cv2
import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

Kp_x = 0.002
Kp_y = 0.002
Kp_z = 0.5
MAX_SPEED = 2.0
TAKEOFF_ALT = 5.0
INIS_ALAN_ESIK = 15000
MERKEZ_TOLERANS = 30
DRONE_BAGLANTI = "udp://:14540"

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
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.onceki_gri = None
        self.onceki_noktalar = None
        self.takip_aktif = False
        self.kayip_sayac = 0
        self.max_kayip = 8

    def kalman_tahmin(self):
        tahmin = self.kalman.predict()
        return int(tahmin[0]), int(tahmin[1])

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

        yeni_noktalar, durum, _ = cv2.calcOpticalFlowPyrLK(
            self.onceki_gri, gri_frame, self.onceki_noktalar, None, **LK_PARAMS
        )

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
            self.onceki_noktalar = None
            return False
        return True


class DroneController:
    def __init__(self):
        self.drone = System()
        self.aktif = False

    async def baglan(self):
        print("Drone'a bağlanılıyor...")
        await self.drone.connect(system_address=DRONE_BAGLANTI)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("-- Drone bağlandı!")
                break

        print("GPS kilidi bekleniyor...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- GPS kilidi alındı.")
                break

    async def kalkis(self):
        print(f"-- Arm ediliyor")
        await self.drone.action.arm()

        print(f"-- Kalkış yapılıyor ({TAKEOFF_ALT}m)")
        await self.drone.action.set_takeoff_altitude(TAKEOFF_ALT)
        await self.drone.action.takeoff()
        await asyncio.sleep(8)

        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )
        try:
            await self.drone.offboard.start()
            self.aktif = True
            print("-- Offboard mod başlatıldı.")
        except OffboardError as e:
            print(f"Offboard başlatılamadı: {e._result.result}")
            await self.drone.action.disarm()

    async def hedefe_git(self, error_x, error_y, alan):
        if not self.aktif:
            return

        vx = np.clip(Kp_y * error_y, -MAX_SPEED, MAX_SPEED)
        vy = np.clip(Kp_x * error_x, -MAX_SPEED, MAX_SPEED)

        vz = 0.0
        if abs(error_x) < MERKEZ_TOLERANS and abs(error_y) < MERKEZ_TOLERANS:
            vz = Kp_z

        if alan > INIS_ALAN_ESIK:
            vz = 1.0

        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, 0.0)
        )

    async def hover(self):
        if not self.aktif:
            return
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )

    async def inis(self):
        if self.aktif:
            try:
                await self.drone.offboard.stop()
            except OffboardError:
                pass
            self.aktif = False

        print("-- İniş yapılıyor...")
        await self.drone.action.land()


def renk_tespit(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_red1 = [0, 150, 100]
    upper_red1 = [10, 255, 255]
    lower_red2 = [170, 150, 100]
    upper_red2 = [180, 255, 255]

    red_stack = MaskStack(hsv_frame, [lower_red1, upper_red1], [lower_red2, upper_red2])

    contours, _ = cv2.findContours(
        red_stack.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    en_buyuk = None
    en_buyuk_alan = 0

    for contour in contours:
        alan = cv2.contourArea(contour)
        if alan > 3000 and alan > en_buyuk_alan:
            en_buyuk = contour
            en_buyuk_alan = alan

    hedef_bilgi = None

    if en_buyuk is not None:
        peri = cv2.arcLength(en_buyuk, True)
        approx = cv2.approxPolyDP(en_buyuk, 0.04 * peri, True)

        kose_sayisi = len(approx)
        sekil_ismi = "Bilinmiyor"

        if kose_sayisi == 3:
            sekil_ismi = "Ucgen"
        elif kose_sayisi == 4:
            x, y, w, h = cv2.boundingRect(approx)
            oran = float(w) / h
            sekil_ismi = "Kare" if 0.9 <= oran <= 1.1 else "Dikdortgen"
        elif kose_sayisi == 5:
            sekil_ismi = "Besgen"
        elif kose_sayisi == 6:
            sekil_ismi = "Altigen"
        elif kose_sayisi > 8:
            sekil_ismi = "Daire/Elips"

        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

        M = cv2.moments(en_buyuk)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            hedef_bilgi = {
                "cX": cX, "cY": cY,
                "alan": en_buyuk_alan,
                "sekil": sekil_ismi,
                "contour": en_buyuk
            }

        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(frame, sekil_ismi, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, red_stack.mask, hedef_bilgi


async def main():
    drone_ctrl = DroneController()
    await drone_ctrl.baglan()
    await drone_ctrl.kalkis()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Hata: Kamera bağlantısı kurulamadı.")
        await drone_ctrl.inis()
        return

    takipci = HedefTakipci()

    print("Hedef aranıyor... Çıkış için 'q' tuşuna basın.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            merkez_x = frame_w // 2
            merkez_y = frame_h // 2
            gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame, mask, renk_hedef = renk_tespit(frame)

            hedef_cx, hedef_cy = None, None
            hedef_alan = 0
            kaynak = ""

            if renk_hedef:
                hedef_cx = renk_hedef["cX"]
                hedef_cy = renk_hedef["cY"]
                hedef_alan = renk_hedef["alan"]
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
                    devam = takipci.kayip_artir()
                    if devam:
                        hedef_cx, hedef_cy = takipci.kalman_tahmin()
                        kaynak = "KALMAN TAHMIN"

            if hedef_cx is not None and hedef_cy is not None:
                tahmin_x, tahmin_y = takipci.kalman_tahmin()

                cv2.circle(frame, (hedef_cx, hedef_cy), 7, (255, 255, 255), -1)
                cv2.putText(frame, "HEDEF", (hedef_cx - 20, hedef_cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.circle(frame, (tahmin_x, tahmin_y), 10, (255, 0, 255), 2)
                cv2.putText(frame, "TAHMIN", (tahmin_x + 12, tahmin_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                error_x = tahmin_x - merkez_x
                error_y = tahmin_y - merkez_y

                durum = f"[{kaynak}] X:{error_x:+d} Y:{error_y:+d} Alan:{hedef_alan}"
                cv2.putText(frame, durum, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.line(frame, (merkez_x, merkez_y), (tahmin_x, tahmin_y),
                         (0, 0, 255), 2)

                if takipci.onceki_noktalar is not None:
                    for pt in takipci.onceki_noktalar:
                        a, b = pt.ravel().astype(int)
                        cv2.circle(frame, (a, b), 3, (0, 200, 255), -1)

                await drone_ctrl.hedefe_git(error_x, error_y, hedef_alan)

                if hedef_alan > INIS_ALAN_ESIK and abs(error_x) < MERKEZ_TOLERANS and abs(error_y) < MERKEZ_TOLERANS:
                    print("HEDEF ÜZERİNDE — İNİŞ BAŞLATIYOR!")
                    cv2.putText(frame, "INIS BASLATILIYOR", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("REDi", frame)
                    cv2.waitKey(1000)
                    await drone_ctrl.inis()
                    break
            else:
                cv2.putText(frame, "Hedef yok - Hover", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                await drone_ctrl.hover()
                takipci.onceki_gri = gri.copy()

            cv2.circle(frame, (merkez_x, merkez_y), 5, (0, 255, 255), -1)

            cv2.imshow("REDi", frame)
            cv2.imshow("Maske", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Kullanıcı çıkış yaptı — iniş başlatılıyor.")
                await drone_ctrl.inis()
                break

            await asyncio.sleep(0.03)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())