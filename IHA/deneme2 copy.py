import cv2
import numpy as np
import asyncio
import threading
import time
import math
from ultralytics import YOLO
# --- SIKILASTIRILMIS GEOMETRI ---
MIN_CONTOUR_AREA  = 500
POLY_APPROX_EPS   = 0.035
MIN_SOLIDITY      = 0.85
SIDE_TOLERANCE    = 0.20
ANGLE_TOLERANCE   = 15.0
SQUARE_AR_TOL     = 0.15
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_MODEL = "yolov8n.pt"
class GeometryEngine:
    @staticmethod
    def interior_angle_deg(p1, vertex, p2):
        v1 = (p1 - vertex).astype(float); v2 = (p2 - vertex).astype(float)
        den = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return math.degrees(math.acos(np.clip(np.dot(v1, v2) / den, -1.0, 1.0)))
    @staticmethod
    def validate_strict(cnt, approx, expected_v, ideal_ang, is_sq):
        if len(approx) != expected_v: return False
        if not cv2.isContourConvex(approx): return False
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area < 1 or (cv2.contourArea(cnt) / hull_area) < MIN_SOLIDITY: return False
        
        pts = approx.reshape(-1, 2).astype(float)
        sides = [np.linalg.norm(pts[(i+1)%expected_v] - pts[i]) for i in range(expected_v)]
        mean_s = np.mean(sides)
        if mean_s < 1e-6: return False
        
        if any(abs(s - mean_s)/mean_s > SIDE_TOLERANCE for s in sides): return False
        for i in range(expected_v):
            ang = GeometryEngine.interior_angle_deg(pts[(i-1)%expected_v], pts[i], pts[(i+1)%expected_v])
            if abs(ang - ideal_ang) > ANGLE_TOLERANCE: return False
        if is_sq:
            _, _, w, h = cv2.boundingRect(approx)
            if abs((w/h) - 1.0) > SQUARE_AR_TOL: return False
        return True
class ThreadedCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()
    def update(self):
        while self.running:
            g, f = self.cap.read()
            if g: self.grabbed, self.frame = g, f
    def read(self): return self.grabbed, self.frame.copy() if self.frame is not None else None
    def stop(self): self.running = False; self.cap.release()
class ThreadedYOLO:
    def __init__(self, path):
        self.model = YOLO(path)
        self.frame = None; self.results = None
        self.enabled = False; self.running = True
        threading.Thread(target=self.infer, daemon=True).start()
    def infer(self):
        while self.running:
            if self.enabled and self.frame is not None:
                self.results = self.model.predict(self.frame, verbose=False, imgsz=320)
            time.sleep(0.01)
async def main():
    cam = ThreadedCamera(); yolo = ThreadedYOLO(YOLO_MODEL)
    prev_t = time.time()
    
    print("\n--- SISTEM GUNCELLENDI: ANTI-FALSE POSITIVE MODU ---")
    while True:
        grabbed, frame = cam.read()
        if not grabbed: break
        yolo.frame = frame
        
        # --- ROBUST KANAL ANALIZI ---
        # Gürültüyü silmek için hafif blur
        denoised = cv2.GaussianBlur(frame, (5,5), 0)
        b, g, r = cv2.split(denoised.astype(np.float32))
        
        # Kırmızı Şartı: R kanalı, G ve B'den en az 2 kat baskın olmalı
        mask_red = np.where((r > g * 2.0) & (r > b * 2.0), 255, 0).astype(np.uint8)
        
        # Mavi Şartı: B kanalı, R ve G'den en az 2 kat baskın olmalı
        mask_blue = np.where((b > r * 2.0) & (b > g * 2.0), 255, 0).astype(np.uint8)
        hedef = None
        targets = [(mask_red, "Kirmizi Kare", 4, 90.0, True), (mask_blue, "Mavi Altigen", 6, 120.0, False)]
        
        for mask, name, v_count, i_angle, is_sq in targets:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < MIN_CONTOUR_AREA: continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, POLY_APPROX_EPS * peri, True)
                if GeometryEngine.validate_strict(c, approx, v_count, i_angle, is_sq):
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        hedef = {"cx": cx, "cy": cy, "name": name, "cnt": c}
                        break
            if hedef: break
        # HUD
        fps = 1 / (time.time() - prev_t + 1e-6); prev_t = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 1, 1.2, (255, 255, 0), 2)
        
        if hedef:
            cv2.drawContours(frame, [hedef["cnt"]], -1, (0, 255, 0), 3)
            cv2.putText(frame, hedef["name"], (hedef["cx"]-30, hedef["cy"]-30), 1, 1.3, (0,255,0), 2)
        else:
            cv2.putText(frame, "HEDEF ARANIYOR (STRICT COLOR+GEOM)...", (10, 60), 1, 1.1, (0, 0, 255), 2)
        if yolo.enabled and yolo.results:
            for r in yolo.results[0].boxes:
                bx = r.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (255,0,255), 2)
        cv2.imshow("IHA Anti-False Positive", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('y'): yolo.enabled = not yolo.enabled
        await asyncio.sleep(0.01)
    cam.stop(); yolo.running = False; cv2.destroyAllWindows()
if __name__ == "__main__":
    asyncio.run(main())