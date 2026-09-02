"""
Mavi altıgenin üzerine fareyi getir → terminalde HSV değerlerini göster.
Doğru aralığı bulduktan sonra drone_takip.py'deki HSV_BLUE'yu güncelle.

Kullanım: python hsv_bulucu.py
"""
import cv2
import numpy as np

CAM_ID = 0
hsv_val = (0, 0, 0)

def mouse_cb(event, x, y, flags, param):
    global hsv_val
    if event == cv2.EVENT_MOUSEMOVE:
        hsv_val = param[y, x]

cap = cv2.VideoCapture(CAM_ID)
cv2.namedWindow("Kamera")
cv2.namedWindow("Maske")

# Trackbar ile aralık ayarı
cv2.namedWindow("HSV Ayar")
for name, val in [("H_alt",100),("H_ust",135),
                  ("S_alt",80), ("S_ust",255),
                  ("V_alt",50), ("V_ust",255)]:
    cv2.createTrackbar(name, "HSV Ayar", val, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.setMouseCallback("Kamera", mouse_cb, hsv)

    h_alt = cv2.getTrackbarPos("H_alt","HSV Ayar")
    h_ust = cv2.getTrackbarPos("H_ust","HSV Ayar")
    s_alt = cv2.getTrackbarPos("S_alt","HSV Ayar")
    s_ust = cv2.getTrackbarPos("S_ust","HSV Ayar")
    v_alt = cv2.getTrackbarPos("V_alt","HSV Ayar")
    v_ust = cv2.getTrackbarPos("V_ust","HSV Ayar")

    mask = cv2.inRange(hsv,
                       np.array([h_alt, s_alt, v_alt]),
                       np.array([h_ust, s_ust, v_ust]))

    h, s, v = hsv_val
    info = f"Fare HSV: H={h}  S={s}  V={v}"
    cv2.putText(frame, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Aralik: [{h_alt},{s_alt},{v_alt}] - [{h_ust},{s_ust},{v_ust}]",
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)

    cv2.imshow("Kamera", frame)
    cv2.imshow("Maske",  mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n✓ Kopyala → drone_takip.py:")
        print(f"HSV_BLUE = (np.array([{h_alt}, {s_alt}, {v_alt}]), "
              f"np.array([{h_ust}, {s_ust}, {v_ust}]))")
        break

cap.release()
cv2.destroyAllWindows()