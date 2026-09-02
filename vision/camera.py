import queue
import threading
import time

import cv2


class CameraWorker:
    def __init__(self, camera=0, width=640, height=480, use_picamera2=True):
        self.camera = camera
        self.width = int(width)
        self.height = int(height)
        self.use_picamera2 = use_picamera2
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = None
        self.cap = None
        self.picam2 = None
        self.backend = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="CameraWorker", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()

    def read_latest(self, timeout=0.5):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        if self.use_picamera2 and self._open_picamera2():
            self._picamera_loop()
            return
        self._opencv_loop()

    def _open_picamera2(self):
        try:
            from picamera2 import Picamera2
        except Exception:
            return False

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.backend = "picamera2"
            return True
        except Exception:
            self.picam2 = None
            return False

    def _picamera_loop(self):
        while not self.stop_event.is_set():
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._put_latest((time.time(), frame_bgr))

    def _opencv_loop(self):
        self.cap = cv2.VideoCapture(self.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.backend = "opencv"

        if not self.cap.isOpened():
            self._put_latest((time.time(), None))
            return

        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if ok:
                self._put_latest((time.time(), frame))
            else:
                time.sleep(0.02)

    def _put_latest(self, item):
        try:
            self.frame_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            pass
