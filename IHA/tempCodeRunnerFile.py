    def update(self, frame: np.ndarray) -> YoloResult:
        if not self.ready:
            return None
        self._frame_counter = (self._frame_counter + 1) % YOLO_SKIP
        if self._frame_counter == 0:
            self._last_result = self._run_inference(frame)
        return self._last_result

