import cv2
import threading
import queue
import time
import logging

class VideoWorker(threading.Thread):
    def __init__(self, source, out_q: queue.Queue, name: str = 'camera'):
        super().__init__(daemon=True)
        self.source = source
        self.out_q = out_q
        self.name = name
        self.running = False
        self.cap = None
        self.fps = 30
        
    def _open_camera(self):
        try:
            # Handle integer sources (webcams) vs string sources (files/streams)
            src = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source {self.source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        except Exception as e:
            logging.error(f"Error opening camera {self.name}: {e}")
            self.cap = None

    def run(self):
        self.running = True
        self._open_camera()
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._open_camera()
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                # If file, loop it; if camera, wait and retry
                if isinstance(self.source, str) and not str(self.source).isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.1)
                    continue
            
            # Resize for consistent processing
            frame = cv2.resize(frame, (640, 360))
            
            try:
                # Non-blocking put to avoid stalls
                self.out_q.put(frame, block=False)
            except queue.Full:
                pass  # Drop frame if queue is full
                
            # Control frame rate
            time.sleep(1.0 / self.fps)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
