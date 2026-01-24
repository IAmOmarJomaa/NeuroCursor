import cv2
import threading
import time

class CameraStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # OPTIMIZATION: Low Resolution for Max FPS (AI doesn't need 4K)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 60) # Request 60 FPS
        
        self.status, self.frame = self.capture.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    with self.lock:
                        self.frame = frame
                        self.status = status
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            # Mirror frame immediately for intuitive interaction
            return self.status, cv2.flip(self.frame, 1) if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.capture.release()