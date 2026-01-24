import cv2
import threading
import queue

class FastCamera:
    def __init__(self, width=640, height=480, fps=60):
        self.cap = cv2.VideoCapture(0)
        # Force low res for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.q = queue.Queue(maxsize=1) # Only keep latest frame
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            if not self.q.empty():
                try: self.q.get_nowait()
                except: pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.running = False
        self.cap.release()