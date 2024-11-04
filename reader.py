import cv2
import threading
import time


class Camera:

    def __init__(self, rtsp_link, width=1920, height=1080):
        self.running = False
        self.last_frame = None
        self.last_ready = None
        self.lock = threading.Lock()
        self.capture = cv2.VideoCapture(rtsp_link)
        self.width = width
        self.height = height
        self.thread = threading.Thread(target=self.rtsp_cam_buffer, args=(self.capture,), name="rtsp_read_thread")
        self.thread.daemon = True
        self.thread.start()

    def rtsp_cam_buffer(self, capture_):
        self.running = True
        while self.running:
            with self.lock:
                self.last_ready, self.last_frame = capture_.read()
            time.sleep(0.0001)

    def getFrame(self):
        if (self.last_ready is True) and (self.last_frame is not None):
            return cv2.resize(self.last_frame.copy(), (self.width, self.height))
        else:
            return None

    def isOpened(self):
        with self.lock:
            return self.capture.isOpened()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()