import cv2

class VideoStream:
    """OpenCV VideoCapture wrapper with convenience helpers."""

    def __init__(self, source=0, window_name: str = "Video"):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.window_name = window_name

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from stream")
        return frame

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
