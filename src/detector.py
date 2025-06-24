import torch

class PersonDetector:
    """Tiny wrapper around a YOLOv5 model that returns annotated frames and count."""

    def __init__(self,
                model_name: str = "yolov5s",
                classes=(0,),
                conf_threshold: float = 0.5):
        # Загрузка модели из Ultralytics hub
        self.model = torch.hub.load('ultralytics/yolov5',
                                    model_name,
                                    pretrained=True)
        self.model.classes = list(classes)
        self.model.conf = conf_threshold

    @torch.no_grad()
    def detect(self, frame):
        results = self.model(frame)
        # делаем копию, иначе OpenCV будет жаловаться на readonly
        annotated = results.render()[0].copy()
        num_people = len(results.xyxy[0])
        return annotated, num_people
