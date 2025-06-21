import os
import numpy as np
from PIL import Image
from ultralytics import YOLO


class StreamlitBeverageDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):  # Fixed the typo here
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        return False


    def detect_beverages(self, image):
        if self.model is None:
            return []
        results = self.model(image, conf=self.confidence_threshold)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_name': class_name
                    })
        return detections