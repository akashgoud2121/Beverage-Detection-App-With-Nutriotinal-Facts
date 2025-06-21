from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import numpy as np
import cv2
import os

from .detector import StreamlitBeverageDetector

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "api/models/model_final.pt")  # Set your model path
detector = StreamlitBeverageDetector(model_path=MODEL_PATH, confidence_threshold=0.6)
detector.load_model()

class DetectionResult(BaseModel):
    bbox: List[int]
    confidence: float
    class_name: str

@app.post("/detect", response_model=List[DetectionResult])
async def detect_beverage(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detections = detector.detect_beverages(opencv_image)
    return detections

@app.get("/")
def root():
    return {"message": "Smart Beverage Health Scanner API is running."}
