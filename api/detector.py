import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Nutrition database (copy from your original code)
NUTRITION_DATABASE = {
    "pepsi": {
        "name": "Pepsi Cola",
        "sugar_per_100ml": 11.0,
        "calories_per_100ml": 42,
        "caffeine_per_100ml": 10.4,
        "typical_volume": 330,
        "color": (255, 0, 0),
        "health_warning": "High sugar content"
    },
    "7up_320ml": {
        "name": "7up 320ml",
        "sugar_per_100ml": 10.5,
        "calories_per_100ml": 40,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (0, 255, 0),
        "health_warning": "Contains high sugar"
    },
   "7up_c1_5l": {
        "name": "7up 1.5L",
        "sugar_per_100ml": 9.8,
        "calories_per_100ml": 38,
        "caffeine_per_100ml": 0,
        "typical_volume": 1500,
        "color": (0, 255, 0),
        "health_warning": "Contains high sugar"
    },
    "7up_c390ml": {
        "name": "7up 390ml",
        "sugar_per_100ml": 10.3,
        "calories_per_100ml": 39,
        "caffeine_per_100ml": 0,
        "typical_volume": 390,
        "color": (0, 255, 0),
        "health_warning": "Contains high sugar"
    },
    "7up_chat_xo_320ml": {
        "name": "7up Chat XO 320ml",
        "sugar_per_100ml": 10.8,
        "calories_per_100ml": 41,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (0, 255, 0),
        "health_warning": "Contains high sugar"
    },
    "boss_180ml": {
        "name": "Boss 180ml",
        "sugar_per_100ml": 9.5,
        "calories_per_100ml": 37,
        "caffeine_per_100ml": 30.0,
        "typical_volume": 180,
        "color": (255, 165, 0),
        "health_warning": "Energy drink - High caffeine"
    },
    "lipton_c450ml": {
        "name": "Lipton Ice Tea 450ml",
        "sugar_per_100ml": 7.8,
        "calories_per_100ml": 30,
        "caffeine_per_100ml": 8.0,
        "typical_volume": 450,
        "color": (255, 215, 0),
        "health_warning": "Moderate sugar content"
    },
    "mirinda_cam_320ml": {
        "name": "Mirinda Cam 320ml",
        "sugar_per_100ml": 11.2,
        "calories_per_100ml": 45,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (0, 165, 255),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_cam_c1_5l": {
        "name": "Mirinda Cam 1.5L",
        "sugar_per_100ml": 10.9,
        "calories_per_100ml": 44,
        "caffeine_per_100ml": 0,
        "typical_volume": 1500,
        "color": (0, 165, 255),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_cam_c390ml": {
        "name": "Mirinda Cam 390ml",
        "sugar_per_100ml": 11.0,
        "calories_per_100ml": 43,
        "caffeine_per_100ml": 0,
        "typical_volume": 390,
        "color": (0, 165, 255),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_soda_kem_320ml": {
        "name": "Mirinda Soda Kem 320ml",
        "sugar_per_100ml": 11.5,
        "calories_per_100ml": 46,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (0, 165, 255),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_soda_kem_c1_5l": {
        "name": "Mirinda Soda Kem 1.5L",
        "sugar_per_100ml": 10.8,
        "calories_per_100ml": 45,
        "caffeine_per_100ml": 0,
        "typical_volume": 1500,
        "color": (0, 165, 255),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_soda_kem_c390ml": {
        "name": "Mirinda Soda Kem 390ml",
        "sugar_per_100ml": 11.1,
        "calories_per_100ml": 44,
        "caffeine_per_100ml": 0,
        "typical_volume": 390,
        "color": (0, 165, 255),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_viet_quat_320ml": {
        "name": "Mirinda Viet Quat 320ml",
        "sugar_per_100ml": 11.2,
        "calories_per_100ml": 45,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (255, 140, 0),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_viet_quat_c390ml": {
        "name": "Mirinda Viet Quat 390ml",
        "sugar_per_100ml": 10.9,
        "calories_per_100ml": 44,
        "caffeine_per_100ml": 0,
        "typical_volume": 390,
        "color": (255, 140, 0),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_xa_xi_320ml": {
        "name": "Mirinda Xa Xi 320ml",
        "sugar_per_100ml": 11.3,
        "calories_per_100ml": 46,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (255, 140, 0),
        "health_warning": "High sugar, no caffeine"
    },
    "mirinda_xa_xi_c390ml": {
        "name": "Mirinda Xa Xi 390ml",
        "sugar_per_100ml": 11.0,
        "calories_per_100ml": 45,
        "caffeine_per_100ml": 0,
        "typical_volume": 390,
        "color": (255, 140, 0),
        "health_warning": "High sugar, no caffeine"
    },
    "olong_tea_plus_320ml": {
        "name": "Olong Tea Plus 320ml",
        "sugar_per_100ml": 7.0,
        "calories_per_100ml": 30,
        "caffeine_per_100ml": 8.0,
        "typical_volume": 320,
        "color": (255, 165, 0),
        "health_warning": "Moderate sugar, contains caffeine"
    },
    "olong_tea_plus_c1l": {
        "name": "Olong Tea Plus 1L",
        "sugar_per_100ml": 6.5,
        "calories_per_100ml": 28,
        "caffeine_per_100ml": 7.5,
        "typical_volume": 1000,
        "color": (255, 165, 0),
        "health_warning": "Moderate sugar, contains caffeine"
    },
    "olong_tea_plus_c450ml": {
        "name": "Olong Tea Plus 450ml",
        "sugar_per_100ml": 7.3,
        "calories_per_100ml": 31,
        "caffeine_per_100ml": 8.2,
        "typical_volume": 450,
        "color": (255, 165, 0),
        "health_warning": "Moderate sugar, contains caffeine"
    },
    "olong_tea_plus_chanh_c450ml": {
        "name": "Olong Tea Plus Chan 450ml",
        "sugar_per_100ml": 7.5,
        "calories_per_100ml": 32,
        "caffeine_per_100ml": 8.5,
        "typical_volume": 450,
        "color": (255, 165, 0),
        "health_warning": "Moderate sugar, contains caffeine"
    },
    "olong_tea_plus_zero_c450ml": {
        "name": "Olong Tea Plus Zero 450ml",
        "sugar_per_100ml": 0,
        "calories_per_100ml": 0,
        "caffeine_per_100ml": 9.0,
        "typical_volume": 450,
        "color": (255, 165, 0),
        "health_warning": "Zero sugar, contains caffeine"
    },
    "pepsi_320ml": {
        "name": "Pepsi 320ml",
        "sugar_per_100ml": 11.0,
        "calories_per_100ml": 42,
        "caffeine_per_100ml": 10.4,
        "typical_volume": 320,
        "color": (255, 0, 0),
        "health_warning": "High sugar content"
    },
    "pepsi_c1_5l": {
        "name": "Pepsi 1.5L",
        "sugar_per_100ml": 10.9,
        "calories_per_100ml": 41,
        "caffeine_per_100ml": 10.2,
        "typical_volume": 1500,
        "color": (255, 0, 0),
        "health_warning": "High sugar content"
    },
    "pepsi_c390ml": {
        "name": "Pepsi 390ml",
        "sugar_per_100ml": 11.1,
        "calories_per_100ml": 42,
        "caffeine_per_100ml": 10.4,
        "typical_volume": 390,
        "color": (255, 0, 0),
        "health_warning": "High sugar content"
    },
    "pepsi_zero_320ml": {
        "name": "Pepsi Zero 320ml",
        "sugar_per_100ml": 0,
        "calories_per_100ml": 0,
        "caffeine_per_100ml": 10.5,
        "typical_volume": 320,
        "color": (0, 0, 0),
        "health_warning": "Zero sugar"
    },
    "pepsi_zero_chanh_320ml": {
        "name": "Pepsi Zero Chan 320ml",
        "sugar_per_100ml": 0,
        "calories_per_100ml": 0,
        "caffeine_per_100ml": 10.4,
        "typical_volume": 320,
        "color": (0, 0, 0),
        "health_warning": "Zero sugar"
    },
    "pepsi_zero_chanh_c390ml": {
        "name": "Pepsi Zero Chan 390ml",
        "sugar_per_100ml": 0,
        "calories_per_100ml": 0,
        "caffeine_per_100ml": 10.5,
        "typical_volume": 390,
        "color": (0, 0, 0),
        "health_warning": "Zero sugar"
    },
    "rockstar_250ml": {
        "name": "Rockstar 250ml",
        "sugar_per_100ml": 11.5,
        "calories_per_100ml": 45,
        "caffeine_per_100ml": 32.0,
        "typical_volume": 250,
        "color": (255, 215, 0),
        "health_warning": "High caffeine"
    },
    "sting_do_320ml": {
        "name": "Sting Do 320ml",
        "sugar_per_100ml": 12.0,
        "calories_per_100ml": 48,
        "caffeine_per_100ml": 30.0,
        "typical_volume": 320,
        "color": (255, 0, 255),
        "health_warning": "High caffeine & sugar"
    },
    "sting_do_c330ml": {
        "name": "Sting Do 330ml",
        "sugar_per_100ml": 12.1,
        "calories_per_100ml": 49,
        "caffeine_per_100ml": 31.0,
        "typical_volume": 330,
        "color": (255, 0, 255),
        "health_warning": "High caffeine & sugar"
    },
    "sting_vang_320ml": {
        "name": "Sting Vang 320ml",
        "sugar_per_100ml": 12.3,
        "calories_per_100ml": 50,
        "caffeine_per_100ml": 30.0,
        "typical_volume": 320,
        "color": (255, 0, 255),
        "health_warning": "High caffeine & sugar"
    },
    "sting_vang_c330ml": {
        "name": "Sting Vang 330ml",
        "sugar_per_100ml": 12.5,
        "calories_per_100ml": 52,
        "caffeine_per_100ml": 31.0,
        "typical_volume": 330,
        "color": (255, 0, 255),
        "health_warning": "High caffeine & sugar"
    },
    "twister_c450ml": {
        "name": "Twister 450ml",
        "sugar_per_100ml": 9.0,
        "calories_per_100ml": 36,
        "caffeine_per_100ml": 0,
        "typical_volume": 450,
        "color": (0, 255, 255),
        "health_warning": "Moderate sugar content"
    },
    "twister_cam_320ml": {
        "name": "Twister Cam 320ml",
        "sugar_per_100ml": 9.3,
        "calories_per_100ml": 37,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (0, 255, 255),
        "health_warning": "Moderate sugar content"
    },
    "twister_cam_c1l": {
        "name": "Twister Cam 1L",
        "sugar_per_100ml": 9.0,
        "calories_per_100ml": 36,
        "caffeine_per_100ml": 0,
        "typical_volume": 1000,
        "color": (0, 255, 255),
        "health_warning": "Moderate sugar content"
    },
    "twister_cam_c320ml": {
        "name": "Twister Cam 320ml",
        "sugar_per_100ml": 9.4,
        "calories_per_100ml": 38,
        "caffeine_per_100ml": 0,
        "typical_volume": 320,
        "color": (0, 255, 255),
        "health_warning": "Moderate sugar content"
    },
    "twister_sua_cam_c290ml": {
        "name": "Twister Sua Cam 290ml",
        "sugar_per_100ml": 8.5,
        "calories_per_100ml": 34,
        "caffeine_per_100ml": 0,
        "typical_volume": 290,
        "color": (255, 255, 0),
        "health_warning": "Moderate sugar content"
    },
    "twister_sua_dau_c290ml": {
        "name": "Twister Sua Dau 290ml",
        "sugar_per_100ml": 8.7,
        "calories_per_100ml": 35,
        "caffeine_per_100ml": 0,
        "typical_volume": 290,
        "color": (255, 255, 0),
        "health_warning": "Moderate sugar content"
    }
}

class StreamlitBeverageDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
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

    def get_nutrition_info(self, class_name, volume=None):
        clean_name = class_name.lower().strip()
        if clean_name in NUTRITION_DATABASE:
            data = NUTRITION_DATABASE[clean_name]
            if volume is None:
                volume = data['typical_volume']
            sugar_g = round((data['sugar_per_100ml'] * volume) / 100, 1)
            sugar_grams_daily_limit = 25
            comparison_message = ""
            if sugar_g <= sugar_grams_daily_limit:
                comparison_message = f"✅ Within daily sugar limit"
            else:
                excess = sugar_g - sugar_grams_daily_limit
                comparison_message = f"⚠️ Exceeds daily limit by {excess}g"
            return {
                "name": data['name'],
                "volume_ml": volume,
                "total_sugar_g": sugar_g,
                "total_calories": round((data['calories_per_100ml'] * volume) / 100),
                "total_caffeine_mg": round((data['caffeine_per_100ml'] * volume) / 100, 1),
                "health_warning": data.get('health_warning', ''),
                "comparison_message": comparison_message,
                "sugar_per_100ml": data['sugar_per_100ml'],
                "calories_per_100ml": data['calories_per_100ml'],
                "caffeine_per_100ml": data['caffeine_per_100ml']
            }
        return None

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
                    nutrition = self.get_nutrition_info(class_name)
                    if nutrition:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_name': class_name,
                            'nutrition': nutrition
                        })
        return detections