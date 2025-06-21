import streamlit as st
import cv2
import torch
import os
import numpy as np
from datetime import datetime
from time import sleep
from collections import Counter
from PIL import Image
from ultralytics import YOLO
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import requests
from pathlib import Path
import io
import base64

# Sample images for testing (base64 encoded or you can upload them to your app)


import streamlit as st

# List of raw image URLs from GitHub
# List of raw image URLs from GitHub
SAMPLE_IMAGES = {
    "7up.jpg": "https://raw.githubusercontent.com/akashgoud2121/Beverage-Detection-App-With-Nutriotinal-Facts/main/Sample_Images/7up.jpg",
    "Mirinda.jpg": "https://raw.githubusercontent.com/akashgoud2121/Beverage-Detection-App-With-Nutriotinal-Facts/main/Sample_Images/Mirinda.jpg",
    "Pepsi_normal.jpg": "https://raw.githubusercontent.com/akashgoud2121/Beverage-Detection-App-With-Nutriotinal-Facts/main/Sample_Images/Pepsi_normal.jpg",
    "Pepsi_tin_zero_soda.jpg": "https://raw.githubusercontent.com/akashgoud2121/Beverage-Detection-App-With-Nutriotinal-Facts/main/Sample_Images/Pepsi_tin_zero_soda.jpg",
    "Sting.jpg": "https://raw.githubusercontent.com/akashgoud2121/Beverage-Detection-App-With-Nutriotinal-Facts/main/Sample_Images/Sting.jpg",
    "Tea+.jpg": "https://raw.githubusercontent.com/akashgoud2121/Beverage-Detection-App-With-Nutriotinal-Facts/main/Sample_Images/Tea+.jpg",
    "Twister_juice.jpg": "https://raw.githubusercontent.com/akashgoud2121/Beverage-Detection-App-With-Nutriotinal-Facts/main/Sample_Images/Twister_juice.jpg"
}


class AutoModelLoader:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.default_model_url = "https://drive.google.com/uc?export=download&id=10APdf_86N9JLIlpz1vvVL69suHV68VeY"
        self.model_filename = "model_final.pt"

    @st.cache_data
    def get_default_model(_self):
        model_path = _self.models_dir / _self.model_filename
        if not model_path.exists():
            st.info("🤖 Downloading default model... This will only happen once!")
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(_self.default_model_url, stream=True, headers=headers, timeout=60)
                response.raise_for_status()
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    st.error("❌ Invalid model URL - received HTML instead of model file")
                    return None
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress_bar.progress(min(downloaded / total_size, 1.0))
                progress_bar.empty()
                st.success("✅ Default model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {str(e)}")
                if model_path.exists():
                    model_path.unlink()
                return None
        return str(model_path)

    def validate_model_file(self, model_path):
        try:
            if os.path.getsize(model_path) < 1000:
                return False
            torch.load(model_path, map_location='cpu')
            return True
        except Exception:
            return False

# Page configuration
st.set_page_config(
    page_title="🥤 Beverage Detection App With Nutritional Facts",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #260404;
        border-left: 5px solid #FF6B6B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #00420C;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #070FF0;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Fixed sample image gallery styling */
    .sample-gallery {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .sample-image-container {
        flex: 1;
        min-width: 120px;
        max-width: 150px;
        text-align: center;
    }
    
    .sample-image {
        width: 100%;
        height: 100px;
        object-fit: cover;
        border: 2px solid #ddd;
        border-radius: 8px;
        cursor: pointer;
        transition: border-color 0.3s, transform 0.2s;
    }
    
    .sample-image:hover {
        border-color: #4ECDC4;
        transform: scale(1.05);
    }
    
    .sample-button {
        width: 100%;
        margin-top: 5px;
        padding: 5px;
        font-size: 0.8rem;
    }

    /* Dark mode overrides */
    body[data-theme="dark"] .info-box {
        background-color: #1e3a8a;
        color: #e0f2fe;
        border-left: 5px solid #60a5fa;
    }

    body[data-theme="dark"] .warning-box {
        background-color: #7f1d1d;
        color: #fca5a5;
        border-left: 5px solid #ef4444;
    }

    body[data-theme="dark"] .success-box {
        background-color: #14532d;
        color: #bbf7d0;
        border-left: 5px solid #22c55e;
    }

    body[data-theme="dark"] .main-header {
        color: #e5e5e5;
    }

    body[data-theme="dark"] ul {
        color: #e5e5e5;
    }

    body[data-theme="dark"] .info-box ul li {
        color: #e5e5e5;
        font-weight: normal;
    }
</style>
""", unsafe_allow_html=True)


# Nutrition Database (same as your original)
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

# Ensure session state is initialized
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None
if 'sample_name' not in st.session_state:
    st.session_state.sample_name = None


class StreamlitBeverageDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Fixed: Better error handling and validation
                self.model = YOLO(self.model_path)
                # Test the model with a dummy prediction to ensure it works
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy_image, verbose=False)
                return True
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                # Additional debugging info
                if "invalid load key" in str(e):
                    st.error("The model file appears to be corrupted or not a valid PyTorch model file.")
                return False
        return False
    
    def get_nutrition_info(self, class_name, volume=None):
    # Search for the class_name in the NUTRITION_DATABASE
        beverage = NUTRITION_DATABASE.get(class_name.lower().replace(" ", "_"), None)
        
        if beverage:
            # Return the nutrition info for the found beverage
            return {
                "name": beverage["name"],
                "volume_ml": beverage["typical_volume"],
                "total_sugar_g": (beverage["sugar_per_100ml"] * beverage["typical_volume"]) / 100,
                "total_calories": (beverage["calories_per_100ml"] * beverage["typical_volume"]) / 100,
                "total_caffeine_mg": (beverage["caffeine_per_100ml"] * beverage["typical_volume"]) / 100,
                "health_warning": beverage["health_warning"],
                "comparison_message": "⚠️ Exceeds daily limit" if (beverage["sugar_per_100ml"] * beverage["typical_volume"]) / 100 > 25 else "",
                "sugar_per_100ml": beverage["sugar_per_100ml"],
                "calories_per_100ml": beverage["calories_per_100ml"],
                "caffeine_per_100ml": beverage["caffeine_per_100ml"]
            }
        else:
            # If not found, return None or placeholder
            return None

    def detect_beverages(self, image):
        if self.model is None:
            return []
        
        try:
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
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
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_name': class_name,
                                'nutrition': nutrition
                            })
            
            return detections
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            return []

def create_sample_image_gallery():
    """Create a compact sample image gallery"""
    st.markdown("### 🖼️ Sample Images for Testing")
    
    if not SAMPLE_IMAGES:
        st.info("No sample images found.")
        return
    
    with st.expander("Click to view sample images", expanded=False):
        st.markdown("Select any sample image below to test the scanner:")
        
        # Create columns for compact layout
        cols = st.columns(4)  # Reduced to 4 columns for better fit
        
        for i, (name, url) in enumerate(SAMPLE_IMAGES.items()):
            with cols[i % 4]:
                try:
                    st.image(url, caption=name, width=150)
                    if st.button(f"Use {name}", key=f"sample_{i}"):
                        # Set the selected image as the session state image
                        st.session_state.selected_sample = url
                        st.session_state.sample_name = name
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading {name}: {str(e)}")

def main():
    # Initialize session state
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None

    # Header
    st.markdown('<h1 class="main-header">🥤 Beverage Detection App With Nutritional Facts</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover what\'s really in your drink! 🔍</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Controls")
        auto_loader = AutoModelLoader()
        st.markdown("### AI Model Setup")
        if st.button("🚀 Quick Start (Use Default Model)"):
            with st.spinner("Downloading model..."):
                default_model_path = auto_loader.get_default_model()
                if default_model_path and auto_loader.validate_model_file(default_model_path):
                    st.session_state.model_path = default_model_path
                    st.success("✅ Ready to scan beverages!")
                    st.rerun()
                else:
                    st.error("❌ Failed to download or validate model")
        st.markdown("**Or upload your own model:**")
        uploaded_model = st.file_uploader("Upload YOLO model (.pt file)", type=['pt'])
        if uploaded_model:
            model_path = f"temp_model_{uploaded_model.name}"
            with open(model_path, "wb") as f:
                f.write(uploaded_model.read())
            if auto_loader.validate_model_file(model_path):
                st.session_state.model_path = model_path
                st.success("✅ Custom model uploaded successfully!")
            else:
                st.error("❌ Invalid model file uploaded")
                os.remove(model_path)
        st.markdown("### Detection Settings")
        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.6, 0.1)
        show_health_info = st.checkbox("Show Health Warnings", True)
        show_comparisons = st.checkbox("Show Health Comparisons", True)
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if len(NUTRITION_DATABASE) > 0:
            st.metric("Beverages in Database", len(NUTRITION_DATABASE))
            avg_sugar = np.mean([v['sugar_per_100ml'] for v in NUTRITION_DATABASE.values()])
            st.metric("Avg Sugar (per 100ml)", f"{avg_sugar:.1f}g")

    model_path = st.session_state.get('model_path', None)
    detector = StreamlitBeverageDetector(model_path, confidence)
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Scan Your Drink", "📊 Health Dashboard", "🧠 Learn More", "📈 Compare Beverages"])

    with tab1:
        if st.session_state.get('model_path'):
            st.markdown("## Upload Your Beverage Photo")
            create_sample_image_gallery()
            st.markdown("---")
            st.markdown("## Or Upload Your Own Image")
        else:
            st.info("👈 Click 'Quick Start' in the sidebar to begin!")
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📝 Tips for Best Results:</h4>
            <ul>
            <li>Take a clear, well-lit photo</li>
            <li>Show the beverage label clearly</li>
            <li>Avoid shadows and reflections</li>
            <li>Hold the camera steady</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        with col1:
            image = None
            image_source = ""
            if st.session_state.selected_sample is not None:
                sample = st.session_state.selected_sample
                # If it's a string (URL or path), load as image
                if isinstance(sample, str):
                    if sample.startswith("http"):
                        image = Image.open(requests.get(sample, stream=True).raw)
                    else:
                        image = Image.open(sample)
                else:
                    image = sample
                image_source = f"Sample: {st.session_state.get('sample_name', 'Unknown')}"
                st.info(f"Using sample image: {st.session_state.get('sample_name', 'Unknown')}")
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a photo of your beverage"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_source = "Uploaded file"
                # Clear sample selection when new file is uploaded
                st.session_state.selected_sample = None
            if image is not None:
                # Fixed: Remove use_container_width parameter
                st.image(image, caption=f"Your image ({image_source})")

                # Use the model_path variable we defined earlier
                if model_path and detector.load_model():
                    with st.spinner("🔍 Analyzing your beverage..."):
                        # Convert PIL to OpenCV format
                        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        detections = detector.detect_beverages(opencv_image)
                    
                    if detections:
                        st.success(f"🎉 Found {len(detections)} beverage(s)!")
                        
                        for i, detection in enumerate(detections):
                            nutrition = detection['nutrition']
                            confidence = detection['confidence']
                            
                            st.markdown(f"### 🥤 Detection #{i+1}: {nutrition['name']}")
                            
                            # Create metrics columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Confidence", f"{confidence:.1%}")
                            with col2:
                                st.metric("Sugar", f"{nutrition['total_sugar_g']}g")
                            with col3:
                                st.metric("Calories", f"{nutrition['total_calories']}")
                            with col4:
                                st.metric("Caffeine", f"{nutrition['total_caffeine_mg']}mg")
                            
                            # Health warnings
                            if show_health_info:
                                if nutrition['total_sugar_g'] > 25:
                                    st.markdown(f"""
                                    <div class="warning-box">
                                    <h4>⚠️ High Sugar Alert!</h4>
                                    <p>This beverage contains <strong>{nutrition['total_sugar_g']}g</strong> of sugar, 
                                    which exceeds the WHO daily recommendation of 25g.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif nutrition['total_sugar_g'] == 0:
                                    st.markdown(f"""
                                    <div class="success-box">
                                    <h4>✅ Sugar-Free Choice!</h4>
                                    <p>Great choice! This beverage contains no sugar.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            if show_comparisons:
                                st.markdown(f"**Health Comparison:** {nutrition['comparison_message']}")
                            

                            # Create a nutrition chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=['Sugar (g)', 'Calories', 'Caffeine (mg)'],
                                y=[nutrition['total_sugar_g'], nutrition['total_calories'], nutrition['total_caffeine_mg']],
                                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                            ))
                            fig.update_layout(
                                title=f"Nutritional Content - {nutrition['name']}",
                                yaxis_title="Amount",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("🤔 No beverages detected. Try uploading a clearer image or check if your beverage is in our database.")
                else:
                    st.error("❌ Please upload a valid YOLO model file (.pt) in the sidebar to start detection.")
    
    with tab2:
        st.markdown("## 📊 Health Dashboard")
        
        if len(NUTRITION_DATABASE) > 0:
            # Create DataFrame for analysis
            df_data = []
            for key, value in NUTRITION_DATABASE.items():
                df_data.append({
                    'Name': value['name'],
                    'Sugar (g/100ml)': value['sugar_per_100ml'],
                    'Calories (per 100ml)': value['calories_per_100ml'],
                    'Caffeine (mg/100ml)': value['caffeine_per_100ml'],
                    'Volume (ml)': value['typical_volume']
                })
            
            df = pd.DataFrame(df_data)
            
            # Sugar content distribution
            fig1 = px.histogram(df, x='Sugar (g/100ml)', nbins=15, 
                              title="Sugar Content Distribution Across Beverages")
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Top 10 highest sugar beverages
            top_sugar = df.nlargest(10, 'Sugar (g/100ml)')
            fig2 = px.bar(top_sugar, x='Sugar (g/100ml)', y='Name', orientation='h',
                         title="Top 10 Highest Sugar Beverages (per 100ml)")
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Caffeine vs Sugar scatter plot
            fig3 = px.scatter(df, x='Sugar (g/100ml)', y='Caffeine (mg/100ml)', 
                             hover_name='Name', title="Caffeine vs Sugar Content")
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.markdown("## 🧠 Learn About Beverage Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🍬 Sugar Facts
            
            **Daily Recommendations:**
            - Adults: Maximum 25g per day (WHO)
            - Children: Maximum 19g per day
            
            **Health Impacts:**
            - Weight gain and obesity
            - Tooth decay
            - Increased diabetes risk
            - Energy crashes
            
            **Hidden Sugars:**
            Many beverages contain more sugar than you think!
            A 330ml cola can contain up to 39g of sugar.
            """)
        
        with col2:
            st.markdown("""
            ### ☕ Caffeine Facts
            
            **Daily Safe Limits:**
            - Adults: Up to 400mg per day
            - Pregnant women: Up to 200mg per day
            - Teenagers: Up to 100mg per day
            
            **Effects:**
            - Increased alertness
            - Improved focus
            - Can cause jitters if too much
            - May affect sleep
            
            **Sources:**
            Coffee, tea, energy drinks, some sodas
            """)
        
        st.markdown("""
        ### 🌟 Healthy Alternatives
        
        **Instead of sugary drinks, try:**
        - Water with lemon or cucumber
        - Unsweetened tea
        - Sparkling water with natural flavors
        - Fresh fruit juices (in moderation)
        - Coconut water
        """)
    
    with tab4:
        st.markdown("## 📈 Compare Beverages")
        
        if len(NUTRITION_DATABASE) > 0:
            # Beverage selector
            available_beverages = [v['name'] for v in NUTRITION_DATABASE.values()]
            selected_beverages = st.multiselect(
                "Select beverages to compare (up to 5):",
                available_beverages,
                default=available_beverages[:3] if len(available_beverages) >= 3 else available_beverages
            )
            
            if selected_beverages:
                # Create comparison data
                comparison_data = []
                for bev_name in selected_beverages:
                    for key, value in NUTRITION_DATABASE.items():
                        if value['name'] == bev_name:
                            comparison_data.append({
                                'Beverage': bev_name,
                                'Sugar (g/100ml)': value['sugar_per_100ml'],
                                'Calories (per 100ml)': value['calories_per_100ml'],
                                'Caffeine (mg/100ml)': value['caffeine_per_100ml']
                            })
                            break
                
                df_comparison = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.markdown("### 📋 Nutritional Comparison")
                st.dataframe(df_comparison, use_container_width=True)
                
                # Create comparison charts
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Sugar Content', 'Calories', 'Caffeine'),
                    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
                )
                
                fig.add_trace(
                    go.Bar(x=df_comparison['Beverage'], y=df_comparison['Sugar (g/100ml)'], 
                           name='Sugar', marker_color='#FF6B6B'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=df_comparison['Beverage'], y=df_comparison['Calories (per 100ml)'], 
                           name='Calories', marker_color='#4ECDC4'),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(x=df_comparison['Beverage'], y=df_comparison['Caffeine (mg/100ml)'], 
                           name='Caffeine', marker_color='#45B7D1'),
                    row=1, col=3
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
    <p>🥤 Beverage Detection App With Nutritional Facts - Make informed choices about what you drink!</p>
    <p>Remember: This tool is for educational purposes. Always consult healthcare professionals for personalized advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
