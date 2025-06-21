import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# API endpoints - FIXED
API_URL = "https://beverage-detection-backend.onrender.com/detect"  # Only detection endpoint needed

# Page configuration
st.set_page_config(
    page_title="🥤 Smart Beverage Health Scanner",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (same as your original)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
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
        background-color: #E5F7E5;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #0B008A;
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

# Fallback nutrition database in case API is not available
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

def get_nutrition_info(class_name, volume=None):
    clean_name = class_name.lower().strip()
    data = NUTRITION_DATABASE.get(clean_name)
    if data:
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


# Nutrition database for dashboard and comparison (copy from backend or load via API if needed)


def detect_via_api(image: Image.Image):
    """Detect beverages via API with improved error handling and fallback"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)  # Reset buffer position
    files = {"file": ("image.jpg", buffered.getvalue(), "image/jpeg")}
    
    # Try multiple timeout strategies
    timeouts = [45, 60]  # Increased timeouts for slow API
    
    for timeout in timeouts:
        try:
            st.info(f"🔄 Attempting connection (timeout: {timeout}s)...")
            response = requests.post(API_URL, files=files, timeout=timeout)
            if response.status_code == 200:
                st.success("✅ Successfully connected to detection API!")
                return response.json()
            else:
                st.error(f"❌ API error: {response.status_code} - {response.text}")
                break
        except requests.exceptions.Timeout:
            st.warning(f"⏰ Request timed out after {timeout}s...")
            if timeout == timeouts[-1]:  # Last attempt
                st.error("❌ All connection attempts failed due to timeout")
                return create_demo_detection()
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection error - Backend may be down")
            return create_demo_detection()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {str(e)[:100]}...")
            return create_demo_detection()
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)[:100]}...")
            return create_demo_detection()
    
    return []

def create_demo_detection():
    """Create a demo detection result when API is unavailable"""
    st.info("🔄 API unavailable. Showing demo detection results...")
    return [
        {
            "confidence": 0.85,
            "nutrition": {
                "name": "Demo Beverage (API Unavailable)",
                "total_sugar_g": 35.0,
                "total_calories": 140,
                "total_caffeine_mg": 34,
                "comparison_message": "Demo mode - This is higher in sugar than most beverages in our database."
            }
        }
    ]

def create_nutrition_chart(nutrition_data):
    """Create a nutrition visualization chart"""
    fig = go.Figure()
    
    # Add bars for different nutrition components
    categories = ['Sugar (g)', 'Calories', 'Caffeine (mg)']
    values = [nutrition_data['total_sugar_g'], nutrition_data['total_calories'], nutrition_data['total_caffeine_mg']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Nutritional Content",
        xaxis_title="Components",
        yaxis_title="Amount",
        showlegend=False,
        height=400
    )
    
    return fig

def create_health_dashboard():
    """Create a health dashboard with nutrition database insights"""
    if not NUTRITION_DATABASE:
        st.warning("No nutrition database available for dashboard")
        return
    
    # Convert to DataFrame for easier analysis
    df_data = []
    for name, data in NUTRITION_DATABASE.items():
        df_data.append({
            'name': name,
            'sugar_per_100ml': data['sugar_per_100ml'],
            'calories_per_100ml': data['calories_per_100ml'],
            'caffeine_per_100ml': data['caffeine_per_100ml']
        })
    
    df = pd.DataFrame(df_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sugar content comparison
        fig_sugar = px.bar(
            df.sort_values('sugar_per_100ml', ascending=False).head(10),
            x='sugar_per_100ml',
            y='name',
            orientation='h',
            title='Top 10 Beverages by Sugar Content (per 100ml)',
            color='sugar_per_100ml',
            color_continuous_scale='Reds'
        )
        fig_sugar.update_layout(height=500)
        st.plotly_chart(fig_sugar, use_container_width=True)
    
    with col2:
        # Calories vs Sugar scatter plot
        fig_scatter = px.scatter(
            df,
            x='sugar_per_100ml',
            y='calories_per_100ml',
            hover_name='name',
            title='Calories vs Sugar Content',
            labels={'sugar_per_100ml': 'Sugar (g/100ml)', 'calories_per_100ml': 'Calories (per 100ml)'}
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📊 Database Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Beverages", len(df))
    with col2:
        st.metric("Avg Sugar (per 100ml)", f"{df['sugar_per_100ml'].mean():.1f}g")
    with col3:
        st.metric("Avg Calories (per 100ml)", f"{df['calories_per_100ml'].mean():.0f}")
    with col4:
        st.metric("Avg Caffeine (per 100ml)", f"{df['caffeine_per_100ml'].mean():.1f}mg")

def main():
    # Header
    st.markdown('<h1 class="main-header">🥤 Smart Beverage Health Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover what\'s really in your drink! 🔍</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Controls")
        
        # API Status indicator
        st.markdown("### 🌐 Backend Status")
        if st.button("🔄 Check API Status"):
            with st.spinner("Checking backend..."):
                try:
                    # Only check detection API status
                    response = requests.post(API_URL, files={"file": ("test.jpg", b"test", "image/jpeg")}, timeout=10)
                    if response.status_code == 200:
                        st.success("✅ Detection API is responsive")
                    else:
                        st.warning(f"⚠️ Detection API returned status {response.status_code}")
                except requests.exceptions.Timeout:
                    st.error("❌ Detection API timeout - Backend may be slow")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection failed - Backend may be down")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:50]}...")
        
        st.markdown("---")
        st.markdown("### Detection Settings")
        confidence_threshold = st.slider("Detection Confidence (for display only)", 0.1, 1.0, 0.6, 0.1)
        show_health_info = st.checkbox("Show Health Warnings", True)
        show_comparisons = st.checkbox("Show Health Comparisons", True)
        demo_mode = st.checkbox("Use Demo Mode (if API fails)", False)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if len(NUTRITION_DATABASE) > 0:
            st.metric("Beverages in Database", len(NUTRITION_DATABASE))
            avg_sugar = np.mean([v['sugar_per_100ml'] for v in NUTRITION_DATABASE.values()])
            st.metric("Avg Sugar (per 100ml)", f"{avg_sugar:.1f}g")
        else:
            st.warning("Could not load nutrition database")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Scan Your Drink", "📊 Health Dashboard", "🧠 Learn More", "📈 Compare Beverages"])

    with tab1:
        st.markdown("## Upload Your Beverage Photo")
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
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a photo of your beverage"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your uploaded image", use_column_width=True, output_format="JPEG")
                
                st.markdown("""
                <style>
                    img {
                        max-height: 400px !important;
                        object-fit: contain !important;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                with st.spinner("🔍 Analyzing your beverage..."):
                    if demo_mode:
                        st.info("🎭 Demo mode enabled - Using sample detection")
                        detections = create_demo_detection()
                    else:
                        detections = detect_via_api(image)
                
                if detections:
                    st.success(f"🎉 Found {len(detections)} beverage(s)!")
                    
                    for i, detection in enumerate(detections):
                        nutrition = detection['nutrition']
                        confidence = detection['confidence']
                        
                        st.markdown(f"### 🥤 Detection #{i+1}: {nutrition['name']}")
                        
                        # Metrics row
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
                        
                        # Health comparisons
                        if show_comparisons:
                            st.markdown(f"**Health Comparison:** {nutrition.get('comparison_message', 'No comparison available')}")
                        
                        # Nutrition chart
                        fig = create_nutrition_chart(nutrition)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                else:
                    st.warning("No beverages detected. Please try another image.")

    with tab2:
        st.markdown("## 📊 Health Dashboard")
        st.markdown("Explore nutrition insights from our beverage database")
        create_health_dashboard()

    with tab3:
        st.markdown("## 🧠 Learn More About Beverage Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🍬 Sugar Content Guidelines</h4>
            <ul>
            <li><strong>WHO Recommendation:</strong> Max 25g per day</li>
            <li><strong>Low Sugar:</strong> 0-5g per serving</li>
            <li><strong>Medium Sugar:</strong> 5-15g per serving</li>
            <li><strong>High Sugar:</strong> 15g+ per serving</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>☕ Caffeine Guidelines</h4>
            <ul>
            <li><strong>Safe Daily Limit:</strong> 400mg for adults</li>
            <li><strong>Low Caffeine:</strong> 0-50mg</li>
            <li><strong>Medium Caffeine:</strong> 50-100mg</li>
            <li><strong>High Caffeine:</strong> 100mg+</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 💡 Health Tips
        
        **Choose Wisely:**
        - Water is always the best choice for hydration
        - Look for beverages with natural ingredients
        - Check nutrition labels for hidden sugars
        - Consider portion sizes when calculating daily intake
        
        **Moderation is Key:**
        - Enjoy treats occasionally, not daily
        - Balance high-sugar drinks with physical activity
        - Consider sugar-free alternatives when available
        """)

    with tab4:
        st.markdown("## 📈 Compare Beverages")
        
        if NUTRITION_DATABASE:
            # Beverage selection
            col1, col2 = st.columns(2)
            
            beverage_names = list(NUTRITION_DATABASE.keys())
            
            with col1:
                beverage1 = st.selectbox("Select first beverage:", beverage_names, key="bev1")
            with col2:
                beverage2 = st.selectbox("Select second beverage:", beverage_names, key="bev2")
            
            if beverage1 and beverage2 and beverage1 != beverage2:
                # Get nutrition data
                data1 = NUTRITION_DATABASE[beverage1]
                data2 = NUTRITION_DATABASE[beverage2]
                
                # Comparison metrics
                st.markdown("### Nutrition Comparison (per 100ml)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sugar (g)",
                        f"{data1['sugar_per_100ml']}g vs {data2['sugar_per_100ml']}g",
                        delta=data1['sugar_per_100ml'] - data2['sugar_per_100ml']
                    )
                
                with col2:
                    st.metric(
                        "Calories",
                        f"{data1['calories_per_100ml']} vs {data2['calories_per_100ml']}",
                        delta=data1['calories_per_100ml'] - data2['calories_per_100ml']
                    )
                
                with col3:
                    st.metric(
                        "Caffeine (mg)",
                        f"{data1['caffeine_per_100ml']}mg vs {data2['caffeine_per_100ml']}mg",
                        delta=data1['caffeine_per_100ml'] - data2['caffeine_per_100ml']
                    )
                
                # Comparison chart
                comparison_data = pd.DataFrame({
                    'Metric': ['Sugar (g)', 'Calories', 'Caffeine (mg)'],
                    beverage1: [data1['sugar_per_100ml'], data1['calories_per_100ml'], data1['caffeine_per_100ml']],
                    beverage2: [data2['sugar_per_100ml'], data2['calories_per_100ml'], data2['caffeine_per_100ml']]
                })
                
                fig = px.bar(
                    comparison_data,
                    x='Metric',
                    y=[beverage1, beverage2],
                    title=f"Nutrition Comparison: {beverage1} vs {beverage2}",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No nutrition database available for comparison")

if __name__ == "__main__":
    main()