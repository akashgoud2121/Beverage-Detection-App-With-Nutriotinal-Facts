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
API_URL = "https://beverage-detection-backend.onrender.com/detect"  # Added /detect
NUTRITION_DB_URL = "https://beverage-detection-backend.onrender.com/nutrition_database"  # Added /nutrition_database

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

@st.cache_data
def fetch_nutrition_database():
    try:
        response = requests.get(NUTRITION_DB_URL, timeout=10)  # Added timeout
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Could not fetch nutrition database from backend. Status: {response.status_code}")
            return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend API: {e}")
        return {}
    except Exception as e:
        st.error(f"Error fetching nutrition database: {e}")
        return {}

# Nutrition database for dashboard and comparison (copy from backend or load via API if needed)
NUTRITION_DATABASE = fetch_nutrition_database()

def detect_via_api(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)  # Reset buffer position
    files = {"file": ("image.jpg", buffered.getvalue(), "image/jpeg")}
    try:
        response = requests.post(API_URL, files=files, timeout=30)  # Added timeout
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend API: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

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
        st.markdown("### Detection Settings")
        confidence_threshold = st.slider("Detection Confidence (for display only)", 0.1, 1.0, 0.6, 0.1)
        show_health_info = st.checkbox("Show Health Warnings", True)
        show_comparisons = st.checkbox("Show Health Comparisons", True)
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