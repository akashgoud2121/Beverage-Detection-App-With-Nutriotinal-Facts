import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# API endpoint
API_URL = "https://beverage-detection-backend.onrender.com"  # Change if deploying elsewhere

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
        response = requests.get("https://beverage-detection-backend.onrender.com")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Could not fetch nutrition database from backend.")
            return {}
    except Exception as e:
        st.error(f"Error fetching nutrition database: {e}")
        return {}

# Nutrition database for dashboard and comparison (copy from backend or load via API if needed)
NUTRITION_DATABASE = fetch_nutrition_database()

def detect_via_api(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    files = {"file": ("image.jpg", buffered.getvalue(), "image/jpeg")}
    try:
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("API error: " + response.text)
            return []
    except Exception as e:
        st.error(f"Could not connect to backend API: {e}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🥤 Smart Beverage Health Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover what\'s really in your drink! 🔍</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Controls")
        st.markdown("### Detection Settings")
        confidence = st.slider("Detection Confidence (for display only)", 0.1, 1.0, 0.6, 0.1)
        show_health_info = st.checkbox("Show Health Warnings", True)
        show_comparisons = st.checkbox("Show Health Comparisons", True)
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if len(NUTRITION_DATABASE) > 0:
            st.metric("Beverages in Database", len(NUTRITION_DATABASE))
            avg_sugar = np.mean([v['sugar_per_100ml'] for v in NUTRITION_DATABASE.values()])
            st.metric("Avg Sugar (per 100ml)", f"{avg_sugar:.1f}g")

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
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Confidence", f"{confidence:.1%}")
                        with col2:
                            st.metric("Sugar", f"{nutrition['total_sugar_g']}g")
                        with col3:
                            st.metric("Calories", f"{nutrition['total_calories']}")
                        with col4:
                            st.metric("Caffeine", f"{nutrition['total_caffeine_mg']}mg")
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

    with tab2:
        st.markdown("## 📊 Health Dashboard")
        if len(NUTRITION_DATABASE) > 0:
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
            fig1 = px.histogram(df, x='Sugar (g/100ml)', nbins=15, 
                              title="Sugar Content Distribution Across Beverages")
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            top_sugar = df.nlargest(10, 'Sugar (g/100ml)')
            fig2 = px.bar(top_sugar, x='Sugar (g/100ml)', y='Name', orientation='h',
                         title="Top 10 Highest Sugar Beverages (per 100ml)")
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
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
            available_beverages = [v['name'] for v in NUTRITION_DATABASE.values()]
            selected_beverages = st.multiselect(
                "Select beverages to compare (up to 5):",
                available_beverages,
                default=available_beverages[:3] if len(available_beverages) >= 3 else available_beverages
            )
            if selected_beverages:
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
                st.markdown("### 📋 Nutritional Comparison")
                st.dataframe(df_comparison, use_container_width=True)
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
    <p>🥤 Smart Beverage Health Scanner - Make informed choices about what you drink!</p>
    <p>Remember: This tool is for educational purposes. Always consult healthcare professionals for personalized advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
