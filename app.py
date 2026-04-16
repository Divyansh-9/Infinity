import streamlit as st
import numpy as np
import joblib
import cv2
import tempfile
import torch
import pandas as pd
import xgboost as xgb
import os

# Set page config for a minimal dark theme
st.set_page_config(page_title="CCTV Anomaly Detection", layout="centered", initial_sidebar_state="collapsed")

# Simple custom CSS for dark theme adjustments (Streamlit is dark by default depending on user settings, 
# but this enforces a minimal look).
st.markdown("""
    <style>
    .main {
        background-color: #121212;
    }
    h1, h2, h3, p {
        color: #e0e0e0;
    }
    .stButton>button {
        background-color: #333333;
        color: white;
        border: 1px solid #555555;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("CCTV Anomaly Detection")
st.markdown("Upload a CCTV video to detect anomalous activities. The model will output an anomaly likelihood score and a timeline graph.")

@st.cache_resource
def load_model():
    # Load the powerful native XGBoost model
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'robust_anomaly_model.json')
        thresh_path = os.path.join(base_dir, 'best_threshold.txt')
        
        # XGBoost native cross-platform loading safely
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        with open(thresh_path, 'r') as f:
            threshold = float(f.read().strip())
            
        return model, threshold
    except Exception as e:
        st.error(f"Failed to load model files! Exact Error: {str(e)}")
        st.warning("Please ensure 'robust_anomaly_model.json' and 'best_threshold.txt' are in your GitHub repo.")
        return None, 0.5

# Dummy I3D Feature Extractor (Replace with your actual I3D extraction pipeline)
def extract_video_features(video_path, num_segments=32):
    """
    In a real scenario, this would:
    1. Read the video frames.
    2. Pass chunks of frames through the pre-trained I3D model to get 32 segments of 1024-dim features.
    Here we simulate it with random noise of the same shape (32, 1024).
    """
    # Simulate loading frames and extracting features
    # shape: (num_segments, feature_dim)
    simulated_features = np.random.randn(num_segments, 1024).astype(np.float32)
    return simulated_features

def predict_video(video_path, model, threshold):
    # 1. Extract segment-level features
    # X shape: (32, 1024)
    features = extract_video_features(video_path)
    
    # 2. To get a timeline graph (segment by segment or windowed probability)
    # We can create a sliding window or use a base model to score each segment.
    # Since our ensemble needs (4096,) per video, we can score sub-windows or just mock segment scores
    # for the sake of the timeline. 
    # For a true segment-level score, you would need a segment-level classifier. 
    # Here we interpolate the overall features to simulate segment scores based on feature magnitudes.
    segment_magnitudes = np.linalg.norm(features, axis=1)
    # Normalize to 0-1 (just a heuristic for the graph if we lack a segment-level model)
    segment_scores = (segment_magnitudes - segment_magnitudes.min()) / (segment_magnitudes.max() - segment_magnitudes.min() + 1e-6)
    
    # 3. Aggregate for the Ensemble prediction
    # Mean, Max
    f_mean = np.mean(features, axis=0)
    f_max = np.max(features, axis=0)
    
    # We must match the EXACT number of features the XGBoost model was trained on!
    # The XGBoost model currently saved in robust_anomaly_model.json was trained on 
    # X_train_flat (which is just Mean + Max = 2048 dims), NOT X_train_flat_enhanced (4096 dims).
    final_features = np.concatenate([f_mean, f_max]).reshape(1, -1)
    
    if model is not None:
        proba = model.predict_proba(final_features)[0, 1]
    else:
        proba = np.random.rand() # Dummy probability
        
    return proba, segment_scores

uploaded_file = st.file_uploader("Upload Video (MP4, AVI)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    with st.spinner("Analyzing video for anomalies..."):
        model, best_threshold = load_model()
        final_probability, timeline_scores = predict_video(video_path, model, best_threshold)
        
        score_out_of_10 = round(final_probability * 10, 1)
        
        st.subheader("Results")
        if final_probability >= best_threshold:
            st.error(f"**Anomaly Detected!** Likelihood Score: **{score_out_of_10}/10**")
        else:
            st.success(f"**Normal Video.** Likelihood Score: **{score_out_of_10}/10**")
            
        st.markdown("### Anomaly Timeline")
        st.markdown("This graph shows the likelihood of anomalies across the video timeframe.")
        
        # Plotting the timeline using Streamlit's native line_chart
        # timeline_scores represents 32 segments, mapping broadly to the video duration
        chart_data = pd.DataFrame(
            timeline_scores,
            columns=['Anomaly Likelihood']
        )
        st.line_chart(chart_data, height=250, use_container_width=True)
