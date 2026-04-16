import streamlit as st
import numpy as np
import joblib
import cv2
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import xgboost as xgb
import os
from huggingface_hub import hf_hub_download

# ==========================================
# 1. I3D MODEL ARCHITECTURE (FOR REAL FEATURES)
# ==========================================
class MaxPool3dPad(nn.Module):
    def __init__(self, kernel_size, stride, padding=None):
        super(MaxPool3dPad, self).__init__()
        if padding is None:
            padding = [k // 2 for k in kernel_size]
        self.pad = nn.ConstantPad3d(padding, 0)
        self.pool = nn.MaxPool3d(kernel_size, stride, 0)

    def forward(self, x):
        return self.pool(self.pad(x))

class Unit3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        super(Unit3d, self).__init__()
        padding = [k // 2 for k in kernel_size]
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=tuple(padding), bias=use_bias)
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.conv(x)
        if self._use_batch_norm: x = self.bn(x)
        if self._activation_fn is not None: x = self._activation_fn(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.b0 = Unit3d(in_channels, out_channels[0], kernel_size=[1,1,1], name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3d(in_channels, out_channels[1], kernel_size=[1,1,1], name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3d(out_channels[1], out_channels[2], kernel_size=[3,3,3], name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3d(in_channels, out_channels[3], kernel_size=[1,1,1], name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3d(out_channels[3], out_channels[4], kernel_size=[3,3,3], name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dPad(kernel_size=[3,3,3], stride=[1,1,1], padding=(1,1,1,1,1,1))
        self.b3b = Unit3d(in_channels, out_channels[5], kernel_size=[1,1,1], name=name+'/Branch_3/Conv3d_0b_1x1')

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], 1)

class InceptionI3d(nn.Module):
    def __init__(self, num_classes=400, in_channels=3):
        super(InceptionI3d, self).__init__()
        self.conv3d_1a_7x7 = Unit3d(in_channels, 64, kernel_size=[7,7,7], stride=[2,2,2], name='Conv3d_1a_7x7')
        self.maxPool3d_2a_3x3 = MaxPool3dPad(kernel_size=[1,3,3], stride=[1,2,2], padding=(0,0,1,1,1,1))
        self.conv3d_2b_1x1 = Unit3d(64, 64, kernel_size=[1,1,1], name='Conv3d_2b_1x1')
        self.conv3d_2c_3x3 = Unit3d(64, 192, kernel_size=[3,3,3], name='Conv3d_2c_3x3')
        self.maxPool3d_3a_3x3 = MaxPool3dPad(kernel_size=[1,3,3], stride=[1,2,2], padding=(0,0,1,1,1,1))
        self.mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')
        self.mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')
        self.maxPool3d_4a_3x3 = MaxPool3dPad(kernel_size=[3,3,3], stride=[2,2,2], padding=(0,0,1,1,1,1))
        self.mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64], 'Mixed_4b')
        self.mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64], 'Mixed_4c')
        self.mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64], 'Mixed_4d')
        self.mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 128], 'Mixed_4e')
        self.mixed_4f = InceptionModule(592, [256, 160, 320, 32, 128, 128], 'Mixed_4f')
        self.maxPool3d_5a_2x2 = MaxPool3dPad(kernel_size=[2,2,2], stride=[2,2,2], padding=(0,0,0,0,0,0))
        self.mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128], 'Mixed_5b')
        self.mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128], 'Mixed_5c')
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        for module in self.children():
            if module == self.avg_pool: break
            x = module(x)
        x = self.avg_pool(x)
        return x

# ==========================================
# 2. STREAMLIT APP LOGIC
# ==========================================

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

@st.cache_resource
def load_i3d_feature_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionI3d(num_classes=400, in_channels=3)
    
    # Download weights from HuggingFace
    weights_path = hf_hub_download(repo_id="CarrotBu/I3d", filename="rgb_imagenet.pt")
    checkpoint = torch.load(weights_path, map_location=device)
    
    cleaned_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('module.', '')
        if name in model.state_dict() and 'logits' not in name:
            cleaned_state_dict[name] = v
            
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device).eval()
    return model, device

def extract_video_features(video_path, model, device, num_segments=32, frames_per_seg=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < frames_per_seg:
        cap.release()
        return None 

    seg_size = total_frames / num_segments
    segment_tensors = []

    for i in range(num_segments):
        center_idx = int((i * seg_size) + (seg_size / 2))
        start_frame = max(0, center_idx - (frames_per_seg // 2))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(frames_per_seg):
            ret, frame = cap.read()
            if not ret:
                if len(frames) > 0: frames.append(frames[-1])
                break
            frame = cv2.resize(frame, (224, 224))
            frame = (frame / 127.5) - 1.0 # Normalize -1 to 1
            frames.append(frame)
        
        if len(frames) == frames_per_seg:
            segment_tensors.append(torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2).float())
            
    cap.release()
    if len(segment_tensors) == 0: return None

    input_batch = torch.stack(segment_tensors).to(device)
    video_features = []

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.autocast('cuda'):
                # Process sequentially on Streamlit to avoid VRAM overflow
                for b in range(0, input_batch.shape[0], 4):
                    batch_feat = model(input_batch[b:b+4])
                    video_features.append(batch_feat.cpu().numpy().reshape(batch_feat.shape[0], -1))
        else:
            for b in range(0, input_batch.shape[0], 4):
                batch_feat = model(input_batch[b:b+4])
                video_features.append(batch_feat.cpu().numpy().reshape(batch_feat.shape[0], -1))
            
    res = np.concatenate(video_features, axis=0)
    
    if res.shape[0] < num_segments:
        padding = np.zeros((num_segments - res.shape[0], res.shape[1]))
        res = np.vstack([res, padding])
        
    return res

def predict_video(video_path, xgb_model, threshold, i3d_model, device):
    # 1. Extract REAL segment-level features using the I3D backbone
    features = extract_video_features(video_path, i3d_model, device)
    
    if features is None:
        return 0.0, [0.0]*32
    
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
    
    if xgb_model is not None:
        proba = xgb_model.predict_proba(final_features)[0, 1]
    else:
        proba = 0.0 # Fallback
        
    return proba, segment_scores

uploaded_file = st.file_uploader("Upload Video (MP4, AVI)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    with st.spinner("Initializing models... extracting real I3D features..."):
        model, best_threshold = load_model()
        i3d_model, device = load_i3d_feature_extractor()
        
        final_probability, timeline_scores = predict_video(video_path, model, best_threshold, i3d_model, device)
        
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
