# AI-Powered Video Surveillance for Anomaly Detection
(Submission for Neural Nexus 2.0 by team Infinity)
Visit our [Streamlit app](https://cctv-anomaly-detection.streamlit.app/)


## Overview
Standard CCTV systems are passive. They record everything but require a human to watch the footage to identify a problem. This project changes that by building a system that analyzes video in real-time to detect unusual events like violence, accidents, or theft. The system does not just flag an event; it ranks the severity on a scale of 1 to 10 and provides a visual timeline of the exact moment the anomaly occurred.

## The Problem
Detecting anomalies in a crowded place (like a metro station) is difficult for a machine for three reasons:
1. **Rarity:** Crime and accidents happen much less often than normal activity, meaning the AI has very little bad data to learn from.
2. **Ambiguity:** A person running could be a thief or someone just trying to catch a train.
3. **Data Volume:** Processing every single pixel of 24/7 high-definition video requires more computing power than most organizations have.

## Technical Approach

### 1. Data Processing (Feature Extraction)
Instead of feeding raw video frames directly into a classifier (which is slow and inefficient), we use a technique called Feature Extraction. We use a pre-trained I3D (Inflated 3D ConvNet) model. 
* Unlike standard image models that look at static pictures, I3D uses 3D convolutions to look at blocks of time. 
* It analyzes the movement between frames to understand the difference between a normal walk and an anomaly.
* We split every video into 32 equal segments and extract a mathematical summary (a 1024-dimensional vector) for each. This shrinks the dataset from 37GB of video to roughly 2GB of high-level features.

### 2. Feature Aggregation
To capture complex temporal dynamics like sudden fast movements or variance in action across frames, we aggregate the 32 segments into a single flattened vector per video. We calculate the Mean, Max, Min, and Standard Deviation across time. This approach ensures we capture peak anomaly spikes as well as the overall movement variance, creating a robust 4096-dimensional representation of each video.

### 3. The Learning Logic (Ensemble Modeling)
We handle the anomaly detection classification using a powerful soft-voting ensemble mechanism. By combining multiple robust models, we compensate for individual weaknesses and improve our detection of rare events. The ensemble consists of:
* **XGBoost:** Highly efficient at handling class imbalance through targeted sample weighting.
* **Histogram Gradient Boosting:** Fast and effective at learning complex, non-linear relationships in the aggregated features.
* **Random Forest:** Provides a stable, low-variance baseline that prevents overfitting.

## System Architecture
* **Input:** Raw CCTV video feed (MP4, AVI, MOV).
* **Backbone:** Inception-I3D (Pre-trained on Kinetics-400) for feature extraction.
* **Classifier:** Soft-Voting Ensemble (XGBoost, HistGradientBoosting, RandomForest).
* **Interface:** Streamlit web application.
* **Output:** Real-time severity score (out of 10) and a timeline graph pinpointing the anomaly.

## Deployment and User Interface
We deployed the solution using a Streamlit application (app.py). The interface is built with a minimal, dark theme design to keep the focus entirely on functionality. 
Users can upload their own video files directly to the web app. The application then processes the video through the I3D pipeline and our trained ensemble model to output:
1. **Anomaly Likelihood Score:** A straightforward score out of 10 indicating the probability of an anomalous event.
2. **Timeline Graph:** A visual chart showing the predicted anomaly probability across the duration of the video, showing exactly when the system thinks the event took place.

## Result and Outputs
* The final ensemble model achieves an F1 score of over 80 percent on the dataset, heavily outperforming baseline attempts.
* The system accurately handles the severe class imbalance inherently found in anomaly detection datasets.
* The web app delivers immediate, actionable intelligence to security personnel without overwhelming them with alerts.
