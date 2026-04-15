# AI-Powered Video Surveillance for Anomaly Detection
(Submission for Neural Nexus 2.0 by team Infinity)
## Overview
Standard CCTV systems are passive. They record everything but require a human to watch the footage to identify a problem. This project changes that by building a system that analyzes video in real-time to detect unusual events like violence, accidents, or theft. The system doesn't just flag an event; it ranks the severity on a scale of 1 to 10 and provides a logical explanation for why it triggered an alert.

## The Problem
Detecting anomalies in a crowded place (like a metro station) is difficult for a machine for three reasons:
1. **Rarity:** Crime and accidents happen much less often than "normal" activity, meaning the AI has very little "bad" data to learn from.
2. **Ambiguity:** A person running could be a thief or someone just trying to catch a train.
3. **Data Volume:** Processing every single pixel of 24/7 high-definition video requires more computing power than most organizations have.

## Technical Approach

### 1. Data Processing (Feature Extraction)
Instead of feeding raw video frames directly into a classifier—which is slow and inefficient—we use a technique called **Feature Extraction**. We use a pre-trained **I3D (Inflated 3D ConvNet)** model. 
* Unlike standard image models that look at static pictures, I3D uses 3D convolutions to look at "blocks" of time. 
* It analyzes the movement between frames to understand the difference between a walk and a punch.
* We split every video into 32 equal segments and extract a mathematical summary (a vector) for each. This shrinks the dataset from 37GB of video to roughly 2GB of high-level features.

### 2. The Learning Logic (Multiple Instance Learning)
Because the dataset only tells us that a video "contains a crime" but doesn't tell us exactly which second the crime happens, we use **Multiple Instance Learning (MIL)**.
* We treat a video as a "bag" of segments. 
* If a video is labeled as an anomaly, the model learns that at least one segment in that bag must have a high anomaly score.
* If a video is labeled as normal, it learns that every single segment must have a low score.

### 3. Scoring and Explainability
The system generates a score from 1 to 10. 
* **Scores 1-3:** Routine activity (people walking, standing).
* **Scores 4-7:** Unusual but not necessarily dangerous (running, sudden crowd gathering).
* **Scores 8-10:** High-probability anomalies (physical conflict, vehicle crashes).
The system uses attention weights to identify exactly which segment caused the spike in the score, allowing it to point security staff to the specific moment the event began.

## System Architecture
* **Input:** Raw CCTV video feed.
* **Backbone:** Inception-I3D (Pre-trained on Kinetics-400).
* **Classifier:** A lightweight Temporal MLP (Multi-Layer Perceptron) using Ranking Loss.
* **Output:** Real-time severity score (1-10) and an automated text log explaining the detection.

## Performance Constraints
This project is designed to run on mid-range hardware (like an RTX 4050) by prioritizing "Sparse Sampling." By only analyzing the most important frames in a segment, we achieve real-time performance without needing a massive server farm.
