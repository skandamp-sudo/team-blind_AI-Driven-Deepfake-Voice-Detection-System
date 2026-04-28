# team-blind_AI-Driven-Deepfake-Voice-Detection-System
AI-powered system to detect deepfake voice attacks using signal analysis and machine learning.


# 🔐 AI-Driven Deepfake Voice Detection System

An AI-powered cybersecurity tool that detects **synthetic (deepfake) audio** used in social engineering attacks such as voice phishing, impersonation, and fraud.

This system analyzes audio signals and identifies patterns that distinguish **human speech from AI-generated voices**.

🎯 Problem
AI voice synthesis has made it possible to:
- Impersonate individuals (CEO fraud)
- Bypass voice authentication systems
- Execute highly convincing phishing attacks

Traditional cybersecurity systems do not address these **human-layer threats**

💡 Solution
This project provides:
- 🎙️ Deepfake voice detection (REAL vs FAKE)
- 📊 Confidence score for predictions
- 🔍 Feature-based audio analysis using MFCC
- ⚡ Fast, lightweight, and demo-ready system

🧠 How It Works
1. Upload an audio file
2. Extract MFCC features using Librosa
3. Classify using a trained Random Forest model
4. Output:
   - Prediction (REAL / FAKE)
   - Confidence score

🏗️ Architecture
1. Audio Input

2. Feature Extraction (MFCC)

3. Machine Learning Model

4. Prediction + Confidence

5. Streamlit Interface


---

## 🛠️ Tech Stack
- Python
- Streamlit
- Librosa
- NumPy
- Scikit-learn

## 📁 Project Structure
deepfake-detector/
 backend/
    detector.py

 frontend/
     app.py

model/
  model.pkl

data/
  real/
  fake/
  train.py
  
README.md


---

## ⚙️ Installation
git clone https://github.com/skandamp-sudo/team=blind_AI-Driven-Deepfake-Voice_Detection-System.git
cd deepfake-detector
pip install -r requirements.txt

## 🔐 Cybersecurity Relevance

This system helps detect:
- Voice phishing (vishing)
- AI-based impersonation attacks
- Synthetic identity fraud

## ⚠️ Limitations
Accuracy depends on training data quality
Works best with clean audio samples
Prototype-level implementation

## 🚀 Future Work
Deep learning models (CNN / LSTM)
Real-time detection
API deployment
Integration with communication systems

## 👤 Author
- SKANDA M P
- VINAY T K
- SUJAN G L
- ADHYA T H
  
**DAYANANDA SAGAR ACADEMY OF TECHNOLOGY AND MANAGEMENT**
**AI & Cybersecurity**
