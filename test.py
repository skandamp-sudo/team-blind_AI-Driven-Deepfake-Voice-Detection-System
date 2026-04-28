import joblib
import librosa
import numpy as np

model = joblib.load("model/audio.pkl")

def extract(file):
    y, sr = librosa.load(file, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

file = "data/real/your_file.wav"  # change this
feat = extract(file)

pred = model.predict(feat)[0]
conf = model.predict_proba(feat)[0]

print("Prediction:", "FAKE" if pred == 1 else "REAL")
print("Confidence:", conf)