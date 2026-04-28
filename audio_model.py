import joblib
import librosa
import numpy as np

model = joblib.load("model/audio.pkl")

def predict_audio(file):
    y, sr = librosa.load(file, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat = np.mean(mfcc.T, axis=0).reshape(1, -1)

    pred = model.predict(feat)[0]
    conf = model.predict_proba(feat)[0]

    return int(pred), float(max(conf))