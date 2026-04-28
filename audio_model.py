import joblib
from backend.features_audio import extract_features

model = joblib.load("model/audio.pkl")

def predict_audio(path):
    x = extract_features(path).reshape(1, -1)
    prob = model.predict_proba(x)[0]
    pred = model.predict(x)[0]

    return int(pred), float(max(prob))