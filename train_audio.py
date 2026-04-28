import os, numpy as np, joblib, librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def extract_features(file):
    try:
        y, sr = librosa.load(file, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Skipping {file}: {e}")
        return None

X, y = [], []

for f in os.listdir("data/real"):
    if f.endswith(".wav"):
        path = f"data/real/{f}"
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(0)

for f in os.listdir("data/fake"):
    if f.endswith(".wav"):
        path = f"data/fake/{f}"
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(1)

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

joblib.dump(model, "model/audio.pkl")

print("Audio model trained successfully")