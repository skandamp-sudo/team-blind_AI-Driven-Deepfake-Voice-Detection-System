import os
import numpy as np
import joblib
from backend.features_audio import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = [], []

for f in os.listdir("data/real"):
    X.append(extract_features(f"data/real/{f}"))
    y.append(0)

for f in os.listdir("data/fake"):
    X.append(extract_features(f"data/fake/{f}"))
    y.append(1)

for f in os.listdir("data/real"):
    if f.endswith(".wav"):
        X.append(extract_features(f"data/real/{f}"))
        y.append(0)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10
)

model.fit(X_train, y_train)

joblib.dump(model, "model/audio.pkl")