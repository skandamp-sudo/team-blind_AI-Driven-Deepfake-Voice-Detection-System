import librosa
import numpy as np

def extract_features(file):
    try:
        y, sr = librosa.load(file, sr=22050)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    except Exception as e:
        print(f"Skipping {file}: {e}")
        return None