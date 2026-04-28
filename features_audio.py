import librosa
import numpy as np
from backend.audio_augment import augment_audio

def extract_features(file):
    y, sr = librosa.load(file, sr=22050)
    y = augment_audio(y, sr)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc, chroma, contrast, tonnetz, zcr])