import librosa
import numpy as np
import random

def augment_audio(y, sr):
    choice = random.choice(["noise", "pitch", "stretch", "none"])

    if choice == "noise":
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise

    elif choice == "pitch":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))

    elif choice == "stretch":
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))

    return y