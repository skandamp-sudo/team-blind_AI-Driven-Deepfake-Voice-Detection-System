import librosa
import soundfile as sf
import os

os.makedirs("data/fake", exist_ok=True)

for f in os.listdir("data/real")[:50]:
    if f.endswith(".wav"):
        path = f"data/real/{f}"

        try:
            y, sr = librosa.load(path, sr=22050)

            y_fake = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)

            sf.write(f"data/fake/fake_{f}", y_fake, sr)

            print(f"Generated fake: {f}")

        except Exception as e:
            print(f"Skipping {f}: {e}")