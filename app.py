import streamlit as st
from backend.audio_model import predict_audio
from backend.image_model import predict_image
from backend.video_model import predict_video

st.title("🔐 AI Deepfake Detection Suite")

mode = st.selectbox("Mode", ["Audio", "Image", "Video"])
file = st.file_uploader("Upload File")

if file:
    path = f"temp/input"

    with open(path, "wb") as f:
        f.write(file.read())

    if mode == "Audio":
        pred, conf = predict_audio(path)

    elif mode == "Image":
        pred, conf = predict_image(path)

    else:
        pred, conf = predict_video(path)

    if pred == 1:
        st.error(f"FAKE ({conf:.2f})")
    else:
        st.success(f"REAL ({conf:.2f})")