import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend.predict import run_prediction

st.set_page_config(page_title="Digital Shadow", layout="centered")

# --- HEADER ---
st.title("🔍 Digital Shadow")
st.subheader("AI-Powered Deepfake Detection")
st.caption("Multi-Modal Analysis • Audio • Image • Video")

# --- FILE UPLOAD ---
file = st.file_uploader(
    "Upload Audio / Image / Video",
    type=["wav", "mp3", "jpg", "png", "mp4"]
)

if file:
    file_path = f"temp_{file.name}"

    with open(file_path, "wb") as f:
        f.write(file.read())

    # --- DETERMINE TYPE ---
    file_type = None
    if file.name.endswith(("wav", "mp3")):
        file_type = "audio"
    elif file.name.endswith(("jpg", "png")):
        file_type = "image"
    elif file.name.endswith(("mp4")):
        file_type = "video"

    # --- PREVIEW ---
    st.markdown("### Uploaded File Preview")
    if file_type == "image":
        st.image(file, use_column_width=True)
    elif file_type == "audio":
        st.audio(file)
    elif file_type == "video":
        st.video(file)

    # --- ANALYSIS ---
    st.markdown("---")
    with st.spinner("Running AI Analysis..."):
        st.write("🔍 Extracting features...")
        st.write("🧠 Running detection models...")
        st.write("📊 Evaluating confidence...")
        label, conf = run_prediction(file_path, file_type)

    # --- RESULT ---
    if label:
        st.markdown("---")

        if label == "REAL":
            st.success("🟢 Authentic Content Detected")
        elif label == "FAKE":
            st.error("🔴 Manipulated Content Detected")
        else:
            st.warning("🟡 Uncertain Result — Further Analysis Required")

        st.markdown(f"### Confidence Score: {conf:.2f}")
        st.progress(int(conf * 100))

        # --- ANALYSIS REPORT ---
        st.markdown("---")
        st.subheader("AI Analysis Report")

        if file_type == "image":
            st.write("• Visual consistency evaluated across facial regions")
            st.write("• Texture and artifact patterns analyzed")
        elif file_type == "audio":
            st.write("• Spectral features and voice patterns analyzed")
            st.write("• Temporal inconsistencies evaluated")
        elif file_type == "video":
            st.write("• Frame-level anomaly detection applied")
            st.write("• Temporal coherence across frames analyzed")

        if label == "FAKE":
            st.write("• Synthetic manipulation indicators detected")
            st.write("• High likelihood of generated or altered content")
        elif label == "REAL":
            st.write("• Natural signal patterns detected")
            st.write("• No major manipulation indicators found")
        else:
            st.write("• Ambiguous signal patterns detected")
            st.write("• Recommend deeper multi-modal verification")

        # --- FOOTER INSIGHT ---
        st.markdown("---")
        st.caption("Note: High-quality synthetic content may reduce detectable artifacts, requiring multi-modal validation.")