from backend.audio_model import predict_audio
from backend.image_model import predict_image
from backend.video_model import predict_video

def run_prediction(file, file_type):
    try:
        # --- ROUTING ---
        if file_type == "audio":
            pred, conf = predict_audio(file)

        elif file_type == "image":
            pred, conf = predict_image(file)

        elif file_type == "video":
            pred, conf = predict_video(file)

        else:
            return None, None

        # --- VALIDATION ---
        if conf is None:
            return "UNCERTAIN", 0.0

        # --- UNCERTAINTY HANDLING ---
        if conf < 0.6:
            return "UNCERTAIN", float(conf)

        # --- FINAL LABEL ---
        label = "FAKE" if pred == 1 else "REAL"

        return label, float(conf)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "UNCERTAIN", 0.0