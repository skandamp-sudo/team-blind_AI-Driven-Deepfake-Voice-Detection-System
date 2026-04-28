import cv2
import numpy as np
from backend.image_model import predict_image

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # sample every 10th frame
        if frame_count % 10 == 0:
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            pred, conf = predict_image(temp_path)

            if pred == 1:
                scores.append(conf)
            else:
                scores.append(1 - conf)

        frame_count += 1

    cap.release()

    if len(scores) == 0:
        return 0, 0.0

    avg_score = np.mean(scores)
    final_pred = 1 if avg_score > 0.5 else 0

    return final_pred, float(avg_score)