import cv2
import numpy as np
from backend.image_model import predict_image

def predict_video(path):
    cap = cv2.VideoCapture(path)

    preds = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 10 == 0:
            temp = "temp/frame.jpg"
            cv2.imwrite(temp, frame)

            pred, conf = predict_image(temp)
            preds.append(conf if pred == 1 else (1-conf))

        frame_id += 1

    cap.release()

    if len(preds) == 0:
        return 0, 0

    score = float(np.mean(preds))
    final = 1 if score > 0.5 else 0

    return final, score