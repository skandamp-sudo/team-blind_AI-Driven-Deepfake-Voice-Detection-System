from fastapi import FastAPI, UploadFile, File
import shutil
import os

from backend.audio_model import predict_audio
from backend.image_model import predict_image
from backend.video_model import predict_video

app = FastAPI()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def save_file(file: UploadFile):
    path = os.path.join(TEMP_DIR, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path


@app.post("/predict/audio")
async def audio(file: UploadFile = File(...)):
    path = save_file(file)
    pred, conf = predict_audio(path)
    return {"prediction": int(pred), "confidence": conf}


@app.post("/predict/image")
async def image(file: UploadFile = File(...)):
    path = save_file(file)
    pred, conf = predict_image(path)
    return {"prediction": int(pred), "confidence": conf}


@app.post("/predict/video")
async def video(file: UploadFile = File(...)):
    path = save_file(file)
    pred, conf = predict_video(path)
    return {"prediction": int(pred), "confidence": conf}