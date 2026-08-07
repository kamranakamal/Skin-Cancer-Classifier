from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import numpy as np
from utils import transform, session,img_transform, predict

app = FastAPI()



@app.get("/health")
def health():
    return {"health": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()

    # transform image
    image = BytesIO(contents)
    transformed_img = img_transform(image)
    logit, prob, prediction = predict(transformed_img)

    return {
        "Logit": float(logit),
        "Probability of malignant": float(prob),
        "Predicted Class": int(prediction),
        "Predicted Label": "Malignant" if prediction==1 else "Benign"
    }