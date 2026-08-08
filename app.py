import base64
from io import BytesIO
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
from model.model import CustomCNN, ConvBlock
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from utils import transform, session, img_transform, predict

app = FastAPI()

gradcam_model = CustomCNN()
checkpoint = torch.load('model/best_model.pth', map_location='cpu')
gradcam_model.load_state_dict(checkpoint)
gradcam_model.eval()

gradcam_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.7178, 0.5677, 0.5456],
        std=[0.2214, 0.2109, 0.2252]
    )
])

target_block = cast(ConvBlock, cast(nn.Sequential, gradcam_model.features)[3])
target_layers = [next(iter(cast(nn.Sequential, target_block.block).children()))]


def predict_from_contents(contents: bytes):
    image = BytesIO(contents)
    transformed_img = img_transform(image)
    logit, prob, prediction = predict(transformed_img)
    return logit, prob, prediction


def build_gradcam_html(contents: bytes):
    image = Image.open(BytesIO(contents)).convert('RGB')
    input_tensor = torch.unsqueeze(cast(torch.Tensor, gradcam_transform(image)), 0)

    with torch.inference_mode():
        output = gradcam_model(input_tensor)

    prob = torch.sigmoid(output)
    prediction = int(prob > 0.4)

    cam = GradCAM(
        model=gradcam_model,
        target_layers=target_layers
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    figure = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f'Grad-CAM (Prob={prob.item():.3f})')
    plt.axis('off')

    buffer = BytesIO()
    plt.tight_layout()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(figure)
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"""
    <html>
        <head>
            <title>Grad-CAM Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                .card {{ max-width: 1000px; margin: 0 auto; }}
                img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
                .meta {{ margin-bottom: 1rem; font-size: 1.1rem; }}
            </style>
        </head>
        <body>
            <div class="card">
                <div class="meta">
                    <strong>Prediction:</strong> {'Malignant' if prediction == 1 else 'Benign'}<br>
                    <strong>Probability:</strong> {prob.item():.3f}
                </div>
                <img src="data:image/png;base64,{image_base64}" alt="Grad-CAM visualization" />
            </div>
        </body>
    </html>
    """



@app.get("/health")
def health():
    return {"health": "ok"}


@app.get("/")
def home():
    return {
        "about": "Skin cancer detection api",
        "endpoints":['/health', '/', '/predict', 'predict-gradcam']

    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()

    # transform image
    logit, prob, prediction = predict_from_contents(contents)

    return {
        "Logit": float(logit),
        "Probability of malignant": float(prob),
        "Predicted Class": int(prediction),
        "Predicted Label": "Malignant" if prediction==1 else "Benign"
    }


@app.post("/predict-gradcam", response_class=HTMLResponse)
async def predict_gradcam_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    return build_gradcam_html(contents)