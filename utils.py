from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.7178, 0.5677, 0.5456],
        std=[0.2214, 0.2109, 0.2252]
    )
])

session = ort.InferenceSession("model/model.onnx")

def img_transform(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_array = input_tensor.numpy().astype(np.float32)
    return input_array

def predict(input):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input})
    logit = output[0][0][0]
    prob = 1 / (1 + np.exp(-logit))
    prediction = int(prob>=0.4)
    return logit, prob, prediction








