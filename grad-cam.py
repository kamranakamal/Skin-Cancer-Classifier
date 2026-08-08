import torch
import torch.nn as nn
import numpy as np
from model.model import ConvBlock, CustomCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt


model = CustomCNN()
checkpoint = torch.load('model/best_model.pth', map_location='cpu') 
model.load_state_dict(checkpoint)
model.eval();

target_layers = [model.features[-1].block[0]]
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.7178, 0.5677, 0.5456],
        std=[0.2214, 0.2109, 0.2252]
    )
])
image_path = input("Enter Image Path: ")
image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0)

with torch.inference_mode():
    output = model(input_tensor)

prob = torch.sigmoid(output)

prediction = int(prob > 0.4)



cam = GradCAM(
    model=model,
    target_layers=target_layers
)

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=[ClassifierOutputTarget(0)]
)

grayscale_cam = grayscale_cam[0]



rgb_img = np.array(image.resize((224,224))) / 255.0

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)



plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(rgb_img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(visualization)
plt.title(f"Grad-CAM (Prob={prob.item():.3f})")
plt.axis("off")
plt.show()
